#!/usr/bin/env python3
"""
图像清晰度检测模块
==================

提供两种清晰度检测方法：
1. JPEG编码大小法 - 适用于实时自动对焦，包含抗气泡/黑色区域干扰权重
2. 拉普拉斯算子法 - 适用于景深堆叠，精确像素级清晰度测量

依赖:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def get_anti_noise_clarity_index(cv2_frame, original_sharpness):
    """
    基于原始清晰度进行高亮和黑色区域的权重衰减
    返回调整后的清晰度指标
    
    适用于实时自动对焦场景，基于JPEG编码大小判断清晰度
    
    参数:
        cv2_frame: 输入图像（RGB格式）
        original_sharpness: 原始清晰度值（通常是len(frame)，即JPEG编码大小）
    
    返回:
        调整后的清晰度值
    """
    # 如果原始清晰度太小，直接返回
    if original_sharpness < 1000:  # 阈值可调
        return original_sharpness
    
    # --- 2. 预处理：转换为HSV以检测高亮和黑色区域 ---
    # 缩小图像提升速度
    small_frame = cv2.resize(cv2_frame, (960, 720))
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]  # 饱和度通道
    v_channel = hsv[:, :, 2]  # 亮度通道
    
    # --- 3. 自动化阈值计算（基于统计方法） ---
    # 计算统计信息
    s_mean = np.mean(s_channel)
    s_std = np.std(s_channel)
    s_percentiles = np.percentile(s_channel, [5, 10, 25, 50, 75, 90, 95])
    
    v_mean = np.mean(v_channel)
    v_std = np.std(v_channel)
    v_percentiles = np.percentile(v_channel, [5, 10, 25, 50, 75, 90, 95])
    
    # S阈值：检测低饱和度区域（高亮气泡）
    ret_s, _ = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    T_s_bubble = min(ret_s, s_percentiles[2])  # 25百分位数，低饱和度区域（气泡）
    
    # V下限：检测黑色脏物
    ret_v, _ = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    T_v_black = max(ret_v, v_percentiles[1])  # 10百分位数，黑色区域
    
    # V上限：检测过亮气泡
    T_v_bright = min(v_percentiles[6], v_mean + 2 * v_std)  # 95百分位数或均值+2σ
    
    # --- 4. 生成权重衰减掩码 ---
    # 高亮气泡掩码：低饱和度或过亮
    _, mask_low_s = cv2.threshold(s_channel, T_s_bubble, 255, cv2.THRESH_BINARY_INV)  # 反转，低饱和度区域
    _, mask_high_v = cv2.threshold(v_channel, T_v_bright, 255, cv2.THRESH_BINARY)  # 过亮区域
    mask_bubble = cv2.bitwise_or(mask_low_s, mask_high_v)  # 高亮气泡：低饱和度或过亮
    
    # 黑色脏物掩码：低亮度
    _, mask_black = cv2.threshold(v_channel, T_v_black, 255, cv2.THRESH_BINARY_INV)  # 反转，黑色区域
    
    # --- 5. 计算权重衰减因子 ---
    total_pixels = 960 * 720
    
    # 计算高亮气泡区域占比
    bubble_pixels = np.count_nonzero(mask_bubble)
    bubble_ratio = bubble_pixels / total_pixels
    
    # 计算黑色脏物区域占比
    black_pixels = np.count_nonzero(mask_black)
    black_ratio = black_pixels / total_pixels
    
    # 计算有效区域占比（既不是气泡也不是黑色）
    valid_mask = cv2.bitwise_and(cv2.bitwise_not(mask_bubble), cv2.bitwise_not(mask_black))
    valid_pixels = np.count_nonzero(valid_mask)
    valid_ratio = valid_pixels / total_pixels
    
    # --- 6. 权重衰减计算 ---
    # 高亮气泡权重衰减：气泡区域越多，衰减越大
    # 使用指数衰减：bubble_weight = (1 - bubble_ratio)^alpha
    bubble_alpha = 2.0  # 衰减强度，可调
    bubble_weight = (1.0 - min(bubble_ratio, 0.8)) ** bubble_alpha  # 限制最大衰减到0.2
    
    # 黑色脏物权重衰减：黑色区域越多，衰减越大
    black_alpha = 2.0  # 衰减强度，可调
    black_weight = (1.0 - min(black_ratio, 0.8)) ** black_alpha  # 限制最大衰减到0.2
    
    # 有效区域权重增强：有效区域越多，权重越大
    # 但不要过度增强，保持平衡
    valid_weight = min(1.0 + valid_ratio * 0.5, 1.3)  # 最多增强30%
    
    # 综合权重：高亮和黑色区域的衰减，有效区域的增强
    final_weight = bubble_weight * black_weight * valid_weight
    
    # 限制权重范围，避免过度衰减或增强
    final_weight = max(0.1, min(1.5, final_weight))  # 权重范围：0.1 到 1.5
    
    # --- 7. 返回调整后的清晰度 ---
    adjusted_sharpness = original_sharpness * final_weight
    
    return int(adjusted_sharpness)


class LaplacianFocusMeasure:
    """
    拉普拉斯算子焦点测量类
    
    适用于景深堆叠场景，返回每个像素的清晰度值
    原理：清晰的图像边缘锐利，拉普拉斯算子响应更大
    """
    
    def __init__(self, power: float = 1.5, blur_kernel: int = 5, blur_sigma: float = 1.0):
        """
        初始化拉普拉斯焦点测量器
        
        参数:
            power: 对比度增强指数，默认1.5
            blur_kernel: 高斯模糊核大小，默认5
            blur_sigma: 高斯模糊sigma值，默认1.0
        """
        self.power = power
        self.blur_kernel = (blur_kernel, blur_kernel)
        self.blur_sigma = blur_sigma
    
    def compute(self, img: np.ndarray) -> np.ndarray:
        """
        计算焦点测量图
        
        参数:
            img: 输入灰度图像 (numpy array)
        
        返回:
            焦点测量图 (归一化到0-1的numpy array)
        """
        # 确保是灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 使用拉普拉斯算子
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        focus_measure = np.abs(laplacian)
        
        # 增强对比度，使焦点差异更明显
        if self.power != 1.0:
            focus_measure = np.power(focus_measure, self.power)
        
        # 平滑处理，减少噪声影响
        focus_measure = cv2.GaussianBlur(focus_measure, self.blur_kernel, self.blur_sigma)
        
        # 归一化到0-1范围
        focus_measure = focus_measure / (np.max(focus_measure) + 1e-8)
        
        return focus_measure
    
    def compute_scalar(self, img: np.ndarray) -> float:
        """
        计算单个标量清晰度值
        
        参数:
            img: 输入灰度图像
        
        返回:
            清晰度标量值 (图像均值)
        """
        focus_map = self.compute(img)
        return float(np.mean(focus_map))


class TenengradFocusMeasure:
    """
    Tenengrad (梯度平方和) 焦点测量类
    
    原理：清晰图像有更大的梯度，使用Sobel算子计算梯度
    """
    
    def __init__(self, ksize: int = 3):
        """
        初始化Tenengrad焦点测量器
        
        参数:
            ksize: Sobel算子核大小，默认3
        """
        self.ksize = ksize
    
    def compute(self, img: np.ndarray) -> np.ndarray:
        """
        计算焦点测量图
        
        参数:
            img: 输入灰度图像
        
        返回:
            焦点测量图
        """
        # 确保是灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 计算Sobel梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)
        
        # 梯度平方和
        focus_measure = np.sqrt(sobelx**2 + sobely**2)
        
        # 归一化
        focus_measure = focus_measure / (np.max(focus_measure) + 1e-8)
        
        return focus_measure
    
    def compute_scalar(self, img: np.ndarray) -> float:
        """
        计算单个标量清晰度值
        
        参数:
            img: 输入灰度图像
        
        返回:
            清晰度标量值
        """
        focus_map = self.compute(img)
        return float(np.mean(focus_map))


class VarianceFocusMeasure:
    """
    方差法焦点测量类
    
    原理：清晰的图像灰度变化大，方差也大
    """
    
    def compute_scalar(self, img: np.ndarray) -> float:
        """
        计算标量清晰度值（方差）
        
        参数:
            img: 输入灰度图像
        
        返回:
            清晰度标量值（方差）
        """
        # 确保是灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        return float(np.var(gray))


# ====================== 便捷函数 ======================

def compute_sharpness_jpeg(frame_rgb: np.ndarray, jpeg_size: int) -> int:
    """
    便捷函数：基于JPEG大小的清晰度计算
    
    参数:
        frame_rgb: RGB格式图像
        jpeg_size: JPEG编码后的字节数
    
    返回:
        调整后的清晰度值
    """
    return get_anti_noise_clarity_index(frame_rgb, jpeg_size)


def compute_sharpness_laplacian(img: np.ndarray) -> float:
    """
    便捷函数：基于拉普拉斯的清晰度计算
    
    参数:
        img: BGR格式图像
    
    返回:
        清晰度标量值
    """
    measure = LaplacianFocusMeasure()
    return measure.compute_scalar(img)


def compute_sharpness_tenengrad(img: np.ndarray) -> float:
    """
    便捷函数：基于Tenengrad的清晰度计算
    
    参数:
        img: BGR格式图像
    
    返回:
        清晰度标量值
    """
    measure = TenengradFocusMeasure()
    return measure.compute_scalar(img)


def compute_sharpness_variance(img: np.ndarray) -> float:
    """
    便捷函数：基于方差的清晰度计算
    
    参数:
        img: BGR格式图像
    
    返回:
        清晰度标量值
    """
    measure = VarianceFocusMeasure()
    return measure.compute_scalar(img)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    print("图像清晰度检测模块")
    print("=" * 50)
    print("\n支持的清晰度检测方法:")
    print("1. JPEG编码大小法 - get_anti_noise_clarity_index()")
    print("2. 拉普拉斯算子法 - LaplacianFocusMeasure.compute_scalar()")
    print("3. Tenengrad法    - TenengradFocusMeasure.compute_scalar()")
    print("4. 方差法         - VarianceFocusMeasure.compute_scalar()")
    print("\n使用方法:")
    print("  from focus_measure import get_anti_noise_clarity_index")
    print("  from focus_measure import LaplacianFocusMeasure")
