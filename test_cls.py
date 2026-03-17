import cv2
import torch
import torchvision
import numpy as np
import time
import os
import glob
from collections import Counter
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image, ImageDraw, ImageFont

# ================= ⚙️ 配置区域 =================
# 1. 模型路径
SEG_MODEL_PATH = r'models/22_pollen_50_12/weights/best.pt'      # 分割模型
CLS_MODEL_PATH = r'models/classify/train2/weights/best.pt'         # 分类模型 (Router)

# 2. 输入输出
INPUT_DIR = r'E:\花粉数据集\花粉图册（21种）\2024矮紫杉\矮紫杉40后'
OUTPUT_DIR = r'runs/cascaded_result'

# 3. 推理参数
GLOBAL_IMGSZ = 480       # 直接推理时的尺寸
SLICE_SIZE = 480         # 切片大小
OVERLAP_RATIO = 0.25     # 切片重叠率
CONF_THRES = 0.5         # 置信度阈值
BATCH_SIZE = 16           # 批处理大小

# 4. NMS 参数 (去重用)
MERGE_IOU_THRES = 0.5 
MERGE_IOS_THRES = 0.70 
HARD_IOU_THRES = 0.70 
# =================================================

def nms_area_first(boxes, scores, iou_thres=0.5, ios_thres=0.7, hard_iou_thres=0.7):
    """(保持不变) 你的自定义强力 NMS"""
    if len(boxes) == 0: return []
    np_boxes = boxes.cpu().numpy()
    n = len(np_boxes)
    areas = (np_boxes[:, 2] - np_boxes[:, 0]) * (np_boxes[:, 3] - np_boxes[:, 1])
    sorted_indices = np.argsort(areas)[::-1]
    is_suppressed = np.zeros(n, dtype=bool)
    
    for i in range(n):
        idx_large = sorted_indices[i]
        if is_suppressed[idx_large]: continue
        box_large = np_boxes[idx_large]
        area_large = areas[idx_large]
        
        for j in range(i + 1, n):
            idx_small = sorted_indices[j]
            if is_suppressed[idx_small]: continue
            box_small = np_boxes[idx_small]
            area_small = areas[idx_small]
            
            xx1 = max(box_large[0], box_small[0])
            yy1 = max(box_large[1], box_small[1])
            xx2 = min(box_large[2], box_small[2])
            yy2 = min(box_large[3], box_small[3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            if inter == 0: continue

            ios = inter / area_small
            union = area_large + area_small - inter
            iou = inter / union
            
            should_suppress = False
            if iou > hard_iou_thres: should_suppress = True
            elif ios > ios_thres: should_suppress = True
            elif iou > iou_thres and area_large > 1.2 * area_small: should_suppress = True
            
            if should_suppress: is_suppressed[idx_small] = True

    survivors_indices = [i for i in range(n) if not is_suppressed[i]]
    if not survivors_indices: return torch.tensor([], dtype=torch.long)
    
    survivor_boxes = boxes[survivors_indices]
    survivor_scores = scores[survivors_indices]
    keep_indices = torchvision.ops.nms(survivor_boxes, survivor_scores, iou_thres)
    final_indices = torch.tensor(survivors_indices, device=boxes.device)[keep_indices]
    return final_indices

class CascadedInference:
    def __init__(self, seg_path, cls_path, label_mapping=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 初始化引擎: {self.device}")
        
        # 加载两个模型
        print(f"   - Loading Classifier: {cls_path}")
        self.cls_model = YOLO(cls_path)
        print(f"   - Loading Segmenter: {seg_path}")
        self.seg_model = YOLO(seg_path, task='segment')
        
        # 标签名称映射：短代码 -> 中文名称
        self.label_mapping = label_mapping or {}
        
        # 预热
        self.seg_model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, half=True)

    def get_slices(self, image):
        """生成切片"""
        h, w = image.shape[:2]
        pad_h = max(0, SLICE_SIZE - h)
        pad_w = max(0, SLICE_SIZE - w)
        img_proc = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE) if (pad_h>0 or pad_w>0) else image
            
        stride = int(SLICE_SIZE * (1 - OVERLAP_RATIO))
        slices, offsets = [], []
        h_n, w_n = img_proc.shape[:2]
        
        y_steps = list(range(0, h_n - SLICE_SIZE, stride))
        if (h_n - SLICE_SIZE) not in y_steps: y_steps.append(h_n - SLICE_SIZE)
        x_steps = list(range(0, w_n - SLICE_SIZE, stride))
        if (w_n - SLICE_SIZE) not in x_steps: x_steps.append(w_n - SLICE_SIZE)
        
        if not y_steps: y_steps = [0]
        if not x_steps: x_steps = [0]

        for y in y_steps:
            for x in x_steps:
                slices.append(img_proc[y:y+SLICE_SIZE, x:x+SLICE_SIZE])
                offsets.append((x, y))
        return slices, offsets

    def predict_global(self, img0):
        """直接推理 (用于 200x, 400x)"""
        results = self.seg_model(img0, imgsz=GLOBAL_IMGSZ, conf=CONF_THRES, device=self.device, half=True, verbose=False, retina_masks=True)
        if not results or len(results[0].boxes) == 0:
            return torch.empty((0, 6), device=self.device)
        return results[0].boxes.data.clone()

    def predict_sliced(self, img0):
        """切片推理 (用于 40x, 100x)"""
        h_orig, w_orig = img0.shape[:2]
        slices, offsets = self.get_slices(img0)
        
        all_preds = []
        
        # 批量推理切片
        for i in range(0, len(slices), BATCH_SIZE):
            batch_imgs = slices[i:i+BATCH_SIZE]
            batch_offs = offsets[i:i+BATCH_SIZE]
            
            # 推理
            res = self.seg_model(batch_imgs, imgsz=SLICE_SIZE, conf=CONF_THRES, verbose=False, device=self.device, half=True)
            
            for j, r in enumerate(res):
                if r.boxes is None or len(r.boxes) == 0: continue
                p = r.boxes.data.clone()
                # 坐标映射回原图
                p[:, [0, 2]] += batch_offs[j][0]
                p[:, [1, 3]] += batch_offs[j][1]
                all_preds.append(p)
        
        if not all_preds: 
            return torch.empty((0, 6), device=self.device)
        
        total_preds = torch.cat(all_preds, dim=0)
        
        # 边界截断
        total_preds[:, 0].clamp_(0, w_orig); total_preds[:, 1].clamp_(0, h_orig)
        total_preds[:, 2].clamp_(0, w_orig); total_preds[:, 3].clamp_(0, h_orig)

        # 核心去重
        keep = nms_area_first(total_preds[:, :4], total_preds[:, 4], 
                              iou_thres=MERGE_IOU_THRES, 
                              ios_thres=MERGE_IOS_THRES, 
                              hard_iou_thres=HARD_IOU_THRES)
        
        return total_preds[keep]

    def run(self, image_path):
        img0 = cv2.imread(str(image_path))
        if img0 is None: return None, "Read Error"
        t0 = time.time()
        
        # --- Step 1: 路由 (分类) ---
        cls_res = self.cls_model(img0, imgsz=480, verbose=False)[0]
        top1_cls = cls_res.names[cls_res.probs.top1] # 获取类别名，如 "400", "40x"
        conf = cls_res.probs.top1conf.item()
        
        # 统一转为字符串方便匹配
        cls_str = str(top1_cls).lower()
        
        # --- Step 2: 决策分发 ---
        final_preds = None
        mode = "Unknown"
        
        # 策略 A: 40x 或 100x -> 切片推理
        if ('40' in cls_str and '400' not in cls_str) or ('100' in cls_str):
            mode = f"🍰 Sliced ({top1_cls})"
            final_preds = self.predict_sliced(img0)
            
        # 策略 B: 200x 或 400x -> 直接推理
        elif ('200' in cls_str) or ('400' in cls_str):
            mode = f"🚀 Direct ({top1_cls})"
            final_preds = self.predict_global(img0)
            
        # 策略 C: 兜底 -> 切片推理
        else:
            mode = f"⚠️ Fallback ({top1_cls})"
            final_preds = self.predict_sliced(img0)

        # --- Step 3: 绘图 ---
        annotator = Annotator(img0, line_width=3, example=str(self.seg_model.names))
        species_counts: dict[str, int] = {}
        annotated_img = annotator.result()  # 初始化，即使没有检测结果也返回原图
        
        if len(final_preds) > 0:
            cls_indices = final_preds[:, 5].to(torch.int64).cpu().numpy()
            counts = Counter(int(idx) for idx in cls_indices)
            
            # 先绘制所有框（不绘制文字，避免OpenCV字体不支持中文）
            for *xyxy, conf, cls in final_preds.cpu().numpy():
                annotator.box_label(xyxy, "", color=colors(int(cls), True))
            
            # 获取绘制结果
            annotated_img = annotator.result()
            
            # 使用PIL绘制中文标签
            pil_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # 尝试加载中文字体，如果失败则使用默认字体
            try:
                # Windows系统字体路径
                font_paths = [
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                    "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
                ]
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(font_path, 20)
                            break
                        except:
                            continue
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # 绘制中文标签
            for *xyxy, conf, cls in final_preds.cpu().numpy():
                class_name = self.seg_model.names[int(cls)]
                # 使用映射后的名称（如果存在），否则使用原名称
                display_name = self.label_mapping.get(class_name, class_name)
                label = f"{display_name} {conf:.2f}"
                
                # 计算文本位置（框的左上角）
                x1, y1, x2, y2 = xyxy
                text_x = int(x1)
                text_y = int(y1) - 20 if int(y1) > 20 else int(y1)
                
                # 绘制文本背景（白色）
                bbox = draw.textbbox((text_x, text_y), label, font=font)
                padding = 2
                draw.rectangle(
                    [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
                    fill=(255, 255, 255)
                )
                
                # 绘制文本（黑色）
                draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
            
            # 转换回OpenCV格式
            annotated_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            for cls_idx, cnt in counts.items():
                class_name = self.seg_model.names[int(cls_idx)]
                # 统计时使用原始名称（保持数据一致性），但显示时使用映射名称
                species_counts[class_name] = cnt

        return annotated_img, f"{mode} | Obj: {len(final_preds)} | {time.time()-t0:.3f}s", species_counts

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.[jJ][pP][gG]")))
    print(f"🚀 Starting Cascaded Inference...")
    
    engine = CascadedInference(SEG_MODEL_PATH, CLS_MODEL_PATH)
    
    for f in files:
        name = os.path.basename(f)
        try:
            res_img, status = engine.run(f)
            if res_img is not None:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"res_{name}"), res_img)
                print(f"Processed {name}: {status}")
        except Exception as e:
            print(f"Error processing {name}: {e}")

if __name__ == "__main__":
    main()