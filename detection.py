from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import test_cls
from PyQt5.QtCore import QObject, pyqtSignal


class MissingDependencyError(RuntimeError):
    """Raised when YOLO detection dependencies are missing."""


class ModelLoadError(RuntimeError):
    """Raised when detection models are missing or fail to initialize."""


def _ensure_detection_dependencies() -> None:
    """Ensure required YOLO dependencies are available."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            "缺少检测依赖，请先安装 torch、torchvision、ultralytics。"
        ) from exc
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError("缺少检测依赖，请安装 opencv-python、numpy。") from exc


def load_yolo_model(
    seg_weight_path: Path | str,
    cls_weight_path: Path | str | None = None,
    *,
    conf_threshold: float = 0.5,
    merge_iou: float | None = None,
    label_mapping: dict[str, str] | None = None,
):
    """Load YOLO models using the cascaded pipeline defined in test_cls.py."""
    seg_path = Path(seg_weight_path)
    if not seg_path.is_file():
        raise ModelLoadError(f"分割模型文件不存在：{seg_path}")

    if cls_weight_path is None:
        raise ModelLoadError("未提供分类模型路径，请选择 cls/pt 模型。")

    cls_path = Path(cls_weight_path)
    if not cls_path.is_file():
        raise ModelLoadError(f"分类模型文件不存在：{cls_path}")

    _ensure_detection_dependencies()

    try:
        if hasattr(test_cls, "CONF_THRES"):
            test_cls.CONF_THRES = conf_threshold
        if merge_iou is not None and hasattr(test_cls, "MERGE_IOU_THRES"):
            test_cls.MERGE_IOU_THRES = merge_iou
        model = test_cls.CascadedInference(seg_path, cls_path, label_mapping=label_mapping)
    except ImportError as exc:
        raise MissingDependencyError(f"缺少检测依赖：{exc}") from exc
    except Exception as exc:
        raise ModelLoadError(f"初始化模型失败：{exc}") from exc

    return model


class YoloDetectionWorker(QObject):
    """PyQt worker that runs cascaded detection via test_cls."""

    progress = pyqtSignal(int, int, str, str)
    finished = pyqtSignal(bool, str, object)

    def __init__(
        self,
        model: test_cls.CascadedInference,
        image_paths: List[Path],
        output_dir: Path,
        conf: float = 0.5,
        iou: float = 0.5,
        project_name: str | None = None,
        label_mapping: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._image_paths = [Path(p) for p in image_paths]
        self._base_output_dir = Path(output_dir)
        self._conf = conf
        self._iou = iou
        self._project_name = project_name
        self._label_mapping = label_mapping or {}
        self._cancelled = False
        self._session_dir = self._create_session_dir()
        self._log_file = self._session_dir / "detection_log.log"
        self._logger = self._create_logger()

    def _create_session_dir(self) -> Path:
        # 如果有项目名称，使用项目名称；否则使用时间戳
        if self._project_name:
            # 确保项目名称安全（只保留字母数字、空格、连字符、下划线）
            safe_name = "".join(c for c in self._project_name if c.isalnum() or c in (' ', '-', '_')).strip()
            if safe_name:
                dir_name = safe_name
            else:
                # 如果项目名称无效，回退到时间戳
                dir_name = time.strftime("%Y%m%d_%H%M%S")
        else:
            dir_name = time.strftime("%Y%m%d_%H%M%S")
        
        candidate = self._base_output_dir / dir_name
        counter = 1
        while candidate.exists():
            candidate = self._base_output_dir / f"{dir_name}_{counter}"
            counter += 1
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _create_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"cascaded_detection.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        try:
            file_handler = logging.FileHandler(self._log_file, mode="w", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        total = len(self._image_paths)
        if total == 0:
            self.finished.emit(False, "没有可供检测的历史图像", None)
            return

        last_output: Optional[str] = None
        self._logger.info("检测开始，输出目录: %s", self._session_dir)
        self._logger.info("日志文件: %s", self._log_file)

        for idx, image_path in enumerate(self._image_paths, start=1):
            if self._cancelled:
                self.finished.emit(False, "检测已取消", last_output)
                return
            try:
                run_result = self._model.run(image_path)
                annotated_bgr: Optional[cv2.typing.MatLike]
                status: str
                stats: Optional[Dict[str, int]] = None

                if isinstance(run_result, tuple) and len(run_result) == 3:
                    annotated_bgr, status, stats = run_result
                else:
                    annotated_bgr, status = run_result  # type: ignore[misc]
                    stats = None

                if annotated_bgr is None:
                    raise RuntimeError("未生成检测结果")
                annotated_path = self._save_outputs(image_path, annotated_bgr, status, stats)
            except Exception as exc:
                self._logger.exception("检测 %s 时出错", image_path)
                self.finished.emit(False, f"{image_path.name} 处理失败：{exc}", last_output)
                return

            last_output = str(annotated_path)
            self._logger.info("%s | %s", image_path.name, status)
            self.progress.emit(idx, total, str(image_path), str(annotated_path))
        # 单次任务的统计与报表生成
        try:
            self._generate_run_reports()
        except Exception as exc:  # pragma: no cover - 统计报表失败不影响主流程
            self._logger.warning("生成检测统计报表失败: %s", exc, exc_info=True)

        self.finished.emit(True, f"目标检测完成，共处理{total}张图片", last_output)

    def _save_outputs(
        self,
        image_path: Path,
        annotated_bgr,
        status: str,
        stats: Optional[Dict[str, int]] = None,
    ) -> Path:
        image_output_dir = self._session_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = image_output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_bgr)

        summary_file = image_output_dir / "summary.txt"
        with summary_file.open("w", encoding="utf-8") as fp:
            fp.write(f"图像: {image_path.name}\n")
            fp.write(f"状态: {status}\n")
            fp.write(f"阈值: conf={self._conf}, iou={self._iou}\n")
            stats_dict = stats or {}
            fp.write("花粉统计:\n")
            if stats_dict:
                for name, count in stats_dict.items():
                    fp.write(f"- {name}: {count}\n")
            else:
                fp.write("- 未检测到花粉目标\n")
            fp.write(f"统计: {json.dumps(stats_dict, ensure_ascii=False)}\n")

        return annotated_path

    # ------------------------------------------------------------------
    # Run-level reports (charts + Excel)
    # ------------------------------------------------------------------
    def _collect_run_stats(self) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        """遍历本次 run 目录下所有 summary.txt，汇总每个种类的花粉数量。"""
        species_totals: Dict[str, int] = defaultdict(int)
        per_image_stats: Dict[str, Dict[str, int]] = {}

        for image_dir in self._session_dir.iterdir():
            if not image_dir.is_dir():
                continue
            summary_file = image_dir / "summary.txt"
            if not summary_file.is_file():
                continue
            try:
                stats_dict: Dict[str, int] = {}
                with summary_file.open("r", encoding="utf-8") as fp:
                    image_name = None
                    for line in fp:
                        if line.startswith("图像:"):
                            image_name = line.split(":", 1)[1].strip()
                        if line.startswith("统计:"):
                            payload = line.split(":", 1)[1].strip()
                            if payload:
                                data = json.loads(payload)
                                if isinstance(data, dict):
                                    stats_dict = {
                                        str(name): int(count)
                                        for name, count in data.items()
                                    }
                            break
                if not stats_dict:
                    continue
                key = image_name or image_dir.name
                per_image_stats[key] = stats_dict
                for name, count in stats_dict.items():
                    species_totals[name] += int(count)
            except Exception as exc:  # pragma: no cover - 读取单个 summary 失败不影响整体
                self._logger.warning("解析 summary 失败 (%s): %s", summary_file, exc)
                continue

        return dict(species_totals), per_image_stats

    def _generate_run_reports(self) -> None:
        """为本次检测任务生成饼状图、柱状图和 Excel 统计表。"""
        species_totals, per_image_stats = self._collect_run_stats()
        if not species_totals:
            self._logger.info("本次检测未汇总到花粉统计数据，跳过报表生成")
            return

        # 1) 生成饼图和柱状图
        try:
            import matplotlib.pyplot as plt  # type: ignore[import]
            import matplotlib.font_manager as fm  # type: ignore[import]
            import os

            # 配置matplotlib使用中文字体
            # 首先尝试从Windows字体目录直接加载
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
                "C:/Windows/Fonts/simsun.ttc",   # 宋体
            ]
            chinese_font_prop = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        chinese_font_prop = fm.FontProperties(fname=font_path)
                        break
                    except:
                        continue
            
            # 如果找不到字体文件，尝试使用系统字体名称
            if chinese_font_prop is None:
                font_names = ['SimHei', 'Microsoft YaHei', 'SimSun']
                for font_name in font_names:
                    try:
                        plt.rcParams['font.sans-serif'] = [font_name]
                        plt.rcParams['axes.unicode_minus'] = False
                        chinese_font_prop = None  # 使用系统默认
                        break
                    except:
                        continue
            
            # 设置全局字体参数
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            # 将标签名称转换为中文
            original_labels = list(species_totals.keys())
            chinese_labels = [
                self._label_mapping.get(name, name) for name in original_labels
            ]
            counts = [species_totals[name] for name in original_labels]

            # 饼图
            plt.figure(figsize=(6, 6))
            text_props = {"fontsize": 8}
            if chinese_font_prop:
                text_props["fontproperties"] = chinese_font_prop
            plt.pie(
                counts,
                labels=chinese_labels,
                autopct="%1.1f%%",
                startangle=90,
                textprops=text_props,
            )
            title_font = chinese_font_prop if chinese_font_prop else None
            plt.title("花粉种类占比", fontproperties=title_font)
            pie_path = self._session_dir / "pollen_pie.png"
            plt.tight_layout()
            plt.savefig(pie_path, dpi=150)
            plt.close()

            # 柱状图
            plt.figure(figsize=(8, 5))
            x = range(len(chinese_labels))
            plt.bar(x, counts, color="#4f46e5")
            if chinese_font_prop:
                plt.xticks(x, chinese_labels, rotation=30, ha="right", fontsize=8, fontproperties=chinese_font_prop)
            else:
                plt.xticks(x, chinese_labels, rotation=30, ha="right", fontsize=8)
            plt.ylabel("数量", fontproperties=title_font)
            plt.title("花粉种类数量统计", fontproperties=title_font)
            bar_path = self._session_dir / "pollen_bar.png"
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150)
            plt.close()

            self._logger.info("统计图已生成: %s, %s", pie_path, bar_path)
        except ImportError:
            self._logger.warning("未安装 matplotlib，跳过饼图 / 柱状图生成")
        except Exception as exc:
            self._logger.warning("生成统计图时出错: %s", exc, exc_info=True)

        # 2) 生成 Excel（若无 pandas，则退化为 CSV）
        rows = []
        for image_name, stats in per_image_stats.items():
            for species, count in stats.items():
                rows.append(
                    {
                        "图像": image_name,
                        "花粉种类": species,
                        "数量": int(count),
                    }
                )

        # 按总数汇总一份
        for species, count in species_totals.items():
            rows.append(
                {
                    "图像": "汇总",
                    "花粉种类": species,
                    "数量": int(count),
                }
            )

        try:
            import pandas as pd  # type: ignore[import]

            df = pd.DataFrame(rows)
            excel_path = self._session_dir / "pollen_stats.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:  # type: ignore[arg-type]
                df.to_excel(writer, index=False, sheet_name="明细")
                # 额外写一张汇总表
                summary_rows = [
                    {"花粉种类": s, "总数量": int(c)}
                    for s, c in species_totals.items()
                ]
                df_summary = pd.DataFrame(summary_rows)
                df_summary.to_excel(writer, index=False, sheet_name="汇总")
            self._logger.info("Excel 统计已生成: %s", excel_path)
        except ImportError:
            csv_path = self._session_dir / "pollen_stats.csv"
            try:
                # 简单 CSV 导出
                import csv  # type: ignore[import]

                with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["图像", "花粉种类", "数量"])
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                self._logger.warning(
                    "未安装 pandas/openpyxl，仅生成 CSV 统计文件: %s", csv_path
                )
            except Exception as exc:
                self._logger.warning("生成 CSV 统计失败: %s", exc)
