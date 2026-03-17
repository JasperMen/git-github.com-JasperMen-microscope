from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import nncam
import numpy
from PIL import Image
from PyQt5.QtCore import QThread, QTimer, QUrl, QSize, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from camera_service import Frame, MicroscopeCameraService, ensure_logging_configured
from detection import (
    MissingDependencyError,
    ModelLoadError,
    YoloDetectionWorker,
    load_yolo_model,
)


class CameraWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_logging_configured()
        self.logger = logging.getLogger(__name__)

        self.setWindowTitle("花粉自动识别操控平台")
        self.resize(1600, 960)
        self.setMinimumSize(1280, 800)
        self.statusBar().showMessage("正在检测摄像头...")

        self.service = MicroscopeCameraService()
        self.capture_dir = Path("captures")
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前项目的照片目录（在设置项目时动态更新）
        self.current_project_capture_dir: Optional[Path] = None

        # 项目管理（气象站站点）本地配置
        self.project_config_path = Path("projects.json")
        self.projects: List[Dict[str, str]] = []
        self.current_project: Optional[Dict[str, str]] = None

        self.camera_ready = False
        self.scan_direction = 1  # 1 => 正向, -1 => 反向

        self.frame_rate: Optional[float] = None
        self.live_resolution: Optional[Dict[str, int]] = None
        self.available_resolutions: list[Dict[str, int]] = []
        self._auto_calibration_done = False

        self.focus_raw_dir = self.capture_dir / "focus_raw"
        self.focus_raw_dir.mkdir(parents=True, exist_ok=True)
        self.sharpness_threshold = 150.0

        # 性能优化：异步保存和计算
        self.save_queue = queue.Queue()  # 图像保存队列
        self.save_thread = None  # 保存线程

        # 防止重复点击拍照按钮
        self.capture_in_progress = False
        self.capture_lock = threading.Lock()

        # 目标检测结果预览
        self.latest_detection_result: Optional[Path] = None
        self.detection_preview_label: Optional[QLabel] = None
        self.detection_result_path_label: Optional[QLabel] = None
        self.detection_stats_label: Optional[QLabel] = None
        self.page_tabs: Optional[QTabWidget] = None
        self.detection_model_path: Optional[Path] = None
        self.detection_cls_model_path: Optional[Path] = None
        self.detection_model = None
        self.detection_output_dir = self.capture_dir / "detections"
        self.detection_output_dir.mkdir(parents=True, exist_ok=True)
        self.detection_results: List[Path] = []
        self.detection_result_stats: Dict[str, Dict[str, int]] = {}
        
        # 标签名称映射：短代码 -> 中文名称
        self.label_name_mapping: Dict[str, str] = {
            "ys1": "云杉",
            "bjys": "北京云杉",
            "dzh": "大籽蒿",
            "ys2": "杨树",
            "sb": "松柏",
            "ls": "柳树",
            "gsh": "格桑花",
            "ych": "油菜花",
            "gwbc": "狗尾巴草",
            "qxlm": "球悬铃木",
            "azs": "矮紫杉",
            "syh": "芍药花",
            "cmx": "草木犀",
            "jkzhkmc": "菊科中华苦荬菜",
            "jkpgy": "菊科蒲公英",
            "qwkpg": "蔷薇科苹果",
            "sk": "蜀葵",
            "jyh": "金银花",
            "hcm": "黄刺玫",
            "yb": "圆柏",
            "lk": "藜科",
        }
        self.stats_run_list: Optional[QListWidget] = None
        self.stats_artifact_list: Optional[QListWidget] = None
        self.stats_preview_label: Optional[QLabel] = None
        self.stats_preview_path_label: Optional[QLabel] = None
        self.detection_conf = 0.25
        self.detection_iou = 0.45
        self.detection_in_progress = False
        self._detection_thread: Optional[QThread] = None
        self._detection_worker: Optional[YoloDetectionWorker] = None

        # 加载本地项目（气象站站点）配置
        self._load_projects()

        self._build_ui()
        self._load_captures()
        self._auto_load_default_models()

        # 启动异步保存线程
        self._start_save_thread()

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self._update_frame)
        self.frame_timer.start(80)

        self.camera_retry_timer = QTimer(self)
        self.camera_retry_timer.setInterval(4000)
        self.camera_retry_timer.timeout.connect(self._ensure_camera_ready)
        self.camera_retry_timer.start()

        QTimer.singleShot(100, self._ensure_camera_ready)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(18, 12, 18, 18)
        outer_layout.setSpacing(12)

        self.page_tabs = QTabWidget()
        self.page_tabs.setTabPosition(QTabWidget.North)
        self.page_tabs.setDocumentMode(True)
        self.page_tabs.setMovable(False)
        self.page_tabs.setStyleSheet(
            "QTabBar::tab{padding:10px 22px; font-size:16px; font-weight:600;} "
            "QTabBar::tab:selected{color:#111827;}"
        )
        outer_layout.addWidget(self.page_tabs, 1)
        self.page_tabs.currentChanged.connect(self._on_tab_changed)

        capture_tab = QWidget()
        main_layout = QHBoxLayout(capture_tab)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(18)
        self.page_tabs.addTab(capture_tab, "实时采集")

        base_font = QFont(self.font())
        base_font.setPointSize(11)
        self.setFont(base_font)

        # Camera section -------------------------------------------------
        camera_group = QGroupBox("摄像头分区")
        camera_group.setStyleSheet(
            "QGroupBox::title{font-size:20px;font-weight:700;padding:6px 10px;}"
        )
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(14)

        self.image_label = QLabel("等待摄像头...")
        self.image_label.setAlignment(Qt.AlignCenter)
        # 设置画面尺寸，铺满上半部分
        self.image_label.setMinimumSize(1000, 700)  # 增大尺寸以铺满上半部分
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展
        self.image_label.setScaledContents(False)  # 确保使用KeepAspectRatio缩放，避免拉伸
        self.image_label.setStyleSheet(
            "background:#0f172a; color:#f1f5f9; border-radius:10px; font-size:28px;"
        )
        camera_layout.addWidget(self.image_label, 1)

        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(18)

        self.frame_rate_label = QLabel("帧率：-- fps")
        self.frame_rate_label.setStyleSheet("color:#263238; font-size:25px; font-weight:600;")
        stats_layout.addWidget(self.frame_rate_label)

        self.resolution_label = QLabel("当前尺寸：--")
        self.resolution_label.setStyleSheet("color:#263238; font-size:25px; font-weight:600;")
        stats_layout.addWidget(self.resolution_label)

        stats_layout.addStretch(1)

        res_label = QLabel("图像大小：")
        res_label.setStyleSheet("color:#263238; font-size:25px;")
        stats_layout.addWidget(res_label)

        self.resolution_combo = QComboBox()
        self.resolution_combo.setEnabled(False)
        self.resolution_combo.setMinimumWidth(220)
        self.resolution_combo.currentIndexChanged.connect(self._handle_resolution_combo)
        stats_layout.addWidget(self.resolution_combo)

        camera_layout.addWidget(stats_widget)

        # 摄像头下拉列表
        device_row = QHBoxLayout()
        device_label = QLabel("摄像头列表：")
        device_label.setStyleSheet("color:#263238; font-size:14px;")
        device_row.addWidget(device_label)
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(260)
        self.device_combo.currentIndexChanged.connect(self._on_device_selected)
        device_row.addWidget(self.device_combo, 1)
        self.reconnect_button = QPushButton("刷新")
        self.reconnect_button.setMinimumHeight(32)
        self.reconnect_button.clicked.connect(self._handle_reconnect)
        device_row.addWidget(self.reconnect_button)
        camera_layout.addLayout(device_row)

        main_layout.addWidget(camera_group, 6)

        # Feature section -----------------------------------------------
        feature_group = QGroupBox("功能分区")
        feature_group.setStyleSheet(
            "QGroupBox::title{font-size:20px;font-weight:700;padding:6px 10px;}"
        )
        feature_group_layout = QVBoxLayout(feature_group)
        feature_group_layout.setContentsMargins(12, 12, 12, 12)
        feature_group_layout.setSpacing(16)

        # 仅保留三个核心按钮：校准 / 拍照 / 执行目标检测
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(14)

        self.auto_color_button = QPushButton("校准")
        self.auto_color_button.setMinimumHeight(48)
        self.auto_color_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.auto_color_button.clicked.connect(self._handle_auto_color)
        buttons_layout.addWidget(self.auto_color_button)

        self.capture_button = QPushButton("拍照")
        self.capture_button.setMinimumHeight(48)
        self.capture_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.capture_button.clicked.connect(self._handle_capture)
        self.capture_button.setEnabled(False)  # 初始禁用，需要先新建项目
        buttons_layout.addWidget(self.capture_button)

        self.detect_image_button = QPushButton("执行目标检测")
        self.detect_image_button.setMinimumHeight(48)
        self.detect_image_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.detect_image_button.clicked.connect(self._handle_run_detection)
        buttons_layout.addWidget(self.detect_image_button)

        feature_group_layout.addLayout(buttons_layout)

        # 项目管理（气象站站点）
        project_group = QGroupBox("项目管理")
        project_group.setStyleSheet(
            "QGroupBox::title{font-size:17px;font-weight:700;padding:6px 10px;color:#2c3e50;} "
            "QGroupBox{margin-top:12px;border:1px solid #e0e0e0;border-radius:6px;padding-top:8px;background-color:#fafafa;}"
        )
        project_layout = QVBoxLayout(project_group)
        project_layout.setContentsMargins(16, 20, 16, 16)
        project_layout.setSpacing(14)

        self.project_status_label = QLabel("当前项目：未选择")
        self.project_status_label.setStyleSheet(
            "color:#34495e; font-size:14px; font-weight:500; padding:8px 12px; "
            "background-color:#ffffff; border:1px solid #e0e0e0; border-radius:4px;"
        )
        self.project_status_label.setWordWrap(True)
        project_layout.addWidget(self.project_status_label)

        project_btn_row = QHBoxLayout()
        project_btn_row.setSpacing(14)

        self.new_project_button = QPushButton("新建项目")
        self.new_project_button.setMinimumHeight(48)
        self.new_project_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.new_project_button.clicked.connect(self._handle_new_project)
        project_btn_row.addWidget(self.new_project_button)

        self.select_project_button = QPushButton("选择项目")
        self.select_project_button.setMinimumHeight(48)
        self.select_project_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.select_project_button.clicked.connect(self._handle_select_project)
        project_btn_row.addWidget(self.select_project_button)

        self.delete_project_button = QPushButton("删除项目")
        self.delete_project_button.setMinimumHeight(48)
        self.delete_project_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.delete_project_button.clicked.connect(self._handle_delete_project)
        project_btn_row.addWidget(self.delete_project_button)

        project_layout.addLayout(project_btn_row)

        feature_group_layout.addWidget(project_group)
        feature_group_layout.addStretch(1)

        feature_layout_final = QVBoxLayout()
        feature_layout_final.setContentsMargins(0, 0, 0, 0)
        feature_layout_final.addWidget(feature_group)
        feature_widget = QWidget()
        feature_widget.setLayout(feature_layout_final)
        feature_widget.setMaximumWidth(520)
        main_layout.addWidget(feature_widget, 4)

        # History section -----------------------------------------------
        history_group = QGroupBox("历史照片分区")
        history_group.setStyleSheet(
            "QGroupBox::title{font-size:20px;font-weight:700;padding:6px 10px;}"
        )
        history_layout = QVBoxLayout(history_group)
        history_layout.setSpacing(12)

        self.history_list = QListWidget()
        self.history_list.setViewMode(QListView.ListMode)
        self.history_list.setSpacing(10)
        self.history_list.setResizeMode(QListWidget.Adjust)
        self.history_list.setUniformItemSizes(True)
        self.history_list.setStyleSheet("font-size:14px;")
        self.history_list.itemDoubleClicked.connect(self._open_capture)
        history_layout.addWidget(self.history_list, 1)

        main_layout.addWidget(history_group, 2)

        # 检测结果预览页面 -----------------------------------------------
        detection_tab = QWidget()
        self.detection_tab = detection_tab
        detection_layout = QVBoxLayout(detection_tab)
        detection_layout.setContentsMargins(18, 18, 18, 18)
        detection_layout.setSpacing(16)

        detection_title = QLabel("目标检测结果预览")
        detection_title.setStyleSheet("font-size:24px; font-weight:700; color:#111827;")
        detection_layout.addWidget(detection_title)

        detection_hint = QLabel("加载模型并完成检测后，可在左侧列表中选择不同图片查看对应的目标检测结果。")
        detection_hint.setWordWrap(True)
        detection_hint.setStyleSheet("color:#4b5563; font-size:15px;")
        detection_layout.addWidget(detection_hint)

        result_split = QHBoxLayout()
        result_split.setSpacing(18)

        result_list_container = QVBoxLayout()
        result_list_container.setSpacing(8)

        run_list_label = QLabel("检测任务（run_...）")
        run_list_label.setStyleSheet("font-size:15px; font-weight:600; color:#111827;")
        result_list_container.addWidget(run_list_label)

        self.detection_run_list = QListWidget()
        self.detection_run_list.setMinimumWidth(260)
        self.detection_run_list.setStyleSheet("font-size:13px;")
        self.detection_run_list.itemSelectionChanged.connect(self._on_detection_run_selected)
        result_list_container.addWidget(self.detection_run_list, 1)
        
        # 删除按钮放在列表下方，更醒目
        self.delete_detection_run_button = QPushButton("删除检测任务")
        self.delete_detection_run_button.setMinimumHeight(40)
        self.delete_detection_run_button.setStyleSheet(
            "QPushButton{"
            "font-size:13px;font-weight:700;padding:8px 16px;border-radius:4px;"
            "background-color:#dc2626;color:#ffffff;"
            "}"
            "QPushButton:hover{"
            "background-color:#b91c1c;"
            "}"
            "QPushButton:pressed{"
            "background-color:#991b1b;"
            "}"
        )
        self.delete_detection_run_button.clicked.connect(self._handle_delete_detection_run)
        result_list_container.addWidget(self.delete_detection_run_button)

        result_list_label = QLabel("检测结果列表")
        result_list_label.setStyleSheet("font-size:15px; font-weight:600; color:#111827;")
        result_list_container.addWidget(result_list_label)

        self.detection_result_list = QListWidget()
        self.detection_result_list.setMinimumWidth(260)
        self.detection_result_list.setStyleSheet("font-size:13px;")
        self.detection_result_list.itemSelectionChanged.connect(self._on_detection_result_selected)
        result_list_container.addWidget(self.detection_result_list, 2)

        result_split.addLayout(result_list_container, 0)

        self.detection_result_path_label = QLabel("当前展示：无")
        self.detection_result_path_label.setStyleSheet("color:#334155; font-size:14px;")
        result_right = QVBoxLayout()
        result_right.setSpacing(12)
        result_right.addWidget(self.detection_result_path_label)

        self.detection_stats_label = QLabel("花粉统计：--")
        self.detection_stats_label.setStyleSheet("color:#1f2933; font-size:14px;")
        self.detection_stats_label.setWordWrap(True)
        result_right.addWidget(self.detection_stats_label)

        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setStyleSheet("border:1px solid #e2e8f0; border-radius:8px;")

        self.detection_preview_label = QLabel()
        self.detection_preview_label.setAlignment(Qt.AlignCenter)
        self.detection_preview_label.setWordWrap(True)
        self.detection_preview_label.setMinimumSize(960, 600)
        self.detection_preview_label.setStyleSheet(
            "background:#0f172a; color:#f1f5f9; border-radius:10px; font-size:20px;"
        )
        self.detection_preview_label.setScaledContents(False)

        preview_scroll.setWidget(self.detection_preview_label)
        result_right.addWidget(preview_scroll, 1)

        result_split.addLayout(result_right, 1)
        detection_layout.addLayout(result_split, 1)

        self.page_tabs.addTab(detection_tab, "检测结果")
        self._clear_detection_preview("目标检测完成后，处理后的图像将显示在此区域。")
        self._load_detection_history_results()

        # 新增“统计数据”页面，查看每次检测生成的饼状图和柱状图
        self._build_stats_tab()
        # 新增“相机参数设置”页面，将详细的参数控制移动到该页签
        self._build_camera_settings_tab()

        # 在所有控件创建完成后再刷新一次可用状态
        self._update_control_states()

    def _build_camera_settings_tab(self) -> None:
        """构建“相机参数设置”页，将原功能分区中的详细相机参数集中展示。"""
        if self.page_tabs is None:
            return

        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(18, 18, 18, 18)
        settings_layout.setSpacing(16)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        control_scroll.setStyleSheet("QScrollArea{border:0;}")

        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setSpacing(18)
        control_layout.setContentsMargins(12, 12, 12, 12)

        control_scroll.setWidget(control_container)
        settings_layout.addWidget(control_scroll)

        # 拍照控制（行列与栅格设置）
        capture_group = QGroupBox("拍照控制")
        capture_group.setStyleSheet(
            "QGroupBox::title{font-size:16px;font-weight:700;padding:4px 8px;} "
            "QGroupBox{margin-top:8px;}"
        )
        capture_layout = QVBoxLayout(capture_group)
        capture_layout.setContentsMargins(12, 20, 12, 12)
        capture_layout.setSpacing(12)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(12)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        column_label = QLabel("列 (ccc):")
        column_label.setStyleSheet("color:#1f2933; font-weight:600;")
        self.column_spin = QSpinBox()
        self.column_spin.setRange(0, 999)
        self.column_spin.setValue(0)
        self.column_spin.setFixedHeight(34)
        grid_layout.addWidget(column_label, 0, 0)
        grid_layout.addWidget(self.column_spin, 0, 1)

        row_label = QLabel("行 (rrr):")
        row_label.setStyleSheet("color:#1f2933; font-weight:600;")
        self.row_spin = QSpinBox()
        self.row_spin.setRange(0, 999)
        self.row_spin.setValue(0)
        self.row_spin.setFixedHeight(34)
        grid_layout.addWidget(row_label, 0, 2)
        grid_layout.addWidget(self.row_spin, 0, 3)

        max_columns_label = QLabel("列总数:")
        max_columns_label.setStyleSheet("color:#1f2933; font-weight:600;")
        self.max_columns_spin = QSpinBox()
        self.max_columns_spin.setRange(1, 999)
        self.max_columns_spin.setValue(5)
        self.max_columns_spin.setFixedHeight(34)
        grid_layout.addWidget(max_columns_label, 1, 0)
        grid_layout.addWidget(self.max_columns_spin, 1, 1)

        max_rows_label = QLabel("行总数:")
        max_rows_label.setStyleSheet("color:#1f2933; font-weight:600;")
        self.max_rows_spin = QSpinBox()
        self.max_rows_spin.setRange(1, 999)
        self.max_rows_spin.setValue(5)
        self.max_rows_spin.setFixedHeight(34)
        grid_layout.addWidget(max_rows_label, 1, 2)
        grid_layout.addWidget(self.max_rows_spin, 1, 3)

        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 2)
        grid_layout.setColumnStretch(2, 1)
        grid_layout.setColumnStretch(3, 2)

        capture_layout.addLayout(grid_layout)

        self.auto_advance_check = QCheckBox("拍照后自动Z字前进")
        self.auto_advance_check.setChecked(True)
        capture_layout.addWidget(self.auto_advance_check)

        control_layout.addWidget(capture_group)

        # AI 模型状态
        detection_group = QGroupBox("AI目标检测")
        detection_group.setStyleSheet(
            "QGroupBox::title{font-size:16px;font-weight:700;padding:4px 8px;} "
            "QGroupBox{margin-top:8px;}"
        )
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.setSpacing(12)
        detection_layout.setContentsMargins(12, 20, 12, 12)

        self.model_status_label = QLabel("当前模型: 未加载")
        self.model_status_label.setStyleSheet("color:#1f2933; font-weight:600;")
        detection_layout.addWidget(self.model_status_label)

        control_layout.addWidget(detection_group)

        # 摄像头调节
        color_group = QGroupBox("摄像头调节")
        color_group.setStyleSheet(
            "QGroupBox::title{font-size:16px;font-weight:700;padding:4px 8px;}"
        )
        color_layout = QVBoxLayout(color_group)
        color_layout.setSpacing(12)

        self.brightness_slider, self.brightness_value = self._create_slider(
            nncam.NNCAM_BRIGHTNESS_MIN,
            nncam.NNCAM_BRIGHTNESS_MAX,
            "亮度",
            color_layout,
            self._on_brightness_changed,
            self._commit_brightness,
        )
        self.hue_slider, self.hue_value = self._create_slider(
            nncam.NNCAM_HUE_MIN,
            nncam.NNCAM_HUE_MAX,
            "颜色",
            color_layout,
            self._on_hue_changed,
            self._commit_hue,
        )
        self.saturation_slider, self.saturation_value = self._create_slider(
            nncam.NNCAM_SATURATION_MIN,
            nncam.NNCAM_SATURATION_MAX,
            "饱和度",
            color_layout,
            self._on_saturation_changed,
            self._commit_saturation,
        )

        control_layout.addWidget(color_group)

        # 曝光与增益控制组
        exposure_group = QGroupBox("曝光与增益")
        exposure_group.setStyleSheet(
            "QGroupBox::title{font-size:16px;font-weight:700;padding:4px 8px;}"
        )
        exposure_layout = QVBoxLayout(exposure_group)
        exposure_layout.setSpacing(12)

        self.auto_exposure_check = QCheckBox("自动曝光")
        self.auto_exposure_check.setChecked(True)
        self.auto_exposure_check.stateChanged.connect(self._on_auto_exposure_changed)
        exposure_layout.addWidget(self.auto_exposure_check)

        self.exposure_target_slider, self.exposure_target_value = self._create_slider(
            nncam.NNCAM_AETARGET_MIN,
            nncam.NNCAM_AETARGET_MAX,
            "曝光目标:",
            exposure_layout,
            self._on_exposure_target_changed,
            self._commit_exposure_target,
        )

        self.exposure_time_slider, self.exposure_time_value = self._create_slider(
            1,
            100000,
            "曝光时间:",
            exposure_layout,
            self._on_exposure_time_changed,
            self._commit_exposure_time,
        )

        self.gain_slider, self.gain_value = self._create_slider(
            100,
            1000,
            "增益:",
            exposure_layout,
            self._on_gain_changed,
            self._commit_gain,
        )

        self.speed_slider, self.speed_value = self._create_slider(
            0,
            10,
            "帧率级别:",
            exposure_layout,
            self._on_speed_changed,
            self._commit_speed,
        )

        auto_expo_button = QPushButton("一键自动曝光")
        auto_expo_button.setMinimumHeight(40)
        auto_expo_button.clicked.connect(self._handle_auto_exposure_once)
        exposure_layout.addSpacing(4)
        exposure_layout.addWidget(auto_expo_button)

        control_layout.addWidget(exposure_group)

        # 锐化控制组
        sharpening_group = QGroupBox("锐化")
        sharpening_group.setStyleSheet(
            "QGroupBox::title{font-size:16px;font-weight:700;padding:4px 8px;}"
        )
        sharpening_layout = QVBoxLayout(sharpening_group)
        sharpening_layout.setSpacing(12)

        self.sharpening_strength_slider, self.sharpening_strength_value = self._create_slider(
            nncam.NNCAM_SHARPENING_STRENGTH_MIN,
            nncam.NNCAM_SHARPENING_STRENGTH_MAX,
            "强度:",
            sharpening_layout,
            self._on_sharpening_strength_changed,
            self._commit_sharpening_strength,
        )

        self.sharpening_radius_slider, self.sharpening_radius_value = self._create_slider(
            nncam.NNCAM_SHARPENING_RADIUS_MIN,
            nncam.NNCAM_SHARPENING_RADIUS_MAX,
            "半径:",
            sharpening_layout,
            self._on_sharpening_radius_changed,
            self._commit_sharpening_radius,
        )

        self.sharpening_threshold_slider, self.sharpening_threshold_value = self._create_slider(
            nncam.NNCAM_SHARPENING_THRESHOLD_MIN,
            nncam.NNCAM_SHARPENING_THRESHOLD_MAX,
            "阈值:",
            sharpening_layout,
            self._on_sharpening_threshold_changed,
            self._commit_sharpening_threshold,
        )

        sharpening_default_button = QPushButton("默认值")
        sharpening_default_button.setMinimumHeight(40)
        sharpening_default_button.clicked.connect(self._handle_sharpening_default)
        sharpening_layout.addSpacing(4)
        sharpening_layout.addWidget(sharpening_default_button)

        control_layout.addWidget(sharpening_group)

        # 杂项控制组
        misc_group = QGroupBox("*杂项")
        misc_group.setStyleSheet(
            "QGroupBox::title{font-size:16px;font-weight:700;padding:4px 8px;} "
            "QGroupBox{margin-top:8px;}"
        )
        misc_layout = QVBoxLayout(misc_group)
        misc_layout.setSpacing(14)
        misc_layout.setContentsMargins(12, 20, 12, 12)

        checkbox_container = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_container)
        checkbox_layout.setSpacing(10)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)

        self.negative_check = QCheckBox("负片")
        self.negative_check.stateChanged.connect(self._on_negative_changed)
        checkbox_layout.addWidget(self.negative_check)

        self.low_noise_check = QCheckBox("低噪声(更高的信噪比,更低的帧率)")
        self.low_noise_check.stateChanged.connect(self._on_low_noise_changed)
        checkbox_layout.addWidget(self.low_noise_check)

        self.low_power_check = QCheckBox("低功耗")
        self.low_power_check.stateChanged.connect(self._on_low_power_changed)
        checkbox_layout.addWidget(self.low_power_check)

        self.remove_shutter_effect_check = QCheckBox("去快门效应")
        self.remove_shutter_effect_check.stateChanged.connect(self._on_remove_shutter_effect_changed)
        checkbox_layout.addWidget(self.remove_shutter_effect_check)

        misc_layout.addWidget(checkbox_container)

        dropdown_container = QWidget()
        dropdown_layout = QVBoxLayout(dropdown_container)
        dropdown_layout.setSpacing(12)
        dropdown_layout.setContentsMargins(0, 0, 0, 0)

        debayer_layout = QHBoxLayout()
        debayer_label = QLabel("Debayer:")
        debayer_label.setStyleSheet("color:#1f2933; font-weight:600; min-width:80px;")
        debayer_layout.addWidget(debayer_label)
        self.debayer_combo = QComboBox()
        self.debayer_combo.setMinimumHeight(32)
        self.debayer_combo.addItem("双线性(Bilinear)", 0)
        self.debayer_combo.addItem("VNG", 1)
        self.debayer_combo.addItem("PPG", 2)
        self.debayer_combo.addItem("AHD", 3)
        self.debayer_combo.addItem("边缘感知(Edge Aware)", 4)
        self.debayer_combo.setCurrentIndex(4)
        self.debayer_combo.currentIndexChanged.connect(self._on_debayer_changed)
        debayer_layout.addWidget(self.debayer_combo, 1)
        dropdown_layout.addLayout(debayer_layout)

        tone_layout = QHBoxLayout()
        tone_label = QLabel("色调映射:")
        tone_label.setStyleSheet("color:#1f2933; font-weight:600; min-width:80px;")
        tone_layout.addWidget(tone_label)
        self.tone_mapping_combo = QComboBox()
        self.tone_mapping_combo.setMinimumHeight(32)
        self.tone_mapping_combo.addItem("关闭", 0)
        self.tone_mapping_combo.addItem("多项式", 1)
        self.tone_mapping_combo.addItem("对数", 2)
        self.tone_mapping_combo.setCurrentIndex(2)
        self.tone_mapping_combo.currentIndexChanged.connect(self._on_tone_mapping_changed)
        tone_layout.addWidget(self.tone_mapping_combo, 1)
        dropdown_layout.addLayout(tone_layout)

        shutter_layout = QHBoxLayout()
        shutter_label = QLabel("快门模式:")
        shutter_label.setStyleSheet("color:#1f2933; font-weight:600; min-width:80px;")
        shutter_layout.addWidget(shutter_label)
        self.shutter_mode_combo = QComboBox()
        self.shutter_mode_combo.setMinimumHeight(32)
        self.shutter_mode_combo.addItem("卷帘快门", 0)
        self.shutter_mode_combo.addItem("全局快门", 1)
        self.shutter_mode_combo.currentIndexChanged.connect(self._on_shutter_mode_changed)
        shutter_layout.addWidget(self.shutter_mode_combo, 1)
        dropdown_layout.addLayout(shutter_layout)

        readout_layout = QHBoxLayout()
        readout_label = QLabel("读出模式:")
        readout_label.setStyleSheet("color:#1f2933; font-weight:600; min-width:80px;")
        readout_layout.addWidget(readout_label)
        self.readout_mode_combo = QComboBox()
        self.readout_mode_combo.setMinimumHeight(32)
        self.readout_mode_combo.addItem("IWR(边读出边积分)", 0)
        self.readout_mode_combo.addItem("TWR(边读出边积分)", 1)
        self.readout_mode_combo.currentIndexChanged.connect(self._on_readout_mode_changed)
        readout_layout.addWidget(self.readout_mode_combo, 1)
        dropdown_layout.addLayout(readout_layout)

        misc_layout.addWidget(dropdown_container)

        misc_default_button = QPushButton("默认值")
        misc_default_button.setMinimumHeight(40)
        misc_default_button.clicked.connect(self._handle_misc_default)
        misc_layout.addSpacing(4)
        misc_layout.addWidget(misc_default_button)

        control_layout.addWidget(misc_group)

        self.file_hint_label = QLabel("文件命名：img_r{rrr}_c{ccc}.jpg")
        self.file_hint_label.setStyleSheet("color:#546e7a; font-size:25px; padding:8px 0;")
        control_layout.addSpacing(8)
        control_layout.addWidget(self.file_hint_label)

        control_layout.addStretch(1)

        self.page_tabs.addTab(settings_tab, "相机参数设置")

    def _build_stats_tab(self) -> None:
        """构建“统计数据”页，浏览每次检测生成的饼状图和柱状图。"""
        if self.page_tabs is None:
            return

        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.setContentsMargins(18, 18, 18, 18)
        stats_layout.setSpacing(16)

        title = QLabel("检测统计数据")
        title.setStyleSheet("font-size:24px; font-weight:700; color:#111827;")
        stats_layout.addWidget(title)

        hint = QLabel("左侧选择 run_* 文件夹，再选择统计图类型（饼状图 / 柱状图），即可在右侧查看。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#4b5563; font-size:15px;")
        stats_layout.addWidget(hint)

        split = QHBoxLayout()
        split.setSpacing(18)

        # 左侧 run 列表 + 统计项列表
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        run_label = QLabel("检测任务列表")
        run_label.setStyleSheet("font-size:15px; font-weight:600; color:#111827;")
        left_layout.addWidget(run_label)

        self.stats_run_list = QListWidget()
        self.stats_run_list.setMinimumWidth(260)
        self.stats_run_list.setStyleSheet("font-size:13px;")
        self.stats_run_list.itemSelectionChanged.connect(self._on_stats_run_selected)
        left_layout.addWidget(self.stats_run_list, 1)
        
        # 删除按钮放在列表下方，更醒目
        self.delete_stats_run_button = QPushButton("删除检测任务")
        self.delete_stats_run_button.setMinimumHeight(40)
        self.delete_stats_run_button.setStyleSheet(
            "QPushButton{"
            "font-size:13px;font-weight:700;padding:8px 16px;border-radius:4px;"
            "background-color:#dc2626;color:#ffffff;"
            "}"
            "QPushButton:hover{"
            "background-color:#b91c1c;"
            "}"
            "QPushButton:pressed{"
            "background-color:#991b1b;"
            "}"
        )
        self.delete_stats_run_button.clicked.connect(self._handle_delete_stats_run)
        left_layout.addWidget(self.delete_stats_run_button)

        artifact_label = QLabel("统计图列表")
        artifact_label.setStyleSheet("font-size:15px; font-weight:600; color:#111827;")
        left_layout.addWidget(artifact_label)

        self.stats_artifact_list = QListWidget()
        self.stats_artifact_list.setMinimumWidth(260)
        self.stats_artifact_list.setStyleSheet("font-size:13px;")
        self.stats_artifact_list.itemSelectionChanged.connect(self._on_stats_artifact_selected)
        left_layout.addWidget(self.stats_artifact_list, 1)

        split.addLayout(left_layout, 0)

        # 右侧预览
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)

        # 顶部工具栏：路径标签 + 下载PDF按钮
        top_toolbar = QHBoxLayout()
        top_toolbar.setSpacing(12)
        
        self.stats_preview_path_label = QLabel("当前展示：无")
        self.stats_preview_path_label.setStyleSheet("color:#334155; font-size:14px;")
        top_toolbar.addWidget(self.stats_preview_path_label)
        
        top_toolbar.addStretch()
        
        self.download_pdf_button = QPushButton("下载PDF报告")
        self.download_pdf_button.setMinimumHeight(36)
        self.download_pdf_button.setStyleSheet(
            "QPushButton{font-size:13px;font-weight:600;padding:6px 16px;border-radius:4px;}"
        )
        self.download_pdf_button.clicked.connect(self._handle_download_pdf)
        top_toolbar.addWidget(self.download_pdf_button)
        
        right_layout.addLayout(top_toolbar)

        self.stats_preview_label = QLabel("暂无数据")
        self.stats_preview_label.setAlignment(Qt.AlignCenter)
        self.stats_preview_label.setMinimumSize(600, 420)
        self.stats_preview_label.setStyleSheet(
            "background:#0f172a; color:#f1f5f9; border-radius:10px; font-size:18px;"
        )
        self.stats_preview_label.setScaledContents(False)
        right_layout.addWidget(self.stats_preview_label, 1)

        split.addLayout(right_layout, 1)

        stats_layout.addLayout(split, 1)

        self.page_tabs.addTab(stats_tab, "统计数据")

    # ------------------------------------------------------------------
    # 项目管理（气象站站点）
    # ------------------------------------------------------------------
    def _load_projects(self) -> None:
        """从本地文件加载项目配置，如果不存在则创建一个默认示例列表。"""
        self.projects = []
        self.current_project = None

        try:
            if self.project_config_path.is_file():
                import json

                with self.project_config_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.projects = [p for p in data if isinstance(p, dict) and "name" in p]
        except Exception as exc:
            self.logger.warning("Failed to load projects config: %s", exc)
            self.projects = []

        # 如果没有任何项目，初始化一些默认的气象站示例
        if not self.projects:
            self.projects = [
                {"name": "默认站点A", "station": "气象站A"},
                {"name": "默认站点B", "station": "气象站B"},
            ]
            self._save_projects()

        # 不自动选中项目，需要用户明确选择或新建项目后才能使用拍照功能
        # 即使有项目列表，也不自动设置 current_project
        self.current_project = None

        if hasattr(self, "project_status_label"):
            self.project_status_label.setText("当前项目：未选择")
        
        # 不启用拍照按钮，必须用户明确操作（新建或选择项目）后才能启用
        # 这里不调用 _update_capture_button_state()，保持按钮禁用状态

    def _save_projects(self) -> None:
        """将当前项目列表保存到本地 JSON 文件。"""
        try:
            import json

            with self.project_config_path.open("w", encoding="utf-8") as f:
                json.dump(self.projects, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.logger.warning("Failed to save projects config: %s", exc)

    def _update_capture_button_state(self) -> None:
        """根据项目状态和相机状态更新拍照按钮的启用/禁用状态。"""
        if hasattr(self, "capture_button"):
            # 只有在有当前项目且相机就绪时才能启用拍照按钮
            self.capture_button.setEnabled(
                self.current_project is not None and self.camera_ready
            )
    
    def _get_project_capture_dir(self, project: Optional[Dict[str, str]]) -> Optional[Path]:
        """获取项目的照片保存目录。"""
        if project is None:
            return None
        project_name = project.get("name", "")
        if not project_name:
            return None
        # 使用项目名称作为目录名，确保目录名安全
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            return None
        project_dir = self.capture_dir / safe_name
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def _handle_new_project(self) -> None:
        """新建项目：输入项目名称和所属气象站，写入本地配置。"""
        name, ok = QInputDialog.getText(self, "新建项目", "请输入项目名称：")
        if not ok or not name.strip():
            return
        name = name.strip()

        station, ok2 = QInputDialog.getText(self, "新建项目", "请输入气象站站点名称：")
        if not ok2 or not station.strip():
            return
        station = station.strip()

        # 给项目名称添加时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_with_timestamp = f"{name}_{timestamp}"

        project = {"name": name_with_timestamp, "station": station}
        self.projects.append(project)
        self.current_project = project
        self._save_projects()

        if hasattr(self, "project_status_label"):
            self.project_status_label.setText(f"当前项目：{name_with_timestamp}（{station}）")

        self.logger.info("New project created: %s (%s)", name_with_timestamp, station)
        self.statusBar().showMessage(f"已新建项目：{name_with_timestamp}")
        
        # 更新项目照片目录
        self.current_project_capture_dir = self._get_project_capture_dir(self.current_project)
        
        # 启用拍照按钮并重新加载照片
        self._update_capture_button_state()
        self._load_captures()

    def _handle_select_project(self) -> None:
        """选择已存在的项目（气象站站点）。"""
        if not self.projects:
            QMessageBox.information(self, "提示", "当前没有可选择的项目，请先新建项目。")
            return

        items = [f"{p.get('name', '未命名')}（{p.get('station', '未知气象站')}）" for p in self.projects]
        current_index = 0
        if self.current_project in self.projects:
            current_index = self.projects.index(self.current_project)

        item, ok = QInputDialog.getItem(self, "选择项目", "请选择气象站项目：", items, current_index, False)
        if not ok or not item:
            return

        index = items.index(item)
        self.current_project = self.projects[index]
        name = self.current_project.get("name", "未命名")
        station = self.current_project.get("station", "未知气象站")

        if hasattr(self, "project_status_label"):
            self.project_status_label.setText(f"当前项目：{name}（{station}）")

        self.logger.info("Project selected: %s (%s)", name, station)
        self.statusBar().showMessage(f"已选择项目：{name}")
        
        # 更新项目照片目录
        self.current_project_capture_dir = self._get_project_capture_dir(self.current_project)
        
        # 启用拍照按钮并重新加载照片
        self._update_capture_button_state()
        self._load_captures()
    
    def _handle_delete_project(self) -> None:
        """删除选中的项目。"""
        if not self.projects:
            QMessageBox.information(self, "提示", "当前没有可删除的项目。")
            return
        
        items = [f"{p.get('name', '未命名')}（{p.get('station', '未知气象站')}）" for p in self.projects]
        current_index = 0
        if self.current_project in self.projects:
            current_index = self.projects.index(self.current_project)

        item, ok = QInputDialog.getItem(self, "删除项目", "请选择要删除的项目：", items, current_index, False)
        if not ok or not item:
            return

        index = items.index(item)
        project_to_delete = self.projects[index]
        project_name = project_to_delete.get("name", "未命名")
        
        # 确认删除
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除项目：{project_name} 吗？\n\n此操作将删除项目配置和项目照片目录，且无法恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 删除项目配置
        self.projects.pop(index)
        self._save_projects()
        
        # 如果删除的是当前项目，清空当前项目
        if self.current_project == project_to_delete:
            self.current_project = None
            self.current_project_capture_dir = None
            if hasattr(self, "project_status_label"):
                self.project_status_label.setText("当前项目：未选择")
            self._update_capture_button_state()
            self._load_captures()
        
        # 删除项目照片目录
        project_capture_dir = self._get_project_capture_dir(project_to_delete)
        if project_capture_dir and project_capture_dir.exists():
            try:
                import shutil
                shutil.rmtree(project_capture_dir)
                self.logger.info("Deleted project capture directory: %s", project_capture_dir)
            except Exception as exc:
                self.logger.warning("Failed to delete project capture directory: %s", exc)
                QMessageBox.warning(self, "警告", f"项目配置已删除，但删除照片目录时出错：{exc}")
        
        self.logger.info("Project deleted: %s", project_name)
        self.statusBar().showMessage(f"已删除项目：{project_name}")
        QMessageBox.information(self, "成功", f"项目 {project_name} 已删除。")

    def _clear_detection_preview(self, message: str = "暂无检测结果") -> None:
        """重置检测结果展示内容"""
        if self.detection_preview_label is None:
            return
        self.detection_preview_label.setPixmap(QPixmap())
        self.detection_preview_label.setText(message)
        self.detection_preview_label.setAlignment(Qt.AlignCenter)
        if self.detection_result_path_label is not None:
            self.detection_result_path_label.setText("当前展示：无")
        if self.detection_stats_label is not None:
            self.detection_stats_label.setText("花粉统计：--")
        if hasattr(self, "detection_result_list"):
            self.detection_result_list.blockSignals(True)
            self.detection_result_list.clear()
            self.detection_result_list.blockSignals(False)
        self.detection_results.clear()
        self.detection_result_stats.clear()
        self.latest_detection_result = None

    def _set_detection_preview_pixmap(self, pixmap: QPixmap, source_label: str = "临时数据") -> None:
        """展示检测结果图像"""
        if self.detection_preview_label is None:
            return
        self.detection_preview_label.setPixmap(pixmap)
        self.detection_preview_label.setText("")
        self.detection_preview_label.setAlignment(Qt.AlignCenter)
        if self.detection_result_path_label is not None:
            self.detection_result_path_label.setText(f"当前展示：{source_label}")
        if self.latest_detection_result is not None:
            self._update_detection_stats_label(self.latest_detection_result)

    def _update_detection_preview_from_path(self, image_path: Path, *, add_entry: bool = True) -> None:
        if self.detection_preview_label is None or not image_path.exists():
            return
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    qimage = QImage(
                        img.tobytes("raw", "RGB"),
                        img.width,
                        img.height,
                        img.width * 3,
                        QImage.Format_RGB888,
                    )
                    pixmap = QPixmap.fromImage(qimage)
            except Exception as exc:
                self.logger.warning("Failed to update detection preview for %s: %s", image_path, exc)
                return

        target_size = self.detection_preview_label.size()
        if target_size.width() > 0 and target_size.height() > 0:
            pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.latest_detection_result = image_path
        self._set_detection_preview_pixmap(pixmap, image_path.name)
        if add_entry:
            self._add_detection_result_entry(image_path)

    def _add_detection_result_entry(self, image_path: Path, select: bool = True) -> None:
        """将检测结果添加到列表，避免重复，并保持最新项选中。"""
        if not hasattr(self, "detection_result_list"):
            return
        resolved = image_path.resolve()
        stats = self._load_detection_stats(resolved)
        self.detection_result_stats[str(resolved)] = stats
        if any(resolved == existing.resolve() for existing in self.detection_results):
            # 已存在时只需同步选中
            count = self.detection_result_list.count()
            for idx in range(count):
                item = self.detection_result_list.item(idx)
                if item and Path(item.data(Qt.UserRole)).resolve() == resolved:
                    if select:
                        self.detection_result_list.blockSignals(True)
                        self.detection_result_list.setCurrentRow(idx)
                        self.detection_result_list.blockSignals(False)
                    break
            self._update_detection_stats_label(resolved)
            return

        self.detection_results.append(resolved)
        item = QListWidgetItem(resolved.name)
        item.setData(Qt.UserRole, str(resolved))
        self.detection_result_list.blockSignals(True)
        self.detection_result_list.addItem(item)
        if select:
            self.detection_result_list.setCurrentItem(item)
        self.detection_result_list.blockSignals(False)
        if select:
            self._update_detection_stats_label(resolved)

    def _on_detection_result_selected(self) -> None:
        if not hasattr(self, "detection_result_list"):
            return
        item = self.detection_result_list.currentItem()
        if not item:
            return
        path_str = item.data(Qt.UserRole)
        if not path_str:
            return
        path = Path(path_str)
        self._update_detection_preview_from_path(path, add_entry=False)

    def _load_detection_stats(self, annotated_image_path: Path) -> Dict[str, int]:
        summary_path = annotated_image_path.parent / "summary.txt"
        stats: Dict[str, int] = {}
        if not summary_path.exists():
            return stats
        try:
            with summary_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    if line.startswith("统计:"):
                        payload = line.split(":", 1)[1].strip()
                        if payload:
                            data = json.loads(payload)
                            if isinstance(data, dict):
                                stats = {
                                    str(name): int(count)
                                    for name, count in data.items()
                                }
                        break
        except Exception as exc:
            self.logger.warning("Failed to load detection stats for %s: %s", annotated_image_path.name, exc)
        return stats

    def _update_detection_stats_label(self, image_path: Optional[Path]) -> None:
        if self.detection_stats_label is None:
            return
        if image_path is None:
            self.detection_stats_label.setText("花粉统计：--")
            return
        stats = self.detection_result_stats.get(str(image_path.resolve()))
        if not stats:
            self.detection_stats_label.setText("花粉统计：未检测到花粉目标")
            return
        # 将标签名称从短代码转换为中文名称
        parts = []
        for name, count in stats.items():
            # 查找映射，如果找不到则使用原名称
            display_name = self.label_name_mapping.get(name, name)
            parts.append(f"{display_name} × {count}")
        self.detection_stats_label.setText("花粉统计：" + "；".join(parts))

    def _load_detection_history_results(self) -> None:
        """加载历史检测 run 列表，并默认展示最新一次的检测结果。"""
        if not hasattr(self, "detection_run_list") or not hasattr(self, "detection_result_list"):
            return
        if not self.detection_output_dir.exists():
            return

        runs: List[tuple[float, Path]] = []
        try:
            # 查找所有目录（支持 run_* 格式和项目名称格式）
            for run_dir in sorted(self.detection_output_dir.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                # 跳过系统目录
                if run_dir.name.startswith('.'):
                    continue
                try:
                    mtime = run_dir.stat().st_mtime
                except OSError:
                    mtime = 0.0
                runs.append((mtime, run_dir))
        except Exception as exc:
            self.logger.warning("Failed to enumerate detection history runs: %s", exc)
            return

        if not runs:
            self.detection_run_list.blockSignals(True)
            self.detection_run_list.clear()
            self.detection_run_list.blockSignals(False)
            self.detection_result_list.blockSignals(True)
            self.detection_result_list.clear()
            self.detection_result_list.blockSignals(False)
            self.detection_results.clear()
            self.detection_result_stats.clear()
            self._clear_detection_preview("暂无检测结果")
            return

        runs.sort(key=lambda item: item[0], reverse=True)

        self.detection_run_list.blockSignals(True)
        self.detection_run_list.clear()
        for _, run_dir in runs:
            item = QListWidgetItem(run_dir.name)
            item.setData(Qt.UserRole, str(run_dir))
            self.detection_run_list.addItem(item)
        self.detection_run_list.blockSignals(False)

        # 默认选中最新 run，并加载该 run 下的图片
        self.detection_run_list.blockSignals(True)
        self.detection_run_list.setCurrentRow(0)
        self.detection_run_list.blockSignals(False)

        first_item = self.detection_run_list.item(0)
        if first_item is not None:
            data = first_item.data(Qt.UserRole)
            if data:
                self._load_detection_results_for_run(Path(data))

    def _load_detection_results_for_run(self, run_dir: Path) -> None:
        """根据指定 run 目录加载该批次下的检测结果图片列表。"""
        if not hasattr(self, "detection_result_list"):
            return
        if not run_dir.exists() or not run_dir.is_dir():
            return

        entries: List[tuple[float, Path]] = []
        try:
            for image_dir in run_dir.iterdir():
                if not image_dir.is_dir():
                    continue
                annotated = next(image_dir.glob("*_annotated.jpg"), None)
                if annotated and annotated.exists():
                    try:
                        mtime = annotated.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    entries.append((mtime, annotated))
        except Exception as exc:
            self.logger.warning("Failed to enumerate detection results for %s: %s", run_dir, exc)
            return

        self.detection_result_list.blockSignals(True)
        self.detection_result_list.clear()
        self.detection_result_list.blockSignals(False)
        self.detection_results.clear()
        self.detection_result_stats.clear()

        if not entries:
            self._clear_detection_preview("该检测任务下暂无检测结果图像")
            return

        entries.sort(key=lambda item: item[0], reverse=True)

        for _, annotated_path in entries:
            self._add_detection_result_entry(annotated_path, select=False)

        if self.detection_result_list.count() > 0:
            self.detection_result_list.blockSignals(True)
            self.detection_result_list.setCurrentRow(0)
            self.detection_result_list.blockSignals(False)
            self._update_detection_preview_from_path(entries[0][1], add_entry=False)

    def _on_detection_run_selected(self) -> None:
        """当选择不同的 run_... 时，刷新下方的检测结果列表。"""
        if not hasattr(self, "detection_run_list"):
            return
        item = self.detection_run_list.currentItem()
        if not item:
            return
        data = item.data(Qt.UserRole)
        if not data:
            return
        self._load_detection_results_for_run(Path(data))
    
    def _handle_delete_detection_run(self) -> None:
        """删除选中的检测任务。"""
        if not hasattr(self, "detection_run_list"):
            return
        item = self.detection_run_list.currentItem()
        if not item:
            QMessageBox.warning(self, "提示", "请先选择一个检测任务。")
            return
        
        data = item.data(Qt.UserRole)
        if not data:
            return
        
        run_dir = Path(data)
        if not run_dir.exists():
            QMessageBox.warning(self, "提示", "检测任务目录不存在。")
            return
        
        # 确认删除
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除检测任务：{run_dir.name} 吗？\n\n此操作将删除所有检测结果，且无法恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            import shutil
            shutil.rmtree(run_dir)
            self.logger.info("Deleted detection run: %s", run_dir)
            self.statusBar().showMessage(f"已删除检测任务：{run_dir.name}")
            
            # 重新加载检测任务列表
            self._load_detection_history_results()
            QMessageBox.information(self, "成功", f"检测任务 {run_dir.name} 已删除。")
        except Exception as exc:
            self.logger.error("Failed to delete detection run: %s", exc, exc_info=True)
            QMessageBox.critical(self, "错误", f"删除检测任务失败：{exc}")

    def _reload_stats_runs(self) -> None:
        """加载所有 run_* 目录，用于统计数据页的列表展示。"""
        if not hasattr(self, "stats_run_list"):
            return
        if not self.detection_output_dir.exists():
            return
        runs = []
        try:
            # 查找所有目录（支持 run_* 格式和项目名称格式）
            for run_dir in sorted(self.detection_output_dir.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                # 跳过系统目录
                if run_dir.name.startswith('.'):
                    continue
                pie = run_dir / "pollen_pie.png"
                bar = run_dir / "pollen_bar.png"
                if pie.exists() or bar.exists():
                    try:
                        mtime = run_dir.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    runs.append((mtime, run_dir, pie, bar))
        except Exception as exc:
            self.logger.warning("Failed to enumerate stats runs: %s", exc)
            return

        if not runs:
            self.stats_run_list.blockSignals(True)
            self.stats_run_list.clear()
            self.stats_run_list.blockSignals(False)
            if hasattr(self, "stats_artifact_list"):
                self.stats_artifact_list.blockSignals(True)
                self.stats_artifact_list.clear()
                self.stats_artifact_list.blockSignals(False)
            self._set_stats_preview(None, "暂无检测统计数据")
            if getattr(self, "stats_preview_path_label", None) is not None:
                self.stats_preview_path_label.setText("当前展示：无")
            return

        runs.sort(key=lambda item: item[0], reverse=True)

        self.stats_run_list.blockSignals(True)
        self.stats_run_list.clear()
        for _, run_dir, pie, bar in runs:
            text = run_dir.name
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, str(run_dir))
            self.stats_run_list.addItem(item)
        self.stats_run_list.blockSignals(False)

        # 默认选中最新一次
        if self.stats_run_list.count() > 0:
            self.stats_run_list.blockSignals(True)
            self.stats_run_list.setCurrentRow(0)
            self.stats_run_list.blockSignals(False)
            current = self.stats_run_list.currentItem()
            if current is not None:
                data = current.data(Qt.UserRole)
                if data:
                    self._load_stats_artifacts_for_run(Path(data))

    def _on_stats_run_selected(self) -> None:
        if not hasattr(self, "stats_run_list"):
            return
        item = self.stats_run_list.currentItem()
        if not item:
            return
        data = item.data(Qt.UserRole)
        if not data:
            return
        self._load_stats_artifacts_for_run(Path(data))
    
    def _handle_delete_stats_run(self) -> None:
        """删除统计数据页面中选中的检测任务。"""
        run_dir = self._get_current_stats_run_dir()
        if not run_dir or not run_dir.exists():
            QMessageBox.warning(self, "提示", "请先选择一个检测任务。")
            return
        
        # 确认删除
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除检测任务：{run_dir.name} 吗？\n\n此操作将删除所有检测结果，且无法恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            import shutil
            shutil.rmtree(run_dir)
            self.logger.info("Deleted detection run: %s", run_dir)
            self.statusBar().showMessage(f"已删除检测任务：{run_dir.name}")
            
            # 重新加载检测任务列表
            self._reload_stats_runs()
            QMessageBox.information(self, "成功", f"检测任务 {run_dir.name} 已删除。")
        except Exception as exc:
            self.logger.error("Failed to delete detection run: %s", exc, exc_info=True)
            QMessageBox.critical(self, "错误", f"删除检测任务失败：{exc}")

    def _load_stats_artifacts_for_run(self, run_dir: Path) -> None:
        """填充统计图列表（饼状图、柱状图），并默认展示第一个可用图。"""
        if not hasattr(self, "stats_artifact_list"):
            return
        artifacts: List[tuple[str, Path]] = []
        pie_path = run_dir / "pollen_pie.png"
        bar_path = run_dir / "pollen_bar.png"
        if pie_path.exists():
            artifacts.append(("饼状图（花粉占比）", pie_path))
        if bar_path.exists():
            artifacts.append(("柱状图（花粉数量）", bar_path))

        self.stats_artifact_list.blockSignals(True)
        self.stats_artifact_list.clear()
        for label, path in artifacts:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, str(path))
            self.stats_artifact_list.addItem(item)
        self.stats_artifact_list.blockSignals(False)

        if not artifacts:
            if getattr(self, "stats_preview_path_label", None) is not None:
                self.stats_preview_path_label.setText("当前展示：无")
            self._set_stats_preview(None, "该检测任务下暂无统计图")
            return

        self.stats_artifact_list.blockSignals(True)
        self.stats_artifact_list.setCurrentRow(0)
        self.stats_artifact_list.blockSignals(False)
        first = self.stats_artifact_list.item(0)
        if first is not None:
            data = first.data(Qt.UserRole)
            if data:
                self._show_stats_preview(Path(data))

    def _on_stats_artifact_selected(self) -> None:
        if not hasattr(self, "stats_artifact_list"):
            return
        item = self.stats_artifact_list.currentItem()
        if not item:
            return
        data = item.data(Qt.UserRole)
        if not data:
            return
        self._show_stats_preview(Path(data))
    
    def _get_current_stats_run_dir(self) -> Optional[Path]:
        """获取当前选中的检测任务目录。"""
        if not hasattr(self, "stats_run_list"):
            return None
        item = self.stats_run_list.currentItem()
        if not item:
            return None
        data = item.data(Qt.UserRole)
        if not data:
            return None
        return Path(data)
    
    def _handle_download_pdf(self) -> None:
        """生成并下载当前检测任务的PDF报告。"""
        run_dir = self._get_current_stats_run_dir()
        if not run_dir or not run_dir.exists():
            QMessageBox.warning(self, "提示", "请先选择一个检测任务。")
            return
        
        # 选择保存位置
        default_filename = f"{run_dir.name}_检测报告.pdf"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存PDF报告",
            default_filename,
            "PDF文件 (*.pdf)"
        )
        if not file_path:
            return
        
        try:
            self._generate_pdf_report(run_dir, Path(file_path))
            QMessageBox.information(self, "成功", f"PDF报告已保存到：\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"生成PDF报告失败：\n{exc}")
            self.logger.exception("Failed to generate PDF report: %s", exc)
    
    def _generate_pdf_report(self, run_dir: Path, output_path: Path) -> None:
        """生成PDF检测报告。"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            import os
        except ImportError:
            raise ImportError("未安装 reportlab 库，请运行: pip install reportlab")
        
        # 创建PDF文档
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # 注册中文字体
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
        ]
        chinese_font_name = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    if font_path.endswith('.ttf'):
                        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                        chinese_font_name = 'ChineseFont'
                        break
                except:
                    continue
        
        # 创建自定义样式
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1f2933'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName=chinese_font_name if chinese_font_name else 'Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#374151'),
            spaceAfter=12,
            spaceBefore=20,
            fontName=chinese_font_name if chinese_font_name else 'Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#111827'),
            fontName=chinese_font_name if chinese_font_name else 'Helvetica'
        )
        
        # 标题
        story.append(Paragraph("花粉检测报告", title_style))
        story.append(Spacer(1, 0.5*cm))
        
        # 检测任务信息
        story.append(Paragraph("检测任务信息", heading_style))
        detection_time = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        info_data = [
            ["任务名称", run_dir.name],
            ["检测时间", detection_time],
        ]
        
        # 尝试读取项目信息
        if self.current_project:
            info_data.append(["项目名称", self.current_project.get("name", "未知")])
            info_data.append(["气象站", self.current_project.get("station", "未知")])
        
        info_table = Table(info_data, colWidths=[4*cm, 12*cm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#111827')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), chinese_font_name if chinese_font_name else 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), chinese_font_name if chinese_font_name else 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.5*cm))
        
        # 收集统计数据
        species_totals: Dict[str, int] = {}
        per_image_stats: Dict[str, Dict[str, int]] = {}
        
        for image_dir in run_dir.iterdir():
            if not image_dir.is_dir():
                continue
            summary_file = image_dir / "summary.txt"
            if not summary_file.exists():
                continue
            try:
                with summary_file.open("r", encoding="utf-8") as fp:
                    image_name = None
                    stats_dict: Dict[str, int] = {}
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
                    if stats_dict and image_name:
                        per_image_stats[image_name] = stats_dict
                        for name, count in stats_dict.items():
                            species_totals[name] = species_totals.get(name, 0) + count
            except Exception as exc:
                self.logger.warning("Failed to read summary: %s", exc)
                continue
        
        if not species_totals:
            story.append(Paragraph("本次检测未发现花粉目标", normal_style))
            doc.build(story)
            return
        
        # 统计摘要
        story.append(Paragraph("统计摘要", heading_style))
        total_count = sum(species_totals.values())
        summary_data = [
            ["检测图片数量", str(len(per_image_stats))],
            ["花粉种类数", str(len(species_totals))],
            ["花粉总数量", str(total_count)],
        ]
        summary_table = Table(summary_data, colWidths=[4*cm, 12*cm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#111827')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), chinese_font_name if chinese_font_name else 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), chinese_font_name if chinese_font_name else 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.5*cm))
        
        # 饼状图和柱状图
        pie_path = run_dir / "pollen_pie.png"
        bar_path = run_dir / "pollen_bar.png"
        
        if pie_path.exists():
            story.append(Paragraph("花粉种类占比", heading_style))
            img = Image(str(pie_path), width=12*cm, height=12*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))
        
        if bar_path.exists():
            story.append(Paragraph("花粉种类数量统计", heading_style))
            img = Image(str(bar_path), width=14*cm, height=8*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))
        
        # 汇总统计表
        story.append(Paragraph("花粉种类汇总", heading_style))
        # 将标签转换为中文
        summary_rows = [["花粉种类", "数量", "占比"]]
        for species, count in sorted(species_totals.items(), key=lambda x: x[1], reverse=True):
            chinese_name = self.label_name_mapping.get(species, species)
            percentage = f"{count / total_count * 100:.1f}%"
            summary_rows.append([chinese_name, str(count), percentage])
        
        summary_table = Table(summary_rows, colWidths=[8*cm, 3*cm, 3*cm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), chinese_font_name if chinese_font_name else 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), chinese_font_name if chinese_font_name else 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ]))
        story.append(summary_table)
        story.append(PageBreak())
        
        # 每张图片的详细统计
        story.append(Paragraph("详细统计", heading_style))
        for image_name, stats in sorted(per_image_stats.items()):
            story.append(Paragraph(f"图像：{image_name}", heading_style))
            detail_rows = [["花粉种类", "数量"]]
            for species, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                chinese_name = self.label_name_mapping.get(species, species)
                detail_rows.append([chinese_name, str(count)])
            
            detail_table = Table(detail_rows, colWidths=[10*cm, 4*cm])
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6b7280')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), chinese_font_name if chinese_font_name else 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), chinese_font_name if chinese_font_name else 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
            ]))
            story.append(detail_table)
            story.append(Spacer(1, 0.3*cm))
        
        # 构建PDF
        doc.build(story)

    def _set_stats_preview(self, pixmap: Optional[QPixmap], message: str = "") -> None:
        if getattr(self, "stats_preview_label", None) is None:
            return
        if pixmap is not None and not pixmap.isNull():
            target = self.stats_preview_label.size()
            if target.width() > 0 and target.height() > 0:
                pixmap = pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.stats_preview_label.setPixmap(pixmap)
            self.stats_preview_label.setText("")
        else:
            self.stats_preview_label.setPixmap(QPixmap())
            self.stats_preview_label.setText(message or "暂无数据")

    def _show_stats_preview(self, artifact_path: Path) -> None:
        if getattr(self, "stats_preview_label", None) is None:
            return
        if getattr(self, "stats_preview_path_label", None) is not None:
            self.stats_preview_path_label.setText(f"当前展示：{artifact_path.name}")
        if not artifact_path.exists():
            self._set_stats_preview(None, "文件不存在")
            return
        pix = QPixmap(str(artifact_path))
        if pix.isNull():
            self._set_stats_preview(None, "图像加载失败")
            return
        self._set_stats_preview(pix)

    def _collect_history_images(self) -> List[Path]:
        images: List[Path] = []
        if self.history_list is None:
            return images
        for idx in range(self.history_list.count()):
            item = self.history_list.item(idx)
            if item is None:
                continue
            data = item.data(Qt.UserRole)
            if not data:
                continue
            path = Path(str(data))
            if path.exists():
                images.append(path)
        return images

    def _on_tab_changed(self, index: int) -> None:
        if self.page_tabs is None:
            return
        widget = self.page_tabs.widget(index)

        if getattr(self, "detection_tab", None) is widget:
            self._load_detection_history_results()
        # 进入“统计数据”页时刷新 run 列表
        if hasattr(self, "stats_run_list") and self.page_tabs.tabText(index) == "统计数据":
            self._reload_stats_runs()

    def _create_slider(
        self,
        minimum: int,
        maximum: int,
        label_text: str,
        parent_layout: QVBoxLayout,
        on_value_changed,
        on_commit,
    ) -> tuple[QSlider, QLabel]:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)  # 增加标签和滑块之间的间距

        header_layout = QHBoxLayout()
        name_label = QLabel(label_text)
        name_label.setStyleSheet("color:#1f2933; font-weight:600;")
        header_layout.addWidget(name_label)
        header_layout.addStretch(1)

        value_label = QLabel("0")
        value_label.setStyleSheet("color:#1f2933;")
        header_layout.addWidget(value_label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.valueChanged.connect(lambda val: on_value_changed(val, value_label))
        slider.sliderReleased.connect(on_commit)

        container_layout.addLayout(header_layout)
        container_layout.addWidget(slider)

        parent_layout.addWidget(container)
        return slider, value_label

    # ------------------------------------------------------------------
    # Camera control and status
    # ------------------------------------------------------------------
    def _ensure_camera_ready(self) -> None:
        if self.camera_ready:
            return

        self.logger.debug("Attempting to start camera")
        try:
            status = self.service.start()
        except RuntimeError as exc:
            self.camera_ready = False
            self.statusBar().showMessage("未检测到摄像头，正在自动重试...")
            self.image_label.setText("未检测到摄像头\n请检查连接后等待自动重试")
            self.logger.warning("Camera not detected: %s", exc)
            self._update_control_states()
            return

        self.logger.info("Camera detected and streaming")
        self.camera_ready = True
        self._apply_status(status)
        self._load_color_controls()
        self._load_exposure_controls()
        self._load_speed_controls()
        self._load_sharpening_controls()
        self._load_misc_controls()
        self._populate_devices()
        self._update_control_states()
        self.statusBar().showMessage("摄像头已连接")
        QTimer.singleShot(200, self._maybe_apply_startup_calibration)

    def _apply_status(self, status: Dict) -> None:
        self.camera_ready = status.get("running", False)
        self.available_resolutions = status.get("available_resolutions", [])
        res = status.get("resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            self.live_resolution = {"width": int(res[0]), "height": int(res[1])}
        else:
            self.live_resolution = None
        self.frame_rate = status.get("frame_rate")

        index = status.get("resolution_index")
        if isinstance(index, int):
            current_index = index
        elif self.available_resolutions:
            current_index = self.available_resolutions[0]["index"]
        else:
            current_index = None

        self._refresh_resolution_combo(current_index)
        self._update_live_info()
        if self.camera_ready:
            self.statusBar().showMessage("摄像头已连接")
        else:
            self.statusBar().showMessage("未检测到摄像头，正在自动重试...")

    def _refresh_status(self) -> None:
        status = self.service.status()
        self._apply_status(status)
        self._populate_devices()

    def _maybe_apply_startup_calibration(self) -> None:
        """Apply calibration once after the camera becomes ready."""
        if not self.camera_ready or self._auto_calibration_done:
            return
        self._auto_calibration_done = True
        self.logger.info("Applying startup auto calibration")
        try:
            values = self.service.auto_color_balance()
            self._sync_color_sliders(values)
            self._refresh_status()
            self.statusBar().showMessage("已自动完成校准")
        except RuntimeError as exc:
            self.logger.warning("Startup auto calibration failed: %s", exc)
            self.statusBar().showMessage("自动校准失败，请手动点击“校准”")

    def _refresh_resolution_combo(self, current_index: Optional[int]) -> None:
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()

        options = []
        if self.available_resolutions:
            options.extend(self.available_resolutions[:2])
            if current_index is not None:
                match = next(
                    (item for item in self.available_resolutions if item["index"] == current_index),
                    None,
                )
                if match:
                    options.append(match)

        unique: Dict[int, Dict[str, int]] = {}
        for option in options:
            unique[option["index"]] = option

        for option in unique.values():
            label = f"{option['width']}×{option['height']}"
            self.resolution_combo.addItem(label, option["index"])

        if current_index is not None:
            idx = self.resolution_combo.findData(current_index)
            if idx != -1:
                self.resolution_combo.setCurrentIndex(idx)

        self.resolution_combo.setEnabled(self.camera_ready and self.resolution_combo.count() > 0)
        self.resolution_combo.blockSignals(False)

    def _update_live_info(self) -> None:
        if self.frame_rate is not None:
            self.frame_rate_label.setText(f"帧率：{self.frame_rate:.1f} fps")
        else:
            self.frame_rate_label.setText("帧率：-- fps")

        if self.live_resolution:
            self.resolution_label.setText(
                f"当前尺寸：{self.live_resolution['width']}×{self.live_resolution['height']}"
            )
        else:
            self.resolution_label.setText("当前尺寸：--")

    def _update_control_states(self) -> None:
        ready = self.camera_ready

        # 拍照按钮的状态由 _update_capture_button_state 单独管理，不在这里控制
        always_enabled = (
            self.auto_color_button,
            self.brightness_slider,
            self.hue_slider,
            self.saturation_slider,
            self.exposure_target_slider,
            self.exposure_time_slider,
            self.gain_slider,
            self.speed_slider,
            self.auto_exposure_check,
            self.sharpening_strength_slider,
            self.sharpening_radius_slider,
            self.sharpening_threshold_slider,
            self.negative_check,
            self.low_noise_check,
            self.low_power_check,
            self.remove_shutter_effect_check,
            self.debayer_combo,
            self.tone_mapping_combo,
            self.shutter_mode_combo,
            self.readout_mode_combo,
        )
        for widget in always_enabled:
            widget.setEnabled(ready)

        self.reconnect_button.setEnabled(True)
        
        # 更新拍照按钮状态（需要同时检查项目状态和相机状态）
        self._update_capture_button_state()

    # ------------------------------------------------------------------
    # Frame handling
    # ------------------------------------------------------------------
    def _update_frame(self) -> None:
        if self.camera_ready and not self.service.is_running():
            # 摄像头在运行过程中被关闭或拔出，立刻提示用户并禁用相关功能
            self.camera_ready = False
            self.frame_rate = None
            self.live_resolution = None
            self.image_label.setText("显微镜已被关闭或拔出\n请检查连接后重新打开")
            self.statusBar().showMessage("显微镜已被关闭或拔出")
            self._update_control_states()
            self._update_live_info()
            return

        frame = self.service.wait_for_frame(0)
        if frame is None:
            frame = self.service.get_latest_frame()

        if frame is None:
            if not self.camera_ready:
                self.image_label.setText("未检测到摄像头\n请检查连接后等待自动重试")
            self._update_live_info()
            return

        self.frame_rate = self.service.get_frame_rate()
        self.live_resolution = {"width": frame.width, "height": frame.height}
        self._update_live_info()

        pixmap = self._frame_to_pixmap(frame)
        # 使用标签的实际尺寸进行缩放，确保画面填满显示区域且不产生黑边
        label_size = self.image_label.size()  # 使用标签的实际尺寸
        scaled = pixmap.scaled(
            label_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        # 如果缩放后的图像大于标签尺寸，进行裁剪以完全填满
        if scaled.width() > label_size.width() or scaled.height() > label_size.height():
            # 计算裁剪区域（居中裁剪）
            x = (scaled.width() - label_size.width()) // 2
            y = (scaled.height() - label_size.height()) // 2
            scaled = scaled.copy(x, y, label_size.width(), label_size.height())
        with_corners = self._draw_corner_overlay(scaled)
        self.image_label.setPixmap(with_corners)


    def _frame_to_pixmap(self, frame: Frame) -> QPixmap:
        bytes_per_line = frame.width * 3
        image = QImage(frame.data, frame.width, frame.height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    def _draw_corner_overlay(self, pixmap: QPixmap) -> QPixmap:
        painter = QPainter(pixmap)
        w, h = pixmap.width(), pixmap.height()
        margin = max(12, w // 20)
        length = max(40, min(w, h) // 6)

        # 当前实时画面的四个角（绿色）
        pen_current = QPen(QColor(76, 175, 80))
        pen_current.setWidth(max(2, pixmap.width() // 150))
        painter.setPen(pen_current)

        # Top-left
        painter.drawLine(margin, margin, margin + length, margin)
        painter.drawLine(margin, margin, margin, margin + length)
        # Top-right
        painter.drawLine(w - margin, margin, w - margin - length, margin)
        painter.drawLine(w - margin, margin, w - margin, margin + length)
        # Bottom-left
        painter.drawLine(margin, h - margin, margin + length, h - margin)
        painter.drawLine(margin, h - margin, margin, h - margin - length)
        # Bottom-right
        painter.drawLine(w - margin, h - margin, w - margin - length, h - margin)
        painter.drawLine(w - margin, h - margin, w - margin, h - margin - length)

        painter.end()
        return pixmap


    # ------------------------------------------------------------------
    # Resolution handling
    # ------------------------------------------------------------------
    def _handle_resolution_combo(self, index: int) -> None:
        data = self.resolution_combo.itemData(index)
        if data is None or not self.camera_ready:
            return
        self._change_resolution(int(data))

    def _change_resolution(self, resolution_index: int) -> None:
        self.logger.info("Requesting resolution change to index %s", resolution_index)
        self.resolution_combo.setEnabled(False)
        try:
            self.service.set_resolution(resolution_index)
            self._refresh_status()
            QMessageBox.information(self, "提示", "分辨率已切换。")
            self.logger.info("Resolution changed successfully")
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"切换分辨率失败：{exc}")
            self.logger.warning("Resolution change failed: %s", exc)
        finally:
            self.resolution_combo.setEnabled(self.camera_ready)

    def _auto_load_default_models(self) -> None:
        seg_path = Path("seg.pt")
        cls_path = Path("classify.pt")
        missing: list[str] = []
        if not seg_path.is_file():
            missing.append(seg_path.name)
        if not cls_path.is_file():
            missing.append(cls_path.name)

        if missing:
            detail = "、".join(missing)
            self.model_status_label.setText(f"当前模型: 缺少默认权重({detail})")
            self.statusBar().showMessage("未找到默认模型，目标检测不可用")
            self.logger.warning("Default detection models missing: %s", detail)
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("正在自动加载默认模型...")
        try:
            model = load_yolo_model(
                seg_path,
                cls_path,
                conf_threshold=self.detection_conf,
                merge_iou=self.detection_iou,
                label_mapping=self.label_name_mapping,
            )
        except MissingDependencyError as exc:
            QMessageBox.warning(self, "缺少依赖", str(exc))
            self.statusBar().showMessage("自动加载默认模型失败")
            self.logger.exception("Failed to auto load detection models: %s", exc)
            return
        except ModelLoadError as exc:
            QMessageBox.critical(self, "错误", f"自动加载模型失败：{exc}")
            self.statusBar().showMessage("自动加载默认模型失败")
            self.logger.exception("Failed to auto load detection models: %s", exc)
            return
        except Exception as exc:  # pragma: no cover - unexpected errors
            QMessageBox.critical(self, "错误", f"自动加载模型失败：{exc}")
            self.statusBar().showMessage("自动加载默认模型失败")
            self.logger.exception("Failed to auto load detection models: %s", exc)
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.detection_model = model
        self.detection_model_path = seg_path
        self.detection_cls_model_path = cls_path
        self.model_status_label.setText(
            f"当前模型: 分割[{self.detection_model_path.name}] | 分类[{self.detection_cls_model_path.name}]"
        )
        self.statusBar().showMessage("已自动加载默认模型，可直接执行目标检测")
        self.logger.info(
            "Loaded default detection models: seg=%s, cls=%s",
            self.detection_model_path,
            self.detection_cls_model_path,
        )

    def _handle_run_detection(self) -> None:
        if self.detection_in_progress:
            QMessageBox.information(self, "提示", "目标检测正在运行，请稍后。")
            return
        if self.detection_model is None:
            QMessageBox.warning(self, "提示", "请先加载 .pt 模型后再执行检测。")
            return

        history_images = self._collect_history_images()
        if not history_images:
            QMessageBox.information(self, "提示", "没有可供检测的历史照片。")
            return

        if self.page_tabs is not None and getattr(self, "detection_tab", None) is not None:
            self.page_tabs.setCurrentWidget(self.detection_tab)

        # 检查是否有当前项目
        project_name = None
        if self.current_project:
            project_name = self.current_project.get("name", "")
        
        self.detection_in_progress = True
        self.detect_image_button.setEnabled(False)
        self._clear_detection_preview("正在执行目标检测，请稍候...")
        self.statusBar().showMessage(f"正在对 {len(history_images)} 张历史图像进行目标检测...")

        self._detection_thread = QThread(self)
        self._detection_worker = YoloDetectionWorker(
            self.detection_model,
            history_images,
            self.detection_output_dir,
            conf=self.detection_conf,
            iou=self.detection_iou,
            project_name=project_name,
            label_mapping=self.label_name_mapping,
        )
        self._detection_worker.moveToThread(self._detection_thread)
        self._detection_thread.started.connect(self._detection_worker.run)
        self._detection_worker.progress.connect(self._on_detection_progress)
        self._detection_worker.finished.connect(self._on_detection_finished)
        self._detection_worker.finished.connect(self._detection_thread.quit)
        self._detection_worker.finished.connect(self._detection_worker.deleteLater)
        self._detection_thread.finished.connect(self._detection_thread.deleteLater)
        self._detection_thread.start()

    def _on_detection_progress(self, current: int, total: int, source_path: str, output_path: str) -> None:
        filename = Path(source_path).name
        self.statusBar().showMessage(f"正在检测 {current}/{total}: {filename}")
        if output_path:
            out_path = Path(output_path)
            if out_path.exists():
                self._update_detection_preview_from_path(out_path)

    def _on_detection_finished(self, ok: bool, message: str, result_path: Optional[str]) -> None:
        self.detection_in_progress = False
        self.detect_image_button.setEnabled(True)
        self.statusBar().showMessage(message)

        # 清理线程引用
        self._detection_worker = None
        self._detection_thread = None

        if ok:
            if result_path:
                self._update_detection_preview_from_path(Path(result_path))
            else:
                self._clear_detection_preview("检测完成，但未生成输出文件。")
        else:
            if not result_path:
                self._clear_detection_preview("目标检测失败，请检查日志后重试。")
            else:
                self._update_detection_preview_from_path(Path(result_path))
            QMessageBox.warning(self, "目标检测失败", message)

    # ------------------------------------------------------------------
    # Capture handling
    # ------------------------------------------------------------------
    def _handle_capture(self) -> None:
        """处理拍照按钮点击事件，防止重复点击和文件命名冲突"""
        # 防止重复点击：如果正在拍照，直接返回
        if self.capture_in_progress:
            self.logger.warning("拍照进行中，忽略重复点击")
            return

        if not self.camera_ready:
            QMessageBox.warning(self, "提示", "摄像头尚未就绪，无法拍照。")
            return

        # 使用锁确保同一时间只有一个拍照操作
        with self.capture_lock:
            if self.capture_in_progress:
                return
            self.capture_in_progress = True

        # 禁用拍照按钮，防止重复点击
        self.capture_button.setEnabled(False)
        try:
            # 在获取图像之前先读取行列值，避免在保存过程中被自动前进改变
            column = self.column_spin.value()
            row = self.row_spin.value()

            self.logger.info(
                "Capturing image at column=%s row=%s",
                column,
                row,
            )

            # 检查是否有当前项目
            if not self.current_project:
                QMessageBox.warning(self, "提示", "请先新建或选择项目后再拍照。")
                return
            
            # 确保项目目录存在
            if not self.current_project_capture_dir:
                self.current_project_capture_dir = self._get_project_capture_dir(self.current_project)
                if not self.current_project_capture_dir:
                    QMessageBox.warning(self, "提示", "无法创建项目照片目录，请检查项目名称。")
                    return
            
            # 生成文件名（使用已读取的行列值）
            filename = f"img_r{row+1:03d}_c{column+1:03d}.jpg"
            path = self.current_project_capture_dir / filename

            # 检查文件是否已存在，如果存在则添加序号后缀
            if path.exists():
                counter = 1
                while path.exists():
                    filename = f"img_r{row+1:03d}_c{column+1:03d}_{counter:02d}.jpg"
                    path = self.capture_dir / filename
                    counter += 1
                self.logger.warning("文件已存在，使用新文件名: %s", filename)

            try:
                # 获取图像帧
                frame = self.service.fetch_recent_frame(1.0)
            except RuntimeError as exc:
                QMessageBox.warning(self, "提示", f"获取图像失败：{exc}")
                self.logger.warning("Failed to fetch frame: %s", exc)
                return

            try:
                # 保存图像
                image = Image.frombytes("RGB", (frame.width, frame.height), frame.data)
                image.save(path, format="JPEG", quality=92)

                # 验证文件是否真正保存成功
                if not path.exists() or path.stat().st_size == 0:
                    raise RuntimeError("文件保存失败：文件不存在或大小为0")

                self.logger.info("Capture saved successfully: %s (size: %d bytes)", filename, path.stat().st_size)

            except Exception as exc:
                QMessageBox.warning(self, "提示", f"保存图像失败：{exc}")
                self.logger.error("Failed to save capture %s: %s", filename, exc, exc_info=True)
                return

            # 只有在保存成功后才执行以下操作
            self._load_captures()
            QMessageBox.information(self, "提示", f"已保存：{filename}")

            # 只有在保存成功后才自动前进位置
            if self.auto_advance_check.isChecked():
                self._advance_scan_position()

        finally:
            # 无论成功或失败，都要释放锁并重新启用按钮
            with self.capture_lock:
                self.capture_in_progress = False
            # 只有在有项目且相机就绪时才能启用拍照按钮
            self._update_capture_button_state()

    def _queue_image_save(self, path: Path, image: numpy.ndarray, quality: int = 92) -> None:
        try:
            if self.save_thread is None or not self.save_thread.is_alive():
                Image.fromarray(numpy.clip(image, 0, 255).astype(numpy.uint8)).save(
                    str(path), format="JPEG", quality=quality
                )
                self.logger.debug("Saved image synchronously: %s", path.name)
                return

            img_array = numpy.ascontiguousarray(numpy.clip(image, 0, 255).astype(numpy.uint8))
            height, width = img_array.shape[:2]
            payload = (str(path), img_array, width, height, quality)
            self.save_queue.put_nowait(payload)
        except queue.Full:
            Image.fromarray(numpy.clip(image, 0, 255).astype(numpy.uint8)).save(
                str(path), format="JPEG", quality=quality
            )
            self.logger.warning("Save queue full, image saved synchronously: %s", path.name)
        except Exception as exc:
            self.logger.error("Failed to queue image save for %s: %s", path.name, exc, exc_info=True)
            Image.fromarray(numpy.clip(image, 0, 255).astype(numpy.uint8)).save(
                str(path), format="JPEG", quality=quality
            )

    def _start_save_thread(self) -> None:
        """启动异步保存线程"""

        def _save_image_to_path(path: Path, image_data, width: int, height: int, quality: int) -> None:
            if isinstance(image_data, bytes):
                image = Image.frombytes("RGB", (width, height), image_data)
            elif isinstance(image_data, numpy.ndarray):
                if image_data.ndim == 3 and image_data.shape[2] == 3:
                    image = Image.fromarray(image_data.astype(numpy.uint8), "RGB")
                else:
                    raise ValueError(f"Unsupported array shape: {image_data.shape}")
            else:
                image = Image.frombytes("RGB", (width, height), bytes(image_data))
            image.save(str(path), format="JPEG", quality=quality)

        def save_worker():
            self.logger.info("Save thread started")
            while True:
                try:
                    item = self.save_queue.get(timeout=1.0)
                    if item is None:  # 退出信号
                        self.logger.info("Save thread received exit signal")
                        break
                    if isinstance(item, tuple):
                        if len(item) == 4:
                            path, image_data, width, height = item
                            quality = 80
                        elif len(item) == 5:
                            path, image_data, width, height, quality = item
                        else:
                            raise ValueError("Unexpected save queue payload length")
                    elif isinstance(item, dict):
                        path = item["path"]
                        image_data = item["image"]
                        width = item.get("width")
                        height = item.get("height")
                        quality = item.get("quality", 92)
                        if width is None or height is None:
                            if isinstance(image_data, numpy.ndarray):
                                height, width = image_data.shape[:2]
                            else:
                                raise ValueError("width/height missing for save payload")
                    else:
                        raise ValueError("Unknown payload type for save queue")

                    try:
                        _save_image_to_path(Path(path), image_data, width, height, quality)
                        self.logger.info("Image saved asynchronously: %s", Path(path).name)
                    except Exception as exc:
                        self.logger.error("Failed to save image %s: %s", Path(path).name, exc, exc_info=True)
                    finally:
                        self.save_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as exc:
                    self.logger.error("Save thread error: %s", exc, exc_info=True)

        try:
            self.save_thread = threading.Thread(target=save_worker, daemon=True, name="ImageSaveThread")
            self.save_thread.start()
            self.logger.info("Save thread created and started")
        except Exception as exc:
            self.logger.error("Failed to start save thread: %s", exc, exc_info=True)
            self.save_thread = None

    def _advance_scan_position(self) -> None:
        max_columns = self.max_columns_spin.value()
        max_rows = self.max_rows_spin.value()

        column = self.column_spin.value()
        row = self.row_spin.value()

        if self.scan_direction == 1:
            if column < max_columns - 1:
                self.column_spin.setValue(column + 1)
            elif row < max_rows - 1:
                self.row_spin.setValue(row + 1)
                self.scan_direction = -1
        else:
            if column > 0:
                self.column_spin.setValue(column - 1)
            elif row < max_rows - 1:
                self.row_spin.setValue(row + 1)
                self.scan_direction = 1

    def _load_captures(self) -> None:
        self.history_list.clear()
        
        # 如果没有当前项目，不显示任何照片
        if not self.current_project or not self.current_project_capture_dir:
            self.logger.info("No project selected, history list cleared")
            return
        
        # 只加载当前项目的照片
        candidates = []
        try:
            if self.current_project_capture_dir.exists():
                for p in self.current_project_capture_dir.iterdir():
                    if not p.is_file():
                        continue
                    name_lower = p.name.lower()
                    if not name_lower.startswith("img_"):
                        continue
                    if name_lower.endswith((".jpg", ".jpeg", ".tif", ".tiff")):
                        candidates.append(p)
        except FileNotFoundError:
            candidates = []
        
        files = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            self._add_history_item_widget(path)

        self.logger.info("Loaded %d capture thumbnails for project: %s", 
                        self.history_list.count(), 
                        self.current_project.get("name", "未知"))

    def _add_history_item_widget(self, path: Path) -> None:
        # Container widget
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # Thumbnail
        thumb = QLabel()
        thumb.setFixedSize(160, 120)
        thumb.setStyleSheet("background:#eceff1; border:1px solid #cfd8dc;")
        thumb.setAlignment(Qt.AlignCenter)
        thumb.setScaledContents(True)
        thumb.setText("缩略图")
        try:
            with Image.open(path) as img:
                # 确保是 RGB 模式
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # 生成缩略图（保持宽高比）
                img.thumbnail((160, 120), Image.Resampling.LANCZOS)
                width, height = img.size
                if width > 0 and height > 0:
                    # 转换为 QPixmap，指定 bytesPerLine
                    bytes_per_line = width * 3
                    q_image = QImage(img.tobytes("raw", "RGB"), width, height, bytes_per_line, QImage.Format_RGB888)
                    pm = QPixmap.fromImage(q_image)
                    if not pm.isNull():
                        thumb.setPixmap(pm)
                        thumb.setText("")
                    else:
                        self.logger.warning("QPixmap is null for %s", path.name)
                else:
                    self.logger.warning("Thumbnail size is zero for %s", path.name)
        except Exception as exc:
            self.logger.warning("Failed to load thumbnail for %s: %s", path.name, exc)
        vbox.addWidget(thumb)

        # Filename + delete row
        row = QHBoxLayout()
        name_label = QLabel(path.name)
        name_label.setStyleSheet("font-size:18px; color:#263238; font-weight:500;")
        name_label.setToolTip(path.name)
        row.addWidget(name_label, 1)
        btn = QPushButton("×")
        btn.setFixedSize(36, 36)  # 从24x24增大到36x36，更容易点击
        btn.setStyleSheet("QPushButton{font-size:24px; font-weight:700; color:#d32f2f; background-color:#ffebee; border:1px solid #ef9a9a; border-radius:4px;} QPushButton:hover{background-color:#ffcdd2; border-color:#e57373;} QPushButton:pressed{background-color:#ef9a9a;}")
        btn.setCursor(Qt.PointingHandCursor)  # 鼠标悬停时显示手型光标
        btn.clicked.connect(lambda _, p=str(path): self._delete_capture(p))
        row.addWidget(btn)
        vbox.addLayout(row)

        item = QListWidgetItem()
        item.setSizeHint(container.sizeHint())
        item.setData(Qt.UserRole, str(path.resolve()))
        self.history_list.addItem(item)
        self.history_list.setItemWidget(item, container)

    def _delete_capture(self, path_str: str) -> None:
        try:
            p = Path(path_str)
            if p.exists():
                p.unlink()
        except Exception as exc:
            self.logger.warning("Failed to delete %s: %s", path_str, exc)
        self._load_captures()

    def _populate_devices(self) -> None:
        try:
            arr = nncam.Nncam.EnumV2()
        except Exception:
            arr = []
        current_id = self.device_combo.currentData() if self.device_combo.count() else None
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for dev in arr:
            self.device_combo.addItem(dev.displayname, dev.id)
        if current_id is not None:
            idx = self.device_combo.findData(current_id)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
        if self.device_combo.count() == 0:
            self.device_combo.addItem("未检测到摄像头", None)
            self.device_combo.setCurrentIndex(0)
        self.device_combo.blockSignals(False)
        combo_data = self.device_combo.itemData(self.device_combo.currentIndex())
        self.device_combo.setEnabled(combo_data is not None)

    def _on_device_selected(self, index: int) -> None:
        devid = self.device_combo.itemData(index) if index >= 0 else None
        if not devid:
            return
        try:
            self.service.stop()
        except Exception:
            pass
        self.camera_ready = False
        self._auto_calibration_done = False
        self._update_control_states()
        try:
            status = self.service.start(camera_identifier=devid)
        except RuntimeError as exc:
            self.statusBar().showMessage("切换摄像头失败")
            self.logger.warning("Switch device failed: %s", exc)
            return
        self.camera_ready = True
        self._apply_status(status)
        self._load_color_controls()
        self._load_exposure_controls()
        self._load_speed_controls()
        self._load_sharpening_controls()
        self._load_misc_controls()
        self._update_control_states()
        self.statusBar().showMessage("已切换摄像头")
        QTimer.singleShot(200, self._maybe_apply_startup_calibration)

    def _open_capture(self, item: QListWidgetItem) -> None:
        path_str = item.data(Qt.UserRole)
        if path_str:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path_str))

    # ------------------------------------------------------------------
    # Color control handlers
    # ------------------------------------------------------------------
    def _load_color_controls(self) -> None:
        if not self.camera_ready:
            return

        try:
            values = self.service.get_color_controls()
        except RuntimeError:
            return

        self._sync_color_sliders(values)

    def _sync_color_sliders(self, values: Dict[str, int]) -> None:
        self._set_slider_value(self.brightness_slider, self.brightness_value, values.get("brightness", 0))
        self._set_slider_value(self.hue_slider, self.hue_value, values.get("hue", 0))
        self._set_slider_value(
            self.saturation_slider,
            self.saturation_value,
            values.get("saturation", nncam.NNCAM_SATURATION_DEF),
        )

    def _set_slider_value(self, slider: QSlider, label: QLabel, value: int) -> None:
        slider.blockSignals(True)
        slider.setValue(value)
        label.setText(str(value))
        slider.blockSignals(False)

    def _on_brightness_changed(self, value: int, label: QLabel) -> None:
        label.setText(str(value))

    def _commit_brightness(self) -> None:
        self._commit_color_change(brightness=self.brightness_slider.value())

    def _on_hue_changed(self, value: int, label: QLabel) -> None:
        label.setText(str(value))

    def _commit_hue(self) -> None:
        self._commit_color_change(hue=self.hue_slider.value())

    def _on_saturation_changed(self, value: int, label: QLabel) -> None:
        label.setText(str(value))

    def _commit_saturation(self) -> None:
        self._commit_color_change(saturation=self.saturation_slider.value())

    def _commit_color_change(self, **kwargs) -> None:
        if not self.camera_ready:
            return
        self.logger.debug("Applying color change: %s", kwargs)
        try:
            values = self.service.set_color_controls(**kwargs)
            self._sync_color_sliders(values)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"调节失败：{exc}")
            self.logger.warning("Color change failed: %s", exc)

    def _handle_auto_color(self) -> None:
        if not self.camera_ready:
            QMessageBox.warning(self, "提示", "摄像头尚未就绪，无法调节颜色。")
            return

        self.logger.info("Starting auto color balance")
        self.auto_color_button.setEnabled(False)
        self.statusBar().showMessage("正在执行校准，请稍候...")

        try:
            # 执行自动白平衡和颜色校准
            values = self.service.auto_color_balance()

            # 等待几帧以确保白平衡生效
            QTimer.singleShot(500, lambda: self._on_auto_color_complete(values))
        except RuntimeError as exc:
            self.auto_color_button.setEnabled(True)
            QMessageBox.warning(self, "提示", f"校准失败：{exc}")
            self.logger.error("Auto color balance failed: %s", exc)
            self.statusBar().showMessage("校准失败")

    def _on_auto_color_complete(self, values: Dict[str, int]) -> None:
        """校准完成后的回调，更新UI"""
        try:
            # 重新获取最新的颜色参数（白平衡可能已更新）
            latest_values = self.service.get_color_controls()
            self._sync_color_sliders(latest_values)
            self._refresh_status()
            self.auto_color_button.setEnabled(True)
            self.statusBar().showMessage("校准完成")
            QMessageBox.information(self, "提示", "已执行白平衡和标准化调节。")
            self.logger.info("Auto color balance completed successfully")
        except Exception as exc:
            self.auto_color_button.setEnabled(True)
            self.logger.error("Failed to update color controls after calibration: %s", exc)
            self.statusBar().showMessage("校准完成，但更新显示失败")

    # ------------------------------------------------------------------
    # Exposure and gain control handlers
    # ------------------------------------------------------------------
    def _load_exposure_controls(self) -> None:
        if not self.camera_ready:
            return

        try:
            values = self.service.get_exposure_controls()
        except RuntimeError:
            return

        self._sync_exposure_controls(values)

    def _sync_exposure_controls(self, values: Dict[str, int]) -> None:
        self.auto_exposure_check.blockSignals(True)
        self.auto_exposure_check.setChecked(bool(values.get("auto_exposure", 1)))
        self.auto_exposure_check.blockSignals(False)

        self._set_slider_value(
            self.exposure_target_slider,
            self.exposure_target_value,
            values.get("exposure_target", nncam.NNCAM_AETARGET_DEF),
        )

        expo_time = values.get("exposure_time", 10000)
        self._set_slider_value(self.exposure_time_slider, self.exposure_time_value, expo_time)
        self.exposure_time_value.setText(f"{expo_time / 1000:.3f}ms")

        # 更新增益滑块范围（如果相机提供了范围）
        gain_min = values.get("gain_min", 100)
        gain_max = values.get("gain_max", 1000)
        if gain_min != self.gain_slider.minimum() or gain_max != self.gain_slider.maximum():
            self.gain_slider.blockSignals(True)
            current_gain = self.gain_slider.value()
            self.gain_slider.setRange(gain_min, gain_max)
            self.gain_slider.setValue(max(gain_min, min(gain_max, current_gain)))
            self.gain_slider.blockSignals(False)

        gain = values.get("gain", 100)
        self._set_slider_value(self.gain_slider, self.gain_value, gain)
        self.gain_value.setText(f"{gain}%")

    def _load_speed_controls(self) -> None:
        if not self.camera_ready:
            return

        try:
            values = self.service.get_speed_controls()
        except RuntimeError:
            return

        self._sync_speed_controls(values)

    def _sync_speed_controls(self, values: Dict[str, int]) -> None:
        # 更新帧率滑块范围（如果相机提供了范围）
        max_speed = values.get("max_speed", 10)
        if max_speed != self.speed_slider.maximum():
            self.speed_slider.blockSignals(True)
            current_speed = self.speed_slider.value()
            self.speed_slider.setRange(0, max_speed)
            self.speed_slider.setValue(max(0, min(max_speed, current_speed)))
            self.speed_slider.blockSignals(False)

        speed = values.get("speed", 0)
        self._set_slider_value(self.speed_slider, self.speed_value, speed)
        self.speed_value.setText(f"{speed}")

    def _on_auto_exposure_changed(self, state: int) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_exposure_controls(auto_exposure=1 if state == Qt.Checked else 0)
            self._sync_exposure_controls(values)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"设置自动曝光失败：{exc}")
            self.logger.warning("Auto exposure change failed: %s", exc)

    def _on_exposure_target_changed(self, value: int, label: QLabel) -> None:
        label.setText(str(value))

    def _commit_exposure_target(self) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_exposure_controls(exposure_target=self.exposure_target_slider.value())
            self._sync_exposure_controls(values)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"设置曝光目标失败：{exc}")
            self.logger.warning("Exposure target change failed: %s", exc)

    def _on_exposure_time_changed(self, value: int, label: QLabel) -> None:
        label.setText(f"{value / 1000:.3f}ms")

    def _commit_exposure_time(self) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_exposure_controls(exposure_time=self.exposure_time_slider.value())
            self._sync_exposure_controls(values)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"设置曝光时间失败：{exc}")
            self.logger.warning("Exposure time change failed: %s", exc)

    def _on_gain_changed(self, value: int, label: QLabel) -> None:
        label.setText(f"{value}%")

    def _commit_gain(self) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_exposure_controls(gain=self.gain_slider.value())
            self._sync_exposure_controls(values)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"设置增益失败：{exc}")
            self.logger.warning("Gain change failed: %s", exc)

    def _on_speed_changed(self, value: int, label: QLabel) -> None:
        label.setText(str(value))

    def _commit_speed(self) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_speed(speed=self.speed_slider.value())
            self._sync_speed_controls(values)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"设置帧率级别失败：{exc}")
            self.logger.warning("Speed change failed: %s", exc)

    def _handle_auto_exposure_once(self) -> None:
        if not self.camera_ready:
            QMessageBox.warning(self, "提示", "摄像头尚未就绪，无法执行自动曝光。")
            return

        self.logger.info("Starting auto exposure once")
        try:
            values = self.service.auto_exposure_once()
            self._sync_exposure_controls(values)
            self.statusBar().showMessage("自动曝光完成")
            QMessageBox.information(self, "提示", "自动曝光已完成。")
            self.logger.info("Auto exposure completed successfully")
        except RuntimeError as exc:
            QMessageBox.warning(self, "提示", f"自动曝光失败：{exc}")
            self.logger.error("Auto exposure failed: %s", exc)
            self.statusBar().showMessage("自动曝光失败")

    # ------------------------------------------------------------------
    # Sharpening control handlers
    # ------------------------------------------------------------------
    def _load_sharpening_controls(self) -> None:
        if not self.camera_ready:
            return

        try:
            values = self.service.get_sharpening_controls()
        except RuntimeError:
            return

        self._sync_sharpening_controls(values)

    def _sync_sharpening_controls(self, values: Dict[str, int]) -> None:
        self._set_slider_value(
            self.sharpening_strength_slider,
            self.sharpening_strength_value,
            values.get("strength", nncam.NNCAM_SHARPENING_STRENGTH_DEF),
        )
        self._set_slider_value(
            self.sharpening_radius_slider,
            self.sharpening_radius_value,
            values.get("radius", nncam.NNCAM_SHARPENING_RADIUS_DEF),
        )
        self._set_slider_value(
            self.sharpening_threshold_slider,
            self.sharpening_threshold_value,
            values.get("threshold", nncam.NNCAM_SHARPENING_THRESHOLD_DEF),
        )

    def _on_sharpening_strength_changed(self, value: int, label: QLabel) -> None:
        label.setText(str(value))

    def _commit_sharpening_strength(self) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_sharpening_controls(strength=self.sharpening_strength_slider.value())
            self._sync_sharpening_controls(values)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"设置锐化强度失败：{exc}")
            self.logger.warning("Sharpening strength change failed: %s", exc)

    def _on_sharpening_radius_changed(self, value: int, label: QLabel) -> None:
        label.setText(str(value))

    def _commit_sharpening_radius(self) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_sharpening_controls(radius=self.sharpening_radius_slider.value())
            self._sync_sharpening_controls(values)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"设置锐化半径失败：{exc}")
            self.logger.warning("Sharpening radius change failed: %s", exc)

    def _on_sharpening_threshold_changed(self, value: int, label: QLabel) -> None:
        label.setText(str(value))

    def _commit_sharpening_threshold(self) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_sharpening_controls(threshold=self.sharpening_threshold_slider.value())
            self._sync_sharpening_controls(values)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "提示", f"设置锐化阈值失败：{exc}")
            self.logger.warning("Sharpening threshold change failed: %s", exc)

    def _handle_sharpening_default(self) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_sharpening_controls(
                strength=nncam.NNCAM_SHARPENING_STRENGTH_DEF,
                radius=nncam.NNCAM_SHARPENING_RADIUS_DEF,
                threshold=nncam.NNCAM_SHARPENING_THRESHOLD_DEF,
            )
            self._sync_sharpening_controls(values)
            self.statusBar().showMessage("锐化参数已重置为默认值")
        except RuntimeError as exc:
            QMessageBox.warning(self, "提示", f"重置锐化参数失败：{exc}")
            self.logger.warning("Reset sharpening failed: %s", exc)

    # ------------------------------------------------------------------
    # Misc control handlers
    # ------------------------------------------------------------------
    def _load_misc_controls(self) -> None:
        if not self.camera_ready:
            return

        try:
            values = self.service.get_misc_controls()
        except RuntimeError:
            return

        self._sync_misc_controls(values)

    def _sync_misc_controls(self, values: Dict[str, int]) -> None:
        self.negative_check.blockSignals(True)
        self.negative_check.setChecked(bool(values.get("negative", 0)))
        self.negative_check.blockSignals(False)

        self.low_noise_check.blockSignals(True)
        self.low_noise_check.setChecked(bool(values.get("low_noise", 0)))
        self.low_noise_check.blockSignals(False)

        debayer = values.get("demosaic", 4)
        idx = self.debayer_combo.findData(debayer)
        if idx >= 0:
            self.debayer_combo.blockSignals(True)
            self.debayer_combo.setCurrentIndex(idx)
            self.debayer_combo.blockSignals(False)

        tone = values.get("tone_mapping", 2)
        idx = self.tone_mapping_combo.findData(tone)
        if idx >= 0:
            self.tone_mapping_combo.blockSignals(True)
            self.tone_mapping_combo.setCurrentIndex(idx)
            self.tone_mapping_combo.blockSignals(False)

    def _on_negative_changed(self, state: int) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_misc_controls(negative=1 if state == Qt.Checked else 0)
            self._sync_misc_controls(values)
        except RuntimeError as exc:
            QMessageBox.warning(self, "提示", f"设置负片失败：{exc}")
            self.logger.warning("Negative change failed: %s", exc)

    def _on_low_noise_changed(self, state: int) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_misc_controls(low_noise=1 if state == Qt.Checked else 0)
            self._sync_misc_controls(values)
        except RuntimeError as exc:
            QMessageBox.warning(self, "提示", f"设置低噪声模式失败：{exc}")
            self.logger.warning("Low noise change failed: %s", exc)

    def _on_low_power_changed(self, state: int) -> None:
        if not self.camera_ready:
            return
        # 低功耗模式需要根据具体相机支持情况实现
        self.logger.debug("Low power mode changed: %s", state == Qt.Checked)

    def _on_remove_shutter_effect_changed(self, state: int) -> None:
        if not self.camera_ready:
            return
        # 去快门效应需要根据具体相机支持情况实现
        self.logger.debug("Remove shutter effect changed: %s", state == Qt.Checked)

    def _on_debayer_changed(self, index: int) -> None:
        if not self.camera_ready:
            return
        try:
            value = self.debayer_combo.itemData(index)
            if value is not None:
                values = self.service.set_misc_controls(demosaic=value)
                self._sync_misc_controls(values)
        except RuntimeError as exc:
            QMessageBox.warning(self, "提示", f"设置Debayer失败：{exc}")
            self.logger.warning("Debayer change failed: %s", exc)

    def _on_tone_mapping_changed(self, index: int) -> None:
        if not self.camera_ready:
            return
        try:
            value = self.tone_mapping_combo.itemData(index)
            if value is not None:
                values = self.service.set_misc_controls(tone_mapping=value)
                self._sync_misc_controls(values)
        except RuntimeError as exc:
            QMessageBox.warning(self, "提示", f"设置色调映射失败：{exc}")
            self.logger.warning("Tone mapping change failed: %s", exc)

    def _on_shutter_mode_changed(self, index: int) -> None:
        if not self.camera_ready:
            return
        # 快门模式需要根据具体相机支持情况实现
        self.logger.debug("Shutter mode changed: %s", index)

    def _on_readout_mode_changed(self, index: int) -> None:
        if not self.camera_ready:
            return
        # 读出模式需要根据具体相机支持情况实现
        self.logger.debug("Readout mode changed: %s", index)

    def _handle_misc_default(self) -> None:
        if not self.camera_ready:
            return
        try:
            values = self.service.set_misc_controls(
                negative=0,
                demosaic=0,  # 双线性
                tone_mapping=2,  # 对数
            )
            self._sync_misc_controls(values)
            self.statusBar().showMessage("杂项参数已重置为默认值")
        except RuntimeError as exc:
            QMessageBox.warning(self, "提示", f"重置杂项参数失败：{exc}")
            self.logger.warning("Reset misc failed: %s", exc)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        # 停止保存线程
        if self.save_thread is not None:
            try:
                self.save_queue.put(None)  # 发送退出信号
                self.save_thread.join(timeout=2.0)  # 等待线程结束，最多2秒
            except Exception as exc:
                self.logger.warning("Failed to stop save thread: %s", exc)
        if self._detection_worker is not None:
            try:
                self._detection_worker.cancel()
            except Exception:
                pass
        if self._detection_thread is not None:
            self._detection_thread.quit()
            self._detection_thread.wait(2000)
            self._detection_thread = None
            self._detection_worker = None
        self.frame_timer.stop()
        if hasattr(self, "camera_retry_timer"):
            self.camera_retry_timer.stop()
        try:
            self.service.stop()
        except Exception as exc:
            self.logger.warning("Error stopping camera service on close: %s", exc)
        super().closeEvent(event)

    def _handle_reconnect(self) -> None:
        self.logger.info("Manual reconnect requested")
        self.reconnect_button.setEnabled(False)
        self.statusBar().showMessage("正在重新获取摄像头...")
        self.image_label.setText("正在重新获取摄像头...")
        try:
            if self.camera_ready:
                try:
                    self.service.stop()
                except Exception as exc:
                    self.logger.warning("Stopping camera during reconnect failed: %s", exc)
            self.camera_ready = False
            self._auto_calibration_done = False
            self._update_control_states()
            self._ensure_camera_ready()
        finally:
            QTimer.singleShot(800, lambda: self.reconnect_button.setEnabled(True))

    def _queue_image_save(self, path: Path, image: numpy.ndarray, quality: int = 92) -> None:
        try:
            if self.save_thread is None or not self.save_thread.is_alive():
                Image.fromarray(numpy.clip(image, 0, 255).astype(numpy.uint8)).save(
                    str(path), format="JPEG", quality=quality
                )
                self.logger.debug("Saved image synchronously: %s", path.name)
                return

            img_array = numpy.ascontiguousarray(numpy.clip(image, 0, 255).astype(numpy.uint8))
            height, width = img_array.shape[:2]
            payload = (str(path), img_array, width, height, quality)
            self.save_queue.put_nowait(payload)
        except queue.Full:
            Image.fromarray(numpy.clip(image, 0, 255).astype(numpy.uint8)).save(
                str(path), format="JPEG", quality=quality
            )
            self.logger.warning("Save queue full, image saved synchronously: %s", path.name)
        except Exception as exc:
            self.logger.error("Failed to queue image save for %s: %s", path.name, exc, exc_info=True)
            Image.fromarray(numpy.clip(image, 0, 255).astype(numpy.uint8)).save(
                str(path), format="JPEG", quality=quality
            )

