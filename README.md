# 显微镜自动识别与检测平台

基于 PyQt5 的显微镜控制系统，集成了实时摄像头操控、批量拍照以及自动化花粉级联检测（分割 + 分类路由）。应用启动后会自动载入默认模型，用户可直接在界面中完成拍照与目标检测任务。

## 主要特性
- **实时显微镜控制**：调节曝光、增益、帧率、锐化等参数，并支持自动对焦/自动彩色校准。
- **历史照片管理**：按栅格命名保存拍摄结果，便于回溯与拼接。
- **级联检测流程**：通过 `seg.pt`（分割）与 `classify.pt`（倍率分类）自动选择切片/整图策略，生成标注图与裁剪结果。
- **自动化输出**：检测日志、统计 `summary.txt`、标注图以及裁剪目标统一存入 `captures/detections/run_时间戳/`。
- **可扩展的 SDK**：自带相机驱动（`nncam`、`SDK/`）和拼接脚本，方便进一步定制。

## 环境准备
1. **Python 版本**：建议 Python 3.9 及以上。
2. **依赖安装**（虚拟环境可选）：
   ```bash
   pip install -r requirements.txt
   ```
   如果第一次运行检测功能，会自动检查并安装 `torch`、`ultralytics`、`sahi` 等核心包，但提前安装可避免 GUI 卡顿。
3. **驱动/权限**：确认显微镜相机驱动已正确安装（Windows 下可使用仓库自带的 `SDK/win/drivers`）。运行应用需要摄像头访问权限。

## 模型文件准备
应用启动时会自动在**项目根目录**（与 `main.py` 同级）查找：

- `seg.pt` ：YOLO 分割模型权重（负责实例分割与定位），完整路径示例：`microscope/seg.pt`；
- `classify.pt` ：倍率/路由分类模型（用于推理策略路由），完整路径示例：`microscope/classify.pt`。

请将训练好的权重复制/命名为上述文件。如果需要切换模型，只需替换这两个文件并重新启动应用。

> 旧版本的“手动加载”入口已移除；若缺少权重，界面会提示“未找到默认模型，目标检测不可用”。

## 启动应用
1. 确保显微镜连接到电脑，且驱动工作正常。
2. 在项目根目录执行：
   ```bash
   python main.py
   ```
3. GUI 启动后会自动尝试加载 `seg.pt` 与 `classify.pt`，状态栏会显示加载结果。
4. 进入“实时采集”页签即可看到摄像头画面，并可调节各项参数或执行拍照。

## 目标检测流程
1. 在“历史记录”列表中准备好待检测的图像（可通过拍照或手动放入 `captures/` 目录）。
2. 点击“执行目标检测”按钮：
   - 应用会创建 `captures/detections/run_YYYYMMDD_HHMMSS/` 文件夹；
- 每个图像生成一个同名子目录，包含：
   - `*_annotated.jpg`：绘制框和类别标签的结果图；
   - `summary.txt`：分类策略/耗时等摘要；
   - `detection_log.log`：整次任务的日志（位于 run 目录根部）。

### 检测结果保存位置
- 根目录：`captures/detections/`
- 单次任务：`captures/detections/run_YYYYMMDD_HHMMSS[/编号]/`
- 单张图片：`captures/detections/run_YYYYMMDD_HHMMSS/原图文件名/`

可以通过该层级快速定位到某次检测和对应的标注、裁剪与日志。
3. 任务过程中可在界面右侧实时预览最新检测结果；若需要中断，可关闭窗口或停止线程。

## 常见问题
- **未找到模型文件**：确认 `seg.pt`、`classify.pt` 与应用位于同一目录，并具有读取权限。
- **显微镜未连接**：在状态栏提示“正在检测摄像头”时检查 USB/电源，必要时重新插拔并点击“刷新”。
- **依赖缺失**：若提示缺少 `torch`/`ultralytics`，请重新运行 `pip install -r requirements.txt` 或手动安装。
- **检测输出为空**：查看 `captures/detections/run_*/summary.txt`，确认分类策略与阈值（`pyqt_app.py` 中 `self.detection_conf`）是否符合预期。

## 目录速览
```
microscope/
├─ main.py                # 应用入口（调用 pyqt_app.main）
├─ pyqt_app.py            # GUI 主逻辑与显微镜控制
├─ test_cls.py            # 级联检测核心逻辑（分割+分类路由）
├─ captures/              # 拍照与检测输出目录
│   ├─ detections/        # run_时间戳/子目录存放检测结果
│   └─ img_rXXX_cXXX.jpg  # 拍照生成的原始图
├─ models (自备)          # 请放置 seg.pt、classify.pt
├─ requirements.txt       # 依赖列表
└─ SDK/, nncam.py         # 相机 SDK 及 Python 封装
```

如需扩展（例如接入其他模型、修改 UI），可直接在 `pyqt_app.py` 与 `test_cls.py` 中调整逻辑。欢迎根据实验需求继续完善！
