# RocoKingdom-Spotting

实时多目标识别框架，用于洛克王国：世界游戏场景。支持插件化检测器，同帧并行运行多种匹配算法。

## 核心能力

| 能力 | 实现 |
|---|---|
| 3D 内容检测 | YOLO 检测器（ultralytics），GPU/CPU 可配，模型可插拔 |
| 平面 label 检测 | OpenCV 模板匹配（TM_CCORR_NORMED + BGR 颜色二次校验） |
| 特征点匹配 | SIFT 特征提取 + BFMatcher + Lowe's ratio test |
| 覆盖层显示 | GDI 透明窗口，绿色=匹配成功 / 黄色=低分，框外标注 name:score |
| 性能目标 | 10fps，单帧处理 <100ms |

## 目录结构

```
RocoKingdom-Spotting/
├── run.py                   # 启动脚本
├── __main__.py              # 入口（支持 python -m / python run.py）
├── config.py                # 统一配置（JSON 驱动）
├── config.json              # 默认运行配置
├── pipeline.py              # 核心流水线：截帧 -> 检测 -> 渲染
│
├── capture/                 # 截图层
│   ├── window.py            # 窗口查找 + DPI 感知
│   ├── grabber.py           # 截图后端（screen-client / screen-window）
│   └── frame_buffer.py      # 环形帧缓冲（截帧与处理解耦）
│
├── detectors/               # 检测器层（核心扩展点）
│   ├── base.py              # Detection 数据类 + DetectorBase + DetectorRegistry
│   ├── template_detector.py # 模板匹配检测器（灰度 + 颜色校验）
│   ├── cuda_template_detector.py # CUDA 加速模板匹配（cv2.cuda.matchTemplate）
│   ├── torch_template_detector.py # PyTorch GPU 模板匹配（conv2d）
│   ├── sift_detector.py     # SIFT 特征点匹配检测器
│   └── yolo_detector.py     # YOLO 检测器（ultralytics）
│
├── display/                 # 渲染层
│   └── overlay.py           # GDI 覆盖层窗口
│
├── tools/                   # 开发工具
│   ├── capture.py           # 公用截图工具（窗口查找、截图、批量截帧）
│   ├── hotkey_capture.py    # 热键截图（F12 触发，Interception 驱动）
│   ├── debug/               # 检测算法诊断
│   │   ├── match.py         #   模板匹配诊断（单模板/批量）
│   │   ├── sift.py          #   SIFT 匹配诊断（含连线可视化）
│   │   └── yolo.py          #   YOLO 检测诊断
│   ├── yolo_tools/          # YOLO 数据集工具
│   │   └── yolo_labeler.py  #   YOLO 数据标注（断点续标 + 自动生成 data.yaml）
│   └── train/               # 模型训练
│       └── yolo.py          #   YOLO 模型训练（ultralytics）
│
├── labels/                  # 模板图片目录
├── models/                  # YOLO 模型目录（.pt 文件）
├── InterceptionCore.py      # Interception 驱动封装（来自 RocoKingdom-Clicker）
└── third/                   # 第三方库
    └── Interception/library/x64/interception.dll
```

## 快速开始

### 依赖安装

```bash
pip install opencv-python pywin32 numpy
# 如需 YOLO 检测器或训练：
pip install ultralytics
```

### GPU 训练环境

项目训练脚本会自动检测 GPU（`--device auto`），但需要安装 **CUDA 版本的 PyTorch**。

检查当前 PyTorch 是否支持 CUDA：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果版本带 `+cpu` 后缀或 `cuda.is_available()` 为 `False`，说明装的是 CPU 版，需要重装。根据 GPU 型号选择对应 CUDA 版本：

```bash
# RTX 50 系列（Blackwell，需 CUDA 12.8+）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# RTX 30/40 系列（Ampere/Ada，CUDA 12.4 即可）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 运行

```bash
# 方式一：启动脚本（推荐）
python run.py

# 方式二：直接执行入口
python __main__.py

# 通用参数
python run.py -v                      # 详细日志
python run.py --config path/to.json   # 指定配置文件
```

在运行窗口中按 `Ctrl+C` 退出。

## 检测算法

项目通过 `DetectorRegistry` 实现检测器插件化，所有算法并行注册，在 `config.json` 中按需启用。

### 模板匹配（type: "template"）

基于 OpenCV `TM_CCORR_NORMED` 灰度匹配 + BGR 颜色二次校验。适合固定 UI 元素（按钮、图标）的精确匹配。

- 支持多尺度搜索（`scales` 参数）
- 支持 ROI 区域限制（`roi` 参数，比例坐标）
- 支持模板数量限制（`max_templates`）

### CUDA 模板匹配（type: "template_cuda"）

与模板匹配算法完全一致，但使用 `cv2.cuda.matchTemplate` 在 GPU 上执行。需要 OpenCV 编译时启用 CUDA 支持。

- 帧只上传 GPU 一次，所有模板复用
- 模板预加载到显存，避免重复传输
- CUDA 不可用时自动回退到 CPU

**安装 CUDA 版 OpenCV：**

```bash
# 先卸载当前版本
pip uninstall opencv-python -y

# 安装 CUDA 版（社区预编译 wheel）
pip install opencv-python --extra-index-url https://jvejix.github.io/opencv-cuda-wheels/

# 或从源码编译（推荐，可控制 CUDA 版本）
# 参考: https://github.com/opencv/opencv/wiki/BuildOpenCVCUDA
```

验证 CUDA 是否可用：

```bash
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
# 输出 > 0 表示成功
```

### PyTorch GPU 模板匹配（type: "template_torch"）

使用 PyTorch `conv2d` 在 GPU 执行模板匹配，无需 OpenCV CUDA 编译。只要 PyTorch 支持 CUDA 即可使用。

- 帧和模板均保持在 GPU 显存，避免重复传输
- 初始化时预加载所有缩放模板
- CUDA 不可用时自动回退到 CPU

**基准测试（RTX 5070 Laptop）：**

| 分辨率 | CPU (matchTemplate) | GPU (PyTorch conv2d) | 加速比 |
|--------|---------------------|----------------------|--------|
| 1280×720 | ~608ms | ~257ms | 2.4x |
| 1920×1080 | ~2207ms | ~549ms | 4.0x |

**要求：** PyTorch 需安装 CUDA 版本。验证：

```bash
python -c "import torch; print(torch.cuda.is_available())"
# 输出 True 表示可用
```

### SIFT 特征匹配（type: "sift"）

基于 SIFT 关键点和描述子，BFMatcher + Lowe's ratio test。适合文字内容差异大、形状相似但语义不同的场景。

- 好匹配点数量作为阈值（`threshold`）
- ratio test 严格度可调（`ratio_threshold`）
- 同样支持多尺度和 ROI

### YOLO 检测（type: "yolo"）

基于 ultralytics，支持 GPU/CPU 自动选择。适合 3D 场景中的物体检测。

- 延迟加载模型，避免启动阻塞
- 支持 FP16 推理加速（`quantize`，兼容旧版 `half` 参数）
- 支持类别过滤（`classes`）

### 选择指南

| 场景 | 推荐算法 | 原因 |
|---|---|---|
| 固定 UI 按钮/图标 | 模板匹配 | 速度快、精度高 |
| 文字内容不同的按钮 | SIFT | 能区分文字差异 |
| 3D 场景物体 | YOLO | 泛化能力强 |
| 缩放/分辨率变化 | SIFT 或 模板+多尺度 | 特征点/多尺度容忍缩放 |

## 配置文件

`config.json` 驱动所有参数：

```json
{
  "window_keyword": "洛克王国：世界",
  "capture_backend": "screen-client",
  "use_frame_buffer": true,
  "frame_buffer_size": 10,
  "foreground_only": true,
  "interval": 0.1,
  "show_overlay": true,
  "print_json": false,
  "debug": false,
  "debug_dir": "debug_frames",
  "debug_save_interval": 10,

  "detectors": {
    "yolo_pets": {
      "type": "yolo",
      "enabled": false,
      "params": {
        "model_path": "models/yolo26s.pt",
        "device": "auto",
        "conf": 0.4,
        "imgsz": 640,
        "quantize": false
      }
    },
    "flat_labels": {
      "type": "template",
      "enabled": true,
      "params": {
        "device": "cpu",
        "default_threshold": 0.85,
        "default_color_threshold": 0.85,
        "scales": [0.85, 1.0, 1.15],
        "templates": [
          {
            "name": "hello",
            "path": "labels/hello.png",
            "threshold": 0.85,
            "color_threshold": 0.85,
            "roi": [0.0, 0.0, 1.0, 1.0]
          }
        ]
      }
    }
  }
}
```

### 关键参数

| 参数 | 说明 |
|---|---|
| `device` | `"auto"` 自动检测 GPU，无 GPU 则降级 CPU |
| `scales` | 缩放档位，每增加一档耗时线性增长 |
| `roi` | 搜索区域 `[left, top, right, bottom]`（比例坐标 0-1） |
| `debug` | 开启后每 N 帧保存带标注的调试图到 `debug_dir` |
| `print_json` | 输出 JSON 格式结果，便于程序解析 |

## 调试工具

`tools/debug/` 下的工具可独立运行，用于诊断单个算法的匹配效果（需要游戏窗口在前台）：

```bash
# 模板匹配诊断（使用 config.json 中所有 template 检测器）
python -m tools.debug.match                    # 打印诊断结果
python -m tools.debug.match --show             # 弹出 OpenCV 窗口显示匹配结果
python -m tools.debug.match --config my.json   # 指定配置文件

# SIFT 匹配诊断
python -m tools.debug.sift                     # 打印诊断结果
python -m tools.debug.sift --show              # 显示匹配结果窗口

# YOLO 检测诊断
python -m tools.debug.yolo                     # 打印检测结果
python -m tools.debug.yolo --show              # 显示检测结果窗口
```

## 数据标注

### 1. 截图

`tools/hotkey_capture.py` 后台监听 F12 热键，按 F12 截取游戏窗口画面。

```bash
# 基本用法
python tools/hotkey_capture.py --class pet1
```

| 操作 | 功能 |
|---|---|
| `F12` | 截取当前窗口画面 |
| `Esc` | 退出 |

截图保存到 `datasets/yolo_dataset/images/<class>/`。

### 2. 标注

`tools/yolo_tools/yolo_labeler.py` 加载指定类别目录的图片并标注，支持断点续标。

```bash
# 基本用法：加载 images/<class>/ 目录下的图片并标注
python tools/yolo_tools/yolo_labeler.py --class pet1
```

**标注界面操作：**

| 操作 | 功能 |
|---|---|
| 左键拖拽 | 画 bounding box |
| 右键点击 | 删除最近的标注 |
| `Z` | 撤销最后一个标注 |
| `X` | 清除当前图片所有标注 |
| `Space` / `D` | 下一张图片 |
| `A` | 上一张图片 |
| `S` | 保存当前进度 |
| `Q` / `Esc` | 保存并退出 |

标注结果输出为 YOLO 格式（`images/` + `labels/` + `data.yaml`），支持断点续标。退出时自动扫描所有类别目录生成 `data.yaml`，可直接用于 `tools/train/yolo.py --data` 训练。

## 模型训练

```bash
python tools/train/yolo.py                                  # 使用默认数据集和参数
python tools/train/yolo.py --epochs 200 --batch 32          # 自定义训练参数
python tools/train/yolo.py --data datasets/other.yaml       # 指定其他数据集
python tools/train/yolo.py --resume                         # 恢复中断的训练
```

训练完成后：
- 原始模型保存在 `runs/detect/<name>/weights/best.pt`
- 最佳模型自动复制到 `models/<name>_<时间戳>.pt`，并生成同名 `.txt` 说明文档（含可识别类别、训练参数、关键指标）

## 扩展新检测器

1. 在 `detectors/` 下创建新文件
2. 继承 `DetectorBase`，实现 `detect()` 和 `warmup()`
3. 添加 `@DetectorRegistry.register("your_type")` 装饰器
4. 在 `config.json` 的 `detectors` 中添加条目，指定 `type` 和 `params`

## 性能预算

| 阶段 | 预估耗时 |
|---|---|
| 截帧（GDI） | 5–15ms |
| YOLO 推理（GPU） | 15–30ms |
| 模板匹配（5个, 单尺度, CPU） | 10–25ms |
| SIFT 匹配（5个, CPU） | 20–40ms |
| 结果合并 + 渲染 | 2–5ms |
| **合计** | **52–115ms** |

无 GPU 时 YOLO CPU 约 50–80ms，可通过降低 `imgsz` 到 416 或换更小的模型缓解。

## 输出格式

每帧打印状态行：

```
[14:30:25] found=2 | hello:0.92, stop:0.88 | cap=8ms det=35ms ren=3ms tot=46ms
```

设置 `print_json: true` 可输出 JSON 格式：

```json
{"time":"14:30:25","capture_ms":8.0,"detect_ms":35.0,"render_ms":3.0,"total_ms":46.0,"detections":[...]}
```

## 致谢

| 项目 | 用途 |
|---|---|
| [ultralytics/yolov5](https://github.com/ultralytics/ultralytics) | YOLO 目标检测框架及预训练模型 |
| [opencv](https://opencv.org/) | 图像处理与模板匹配 |
| [Interception](https://github.com/oblitum/Interception) | 内核级输入设备拦截驱动 |
| [RocoKingdom-Clicker](https://github.com/DN20011219/RocoKingdom-Clicker) | InterceptionCore 模块（已拷贝至本项目） |
