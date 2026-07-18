# sentinel/ — 实时多目标识别（独立项目）

同帧多类型目标实时检测框架，用于洛克王国：世界游戏场景。  
**完全独立**：从 `sentinel/` 目录内部或外部均可运行，无需依赖父项目路径。

## 核心能力

| 能力 | 实现 |
|---|---|
| 3D 内容检测 | YOLO 检测器（ultralytics），GPU/CPU 可配，模型可插拔 |
| 平面 label 检测 | OpenCV 模板匹配（TM_CCOEFF_NORMED + BGR 颜色二次校验） |
| 性能目标 | 10fps，单帧处理 <100ms |
| 覆盖层显示 | GDI 透明窗口，绿色=匹配成功 / 黄色=低分，框外标注 name:score |

## 目录结构

```
sentinel/
├── run.py                   # 启动脚本（从 sentinel/ 内部运行）
├── __init__.py
├── __main__.py              # 入口（支持 python -m / python __main__.py）
├── config.py                # 统一配置（JSON 驱动）
├── config.json              # 默认运行配置
├── pipeline.py              # 核心流水线：截帧 -> 检测 -> 渲染
│
├── capture/                 # 截图层
│   ├── window.py            # 窗口查找 + DPI 感知
│   ├── grabber.py           # 截图后端（screen-client / printwindow）
│   └── frame_buffer.py      # 环形帧缓冲
│
├── detectors/               # 检测器层（核心扩展点）
│   ├── base.py              # Detection 数据类 + DetectorBase + DetectorRegistry
│   ├── template_detector.py # 轻量模板匹配检测器
│   └── yolo_detector.py     # YOLO 检测器
│
├── display/                 # 渲染层
│   └── overlay.py           # GDI 覆盖层
│
├── labels/                  # 模板图片目录
│   └── box/                 # 宝箱相关模板
└── models/                  # YOLO 模型目录（.pt 文件）
```

## 快速开始

### 依赖安装

```bash
pip install opencv-python pywin32
# 如需 YOLO 检测器：
pip install ultralytics
```

### 运行

sentinel 是独立项目，支持从任意位置运行：

```bash
# 方式一：从 sentinel/ 目录内部运行（推荐）
cd sentinel
python run.py
python run.py -v
python run.py --config path/to/config.json

# 方式二：直接执行 __main__.py
cd sentinel
python __main__.py

# 方式三：从父目录使用 python -m
python -m sentinel
```

### 退出

在运行窗口中按 `Q` 键退出。

## 配置文件

`config.json` 驱动所有参数：

```json
{
  "window_keyword": "洛克王国：世界",
  "capture_backend": "screen-client",
  "use_frame_buffer": true,
  "frame_buffer_size": 10,
  "interval": 0.1,
  "detectors": {
    "yolo_pets": {
      "type": "yolo",
      "enabled": false,
      "params": {
        "model_path": "models/yolo26s.pt",
        "device": "auto",
        "conf": 0.4,
        "imgsz": 640
      }
    },
    "flat_labels": {
      "type": "template",
      "enabled": true,
      "params": {
        "default_threshold": 0.75,
        "scales": [0.85, 1.0, 1.15],
        "max_templates": 0,
        "templates": [...]
      }
    }
  }
}
```

关键参数说明：
- `device`: `"auto"` 自动检测 GPU，无 GPU 则降级 CPU
- `scales`: 缩放档位，每增加一档耗时线性增长
- `roi`: 每个模板可指定搜索区域 `[left, top, right, bottom]`（比例坐标）
- `max_templates`: 限制同时匹配的模板数量（0=不限制）

## 扩展新检测器

1. 在 `detectors/` 下创建新文件
2. 继承 `DetectorBase`，实现 `detect()` 和 `warmup()`
3. 添加 `@DetectorRegistry.register("your_type")` 装饰器
4. 在 `config.json` 的 `detectors` 中添加条目，指定 `type` 和 `params`

## 性能预算

| 阶段 | 预估耗时 |
|---|---|
| 截帧（PrintWindow） | 5–15ms |
| YOLO 推理（GPU） | 15–30ms |
| 模板匹配（5个, 单尺度, CPU） | 10–25ms |
| 结果合并 + 渲染 | 2–5ms |
| **合计** | **32–75ms** |

无 GPU 时 YOLO CPU 约 50–80ms，可通过降低 `imgsz` 到 416 或换 YOLOv8n 缓解。

## 输出格式

每帧打印状态行：

```
[14:30:25] found=2 | box-1:0.87, strange-bloodline:0.72 | cap=8ms det=35ms ren=3ms tot=46ms
```

设置 `print_json: true` 可输出 JSON 格式，便于程序解析。

## 与现有模块的关系

| 模块 | 关系 |
|---|---|
| `capture/` | 截图逻辑被提取到 `sentinel/capture/`，模板匹配算法被提取到 `sentinel/detectors/template_detector.py`。原 capture/ 保持不变 |
| `vision-backend/` | 不复用。MSE 算法质量不如 OpenCV TM_CCOEFF_NORMED |
| `RocoPilot/core/` | YOLO 检测器参考 pet_detector.py 写法，不直接依赖 |
| `ocr-backend/` | 暂不集成，后续可作为第三种检测器 `@register("ocr")` 插入 |
