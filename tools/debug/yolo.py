#!/usr/bin/env python3
"""YOLO 检测诊断工具 — 对当前游戏画面执行 YOLO 推理，输出检测结果。

用法:
    cd RocoKingdom-Spotting
    python tools/debug/yolo.py                          # 使用默认模型
    python tools/debug/yolo.py --model models/yolo26s.pt --conf 0.5
    python tools/debug/yolo.py --save                   # 保存标注结果
"""

import sys
import time
from pathlib import Path

# 路径自举：tools/debug/ -> tools/ -> 项目根目录
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import cv2
import numpy as np

from capture.window import find_window_by_keyword, get_client_rect_on_screen, is_foreground
from capture.grabber import FrameGrabber


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """保存图片到可能包含中文的路径。"""
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    return False


def _resolve_device(device: str) -> str:
    """解析 device 参数。'auto' 时自动检测 GPU 可用性。"""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="YOLO 检测诊断")
    parser.add_argument("--model", default="models/yolo26s.pt",
                        help="模型路径 (默认: models/yolo26s.pt)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="置信度阈值 (默认: 0.4)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图片尺寸 (默认: 640)")
    parser.add_argument("--device", default="auto",
                        help="设备: auto / cuda:0 / cpu (默认: auto)")
    parser.add_argument("--half", action="store_true",
                        help="使用 FP16 推理 (GPU 加速)")
    parser.add_argument("--save", action="store_true",
                        help="保存标注结果可视化")
    args = parser.parse_args()

    # 1. 找窗口
    keyword = "洛克王国"
    print(f"[1] 查找窗口：{keyword}")
    hwnd = find_window_by_keyword(keyword)
    if hwnd is None:
        print(f"    未找到窗口! 请确认游戏已打开")
        return
    print(f"    窗口句柄：{hwnd}")

    # 1.5 等待窗口回到前台
    if not is_foreground(hwnd):
        print(f"    窗口不在前台，等待切换到前台...")
        while not is_foreground(hwnd):
            time.sleep(0.5)
        print(f"    窗口已回到前台")

    # 2. 截图
    print(f"[2] 截图...")
    rect = get_client_rect_on_screen(hwnd)
    left, top, width, height = rect
    print(f"    客户区：{width}x{height} @ ({left},{top})")

    grabber = FrameGrabber("screen-client")
    frame = grabber.grab(hwnd, rect)
    if frame is None or frame.size == 0:
        print(f"    截图失败!")
        return
    print(f"    截图尺寸：{frame.shape[1]}x{frame.shape[0]}")

    # 3. 加载模型
    base_dir = Path(__file__).resolve().parent.parent.parent  # 项目根目录
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = base_dir / model_path

    print(f"[3] 加载模型：{model_path.name}")
    if not model_path.exists():
        print(f"    模型文件不存在: {model_path}")
        return

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    resolved_device = _resolve_device(args.device)
    print(f"    设备：{resolved_device}")
    print(f"    置信度阈值：{args.conf}")
    print(f"    推理尺寸：{args.imgsz}")
    print(f"    FP16：{args.half}")

    # 4. 推理
    print(f"[4] 执行推理...")
    tick = time.perf_counter()
    results = model.predict(
        frame,
        device=resolved_device,
        imgsz=args.imgsz,
        conf=args.conf,
        half=args.half,
        verbose=False,
    )
    infer_ms = (time.perf_counter() - tick) * 1000
    print(f"    推理耗时：{infer_ms:.1f}ms")

    # 5. 解析结果
    print(f"[5] 检测结果：")
    detections = []
    class_names = model.names if hasattr(model, 'names') and isinstance(model.names, dict) else {}

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, str(cls_id))

            bx, by = int(x1), int(y1)
            bw, bh = int(x2 - x1), int(y2 - y1)

            if bw > 0 and bh > 0:
                detections.append({
                    "name": cls_name,
                    "cls_id": cls_id,
                    "conf": conf,
                    "x": bx, "y": by, "w": bw, "h": bh,
                })

    if not detections:
        print(f"    未检测到任何目标")
    else:
        for i, det in enumerate(detections, 1):
            print(f"    [{i}] {det['name']} (cls={det['cls_id']}) "
                  f"conf={det['conf']:.4f}  "
                  f"位置=({det['x']},{det['y']})  尺寸={det['w']}x{det['h']}")

    # 6. 汇总
    print(f"\n{'='*60}")
    print(f"[汇总] 检测到 {len(detections)} 个目标, 耗时 {infer_ms:.1f}ms")
    print(f"{'='*60}")

    # 7. 保存可视化
    if args.save:
        out_dir = Path(__file__).resolve().parent / "debug"
        out_dir.mkdir(exist_ok=True)

        _imwrite_unicode(str(out_dir / "frame.png"), frame)

        # 在原图上画所有检测框
        result_img = frame.copy()
        for det in detections:
            cv2.rectangle(result_img,
                          (det["x"], det["y"]),
                          (det["x"] + det["w"], det["y"] + det["h"]),
                          (0, 255, 0), 2)
            label = f"{det['name']}:{det['conf']:.2f}"
            cv2.putText(result_img, label,
                        (det["x"], det["y"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        _imwrite_unicode(str(out_dir / "yolo_result.png"), result_img)

        print(f"\n[图片] 已保存到 {out_dir}/:")
        print(f"    frame.png        - 原始截图")
        print(f"    yolo_result.png  - 所有检测框标注")


if __name__ == "__main__":
    main()
