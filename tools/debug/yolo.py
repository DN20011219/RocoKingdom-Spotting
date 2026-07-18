"""YOLO 目标检测诊断 — 直接调用主流程 YoloDetector。

用法:
    python -m tools.debug.yolo                      # 使用 config.json 中所有 yolo 检测器
    python -m tools.debug.yolo --config my.json     # 指定配置文件
    python -m tools.debug.yolo --show               # 弹出 OpenCV 窗口显示检测结果
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# 路径自举
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from capture.window import find_window_by_keyword, get_client_rect_on_screen
from capture.grabber import FrameGrabber
from config import load_config
from detectors import _ensure_loaded
from detectors.base import Detection, DetectorRegistry


def _find_yolo_detectors(config) -> Dict[str, Any]:
    """从 config 中找到所有 type=yolo 的检测器配置。"""
    return {
        name: det_cfg
        for name, det_cfg in config.detectors.items()
        if det_cfg.get("type") == "yolo" and det_cfg.get("enabled", True)
    }


def _capture_one_frame(keyword: str, backend: str) -> np.ndarray | None:
    """截取一帧。"""
    hwnd = find_window_by_keyword(keyword)
    if hwnd is None:
        print(f"[yolo] 未找到窗口: {keyword}")
        return None

    rect = get_client_rect_on_screen(hwnd)
    grabber = FrameGrabber(backend)
    frame = grabber.grab(hwnd, rect)

    if frame is None or frame.size == 0:
        print("[yolo] 截图失败")
        return None

    print(f"[yolo] 截图: {frame.shape[1]}x{frame.shape[0]}, hwnd={hwnd}")
    return frame


def _print_detection_results(detections: List[Detection], det_name: str) -> None:
    """打印检测结果。"""
    print(f"\n{'='*60}")
    print(f"  检测器: {det_name}")
    print(f"{'='*60}")

    if not detections:
        print("  (未检测到目标)")
        return

    for d in detections:
        status = "低分" if d.is_low_score else "OK"
        print(f"\n  [{status}] {d.name}")
        print(f"    类别: {d.category}")
        print(f"    置信度: {d.score:.4f}")
        print(f"    位置: ({d.x}, {d.y})  尺寸: {d.w}x{d.h}")


def _draw_results(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """在帧上绘制检测结果。"""
    canvas = frame.copy()

    for d in detections:
        x, y, w, h = d.x, d.y, d.w, d.h
        color = (0, 255, 255) if d.is_low_score else (0, 255, 0)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = f"{d.name}:{d.score:.2f}"
        label_y = y - 8 if y - 8 > 15 else y + h + 15
        cv2.putText(canvas, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 目标检测诊断（调用主流程 YoloDetector）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 config.json）")
    parser.add_argument("--show", action="store_true", help="显示检测结果窗口")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # 确保检测器已注册
    _ensure_loaded()

    # 找到所有 yolo 检测器
    yolo_detectors = _find_yolo_detectors(config)
    if not yolo_detectors:
        print("[yolo] 配置中没有启用的 yolo 检测器")
        return

    # 截图
    frame = _capture_one_frame(config.capture.window_keyword, config.capture.backend)
    if frame is None:
        return

    # 对每个检测器执行检测
    all_detections: Dict[str, List[Detection]] = {}
    for name, det_cfg in yolo_detectors.items():
        params = det_cfg.get("params", {})
        try:
            detector = DetectorRegistry.create("yolo", params)
            detector.warmup()
            detections = detector.detect(frame)
            all_detections[name] = detections
            _print_detection_results(detections, name)
        except Exception as e:
            print(f"[yolo] 创建检测器 {name} 失败: {e}")

    # 显示结果
    if args.show:
        for name, detections in all_detections.items():
            canvas = _draw_results(frame, detections)
            cv2.imshow(f"YOLO Debug - {name}", canvas)
        print("\n[yolo] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
