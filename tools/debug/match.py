"""模板匹配诊断 — 直接调用主流程 TemplateDetector。

用法:
    python -m tools.debug.match                     # 使用 config.json 中所有 template 检测器
    python -m tools.debug.match --config my.json    # 指定配置文件
    python -m tools.debug.match --show              # 弹出 OpenCV 窗口显示匹配结果
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from detectors.base import DetectorRegistry


def _find_template_detectors(config) -> Dict[str, Any]:
    """从 config 中找到所有 type=template 的检测器配置。"""
    return {
        name: det_cfg
        for name, det_cfg in config.detectors.items()
        if det_cfg.get("type") == "template" and det_cfg.get("enabled", True)
    }


def _capture_one_frame(keyword: str, backend: str) -> np.ndarray | None:
    """截取一帧。"""
    hwnd = find_window_by_keyword(keyword)
    if hwnd is None:
        print(f"[match] 未找到窗口: {keyword}")
        return None

    rect = get_client_rect_on_screen(hwnd)
    grabber = FrameGrabber(backend)
    frame = grabber.grab(hwnd, rect)

    if frame is None or frame.size == 0:
        print("[match] 截图失败")
        return None

    print(f"[match] 截图: {frame.shape[1]}x{frame.shape[0]}, hwnd={hwnd}")
    return frame


def _print_diagnose_results(results: List[Dict[str, Any]], det_name: str) -> None:
    """打印诊断结果。"""
    print(f"\n{'='*60}")
    print(f"  检测器: {det_name}")
    print(f"{'='*60}")

    for r in results:
        status = "✓ 匹配" if r["found"] else "✗ 未匹配"
        print(f"\n  [{status}] {r['name']}")
        print(f"    灰度分: {r['gray_score']:.4f}  (阈值: {r['threshold']:.4f})")
        print(f"    颜色分: {r['color_score']:.4f}  (阈值: {r['color_threshold']:.4f})")
        if r["found"]:
            print(f"    位置: ({r['x']}, {r['y']})  尺寸: {r['w']}x{r['h']}")


def _draw_results(frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
    """在帧上绘制诊断结果。"""
    canvas = frame.copy()

    for r in results:
        if not r["found"]:
            continue
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{r['name']}:{r['gray_score']:.2f}"
        label_y = y - 8 if y - 8 > 15 else y + h + 15
        cv2.putText(canvas, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="模板匹配诊断（调用主流程 TemplateDetector）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 config.json）")
    parser.add_argument("--show", action="store_true", help="显示匹配结果窗口")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # 确保检测器已注册
    _ensure_loaded()

    # 找到所有 template 检测器
    template_detectors = _find_template_detectors(config)
    if not template_detectors:
        print("[match] 配置中没有启用的 template 检测器")
        return

    # 截图
    frame = _capture_one_frame(config.capture.window_keyword, config.capture.backend)
    if frame is None:
        return

    # 对每个检测器执行诊断
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for name, det_cfg in template_detectors.items():
        params = det_cfg.get("params", {})
        try:
            detector = DetectorRegistry.create("template", params)
            results = detector.diagnose(frame)
            all_results[name] = results
            _print_diagnose_results(results, name)
        except Exception as e:
            print(f"[match] 创建检测器 {name} 失败: {e}")

    # 显示结果
    if args.show:
        for name, results in all_results.items():
            canvas = _draw_results(frame, results)
            cv2.imshow(f"Match Debug - {name}", canvas)
        print("\n[match] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
