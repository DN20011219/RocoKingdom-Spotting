"""SIFT 特征匹配诊断 — 直接调用主流程 SiftDetector。

用法:
    python -m tools.debug.sift                      # 使用 config.json 中所有 sift 检测器
    python -m tools.debug.sift --config my.json     # 指定配置文件
    python -m tools.debug.sift --show               # 弹出 OpenCV 窗口显示匹配结果
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
from detectors.base import DetectorRegistry


def _find_sift_detectors(config) -> Dict[str, Any]:
    """从 config 中找到所有 type=sift 的检测器配置。"""
    return {
        name: det_cfg
        for name, det_cfg in config.detectors.items()
        if det_cfg.get("type") == "sift" and det_cfg.get("enabled", True)
    }


def _capture_one_frame(keyword: str, backend: str) -> np.ndarray | None:
    """截取一帧。"""
    hwnd = find_window_by_keyword(keyword)
    if hwnd is None:
        print(f"[sift] 未找到窗口: {keyword}")
        return None

    rect = get_client_rect_on_screen(hwnd)
    grabber = FrameGrabber(backend)
    frame = grabber.grab(hwnd, rect)

    if frame is None or frame.size == 0:
        print("[sift] 截图失败")
        return None

    print(f"[sift] 截图: {frame.shape[1]}x{frame.shape[0]}, hwnd={hwnd}")
    return frame


def _print_diagnose_results(results: List[Dict[str, Any]], det_name: str) -> None:
    """打印诊断结果。"""
    print(f"\n{'='*60}")
    print(f"  检测器: {det_name}")
    print(f"{'='*60}")

    for r in results:
        status = "✓ 匹配" if r["found"] else "✗ 未匹配"
        print(f"\n  [{status}] {r['name']}")
        print(f"    匹配点数: {r['match_count']}  (阈值: {r['threshold']})")
        print(f"    模板特征点: {r['template_kp_count']}")
        print(f"    帧特征点: {r['frame_kp_count']}")
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
        label = f"{r['name']}:{r['match_count']}"
        label_y = y - 8 if y - 8 > 15 else y + h + 15
        cv2.putText(canvas, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="SIFT 特征匹配诊断（调用主流程 SiftDetector）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认 config.json）")
    parser.add_argument("--show", action="store_true", help="显示匹配结果窗口")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # 确保检测器已注册
    _ensure_loaded()

    # 找到所有 sift 检测器
    sift_detectors = _find_sift_detectors(config)
    if not sift_detectors:
        print("[sift] 配置中没有启用的 sift 检测器")
        return

    # 截图
    frame = _capture_one_frame(config.capture.window_keyword, config.capture.backend)
    if frame is None:
        return

    # 对每个检测器执行诊断
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for name, det_cfg in sift_detectors.items():
        params = det_cfg.get("params", {})
        try:
            detector = DetectorRegistry.create("sift", params)
            results = detector.diagnose(frame)
            all_results[name] = results
            _print_diagnose_results(results, name)
        except Exception as e:
            print(f"[sift] 创建检测器 {name} 失败: {e}")

    # 显示结果
    if args.show:
        for name, results in all_results.items():
            canvas = _draw_results(frame, results)
            cv2.imshow(f"SIFT Debug - {name}", canvas)
        print("\n[sift] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
