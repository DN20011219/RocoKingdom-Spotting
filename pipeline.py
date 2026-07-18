"""核心流水线：截帧 -> 多检测器串行推理 -> 结果合并 -> 覆盖层渲染。

支持两种运行模式：
- 标准模式：按 interval 周期截帧，单线程处理
- 帧缓冲模式：截帧线程连续写入环形缓冲区，匹配线程取最新帧处理
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from capture.frame_buffer import RingFrameBuffer
from capture.grabber import FrameGrabber
from capture.window import (
    find_window_by_keyword,
    get_client_rect_on_screen,
    is_foreground,
)
from config import SentinelConfig, load_config
from detectors.base import Detection, DetectorBase, DetectorRegistry
from display.overlay import OverlayWindow

logger = logging.getLogger(__name__)


def _build_detectors(config: SentinelConfig) -> Dict[str, DetectorBase]:
    """根据 config.json 创建所有启用的检测器。"""
    # 确保 detectors 子模块已导入（触发 register 装饰器）
    from detectors import _ensure_loaded
    _ensure_loaded()

    detectors: Dict[str, DetectorBase] = {}
    for name, det_cfg in config.detectors.items():
        if not det_cfg.get("enabled", True):
            continue
        type_name = det_cfg.get("type")
        if not type_name:
            logger.warning("检测器 %s 缺少 type 字段，跳过", name)
            continue
        params = det_cfg.get("params", {})
        try:
            detectors[name] = DetectorRegistry.create(type_name, params)
            logger.info("检测器已创建: %s (type=%s)", name, type_name)
        except Exception as e:
            logger.error("创建检测器 %s 失败: %s", name, e)

    return detectors


def _warmup_detectors(detectors: Dict[str, DetectorBase]) -> None:
    """预热所有检测器。"""
    for name, det in detectors.items():
        try:
            logger.info("预热检测器: %s", name)
            det.warmup()
        except Exception as e:
            logger.warning("检测器 %s 预热失败: %s", name, e)


def _run_detectors(
    detectors: Dict[str, DetectorBase],
    frame: np.ndarray,
) -> List[Detection]:
    """串行执行所有检测器，合并结果。"""
    all_detections: List[Detection] = []
    for name, det in detectors.items():
        try:
            results = det.detect(frame)
            all_detections.extend(results)
        except Exception as e:
            logger.error("检测器 %s 执行失败: %s", name, e)
    return all_detections


def _save_debug_frame(
    frame: np.ndarray,
    detections: List[Detection],
    debug_dir: Path,
    counter: int,
    save_interval: int,
) -> int:
    """保存带标注的调试帧。"""
    counter += 1
    if save_interval > 0 and counter % save_interval != 0:
        return counter

    debug_dir.mkdir(parents=True, exist_ok=True)
    canvas = frame.copy()
    frame_h, frame_w = canvas.shape[:2]

    for det in detections:
        x, y, w, h = det.x, det.y, det.w, det.h
        score = det.score
        name = det.name

        # 绿色框=匹配成功，黄色=低分
        color = (0, 255, 255) if det.is_low_score else (0, 255, 0)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        # 名字标注（框上方）
        label = f"{name}:{score:.2f}"
        label_y = y - 8 if y - 8 > 15 else y + h + 15
        cv2.putText(canvas, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    filepath = debug_dir / f"debug_{counter:06d}.png"
    cv2.imwrite(str(filepath), canvas)
    return counter


class Pipeline:
    """实时检测流水线。"""

    def __init__(self, config: SentinelConfig) -> None:
        self._config = config
        self._detectors: Dict[str, DetectorBase] = {}
        self._overlay: Optional[OverlayWindow] = None
        self._debug_counter = 0

    def setup(self) -> None:
        """初始化检测器和覆盖层。"""
        self._detectors = _build_detectors(self._config)
        if not self._detectors:
            logger.warning("没有启用的检测器！")

        _warmup_detectors(self._detectors)

        if self._config.display.show_overlay:
            self._overlay = OverlayWindow()

    def run(self) -> None:
        """主循环。"""
        keyword = self._config.capture.window_keyword
        hwnd = find_window_by_keyword(keyword)
        if hwnd is None:
            print(f"[spotting] 未找到窗口: {keyword}")
            return

        grabber = FrameGrabber(self._config.capture.backend)
        print(f"[spotting] 窗口已找到: hwnd={hwnd}")
        print(f"[spotting] 检测器: {', '.join(self._detectors.keys()) or '(无)'}")
        print(f"[spotting] 截图后端: {self._config.capture.backend}")

        if self._config.capture.use_frame_buffer:
            self._run_frame_buffer_mode(hwnd, grabber)
        else:
            self._run_standard_mode(hwnd, grabber)

    # -----------------------------------------------------------------------
    # 标准模式
    # -----------------------------------------------------------------------
    def _run_standard_mode(self, hwnd: int, grabber: FrameGrabber) -> None:
        interval = self._config.capture.interval
        foreground_only = self._config.capture.foreground_only

        print(f"[spotting] 标准模式启动 (interval={interval}s), 按 Q 退出...")

        try:
            while True:
                if foreground_only and not is_foreground(hwnd):
                    if self._overlay:
                        self._overlay.hide()
                    time.sleep(interval)
                    continue

                tick = time.perf_counter()
                rect = get_client_rect_on_screen(hwnd)
                frame = grabber.grab(hwnd, rect)
                capture_ms = (time.perf_counter() - tick) * 1000

                if frame is None or frame.size == 0:
                    time.sleep(interval)
                    continue

                # 检测
                tick_detect = time.perf_counter()
                detections = _run_detectors(self._detectors, frame)
                detect_ms = (time.perf_counter() - tick_detect) * 1000

                # 渲染
                tick_render = time.perf_counter()
                self._render(hwnd, detections, frame.shape[1], frame.shape[0])
                render_ms = (time.perf_counter() - tick_render) * 1000

                total_ms = (time.perf_counter() - tick) * 1000
                self._print_status(detections, capture_ms, detect_ms, render_ms, total_ms)

                # 调试保存
                if self._config.display.debug:
                    self._debug_counter = _save_debug_frame(
                        frame, detections, self._config.debug_dir,
                        self._debug_counter, self._config.display.debug_save_interval,
                    )

                elapsed = time.perf_counter() - tick
                time.sleep(max(0.0, interval - elapsed))

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if self._overlay:
                self._overlay.destroy()
            print("\n[spotting] 已停止")

    # -----------------------------------------------------------------------
    # 帧缓冲模式
    # -----------------------------------------------------------------------
    def _run_frame_buffer_mode(self, hwnd: int, grabber: FrameGrabber) -> None:
        buf_size = self._config.capture.frame_buffer_size
        foreground_only = self._config.capture.foreground_only
        buf = RingFrameBuffer(maxlen=buf_size)

        def capture_fn():
            rect = get_client_rect_on_screen(hwnd)
            return grabber.grab(hwnd, rect)

        buf.start(capture_fn)
        print(f"[spotting] 帧缓冲模式启动 (buffer_size={buf_size}), 按 Q 退出...")

        last_stats_time = time.perf_counter()

        try:
            while True:
                frame_and_ts = buf.wait_for_frame(timeout=0.05)
                if frame_and_ts is None:
                    continue

                frame, _ = frame_and_ts
                if frame is None or frame.size == 0:
                    continue

                if foreground_only and not is_foreground(hwnd):
                    if self._overlay:
                        self._overlay.hide()
                    continue

                tick = time.perf_counter()

                # 检测
                tick_detect = time.perf_counter()
                detections = _run_detectors(self._detectors, frame)
                detect_ms = (time.perf_counter() - tick_detect) * 1000

                buf.mark_processed()

                # 渲染
                tick_render = time.perf_counter()
                self._render(hwnd, detections, frame.shape[1], frame.shape[0])
                render_ms = (time.perf_counter() - tick_render) * 1000

                total_ms = (time.perf_counter() - tick) * 1000
                self._print_status(detections, 0, detect_ms, render_ms, total_ms)

                # 调试保存
                if self._config.display.debug:
                    self._debug_counter = _save_debug_frame(
                        frame, detections, self._config.debug_dir,
                        self._debug_counter, self._config.display.debug_save_interval,
                    )

                # 定期打印统计
                now = time.perf_counter()
                if now - last_stats_time >= 5.0:
                    stats = buf.get_stats()
                    print(f"\n[stats] capture_fps={stats['capture_fps']}, "
                          f"process_fps={stats['process_fps']}, "
                          f"buffer_depth={stats['buffer_depth']}")
                    last_stats_time = now

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            buf.stop()
            if self._overlay:
                self._overlay.destroy()
            print("\n[spotting] 已停止")

    # -----------------------------------------------------------------------
    # 辅助方法
    # -----------------------------------------------------------------------
    def _render(self, hwnd: int, detections: List[Detection],
                frame_w: int, frame_h: int) -> None:
        """更新覆盖层。"""
        if not self._overlay or not self._config.display.show_overlay:
            if self._overlay:
                self._overlay.hide()
            return

        rect = get_client_rect_on_screen(hwnd)
        self._overlay.ensure(hwnd, *rect)
        self._overlay.render(detections, frame_w, frame_h)

    def _print_status(self, detections: List[Detection],
                      capture_ms: float, detect_ms: float,
                      render_ms: float, total_ms: float) -> None:
        """打印状态行。"""
        ts = datetime.now().strftime("%H:%M:%S")
        found = len(detections)
        summary = ", ".join(f"{d.name}:{d.score:.2f}" for d in detections)

        if self._config.display.print_json:
            data = {
                "time": ts,
                "capture_ms": round(capture_ms, 1),
                "detect_ms": round(detect_ms, 1),
                "render_ms": round(render_ms, 1),
                "total_ms": round(total_ms, 1),
                "detections": [
                    {"name": d.name, "category": d.category, "score": round(d.score, 4),
                     "x": d.x, "y": d.y, "w": d.w, "h": d.h}
                    for d in detections
                ],
            }
            print(json.dumps(data, ensure_ascii=False))
        else:
            status = (f"[{ts}] found={found} | {summary} | "
                      f"cap={capture_ms:.0f}ms det={detect_ms:.0f}ms "
                      f"ren={render_ms:.0f}ms tot={total_ms:.0f}ms   ")
            print(f"\r{status}", end="", flush=True)
