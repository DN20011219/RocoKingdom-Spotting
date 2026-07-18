"""环形帧缓冲区 — 捕获线程连续截帧写入，匹配线程取最新帧处理。

从 capture/frame_buffer.py 精简迁移。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np


class RingFrameBuffer:
    """环形帧缓冲区。

    捕获线程以游戏帧率连续截取画面写入缓冲区，
    匹配线程从缓冲区取最新帧进行处理。
    当处理速度慢于捕获速度时，旧帧自动丢弃。
    """

    def __init__(self, maxlen: int = 10) -> None:
        self._buffer: deque[tuple[np.ndarray, float]] = deque(maxlen=maxlen)
        self._new_frame_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_fn: Optional[Callable[[], np.ndarray]] = None
        self._frames_captured = 0
        self._frames_processed = 0
        self._start_time = 0.0

    def start(self, capture_fn: Callable[[], np.ndarray]) -> None:
        """启动连续捕获线程。capture_fn 应返回一帧 BGR 图像。"""
        if self._running:
            return
        self._running = True
        self._capture_fn = capture_fn
        self._start_time = time.perf_counter()
        self._frames_captured = 0
        self._frames_processed = 0
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="sentinel-frame-capture",
        )
        self._capture_thread.start()

    def stop(self) -> None:
        """停止连续捕获。"""
        self._running = False
        self._new_frame_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

    def wait_for_frame(self, timeout: float = 0.05) -> Optional[tuple[np.ndarray, float]]:
        """等待新帧可用。返回 (frame, timestamp) 或 None（超时）。"""
        if self._new_frame_event.wait(timeout=timeout):
            self._new_frame_event.clear()
            return self.get_latest()
        return None

    def get_latest(self) -> Optional[tuple[np.ndarray, float]]:
        """获取缓冲区中最新的帧。"""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def mark_processed(self) -> None:
        """标记一帧已处理完成（用于统计）。"""
        self._frames_processed += 1

    def get_stats(self) -> dict:
        """返回统计信息。"""
        elapsed = time.perf_counter() - self._start_time if self._start_time else 1.0
        return {
            "capture_fps": round(self._frames_captured / max(elapsed, 0.01), 1),
            "process_fps": round(self._frames_processed / max(elapsed, 0.01), 1),
            "buffer_depth": len(self._buffer),
            "frames_captured": self._frames_captured,
            "frames_processed": self._frames_processed,
        }

    def _capture_loop(self) -> None:
        while self._running:
            try:
                frame = self._capture_fn()  # type: ignore[misc]
            except Exception:
                continue

            if frame is None or getattr(frame, "size", 0) == 0:
                time.sleep(0.001)
                continue

            with self._lock:
                self._buffer.append((frame, time.perf_counter()))
            self._frames_captured += 1
            self._new_frame_event.set()
            time.sleep(0.001)
