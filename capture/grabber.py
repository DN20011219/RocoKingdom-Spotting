"""截图后端 — 仅支持前台窗口截图（screen-client / screen-window）。

窗口不在前台时由 pipeline 的 foreground_only 逻辑自动暂停。
从 capture/backends.py 提取核心逻辑。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

try:
    from PIL import ImageGrab
except Exception:
    ImageGrab = None


def grab_region_bgr(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """通过 Pillow ImageGrab 截取屏幕指定区域。

    bbox: (left, top, right, bottom) 屏幕绝对坐标。
    """
    if ImageGrab is None:
        raise RuntimeError("Pillow ImageGrab 不可用")
    image = ImageGrab.grab(bbox=bbox)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


class FrameGrabber:
    """统一截图接口，根据 backend 名称选择实现。

    仅支持 screen-client / screen-window，不支持后台截图。
    """

    def __init__(self, backend: str = "screen-client") -> None:
        if backend not in ("screen-client", "screen-window"):
            raise ValueError(f"unknown capture backend: {backend} (only screen-client / screen-window)")
        self._backend = backend

    def grab(self, hwnd: int, rect_on_screen: Tuple[int, int, int, int]) -> np.ndarray:
        """截取一帧 BGR 图像。

        rect_on_screen: (left, top, width, height) 窗口客户区在屏幕上的位置。
        调用方应确保窗口在前台，否则截取的是其他窗口的内容。
        """
        left, top, width, height = rect_on_screen

        if self._backend == "screen-window":
            # 截取整个窗口（含边框）
            import win32gui as _wg
            wl, wt, wr, wb = _wg.GetWindowRect(hwnd)
            return grab_region_bgr((wl, wt, wr, wb))

        # screen-client（默认）
        if width <= 0 or height <= 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return grab_region_bgr((left, top, left + width, top + height))
