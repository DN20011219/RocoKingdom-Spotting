"""Win32 窗口查找 + DPI 感知。

从 capture/backends.py 提取核心逻辑，去掉与 capture 其他模块的耦合。
"""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

import win32gui


WindowRect = Tuple[int, int, int, int]   # (left, top, width, height)


def enable_dpi_awareness() -> None:
    """让 Win32 坐标与屏幕像素对齐（高 DPI 显示器必须）。"""
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# 模块加载时立即设置，保证后续截图坐标准确
enable_dpi_awareness()


def find_window_by_keyword(keyword: str) -> Optional[int]:
    """枚举可见窗口，返回标题包含 keyword 的最短标题窗口句柄。"""
    matches: list[tuple[int, str]] = []

    def _enum_handler(hwnd: int, _ctx: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        if win32gui.IsIconic(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and keyword in title:
            matches.append((hwnd, title))

    win32gui.EnumWindows(_enum_handler, None)
    if not matches:
        return None
    return min(matches, key=lambda item: len(item[1]))[0]


def get_client_rect_on_screen(hwnd: int) -> WindowRect:
    """返回窗口客户区在屏幕上的 (left, top, width, height)。"""
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    client_w = right - left
    client_h = bottom - top
    screen_left, screen_top = win32gui.ClientToScreen(hwnd, (0, 0))
    return screen_left, screen_top, client_w, client_h


def get_window_rect_on_screen(hwnd: int) -> WindowRect:
    """返回窗口整体在屏幕上的 (left, top, width, height)。"""
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return left, top, right - left, bottom - top


def is_foreground(hwnd: int) -> bool:
    """判断指定窗口是否为当前前台窗口。"""
    try:
        return win32gui.GetForegroundWindow() == hwnd
    except Exception:
        return False
