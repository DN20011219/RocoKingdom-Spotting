"""GDI 覆盖层 — 在游戏窗口上绘制半透明检测框 + 名字标注。

从 capture/realtime_monitor.py 提取 GDI 覆盖层逻辑，增加框外名字标注。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Any, Callable, List, Optional

import win32con
import win32gui

from sentinel.detectors.base import Detection


# ---------------------------------------------------------------------------
# GDI 画布
# ---------------------------------------------------------------------------

class GdiCanvas:
    """简单的 GDI 绘制封装。"""

    def __init__(self, hdc: int, width: int, height: int) -> None:
        self.hdc = hdc
        self.width = width
        self.height = height

    def clear(self, color_key: int = 0x00FF00FF) -> None:
        brush = ctypes.windll.gdi32.CreateSolidBrush(color_key)
        rect = wintypes.RECT(0, 0, self.width, self.height)
        ctypes.windll.user32.FillRect(self.hdc, ctypes.byref(rect), brush)
        ctypes.windll.gdi32.DeleteObject(brush)

    def rectangle(self, x1: int, y1: int, x2: int, y2: int,
                  color: int, width: int = 2) -> None:
        pen = ctypes.windll.gdi32.CreatePen(0, width, color)
        old_pen = ctypes.windll.gdi32.SelectObject(self.hdc, pen)
        null_brush = ctypes.windll.gdi32.GetStockObject(5)
        old_brush = ctypes.windll.gdi32.SelectObject(self.hdc, null_brush)
        ctypes.windll.gdi32.Rectangle(self.hdc, x1, y1, x2, y2)
        ctypes.windll.gdi32.SelectObject(self.hdc, old_brush)
        ctypes.windll.gdi32.SelectObject(self.hdc, old_pen)
        ctypes.windll.gdi32.DeleteObject(pen)

    def filled_rectangle(self, x1: int, y1: int, x2: int, y2: int,
                         color: int) -> None:
        """绘制实心矩形（用于文字背景）。"""
        brush = ctypes.windll.gdi32.CreateSolidBrush(color)
        rect = wintypes.RECT(x1, y1, x2, y2)
        ctypes.windll.user32.FillRect(self.hdc, ctypes.byref(rect), brush)
        ctypes.windll.gdi32.DeleteObject(brush)

    def text(self, x: int, y: int, text_str: str,
             color: int = 0x00FFFFFF, bg_color: int = 0x00000000) -> None:
        """在指定位置绘制文字（使用系统默认字体）。"""
        ctypes.windll.gdi32.SetBkMode(self.hdc, 1)  # TRANSPARENT
        ctypes.windll.gdi32.SetTextColor(self.hdc, color)
        ctypes.windll.user32.DrawTextW(
            self.hdc, text_str, len(text_str),
            ctypes.byref(wintypes.RECT(x, y, x + 300, y + 30)),
            0x00000040,  # DT_NOPREFIX
        )

    def crosshair(self, cx: int, cy: int, size: int = 12,
                  color: int = 0x000000FF, width: int = 1) -> None:
        pen = ctypes.windll.gdi32.CreatePen(0, width, color)
        old_pen = ctypes.windll.gdi32.SelectObject(self.hdc, pen)
        ctypes.windll.gdi32.MoveToEx(self.hdc, cx - size, cy, None)
        ctypes.windll.gdi32.LineTo(self.hdc, cx + size, cy)
        ctypes.windll.gdi32.MoveToEx(self.hdc, cx, cy - size, None)
        ctypes.windll.gdi32.LineTo(self.hdc, cx, cy + size)
        ctypes.windll.gdi32.SelectObject(self.hdc, old_pen)
        ctypes.windll.gdi32.DeleteObject(pen)


# ---------------------------------------------------------------------------
# 覆盖层窗口
# ---------------------------------------------------------------------------

class OverlayWindow:
    """透明覆盖层窗口 — 在游戏窗口上方绘制检测框和名字。"""

    _registered_classes: set = set()

    def __init__(self, class_name: str = "SentinelOverlay") -> None:
        self._class_name = class_name
        self._hwnd: Optional[int] = None
        self._width = 0
        self._height = 0
        self._register_class()

    def _register_class(self) -> None:
        if self._class_name in OverlayWindow._registered_classes:
            return

        def _wnd_proc(hwnd, msg, wparam, lparam):
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = _wnd_proc
        wc.lpszClassName = self._class_name
        wc.hInstance = win32gui.GetModuleHandle(None)
        wc.hbrBackground = win32gui.GetStockObject(5)
        try:
            win32gui.RegisterClass(wc)
        except Exception:
            pass
        OverlayWindow._registered_classes.add(self._class_name)

    def ensure(self, hwnd_parent: int, x: int, y: int, w: int, h: int) -> None:
        """确保覆盖层窗口存在且位置/大小正确。"""
        if self._hwnd is not None and (w, h) != (self._width, self._height):
            self.destroy()

        if self._hwnd is None:
            ex_style = (
                win32con.WS_EX_LAYERED
                | win32con.WS_EX_TRANSPARENT
                | win32con.WS_EX_TOPMOST
                | win32con.WS_EX_TOOLWINDOW
                | win32con.WS_EX_NOACTIVATE
            )
            self._hwnd = win32gui.CreateWindowEx(
                ex_style, self._class_name, "",
                win32con.WS_POPUP, x, y, w, h,
                0, 0, win32gui.GetModuleHandle(None), None,
            )
            win32gui.SetLayeredWindowAttributes(self._hwnd, 0xFF00FF, 0, win32con.LWA_COLORKEY)
            win32gui.ShowWindow(self._hwnd, win32con.SW_SHOWNOACTIVATE)
            self._width, self._height = w, h
        else:
            win32gui.SetWindowPos(
                self._hwnd, win32con.HWND_TOPMOST,
                x, y, w, h,
                win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
            )

    def render(self, detections: List[Detection],
               frame_w: int, frame_h: int) -> None:
        """绘制所有检测结果。"""
        if not self._hwnd:
            return

        hdc = ctypes.windll.user32.GetDC(self._hwnd)
        mem_dc = ctypes.windll.gdi32.CreateCompatibleDC(hdc)
        bmp = ctypes.windll.gdi32.CreateCompatibleBitmap(hdc, self._width, self._height)
        old_bmp = ctypes.windll.gdi32.SelectObject(mem_dc, bmp)

        try:
            canvas = GdiCanvas(mem_dc, self._width, self._height)
            canvas.clear()

            scale_x = self._width / max(frame_w, 1)
            scale_y = self._height / max(frame_h, 1)

            # 绘制边框
            canvas.rectangle(0, 0, self._width - 1, self._height - 1,
                             color=0x0000FFFF, width=3)

            for det in detections:
                x1 = int(det.x * scale_x)
                y1 = int(det.y * scale_y)
                x2 = int((det.x + det.w) * scale_x)
                y2 = int((det.y + det.h) * scale_y)

                # 框颜色：绿色=匹配成功，黄色=低分
                box_color = 0x0000FFFF if det.is_low_score else 0x0000FF00
                canvas.rectangle(x1, y1, x2, y2, color=box_color, width=2)

                # 十字准星
                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                canvas.crosshair(cx, cy, size=10, color=0x000000FF, width=1)

                # 框外名字标注（在矩形框上方）
                label = f"{det.name}:{det.score:.2f}"
                label_y = y1 - 18 if y1 - 18 > 0 else y2 + 2

                # 文字背景
                text_w = len(label) * 7 + 6
                text_h = 16
                canvas.filled_rectangle(
                    x1, label_y, x1 + text_w, label_y + text_h,
                    color=0x00000000,  # 黑色背景
                )
                # 文字（白色）
                canvas.text(x1 + 3, label_y + 1, label, color=0x00FFFFFF)

            # 拷贝到屏幕
            ctypes.windll.gdi32.BitBlt(
                hdc, 0, 0, self._width, self._height,
                mem_dc, 0, 0, 0x00CC0020,
            )
        finally:
            ctypes.windll.gdi32.SelectObject(mem_dc, old_bmp)
            ctypes.windll.gdi32.DeleteObject(bmp)
            ctypes.windll.gdi32.DeleteDC(mem_dc)
            ctypes.windll.user32.ReleaseDC(self._hwnd, hdc)

    def hide(self) -> None:
        if self._hwnd:
            ctypes.windll.user32.ShowWindow(self._hwnd, 0)

    def destroy(self) -> None:
        if self._hwnd:
            ctypes.windll.user32.DestroyWindow(self._hwnd)
            self._hwnd = None
