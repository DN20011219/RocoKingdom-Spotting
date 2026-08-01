#!/usr/bin/env python3
"""公用截图工具 — 封装游戏窗口查找、截图、批量截帧逻辑。

供 tools/debug/ 和 tools/yolo_tools/ 下的工具共用。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 路径自举：tools/ -> 项目根目录
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from capture.window import find_window_by_keyword, get_client_rect_on_screen, is_foreground
from capture.grabber import FrameGrabber


def find_game_window(keyword: str = "洛克王国", wait_foreground: bool = True) -> Optional[int]:
    """查找游戏窗口，可选等待其回到前台。

    返回窗口句柄，未找到返回 None。
    """
    print(f"[capture] 查找窗口：{keyword}")
    hwnd = find_window_by_keyword(keyword)
    if hwnd is None:
        print(f"[capture] 未找到窗口!")
        return None
    print(f"[capture] 窗口句柄：{hwnd}")

    if wait_foreground and not is_foreground(hwnd):
        print(f"[capture] 窗口不在前台，等待切换到前台...")
        while not is_foreground(hwnd):
            time.sleep(0.5)
        print(f"[capture] 窗口已回到前台")

    return hwnd


def capture_frame(hwnd: Optional[int] = None, keyword: str = "洛克王国") -> Optional[np.ndarray]:
    """截取一帧游戏画面。

    如果未提供 hwnd，会自动查找窗口。
    返回 BGR 图像，失败返回 None。
    """
    if hwnd is None:
        hwnd = find_game_window(keyword, wait_foreground=True)
        if hwnd is None:
            return None

    rect = get_client_rect_on_screen(hwnd)
    grabber = FrameGrabber("screen-client")
    frame = grabber.grab(hwnd, rect)

    if frame is None or frame.size == 0:
        print(f"[capture] 截图失败!")
        return None

    return frame


def capture_batch(
    count: int,
    output_dir: Path,
    keyword: str = "洛克王国",
    interval: float = 0.5,
) -> List[Path]:
    """批量截取 N 帧并保存到 output_dir。

    返回保存的图片路径列表。自动跳过空帧。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hwnd = find_game_window(keyword, wait_foreground=True)
    if hwnd is None:
        return []

    grabber = FrameGrabber("screen-client")
    saved_paths: List[Path] = []

    # 从已有图片的最大编号 + 1 开始，避免覆盖
    existing = sorted(output_dir.glob("frame_*.png"))
    start_idx = len(existing) + 1

    print(f"[capture] 开始批量截图：{count} 帧，间隔 {interval}s")
    print(f"[capture] 输出目录：{output_dir}（从 frame_{start_idx:04d} 开始）")

    for i in range(count):
        rect = get_client_rect_on_screen(hwnd)
        frame = grabber.grab(hwnd, rect)

        if frame is None or frame.size == 0:
            print(f"  [{i+1}/{count}] 截图失败，跳过")
            time.sleep(interval)
            continue

        filename = f"frame_{start_idx + i:04d}.png"
        filepath = output_dir / filename

        # 使用 imencode 支持中文路径
        ext = filepath.suffix
        success, buf = cv2.imencode(ext, frame)
        if success:
            with open(filepath, 'wb') as f:
                f.write(buf.tobytes())
            saved_paths.append(filepath)
            print(f"  [{i+1}/{count}] 已保存：{filename} ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"  [{i+1}/{count}] 保存失败")

        if i < count - 1:
            time.sleep(interval)

    print(f"[capture] 完成：{len(saved_paths)}/{count} 帧已保存")
    return saved_paths
