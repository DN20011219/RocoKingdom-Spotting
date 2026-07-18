#!/usr/bin/env python3
"""热键截图工具 — 基于 Interception 驱动监听 F12，自动截取目标窗口画面。

用法:
    python tools/hotkey_capture.py --output ./screenshots
    python tools/hotkey_capture.py --output ./screenshots --keyword "洛克王国"
    python tools/hotkey_capture.py --output ./screenshots --prefix frame --start 1

按 F12 截图（需目标窗口在前台），按 Esc 退出。
需要 Interception 驱动已安装。
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Optional

import cv2

# 路径自举：tools/ -> 项目根目录
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from capture.window import find_window_by_keyword, get_client_rect_on_screen, is_foreground
from capture.grabber import FrameGrabber

# 引入 RocoKingdom-Clicker 的 InterceptionCore
_clicker_dir = r"C:\Users\17676\Desktop\开发中\RocoKingdom-Clicker"
if _clicker_dir not in sys.path:
    sys.path.insert(0, _clicker_dir)

from InterceptionCore import InterceptionCore, InterceptionKeyStroke


# F12 扫描码
_SCANCODE_F12 = 0x58
_SCANCODE_ESC = 0x01

# Interception 按键状态
_KEY_DOWN = 0x00
_KEY_UP = 0x01


def capture_to_file(hwnd: int, output_dir: Path, prefix: str, index: int) -> Optional[Path]:
    """截取一帧并保存到文件。"""
    rect = get_client_rect_on_screen(hwnd)
    grabber = FrameGrabber("screen-client")
    frame = grabber.grab(hwnd, rect)

    if frame is None or frame.size == 0:
        return None

    filename = f"{prefix}_{index:04d}.png"
    filepath = output_dir / filename

    success, buf = cv2.imencode(".png", frame)
    if success:
        with open(filepath, 'wb') as f:
            f.write(buf.tobytes())
        return filepath
    return None


def find_next_index(output_dir: Path, prefix: str) -> int:
    """查找输出目录中下一个可用的索引。"""
    if not output_dir.exists():
        return 1

    max_idx = 0
    for f in output_dir.glob(f"{prefix}_*.png"):
        try:
            idx = int(f.stem.split('_')[-1])
            max_idx = max(max_idx, idx)
        except ValueError:
            continue
    return max_idx + 1


def main():
    import argparse

    parser = argparse.ArgumentParser(description="热键截图工具（Interception 驱动）")
    parser.add_argument("--output", "-o", type=str, default="./screenshots", help="输出目录")
    parser.add_argument("--keyword", "-k", type=str, default="洛克王国", help="窗口标题关键词")
    parser.add_argument("--prefix", "-p", type=str, default="frame", help="文件名前缀")
    parser.add_argument("--start", "-s", type=int, default=0, help="起始编号（0=自动检测）")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Interception
    core = InterceptionCore()
    if not core.is_ready():
        print(f"[hotkey] Interception 初始化失败:")
        print(f"  {core.init_error}")
        sys.exit(1)

    lib = core._lib
    ctx = core._ctx

    # 设置键盘过滤器：监听所有键盘事件
    INTERCEPTION_FILTER_KEY_ALL = 0xFFFF
    is_keyboard_pred = lib._is_keyboard_pred
    lib.interception_set_filter(ctx, is_keyboard_pred, ctypes.c_ushort(INTERCEPTION_FILTER_KEY_ALL))

    # 查找窗口
    print(f"[hotkey] 查找窗口：{args.keyword}")
    hwnd = find_window_by_keyword(args.keyword)
    if hwnd is None:
        print(f"[hotkey] 未找到窗口！请先打开目标窗口。")
        sys.exit(1)
    print(f"[hotkey] 窗口句柄：{hwnd}")

    # 确定起始编号
    if args.start > 0:
        next_idx = args.start
    else:
        next_idx = find_next_index(output_dir, args.prefix)

    print(f"[hotkey] 输出目录：{output_dir}")
    print(f"[hotkey] 文件前缀：{args.prefix}")
    print(f"[hotkey] 起始编号：{next_idx}")
    print(f"\n[hotkey] 按 F12 截图（需窗口在前台），按 Esc 退出\n")
    print("[hotkey] 热键监听已启动（Interception 驱动）...")

    # 分配键盘 stroke 缓冲区
    stroke_buf = (InterceptionKeyStroke * 1)()

    # 保存原始 argtypes 并临时替换为键盘版本
    _orig_receive_argtypes = lib.interception_receive.argtypes
    _orig_receive_restype = lib.interception_receive.restype
    lib.interception_receive.argtypes = [
        ctypes.c_void_p,   # context
        ctypes.c_int,      # device
        ctypes.POINTER(InterceptionKeyStroke),  # stroke
        ctypes.c_uint,     # count
    ]
    lib.interception_receive.restype = ctypes.c_int

    try:
        while True:
            # 等待输入事件（超时 100ms，便于检查退出条件）
            device = lib.interception_wait_with_timeout(ctx, ctypes.c_ulong(100))
            if device <= 0:
                continue

            # 只处理键盘事件
            if not lib.interception_is_keyboard(device):
                continue

            # 接收按键事件
            received = lib.interception_receive(ctx, device, stroke_buf, 1)
            if received <= 0:
                continue

            code = stroke_buf[0].code
            state = stroke_buf[0].state

            # 只响应按下事件（避免重复触发）
            if state != _KEY_DOWN:
                continue

            if code == _SCANCODE_F12:
                if not is_foreground(hwnd):
                    print(f"[hotkey] 窗口不在前台，跳过")
                    continue

                filepath = capture_to_file(hwnd, output_dir, args.prefix, next_idx)
                if filepath:
                    print(f"[hotkey] 已保存：{filepath.name}")
                    next_idx += 1
                else:
                    print(f"[hotkey] 截图失败")

            elif code == _SCANCODE_ESC:
                print("\n[hotkey] 已退出")
                break

    except KeyboardInterrupt:
        print("\n[hotkey] 已退出")

    finally:
        # 恢复原始 argtypes 并清除过滤器
        try:
            lib.interception_receive.argtypes = _orig_receive_argtypes
            lib.interception_receive.restype = _orig_receive_restype
            lib.interception_set_filter(ctx, is_keyboard_pred, ctypes.c_ushort(0))
        except Exception:
            pass


if __name__ == "__main__":
    main()
