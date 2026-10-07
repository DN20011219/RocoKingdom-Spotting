#!/usr/bin/env python3
"""热键截图工具 — F12 触发截图，保存到 datasets/yolo_dataset/images/<class>/。

用法:
    python tools/hotkey_capture.py --class pet1

操作:
    F12: 截取当前窗口画面
    Esc: 退出

原理:
    通过 Interception 驱动拦截所有键盘事件，检测到 F12 时触发截图，
    然后将每个事件原样转发给系统，不影响正常键盘使用。
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 路径自举
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from capture.window import find_window_by_keyword, get_client_rect_on_screen
from capture.grabber import FrameGrabber
from config import load_config

_yolo_tools = str(Path(__file__).resolve().parent / "yolo_tools")
if _yolo_tools not in sys.path:
    sys.path.insert(0, _yolo_tools)

from dataset_registry import DatasetError, register_class


def main():
    import argparse

    parser = argparse.ArgumentParser(description="热键截图工具")
    parser.add_argument("--class", dest="class_name", type=str, required=True,
                        help="类别名称（如 pet1）")
    args = parser.parse_args()

    class_name = args.class_name

    # 加载配置获取窗口关键词
    cfg = load_config()
    keyword = cfg.capture.window_keyword

    # 输出目录
    output_dir = Path(_project_root) / "datasets" / "yolo_dataset" / "images" / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 注册 class id：追加到 classes.txt 末尾，已有类别的 id 永不重排
    try:
        class_id = register_class(class_name)
    except DatasetError as e:
        print(f"[hotkey] {e}")
        return

    # 计算已有帧编号
    existing = sorted(output_dir.glob("frame_*.png"))
    next_idx = len(existing) + 1

    # 查找窗口
    hwnd = find_window_by_keyword(keyword)
    if hwnd is None:
        print(f"[hotkey] 未找到窗口: {keyword}")
        return

    grabber = FrameGrabber("screen-client")

    print(f"\n[hotkey] 热键截图模式")
    print(f"  类别: {class_name} (class id {class_id})")
    print(f"  输出: {output_dir}")
    print(f"  已有: {len(existing)} 张图片，从 frame_{next_idx:04d} 开始编号")
    print(f"  F12: 截图 | Esc: 退出")
    print(f"  (拦截后原样转发，不影响键盘使用)\n")

    # 导入 Interception
    try:
        from InterceptionCore import InterceptionCore, InterceptionKeyStroke
    except ImportError as e:
        print(f"[hotkey] 无法加载 InterceptionCore: {e}")
        print(f"  请确认 InterceptionCore.py 和 third/Interception/library/x64/interception.dll 存在")
        return

    # 初始化 Interception（构造函数自动初始化）
    core = InterceptionCore()
    if not core.is_ready():
        print(f"[hotkey] Interception 初始化失败:")
        print(f"  {core.init_error}")
        return

    lib = core._lib
    ctx = core._ctx

    # 设置过滤器：拦截所有键盘事件（按下 + 释放）
    KEY_DOWN = 0x01
    KEY_UP = 0x02
    FILTER = KEY_DOWN | KEY_UP
    lib.interception_set_filter(ctx, lib._is_keyboard_pred, FILTER)

    # 临时替换 send/receive 的 argtypes 为键盘版本
    orig_send_argtypes = lib.interception_send.argtypes
    orig_send_restype = lib.interception_send.restype
    orig_recv_argtypes = lib.interception_receive.argtypes
    orig_recv_restype = lib.interception_receive.restype

    lib.interception_send.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(InterceptionKeyStroke), ctypes.c_uint,
    ]
    lib.interception_send.restype = ctypes.c_int

    lib.interception_receive.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(InterceptionKeyStroke), ctypes.c_uint,
    ]
    lib.interception_receive.restype = ctypes.c_int

    captured_count = 0

    try:
        while True:
            # 等待下一个键盘事件（阻塞，无延迟）
            device = lib.interception_wait(ctx)

            # 读取事件
            stroke = InterceptionKeyStroke()
            lib.interception_receive(ctx, device, ctypes.byref(stroke), 1)

            # 原样转发给系统（不影响键盘使用）
            lib.interception_send(ctx, device, ctypes.byref(stroke), 1)

            # 检测 F12 按下（scan code 0x58）
            if stroke.code == 0x58 and stroke.state == KEY_DOWN:
                rect = get_client_rect_on_screen(hwnd)
                frame = grabber.grab(hwnd, rect)

                if frame is not None and frame.size > 0:
                    filename = f"frame_{next_idx:04d}.png"
                    filepath = output_dir / filename
                    success, buf = cv2.imencode(".png", frame)
                    if success:
                        with open(filepath, 'wb') as f:
                            f.write(buf.tobytes())
                        captured_count += 1
                        print(f"  [{captured_count}] {filename}")
                        next_idx += 1
                else:
                    print("  [截图失败]")

            # 检测 Esc 按下（scan code 0x01）
            elif stroke.code == 0x01 and stroke.state == KEY_DOWN:
                print("\n[hotkey] 退出")
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 恢复 argtypes
        lib.interception_send.argtypes = orig_send_argtypes
        lib.interception_send.restype = orig_send_restype
        lib.interception_receive.argtypes = orig_recv_argtypes
        lib.interception_receive.restype = orig_recv_restype
        # 清除过滤器
        lib.interception_set_filter(ctx, lib._is_keyboard_pred, 0)
        # 销毁上下文
        try:
            lib.interception_destroy_context(ctx)
        except Exception:
            pass

    print(f"\n[hotkey] 共截取 {captured_count} 张图片到 {output_dir}")


if __name__ == "__main__":
    main()
