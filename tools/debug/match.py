#!/usr/bin/env python3
"""模板匹配诊断工具 — 支持单模板/全部模板批量测试。

用法:
    cd sentinel
    python tools/debug/match.py                     # 测试所有 labels/*.png
    python tools/debug/match.py -t labels/hello.png # 只测单个模板
    python tools/debug/match.py --threshold 0.85 --save
"""

import sys
import time
from pathlib import Path

# 路径自举：tools/debug/ -> tools/ -> sentinel/ -> 父项目
_parent = str(Path(__file__).resolve().parent.parent.parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import argparse
import cv2
import numpy as np

from sentinel.capture.window import find_window_by_keyword, get_client_rect_on_screen, is_foreground
from sentinel.capture.grabber import FrameGrabber


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """保存图片到可能包含中文的路径。"""
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    return False


def _match_template(tpl_gray, tpl_bgr, tpl_name, search_gray, search_bgr,
                    roi_l, roi_t, scales, threshold, color_threshold):
    """对单个模板执行多尺度匹配，返回结果 dict。"""
    th, tw = tpl_gray.shape[:2]

    best_score = -1.0
    best_color = -1.0
    best_x, best_y, best_w, best_h = 0, 0, tw, th
    best_scale = 1.0

    for scale in scales:
        sw = max(1, int(tw * scale))
        sh = max(1, int(th * scale))
        if sw > search_gray.shape[1] or sh > search_gray.shape[0]:
            continue

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized_gray = cv2.resize(tpl_gray, (sw, sh), interpolation=interp)
        resized_bgr = cv2.resize(tpl_bgr, (sw, sh), interpolation=interp)

        # 灰度匹配（TM_CCORR_NORMED 对文字像素值差异更敏感）
        result = cv2.matchTemplate(search_gray, resized_gray, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # 颜色校验
        patch = search_bgr[max_loc[1]:max_loc[1] + sh, max_loc[0]:max_loc[0] + sw]
        if patch.shape == resized_bgr.shape:
            diff = cv2.absdiff(patch, resized_bgr)
            color_score = float(1.0 - np.mean(diff) / 255.0)
        else:
            color_score = 0.0

        if max_val > best_score:
            best_score = float(max_val)
            best_color = color_score
            best_scale = scale
            best_x = roi_l + max_loc[0]
            best_y = roi_t + max_loc[1]
            best_w = sw
            best_h = sh

    # 循环结束后统一判断
    found = best_score >= threshold and best_color >= color_threshold

    return {
        "name": tpl_name,
        "found": found,
        "gray_score": best_score,
        "color_score": best_color,
        "x": best_x, "y": best_y, "w": best_w, "h": best_h,
        "scale": best_scale,
        "threshold": threshold,
        "color_threshold": color_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="模板匹配诊断（支持多模板）")
    parser.add_argument("-t", "--template", default=None,
                        help="单个模板路径，不指定则测试 labels/ 下所有 .png")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="灰度匹配阈值 (默认: 0.85)")
    parser.add_argument("--color-threshold", type=float, default=0.85,
                        help="颜色验证阈值 (默认: 0.85)")
    parser.add_argument("--roi", nargs=4, type=float, default=[0, 0, 1, 1],
                        metavar=("L", "T", "R", "B"),
                        help="ROI 比例坐标 (默认: 0 0 1 1 即全屏)")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0],
                        help="缩放档位 (默认: 1.0)")
    parser.add_argument("--save", action="store_true",
                        help="保存匹配结果可视化")
    args = parser.parse_args()

    # 1. 找窗口
    keyword = "洛克王国"
    print(f"[1] 查找窗口：{keyword}")
    hwnd = find_window_by_keyword(keyword)
    if hwnd is None:
        print(f"    未找到窗口! 请确认游戏已打开")
        return
    print(f"    窗口句柄：{hwnd}")

    # 1.5 等待窗口回到前台
    if not is_foreground(hwnd):
        print(f"    窗口不在前台，等待切换到前台...")
        while not is_foreground(hwnd):
            time.sleep(0.5)
        print(f"    窗口已回到前台")

    # 2. 截图
    print(f"[2] 截图...")
    rect = get_client_rect_on_screen(hwnd)
    left, top, width, height = rect
    print(f"    客户区：{width}x{height} @ ({left},{top})")

    grabber = FrameGrabber("screen-client")
    frame = grabber.grab(hwnd, rect)
    if frame is None or frame.size == 0:
        print(f"    截图失败!")
        return
    print(f"    截图尺寸：{frame.shape[1]}x{frame.shape[0]}")

    # 3. 收集模板列表
    base_dir = Path(__file__).resolve().parent.parent.parent  # sentinel/
    if args.template:
        tpl_paths = [base_dir / args.template]
    else:
        labels_dir = base_dir / "labels"
        tpl_paths = sorted(labels_dir.glob("*.png"))

    if not tpl_paths:
        print(f"[3] 未找到模板文件!")
        return

    print(f"[3] 共 {len(tpl_paths)} 个模板待测试")

    # 4. ROI 裁剪
    frame_h, frame_w = frame.shape[:2]
    roi_l = max(0, int(frame_w * args.roi[0]))
    roi_t = max(0, int(frame_h * args.roi[1]))
    roi_r = min(frame_w, int(frame_w * args.roi[2]))
    roi_b = min(frame_h, int(frame_h * args.roi[3]))
    print(f"[4] ROI：({roi_l},{roi_t})-({roi_r},{roi_b}) = {roi_r-roi_l}x{roi_b-roi_t}")

    search_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_t:roi_b, roi_l:roi_r]
    search_bgr = frame[roi_t:roi_b, roi_l:roi_r]

    # 5. 逐个模板匹配
    print(f"[5] 开始匹配 (threshold={args.threshold}, color_threshold={args.color_threshold})")
    print(f"    scales: {args.scales}")
    print(f"    算法: TM_CCORR_NORMED")

    results = []
    for i, tpl_path in enumerate(tpl_paths, 1):
        if not tpl_path.exists():
            print(f"\n  [{i}/{len(tpl_paths)}] {tpl_path.name} — 文件不存在，跳过")
            continue

        # 加载模板（支持中文路径）
        data = np.fromfile(str(tpl_path), dtype=np.uint8)
        tpl_bgr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if tpl_bgr is None:
            print(f"\n  [{i}/{len(tpl_paths)}] {tpl_path.name} — 读取失败，跳过")
            continue
        if tpl_bgr.ndim == 3 and tpl_bgr.shape[2] == 4:
            tpl_bgr = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGRA2BGR)
        tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

        print(f"\n  [{i}/{len(tpl_paths)}] {tpl_path.name} ({tpl_bgr.shape[1]}x{tpl_bgr.shape[0]})")

        res = _match_template(
            tpl_gray, tpl_bgr, tpl_path.name,
            search_gray, search_bgr,
            roi_l, roi_t, args.scales, args.threshold, args.color_threshold,
        )

        if res["found"]:
            print(f"    [OK] gray={res['gray_score']:.4f}  color={res['color_score']:.4f}  "
                  f"位置=({res['x']},{res['y']})  尺寸={res['w']}x{res['h']}")
        else:
            gray_gap = args.threshold - res['gray_score']
            color_gap = args.color_threshold - res['color_score']
            fail_reason = "灰度不足" if gray_gap > 0 else "颜色不足"
            print(f"    [FAIL] gray={res['gray_score']:.4f}  color={res['color_score']:.4f}  "
                  f"({fail_reason}, 差距: {max(gray_gap, color_gap):.4f})")

        results.append(res)

    # 6. 汇总
    found_count = sum(1 for r in results if r.get("found"))
    print(f"\n{'='*60}")
    print(f"[汇总] {found_count}/{len(results)} 个模板匹配成功")
    print(f"{'='*60}")

    # 7. 保存可视化
    if args.save:
        out_dir = Path(__file__).resolve().parent / "debug"
        out_dir.mkdir(exist_ok=True)

        _imwrite_unicode(str(out_dir / "frame.png"), frame)

        # 在原图上画所有匹配框
        result_img = frame.copy()
        for res in results:
            if res["found"]:
                cv2.rectangle(result_img,
                              (res["x"], res["y"]),
                              (res["x"] + res["w"], res["y"] + res["h"]),
                              (0, 255, 0), 2)
                label = f"{res['name']}:{res['gray_score']:.2f}"
                cv2.putText(result_img, label,
                            (res["x"], res["y"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        _imwrite_unicode(str(out_dir / "match_result.png"), result_img)

        print(f"\n[图片] 已保存到 {out_dir}/:")
        print(f"    frame.png        - 原始截图")
        print(f"    match_result.png - 所有匹配框标注")


if __name__ == "__main__":
    main()
