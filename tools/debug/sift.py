#!/usr/bin/env python3
"""SIFT 特征点匹配诊断工具 — 支持单模板/全部模板测试。

用法:
    cd RocoKingdom-Spotting
    python tools/debug/sift.py                     # 测试所有 labels/*.png
    python tools/debug/sift.py -t labels/hello.png # 只测单个模板
    python tools/debug/sift.py --ratio 0.85 --save
"""

import sys
import time
from pathlib import Path

# 路径自举：tools/debug/ -> tools/ -> 项目根目录
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import cv2
import numpy as np

from capture.window import find_window_by_keyword, get_client_rect_on_screen, is_foreground
from capture.grabber import FrameGrabber


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, img)
    if success:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    return False


def _match_template(sift, matcher, tpl_gray, tpl_bgr, tpl_name,
                    search_gray, search_bgr, search_kp, search_desc,
                    roi_l, roi_t, scales, ratio, threshold):
    """对单个模板执行 SIFT 匹配，返回结果 dict。"""
    th, tw = tpl_gray.shape[:2]

    tpl_kp, tpl_desc = sift.detectAndCompute(tpl_gray, None)
    if tpl_desc is None or len(tpl_kp) == 0:
        return {"name": tpl_name, "found": False, "count": 0,
                "error": "模板未提取到特征点"}

    best_count = 0
    best_matches = []
    best_scale = 1.0
    best_x, best_y, best_w, best_h = 0, 0, tw, th

    for scale in scales:
        sw = max(1, int(tw * scale))
        sh = max(1, int(th * scale))
        if sw > search_gray.shape[1] or sh > search_gray.shape[0]:
            continue

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(tpl_gray, (sw, sh), interpolation=interp)

        scaled_kp, scaled_desc = sift.detectAndCompute(resized, None)
        if scaled_desc is None or len(scaled_kp) == 0:
            continue

        matches = matcher.knnMatch(scaled_desc, search_desc, k=2)
        good = [m for m, n in matches if m.distance < ratio * n.distance]
        count = len(good)

        if count > best_count:
            best_count = count
            best_matches = good
            best_scale = scale

            if good:
                dst_pts = np.float32([search_kp[m.trainIdx].pt for m in good]).reshape(-1, 2)
                cx, cy = np.mean(dst_pts[:, 0]), np.mean(dst_pts[:, 1])
                best_x = roi_l + int(cx - sw / 2)
                best_y = roi_t + int(cy - sh / 2)
                best_w = sw
                best_h = sh

    return {
        "name": tpl_name,
        "found": best_count >= threshold,
        "count": best_count,
        "threshold": threshold,
        "x": best_x, "y": best_y, "w": best_w, "h": best_h,
        "scale": best_scale,
        "tpl_kp": tpl_kp,
        "tpl_bgr": tpl_bgr,
        "matches": best_matches,
        "search_kp": search_kp,
        "search_bgr": search_bgr,
    }


def main():
    parser = argparse.ArgumentParser(description="SIFT 特征点匹配诊断")
    parser.add_argument("-t", "--template", default=None,
                        help="单个模板路径，不指定则测试 labels/ 下所有 .png")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="好匹配点数量阈值 (默认: 5.0)")
    parser.add_argument("--roi", nargs=4, type=float, default=[0, 0, 1, 1],
                        metavar=("L", "T", "R", "B"),
                        help="ROI 比例坐标 (默认: 0 0 1 1 即全屏)")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0],
                        help="缩放档位 (默认: 1.0)")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Lowe's ratio test 阈值 (默认: 0.8, 越小越严格)")
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
    base_dir = Path(__file__).resolve().parent.parent.parent  # 项目根目录
    if args.template:
        tpl_paths = [base_dir / args.template]
    else:
        labels_dir = base_dir / "labels"
        tpl_paths = sorted(labels_dir.glob("*.png"))

    if not tpl_paths:
        print(f"[3] 未找到模板文件!")
        return

    print(f"[3] 共 {len(tpl_paths)} 个模板待测试")

    # 4. ROI
    frame_h, frame_w = frame.shape[:2]
    roi_l = max(0, int(frame_w * args.roi[0]))
    roi_t = max(0, int(frame_h * args.roi[1]))
    roi_r = min(frame_w, int(frame_w * args.roi[2]))
    roi_b = min(frame_h, int(frame_h * args.roi[3]))
    print(f"[4] ROI：({roi_l},{roi_t})-({roi_r},{roi_b}) = {roi_r-roi_l}x{roi_b-roi_t}")

    search_gray = frame[roi_t:roi_b, roi_l:roi_r]
    search_bgr = frame[roi_t:roi_b, roi_l:roi_r]

    # 5. 提取搜索区域 SIFT 特征（只提取一次）
    print(f"[5] 开始 SIFT 匹配 (threshold={args.threshold}, ratio={args.ratio})")
    print(f"    scales: {args.scales}")

    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    search_kp, search_desc = sift.detectAndCompute(search_gray, None)
    print(f"    搜索区域特征点：{len(search_kp) if search_kp else 0}")
    if search_desc is None or len(search_kp) == 0:
        print(f"    搜索区域未提取到特征点!")
        return

    # 6. 逐个模板匹配
    results = []
    for i, tpl_path in enumerate(tpl_paths, 1):
        if not tpl_path.exists():
            print(f"\n  [{i}/{len(tpl_paths)}] {tpl_path.name} — 文件不存在，跳过")
            continue

        data = np.fromfile(str(tpl_path), dtype=np.uint8)
        tpl = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if tpl.ndim == 3 and tpl.shape[2] == 4:
            tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2BGR)
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        print(f"\n  [{i}/{len(tpl_paths)}] {tpl_path.name} ({tpl.shape[1]}x{tpl.shape[0]})")

        res = _match_template(
            sift, matcher, tpl_gray, tpl, tpl_path.name,
            search_gray, search_bgr, search_kp, search_desc,
            roi_l, roi_t, args.scales, args.ratio, args.threshold,
        )

        if res.get("error"):
            print(f"    {res['error']}")
        elif res["found"]:
            print(f"    [OK] 好匹配点={res['count']}  位置=({res['x']},{res['y']})  尺寸={res['w']}x{res['h']}")
        else:
            print(f"    [FAIL] 好匹配点={res['count']} (阈值: {res['threshold']}, 差距: {res['threshold'] - res['count']})")

        results.append(res)

    # 7. 汇总
    found_count = sum(1 for r in results if r.get("found"))
    print(f"\n{'='*60}")
    print(f"[汇总] {found_count}/{len(results)} 个模板匹配成功")
    print(f"{'='*60}")

    # 8. 保存可视化
    if args.save:
        out_dir = Path(__file__).resolve().parent / "debug"
        out_dir.mkdir(exist_ok=True)

        _imwrite_unicode(str(out_dir / "frame.png"), frame)

        # 在原图上画所有匹配框
        result_img = frame.copy()
        for res in results:
            if res.get("found") and res["count"] > 0:
                cv2.rectangle(result_img,
                              (res["x"], res["y"]),
                              (res["x"] + res["w"], res["y"] + res["h"]),
                              (0, 255, 0), 2)
                cv2.putText(result_img, f"{res['name']}:{res['count']}",
                            (res["x"], res["y"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        _imwrite_unicode(str(out_dir / "match_result.png"), result_img)

        # 每个模板的匹配连线图
        for res in results:
            if res.get("matches") and res["count"] > 0:
                match_img = cv2.drawMatches(
                    res["tpl_bgr"], res["tpl_kp"],
                    res["search_bgr"], res["search_kp"],
                    res["matches"][:20], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                _imwrite_unicode(str(out_dir / f"sift_{res['name']}.png"), match_img)

        print(f"\n[图片] 已保存到 {out_dir}/:")
        print(f"    frame.png           - 原始截图")
        print(f"    match_result.png    - 所有匹配框标注")
        for res in results:
            if res.get("matches") and res["count"] > 0:
                print(f"    sift_{res['name']}.png - {res['name']} 匹配连线")


if __name__ == "__main__":
    main()
