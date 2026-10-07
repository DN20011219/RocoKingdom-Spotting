#!/usr/bin/env python3
"""YOLO 数据集标注工具 — 加载指定类别目录的图片并标注。

用法:
    python tools/yolo_tools/yolo_labeler.py --class pet1
    python tools/yolo_tools/yolo_labeler.py --class pet1 --from frame_0050  # 从指定图片开始

流程:
    1. 自动加载 images/<class>/ 目录下所有图片
    2. 自动加载 labels/<class>/ 下已有标注（断点续标）
    3. 自动跳到第一张未标注的图片
    4. 鼠标画框标注，自动保存到 labels/<class>/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 路径自举：tools/yolo_tools/ -> tools/ -> 项目根目录
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from dataset_registry import (DatasetError, check_no_drift, load_classes,
                              register_class, write_split)

# 颜色循环（BGR）
_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    (255, 128, 0), (128, 0, 255),
]

_DATASET_DIR = Path(_project_root) / "datasets" / "yolo_dataset"


class YoloLabeler:
    """YOLO 数据集标注器。"""

    def __init__(self, class_name: str) -> None:
        self.class_name = class_name
        self.dataset_dir = _DATASET_DIR
        self.images_dir = self.dataset_dir / "images" / class_name
        self.labels_dir = self.dataset_dir / "labels" / class_name

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # class_id 取自 classes.txt 注册表（只追加、永不重排）。用目录字母序下标会在
        # 中间插入新类别时把其后所有类别的 id 静默 +1，而已写盘的 .txt 不会更新 ——
        # ultralytics 只校验 id < nc，于是训练不报错，直接产出类别全错的模型。
        # 先审计再注册：已有数据错位时拒绝启动，避免把错位继续扩大。
        check_no_drift(self.dataset_dir)
        self.class_id = register_class(class_name, self.dataset_dir)
        print(f"[labeler] 类别 {class_name} -> class id {self.class_id}")

        # 图片列表和当前索引
        self.image_paths: List[Path] = []
        self.current_idx = 0

        # 标注数据: {image_stem: [(class_id, cx, cy, w, h), ...]}
        self.annotations: Dict[str, List[Tuple[int, float, float, float, float]]] = {}

        # 鼠标状态
        self.drawing = False
        self.draw_start = (0, 0)
        self.draw_end = (0, 0)
        self.temp_box: Optional[Tuple[int, int, int, int]] = None

        # 当前显示的图片尺寸缓存（避免鼠标回调重复加载图片）
        self._cached_img_shape: Optional[Tuple[int, int]] = None

        # 撤销栈: 记录最近添加的标注 (img_stem, index_in_list)
        self._undo_stack: List[Tuple[str, int]] = []

        # 显示参数
        self.window_name = "YOLO Labeler"
        self.status_bar_height = 40

    # ------------------------------------------------------------------
    # 加载阶段
    # ------------------------------------------------------------------
    def load_images(self) -> bool:
        """加载类别目录下所有图片（含断点续标）。返回是否有图片。"""
        imgs = sorted(self.images_dir.glob("*.png")) + sorted(self.images_dir.glob("*.jpg"))
        self.image_paths = [p for p in imgs if p.exists()]

        if not self.image_paths:
            print(f"[labeler] 目录 {self.images_dir} 下没有图片!")
            return False

        print(f"[labeler] 已加载 {len(self.image_paths)} 张图片")

        # 加载已有标注（断点续标）
        loaded = 0
        for img_path in self.image_paths:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            annotations = []
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            annotations.append((int(parts[0]),
                                                float(parts[1]), float(parts[2]),
                                                float(parts[3]), float(parts[4])))
                if annotations:
                    self.annotations[img_path.stem] = annotations
                    loaded += 1
            except Exception as e:
                print(f"[labeler] 加载标注失败 {label_path}: {e}")

        if loaded:
            print(f"[labeler] 断点续标: 已加载 {loaded} 张已有标注")

            # 自动跳到第一张未标注的图片
            for i, img_path in enumerate(self.image_paths):
                if img_path.stem not in self.annotations:
                    self.current_idx = i
                    print(f"[labeler] 从第 {i + 1} 张开始标注: {img_path.name}")
                    return True
            # 全部已标注
            self.current_idx = len(self.image_paths) - 1
            print(f"[labeler] 所有 {len(self.image_paths)} 张图片均已标注!")
        return True

    # ------------------------------------------------------------------
    # 标注阶段
    # ------------------------------------------------------------------
    def run(self) -> None:
        """主循环：逐张标注。"""
        if not self.image_paths:
            print("[labeler] 没有图片可标注!")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print(f"\n[labeler] 标注模式 — 类别: {self.class_name} (class_id={self.class_id})")
        print(f"  Space/D: 下一张 | A: 上一张 | S: 保存")
        print(f"  Z: 撤销 | X: 清除当前图所有标注")
        print(f"  右键: 删除最近标注 | Q/Esc: 退出\n")

        while 0 <= self.current_idx < len(self.image_paths):
            img_path = self.image_paths[self.current_idx]
            img = self._load_image(img_path)
            if img is None:
                self.current_idx += 1
                continue
            self._show_and_wait(img, img_path.stem)

        # 退出前保存
        self._save_progress()
        self._generate_data_yaml()
        cv2.destroyAllWindows()
        print(f"\n[labeler] 标注完成，数据已保存到 {self.dataset_dir}")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """加载图片，支持中文路径。"""
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _show_and_wait(self, img: np.ndarray, img_stem: str) -> None:
        """显示图片并等待用户操作。"""
        h, w = img.shape[:2]
        # 缓存当前图片尺寸，供鼠标回调使用（避免每次事件重新加载图片）
        self._cached_img_shape = (h, w)

        while True:
            canvas = img.copy()
            self._draw_ui(canvas, img_stem, w, h)
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 27 or key == ord('q') or key == ord('Q'):
                return
            elif key == ord(' ') or key == ord('d') or key == ord('D'):
                self._save_current_label(img_stem)
                self.current_idx += 1
                return
            elif key == ord('a') or key == ord('A'):
                self._save_current_label(img_stem)
                self.current_idx -= 1
                return
            elif key == ord('s') or key == ord('S'):
                self._save_progress()
                print("[labeler] 已保存进度")
            elif key == ord('z') or key == ord('Z'):
                # Ctrl+Z 或 Z: 撤销最后一个标注
                self._undo_last(img_stem)
            elif key == ord('x') or key == ord('X'):
                # X: 清除当前图片所有标注
                if img_stem in self.annotations and self.annotations[img_stem]:
                    count = len(self.annotations[img_stem])
                    self.annotations[img_stem] = []
                    # 同时清除撤销栈中该图片的记录
                    self._undo_stack = [(s, i) for s, i in self._undo_stack if s != img_stem]
                    print(f"[labeler] 已清除 {count} 个标注")

    def _draw_ui(self, canvas: np.ndarray, img_stem: str, img_w: int, img_h: int) -> None:
        """绘制状态栏和已有标注。"""
        h, w = canvas.shape[:2]

        # 绘制已有标注
        for class_id, cx, cy, bw, bh in self.annotations.get(img_stem, []):
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = _COLORS[class_id % len(_COLORS)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, self.class_name, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 正在画的框
        if self.temp_box:
            x1, y1, x2, y2 = self.temp_box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 状态栏（顶部半透明条）
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.status_bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        status = f"[{self.current_idx + 1}/{len(self.image_paths)}] {img_stem}"
        cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        class_text = f"Class: {self.class_name}"
        text_w = len(class_text) * 12
        cv2.putText(canvas, class_text, (w - text_w - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _undo_last(self, img_stem: str) -> None:
        """撤销当前图片最后一个标注。"""
        if img_stem in self.annotations and self.annotations[img_stem]:
            self.annotations[img_stem].pop()
            # 清除撤销栈中对应记录
            for i in range(len(self._undo_stack) - 1, -1, -1):
                if self._undo_stack[i][0] == img_stem:
                    self._undo_stack.pop(i)
                    break
            print(f"[labeler] 撤销标注 (剩余 {len(self.annotations[img_stem])} 个)")

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """鼠标事件处理。"""
        if y < self.status_bar_height:
            return

        # 使用缓存的图片尺寸，避免每次鼠标事件都重新加载图片
        if self._cached_img_shape is None:
            return
        h, w = self._cached_img_shape

        img_path = self.image_paths[self.current_idx]

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.draw_start = (x, y)
            self.draw_end = (x, y)
            self.temp_box = (x, y, x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_end = (x, y)
                x1, y1 = self.draw_start
                x2, y2 = self.draw_end
                self.temp_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.draw_start
                x2, y2 = self.draw_end
                self.temp_box = None

                bw = abs(x2 - x1)
                bh = abs(y2 - y1)
                if bw > 5 and bh > 5:
                    cx = (min(x1, x2) + bw / 2) / w
                    cy = (min(y1, y2) + bh / 2) / h
                    nw = bw / w
                    nh = bh / h

                    stem = img_path.stem
                    if stem not in self.annotations:
                        self.annotations[stem] = []
                    self.annotations[stem].append((self.class_id, cx, cy, nw, nh))
                    # 记录到撤销栈
                    self._undo_stack.append((stem, len(self.annotations[stem]) - 1))
                    print(f"[labeler] 添加标注: {self.class_name} (class_id={self.class_id})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            stem = img_path.stem
            if stem in self.annotations and self.annotations[stem]:
                min_dist = float('inf')
                min_idx = -1
                for i, (class_id, cx, cy, bw, bh) in enumerate(self.annotations[stem]):
                    bx = int(cx * w)
                    by = int(cy * h)
                    dist = (x - bx) ** 2 + (y - by) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                if min_idx >= 0 and min_dist < 10000:
                    self.annotations[stem].pop(min_idx)
                    print(f"[labeler] 删除标注")

    def _save_current_label(self, img_stem: str) -> None:
        """保存当前图片的标注。无标注时删除已有 label 文件。"""
        label_path = self.labels_dir / f"{img_stem}.txt"

        annotations = self.annotations.get(img_stem, [])
        if not annotations:
            # 无标注时删除空文件，避免训练时产生空样本
            if label_path.exists():
                label_path.unlink()
            return

        with open(label_path, 'w', encoding='utf-8') as f:
            for class_id, cx, cy, bw, bh in annotations:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _save_progress(self) -> None:
        """保存所有标注。无标注的删除对应 label 文件。"""
        count = 0
        for stem, annotations in self.annotations.items():
            label_path = self.labels_dir / f"{stem}.txt"
            if not annotations:
                # 清除空标注文件
                if label_path.exists():
                    label_path.unlink()
                continue
            with open(label_path, 'w', encoding='utf-8') as f:
                for class_id, cx, cy, bw, bh in annotations:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            count += 1
        if count:
            print(f"[labeler] 已保存 {count} 个标注文件到 {self.labels_dir}")

    def _generate_data_yaml(self) -> None:
        """重新生成 data.yaml + train.txt / val.txt。

        names 取自 classes.txt 注册表而非目录扫描，顺序与标注里的 id 严格对应；
        train/val 用逐文件稳定哈希切分，新标注的图片不会挪动已有图片的归属。
        """
        s = write_split(self.dataset_dir)
        names = load_classes(self.dataset_dir)
        print(f"[labeler] 已更新 {self.dataset_dir / 'data.yaml'}"
              f"（{len(names)} 个类别: {', '.join(names)}）"
              f" train={len(s.train)} val={len(s.val)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLO 数据集标注工具")
    parser.add_argument("--class", dest="class_name", type=str, required=True,
                        help="类别名称（如 pet1）")
    parser.add_argument("--from", dest="from_image", type=str, default=None,
                        help="从指定图片名开始标注（如 frame_0050）")
    args = parser.parse_args()

    try:
        labeler = YoloLabeler(args.class_name)
    except DatasetError as e:
        print(f"[labeler] {e}")
        sys.exit(1)

    if labeler.load_images():
        # --from 参数: 指定起始图片
        if args.from_image:
            stem = args.from_image
            for i, img_path in enumerate(labeler.image_paths):
                if img_path.stem == stem or img_path.stem.startswith(stem):
                    labeler.current_idx = i
                    print(f"[labeler] 从指定图片开始: {img_path.name} (第 {i + 1} 张)")
                    break
            else:
                print(f"[labeler] 未找到图片: {args.from_image}")
                return

        labeler.run()


if __name__ == "__main__":
    main()
