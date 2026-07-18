#!/usr/bin/env python3
"""YOLO 数据集标注工具 — 基于 OpenCV GUI 的 bounding box 标注。

支持从游戏窗口截图或加载本地图片，鼠标画框 + 键盘选类，输出 YOLO 格式。

用法:
    # 从游戏窗口截图后标注
    python tools/yolo_tools/yolo_labeler.py --capture --classes pet1 pet2 pet3

    # 标注本地图片
    python tools/yolo_tools/yolo_labeler.py --images path/to/images --classes pet1 pet2 pet3

    # 指定输出目录
    python tools/yolo_tools/yolo_labeler.py --images ./imgs --classes a b --output datasets/my_data
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 路径自举：tools/yolo_tools/ -> tools/ -> 项目根目录
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入公用截图工具
from tools.capture import capture_batch


# 颜色循环（BGR）
_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    (255, 128, 0), (128, 0, 255),
]


class YoloLabeler:
    """YOLO 数据集标注器。"""

    def __init__(self, classes: List[str], output_dir: Path) -> None:
        self.classes = classes
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        # 确保输出目录存在
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # 图片列表和当前索引
        self.image_paths: List[Path] = []
        self.current_idx = 0

        # 标注数据: {image_stem: [(class_id, cx, cy, w, h), ...]}
        self.annotations: Dict[str, List[Tuple[int, float, float, float, float]]] = {}

        # 当前选中的类别
        self.current_class = 0

        # 鼠标状态
        self.drawing = False
        self.draw_start = (0, 0)
        self.draw_end = (0, 0)
        self.temp_box = None  # (x1, y1, x2, y2) 正在画的框

        # 显示参数
        self.window_name = "YOLO Labeler"
        self.status_bar_height = 40

    def load_images(self, image_paths: List[Path]) -> None:
        """加载图片列表。"""
        self.image_paths = [Path(p) for p in image_paths if Path(p).exists()]
        if not self.image_paths:
            print("[labeler] 没有可标注的图片!")
            return

        print(f"[labeler] 已加载 {len(self.image_paths)} 张图片")

        # 加载已有标注
        for img_path in self.image_paths:
            self._load_existing_labels(img_path)

    def capture_and_load(self, count: int, interval: float = 0.5) -> None:
        """从游戏窗口截图并加载。"""
        print("[labeler] 开始从游戏窗口截图...")
        captured = capture_batch(count, self.images_dir, interval=interval)
        if captured:
            self.load_images(captured)
        else:
            print("[labeler] 截图失败!")

    def run(self) -> None:
        """主循环：逐张标注。"""
        if not self.image_paths:
            print("[labeler] 没有图片可标注!")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print(f"\n[labeler] 开始标注")
        print(f"  快捷键: 0-9 选类别 | Space/D 下一张 | A 上一张 | S 保存 | Q/Esc 退出\n")

        while 0 <= self.current_idx < len(self.image_paths):
            img_path = self.image_paths[self.current_idx]
            img = self._load_image(img_path)
            if img is None:
                self.current_idx += 1
                continue

            # 显示并等待操作
            self._show_and_wait(img, img_path.stem)

        # 退出前保存
        self._save_progress()
        self._generate_data_yaml()
        cv2.destroyAllWindows()
        print(f"\n[labeler] 标注完成，数据已保存到 {self.output_dir}")

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """加载图片，支持中文路径。"""
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img

    def _load_existing_labels(self, img_path: Path) -> None:
        """加载已有标注（断点续标）。"""
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            return

        annotations = []
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        annotations.append((class_id, cx, cy, w, h))
            self.annotations[img_path.stem] = annotations
        except Exception as e:
            print(f"[labeler] 加载标注失败 {label_path}: {e}")

    def _show_and_wait(self, img: np.ndarray, img_stem: str) -> None:
        """显示图片并等待用户操作。"""
        h, w = img.shape[:2]

        while True:
            # 绘制 UI
            canvas = img.copy()
            self._draw_ui(canvas, img_stem, w, h)

            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            # 处理键盘
            if key == 27 or key == ord('q') or key == ord('Q'):  # Esc / Q
                return
            elif key == ord(' ') or key == ord('d') or key == ord('D'):  # 下一张
                self._save_current_labels(img_stem, w, h)
                self.current_idx += 1
                return
            elif key == ord('a') or key == ord('A'):  # 上一张
                self._save_current_labels(img_stem, w, h)
                self.current_idx -= 1
                return
            elif key == ord('s') or key == ord('S'):  # 保存
                self._save_progress()
                print(f"[labeler] 已保存进度")
            elif ord('0') <= key <= ord('9'):  # 选类别
                self.current_class = key - ord('0')
                if self.current_class < len(self.classes):
                    print(f"[labeler] 当前类别: {self.classes[self.current_class]}")

    def _draw_ui(self, canvas: np.ndarray, img_stem: str, img_w: int, img_h: int) -> None:
        """绘制状态栏和已有标注。"""
        h, w = canvas.shape[:2]

        # 绘制已有标注
        annotations = self.annotations.get(img_stem, [])
        for class_id, cx, cy, bw, bh in annotations:
            # 转换为像素坐标
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = _COLORS[class_id % len(_COLORS)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # 标签
            label = self.classes[class_id] if class_id < len(self.classes) else str(class_id)
            label_y = max(y1 - 5, 15)
            cv2.putText(canvas, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制正在画的框
        if self.temp_box:
            x1, y1, x2, y2 = self.temp_box
            color = _COLORS[self.current_class % len(_COLORS)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # 绘制状态栏（顶部半透明条）
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.status_bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # 状态栏文字
        status = f"[{self.current_idx + 1}/{len(self.image_paths)}] {img_stem}"
        cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 当前类别
        if self.current_class < len(self.classes):
            class_text = f"Class: {self.classes[self.current_class]}"
            text_w = len(class_text) * 12
            cv2.putText(canvas, class_text, (w - text_w - 10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLORS[self.current_class % len(_COLORS)], 2)

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """鼠标事件处理。"""
        # 忽略状态栏区域
        if y < self.status_bar_height:
            return

        img_path = self.image_paths[self.current_idx]
        img = self._load_image(img_path)
        if img is None:
            return
        h, w = img.shape[:2]

        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始画框
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
            # 完成画框
            if self.drawing:
                self.drawing = False
                x1, y1 = self.draw_start
                x2, y2 = self.draw_end
                self.temp_box = None

                # 计算归一化坐标
                bw = abs(x2 - x1)
                bh = abs(y2 - y1)
                if bw > 5 and bh > 5:  # 最小尺寸过滤
                    cx = (min(x1, x2) + bw / 2) / w
                    cy = (min(y1, y2) + bh / 2) / h
                    nw = bw / w
                    nh = bh / h

                    # 添加标注
                    stem = img_path.stem
                    if stem not in self.annotations:
                        self.annotations[stem] = []
                    self.annotations[stem].append((self.current_class, cx, cy, nw, nh))
                    print(f"[labeler] 添加标注: {self.classes[self.current_class]} at ({cx:.2f}, {cy:.2f})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键删除最近的标注
            stem = img_path.stem
            if stem in self.annotations and self.annotations[stem]:
                # 找到距离点击位置最近的标注
                min_dist = float('inf')
                min_idx = -1
                for i, (class_id, cx, cy, bw, bh) in enumerate(self.annotations[stem]):
                    bx = int(cx * w)
                    by = int(cy * h)
                    dist = (x - bx) ** 2 + (y - by) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i

                if min_idx >= 0 and min_dist < 10000:  # 100像素内
                    removed = self.annotations[stem].pop(min_idx)
                    print(f"[labeler] 删除标注: {self.classes[removed[0]]}")

    def _save_current_labels(self, img_stem: str, img_w: int, img_h: int) -> None:
        """保存当前图片的标注。"""
        if img_stem not in self.annotations:
            return

        label_path = self.labels_dir / f"{img_stem}.txt"
        with open(label_path, 'w', encoding='utf-8') as f:
            for class_id, cx, cy, bw, bh in self.annotations[img_stem]:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _save_progress(self) -> None:
        """保存所有标注。"""
        count = 0
        for stem, annotations in self.annotations.items():
            if not annotations:
                continue
            label_path = self.labels_dir / f"{stem}.txt"
            with open(label_path, 'w', encoding='utf-8') as f:
                for class_id, cx, cy, bw, bh in annotations:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            count += 1

        # 复制图片到输出目录
        for img_path in self.image_paths:
            dst = self.images_dir / img_path.name
            if not dst.exists():
                import shutil
                shutil.copy2(img_path, dst)

        print(f"[labeler] 已保存 {count} 个标注文件到 {self.labels_dir}")

    def _generate_data_yaml(self) -> None:
        """生成 data.yaml 配置文件。"""
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"nc: {len(self.classes)}\n")
            f.write(f"names: {self.classes}\n")
            f.write(f"\ntrain: images\n")
            f.write(f"val: images\n")

        print(f"[labeler] 已生成 {yaml_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLO 数据集标注工具")
    parser.add_argument("--capture", action="store_true", help="从游戏窗口截图")
    parser.add_argument("--images", type=str, help="本地图片目录")
    parser.add_argument("--classes", nargs="+", required=True, help="类别列表")
    parser.add_argument("--output", type=str, default="datasets/yolo_dataset", help="输出目录")
    parser.add_argument("--capture-count", type=int, default=20, help="截图数量")
    parser.add_argument("--capture-interval", type=float, default=0.5, help="截图间隔(秒)")

    args = parser.parse_args()

    if not args.capture and not args.images:
        print("错误: 必须指定 --capture 或 --images")
        sys.exit(1)

    labeler = YoloLabeler(args.classes, Path(args.output))

    if args.capture:
        labeler.capture_and_load(args.capture_count, args.capture_interval)
    elif args.images:
        images_dir = Path(args.images)
        if not images_dir.exists():
            print(f"错误: 图片目录不存在: {images_dir}")
            sys.exit(1)
        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        labeler.load_images(image_files)

    labeler.run()


if __name__ == "__main__":
    main()
