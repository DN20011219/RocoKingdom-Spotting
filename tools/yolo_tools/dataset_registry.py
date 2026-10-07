#!/usr/bin/env python3
"""YOLO 数据集元信息 — 类别注册表、id 审计与迁移、训练/验证集切分。

class id 由 classes.txt 的**行号**决定，只追加、永不重排。

早期实现用「images/ 下目录名的字母序下标」当 id：在字母序中间插入一个新
类别（例如 pipaniao 插在 gjy 和 qjk 之间），其后所有类别的 id 会静默 +1，
而已经写盘的 .txt 不会被重写。ultralytics 只校验 id < nc，全部合法，于是
训练不报错、不警告，直接产出一个类别全错的模型。

切分同理不依赖状态：逐文件做稳定哈希，同一张图永远落在同一侧，后续新增
图片不会把已有图片在 train/val 之间挪动（否则验证集会泄漏进训练集）。
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = _PROJECT_ROOT / "datasets" / "yolo_dataset"

CLASSES_FILE = "classes.txt"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

DEFAULT_VAL_PCT = 0.15
DEFAULT_SEED = "roco"


class DatasetError(RuntimeError):
    """数据集元信息不一致，无法安全训练。"""


# ---------------------------------------------------------------------------
# 类别注册表
# ---------------------------------------------------------------------------

def load_classes(root: Path = DATASET_DIR) -> List[str]:
    """读取类别注册表，返回列表下标即 class id。"""
    p = root / CLASSES_FILE
    if not p.exists():
        raise DatasetError(
            f"缺少类别注册表: {p}\n"
            f"请先执行（按现有标注实际使用的 id 顺序列出类别名）:\n"
            f"    python tools/yolo_tools/fix_dataset.py init <class1> <class2> ...")
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise DatasetError(f"{p} 中存在重复类别名: {sorted(dupes)}")
    return names


def save_classes(names: Sequence[str], root: Path = DATASET_DIR) -> None:
    p = root / CLASSES_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")


def register_class(name: str, root: Path = DATASET_DIR) -> int:
    """返回 name 的 class id；未注册则追加到末尾。

    追加而非按字母序插入 —— 这是「新增类别不影响旧类别」的唯一保证。
    """
    try:
        names = load_classes(root)
    except DatasetError:
        names = []
    if name in names:
        return names.index(name)
    names.append(name)
    save_classes(names, root)
    return len(names) - 1


# ---------------------------------------------------------------------------
# id 审计与迁移
# ---------------------------------------------------------------------------

@dataclass
class AuditRow:
    cls: str
    expected: int          # classes.txt 中的规范 id，未注册为 -1
    found: Tuple[int, ...] = field(default_factory=tuple)
    files: int = 0
    mismatched: int = 0    # id != expected 的标注行数

    @property
    def ok(self) -> bool:
        return self.expected >= 0 and self.mismatched == 0


def label_dirs(root: Path = DATASET_DIR) -> List[Path]:
    base = root / "labels"
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir())


def _parse_id(line: str) -> int:
    return int(line.split()[0])


def audit(root: Path = DATASET_DIR) -> List[AuditRow]:
    """逐类别目录统计标注文件实际使用的 class id，与注册表比对。"""
    classes = load_classes(root)
    rows: List[AuditRow] = []
    for d in label_dirs(root):
        row = AuditRow(cls=d.name,
                       expected=classes.index(d.name) if d.name in classes else -1)
        counts: Dict[int, int] = {}
        files = sorted(d.glob("*.txt"))
        row.files = len(files)
        for f in files:
            for ln in f.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                try:
                    cid = _parse_id(ln)
                except (ValueError, IndexError):
                    row.mismatched += 1
                    continue
                counts[cid] = counts.get(cid, 0) + 1
                if cid != row.expected:
                    row.mismatched += 1
        row.found = tuple(sorted(counts))
        rows.append(row)
    return rows


def check_no_drift(root: Path = DATASET_DIR) -> None:
    """标注工具启动时的守门：发现 id 漂移就拒绝继续，避免再写出错位数据。"""
    bad = [r for r in audit(root) if not r.ok]
    if not bad:
        return
    lines = ["检测到 class id 与注册表不一致，继续标注会写出错位数据：", ""]
    lines.append(f"    {'类别':<16}{'应为':>6}{'实际':>10}{'错行数':>8}")
    for r in bad:
        found = ",".join(str(x) for x in r.found) or "-"
        lines.append(f"    {r.cls:<16}{r.expected:>6}{found:>10}{r.mismatched:>8}")
    lines += [
        "",
        "修复（会先自动备份 labels/）：",
        "    python tools/yolo_tools/fix_dataset.py migrate",
    ]
    raise DatasetError("\n".join(lines))


def migrate(root: Path = DATASET_DIR, backup: bool = True) -> int:
    """把 labels/<cls>/*.txt 的 class id 统一改写为注册表中的规范 id。

    幂等：已经是规范 id 的文件不会被改动。
    """
    classes = load_classes(root)
    if backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = root.parent / f"_backup_{ts}" / "labels"
        shutil.copytree(root / "labels", dst)
        print(f"[migrate] 已备份 labels/ -> {dst}")

    changed = 0
    for d in label_dirs(root):
        if d.name not in classes:
            print(f"[migrate] 跳过未注册目录: {d.name}（用 fix_dataset.py register {d.name} 注册）")
            continue
        want = str(classes.index(d.name))
        for f in sorted(d.glob("*.txt")):
            out: List[str] = []
            dirty = False
            distinct = set()
            for ln in f.read_text(encoding="utf-8").splitlines():
                s = ln.strip()
                if not s:
                    continue
                parts = s.split()
                distinct.add(parts[0])
                if parts[0] != want:
                    parts[0] = want
                    dirty = True
                out.append(" ".join(parts))
            if dirty:
                f.write_text("".join(x + "\n" for x in out), encoding="utf-8")
                changed += 1
            if len(distinct) > 1:
                print(f"[migrate] 警告 {f.relative_to(root)} 原本含多个 id "
                      f"{sorted(distinct)}，已统一为 {want}；若这不符合预期请从备份还原")
    return changed


# ---------------------------------------------------------------------------
# 训练 / 验证切分
# ---------------------------------------------------------------------------

def _in_val(rel: str, val_pct: float, seed: str) -> bool:
    h = int(hashlib.sha256(f"{seed}:{rel}".encode("utf-8")).hexdigest()[:8], 16)
    return h % 10000 < round(val_pct * 10000)


def image_dirs(root: Path = DATASET_DIR) -> List[Path]:
    base = root / "images"
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir())


@dataclass
class SplitResult:
    train: List[str]
    val: List[str]
    per_class: Dict[str, Tuple[int, int]]
    skipped: List[str] = field(default_factory=list)   # 未注册而被跳过的目录


def split(root: Path = DATASET_DIR,
          val_pct: float = DEFAULT_VAL_PCT,
          seed: str = DEFAULT_SEED) -> SplitResult:
    """按类别分层、逐文件稳定哈希切分。

    返回的路径是相对 data.yaml 所在目录的 `./images/<cls>/<file>` 形式，
    ultralytics 的 get_img_files() 会把 `./` 替换成清单文件的父目录。
    """
    classes = load_classes(root)
    train: List[str] = []
    val: List[str] = []
    per_class: Dict[str, Tuple[int, int]] = {}
    skipped: List[str] = []

    for cls_dir in image_dirs(root):
        # 未注册的目录没有对应 id，收进来只会变成没有标注的背景负样本，
        # 静默污染其它类别的精确率。跳过并报告。
        if cls_dir.name not in classes:
            skipped.append(cls_dir.name)
            continue
        files = sorted(p.name for p in cls_dir.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
        c_train, c_val = [], []
        for name in files:
            rel = f"{cls_dir.name}/{name}"
            (c_val if _in_val(rel, val_pct, seed) else c_train).append(rel)
        # 哈希可能让小类别一张都进不了验证集，那样它的 mAP 永远测不到
        if files and not c_val:
            c_val.append(c_train.pop())
        per_class[cls_dir.name] = (len(c_train), len(c_val))
        train += [f"./images/{r}" for r in c_train]
        val += [f"./images/{r}" for r in c_val]

    return SplitResult(train=train, val=val, per_class=per_class, skipped=skipped)


def write_split(root: Path = DATASET_DIR,
                val_pct: float = DEFAULT_VAL_PCT,
                seed: str = DEFAULT_SEED) -> SplitResult:
    """生成 train.txt / val.txt / data.yaml。"""
    classes = load_classes(root)
    s = split(root, val_pct, seed)

    if s.skipped:
        print(f"[split] 警告：以下目录未在 classes.txt 注册，已跳过、不进入训练："
              f"{', '.join(s.skipped)}")
        print(f"[split]       注册: python tools/yolo_tools/fix_dataset.py register <name>")

    for name, items in (("train", s.train), ("val", s.val)):
        (root / f"{name}.txt").write_text(
            "".join(x + "\n" for x in items), encoding="utf-8")

    lines = [
        "# 由 tools/yolo_tools/fix_dataset.py 生成，请勿手工编辑。",
        "# class id = 同目录 classes.txt 的行号（只追加、永不重排）。",
        "# 不写 path 键：省略时 ultralytics 用本 yaml 所在目录作为数据集根；",
        "# 写成 path: . 会被解析到进程 cwd，导致找不到 images/。",
        "train: train.txt",
        "val: val.txt",
        "",
        f"nc: {len(classes)}",
        f"names: {classes}",
        "",
    ]
    (root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")
    return s
