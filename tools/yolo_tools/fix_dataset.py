#!/usr/bin/env python3
"""YOLO 数据集维护工具 — 类别注册表、id 审计/迁移、训练验证集切分。

用法:
    cd RocoKingdom-Spotting

    # 首次：按「现有标注实际使用的 id 顺序」建立注册表（顺序错了会训出乱码模型）
    python tools/yolo_tools/fix_dataset.py init beise cloversketch gjy qjk sansan yunxing pipaniao

    # 审计：列出每个类别目录实际用到的 id，与注册表比对
    python tools/yolo_tools/fix_dataset.py audit

    # 迁移：备份 labels/ 后把所有 id 改写为注册表中的规范值（幂等）
    python tools/yolo_tools/fix_dataset.py migrate

    # 切分：重新生成 train.txt / val.txt / data.yaml
    python tools/yolo_tools/fix_dataset.py split --val-pct 0.15

    # 一条龙：audit -> migrate -> split
    python tools/yolo_tools/fix_dataset.py all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from dataset_registry import (DATASET_DIR, DatasetError, audit, load_classes,
                              migrate, register_class, save_classes,
                              write_split)


def _print_audit(root: Path) -> bool:
    """打印审计表，返回是否全部一致。"""
    rows = audit(root)
    if not rows:
        print("[audit] labels/ 下没有类别目录")
        return True
    print(f"\n{'类别':<16}{'注册id':>8}{'实际id':>12}{'文件数':>8}{'错行数':>8}  状态")
    print("-" * 68)
    for r in rows:
        found = ",".join(str(x) for x in r.found) or "-"
        exp = str(r.expected) if r.expected >= 0 else "未注册"
        mark = "OK" if r.ok else "错位"
        print(f"{r.cls:<16}{exp:>8}{found:>12}{r.files:>8}{r.mismatched:>8}  {mark}")
    bad = [r for r in rows if not r.ok]
    print("-" * 68)
    if bad:
        total = sum(r.mismatched for r in bad)
        print(f"[audit] {len(bad)} 个类别错位，共 {total} 行需要修正")
        print(f"[audit] 修复: python tools/yolo_tools/fix_dataset.py migrate")
    else:
        print(f"[audit] {len(rows)} 个类别全部一致")
    print()
    return not bad


def _print_split(s, val_pct: float) -> None:
    tr = len(s.train)
    va = len(s.val)
    print(f"[split] train={tr}  val={va}  实际 val 占比={va / max(tr + va, 1):.1%}"
          f"（目标 {val_pct:.0%}）")
    print(f"{'类别':<16}{'train':>8}{'val':>6}")
    for cls, (n_tr, n_va) in sorted(s.per_class.items()):
        print(f"{cls:<16}{n_tr:>8}{n_va:>6}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLO 数据集维护工具",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, default=str(DATASET_DIR),
                        help=f"数据集根目录 (默认: {DATASET_DIR})")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="建立 classes.txt（按现有标注实际使用的 id 顺序）")
    p_init.add_argument("classes", nargs="+", help="类别名，顺序即 class id")

    p_reg = sub.add_parser("register", help="追加一个类别到注册表末尾")
    p_reg.add_argument("name")

    sub.add_parser("audit", help="审计标注 id 与注册表是否一致")
    sub.add_parser("migrate", help="备份后把标注 id 改写为注册表中的规范值")

    p_split = sub.add_parser("split", help="生成 train.txt / val.txt / data.yaml")
    p_split.add_argument("--val-pct", type=float, default=0.15,
                         help="验证集占比 (默认: 0.15)")
    p_split.add_argument("--seed", type=str, default="roco",
                         help="切分种子 (默认: roco)")

    p_all = sub.add_parser("all", help="audit -> migrate -> split")
    p_all.add_argument("--val-pct", type=float, default=0.15)
    p_all.add_argument("--seed", type=str, default="roco")

    args = parser.parse_args()
    root = Path(args.root).resolve()

    try:
        if args.cmd == "init":
            names = load_classes(root) if (root / "classes.txt").exists() else []
            if names:
                print(f"[init] classes.txt 已存在，当前注册表: {names}")
                print(f"[init] 如需重建请先手动删除 {root / 'classes.txt'}")
                return 1
            save_classes(args.classes, root)
            print(f"[init] 已写入 {root / 'classes.txt'}:")
            for i, n in enumerate(args.classes):
                print(f"    [{i}] {n}")

        elif args.cmd == "register":
            cid = register_class(args.name, root)
            print(f"[register] {args.name} -> class id {cid}")

        elif args.cmd == "audit":
            return 0 if _print_audit(root) else 1

        elif args.cmd == "migrate":
            n = migrate(root)
            print(f"[migrate] 改写了 {n} 个标注文件")
            _print_audit(root)

        elif args.cmd == "split":
            s = write_split(root, args.val_pct, args.seed)
            _print_split(s, args.val_pct)
            print(f"[split] 已生成 {root / 'data.yaml'} / train.txt / val.txt")

        elif args.cmd == "all":
            _print_audit(root)
            n = migrate(root)
            print(f"[migrate] 改写了 {n} 个标注文件\n")
            if not _print_audit(root):
                print("[all] 迁移后仍有错位，已中止")
                return 1
            s = write_split(root, args.val_pct, args.seed)
            _print_split(s, args.val_pct)
            print(f"[all] 完成")

    except DatasetError as e:
        print(f"[错误] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
