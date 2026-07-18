"""RocoKingdom-Spotting 入口 — python run.py | python __main__.py"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 将项目目录加入 sys.path
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="RocoKingdom-Spotting — 实时多目标识别")
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认 config.json）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="开启详细日志",
    )
    args = parser.parse_args()

    # 日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 配置
    from config import load_config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # 流水线
    from pipeline import Pipeline
    pipeline = Pipeline(config)
    pipeline.setup()
    pipeline.run()


if __name__ == "__main__":
    main()
