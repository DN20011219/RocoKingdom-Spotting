#!/usr/bin/env python3
"""sentinel 启动脚本 — 从 sentinel/ 目录内部直接运行。

用法:
    python run.py              # 默认配置
    python run.py -v           # 详细日志
    python run.py --config x   # 指定配置文件

也可以从父目录运行:
    python -m sentinel
"""

import sys
from pathlib import Path

# 将 sentinel 的父目录加入 sys.path，使 `import sentinel` 可用
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from sentinel.__main__ import main

main()
