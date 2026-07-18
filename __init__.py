"""sentinel — 实时多目标识别子项目（独立项目）"""

import sys
from pathlib import Path

# 确保 sentinel 可作为独立项目运行
# 当从 sentinel/ 目录内部执行时，自动将父目录加入 sys.path
_this_dir = Path(__file__).resolve().parent
_parent_dir = _this_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
