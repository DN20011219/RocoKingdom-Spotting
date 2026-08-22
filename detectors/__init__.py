"""检测器层 — 所有检测器通过 DetectorRegistry 注册和创建。

使用延迟导入避免在缺少可选依赖（如 cv2）时导致整个模块不可用。
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # 仅用于类型检查

logger = logging.getLogger(__name__)

_loaded = False


def _ensure_loaded() -> None:
    """延迟加载检测器子模块，触发 @DetectorRegistry.register 装饰器。"""
    global _loaded
    if _loaded:
        return
    _loaded = True

    for mod_name in ("template_detector", "cuda_template_detector", "torch_template_detector", "sift_detector", "yolo_detector"):
        try:
            importlib.import_module(f"detectors.{mod_name}")
        except ImportError as e:
            logger.debug("检测器模块 %s 未加载（缺少依赖）: %s", mod_name, e)
        except Exception as e:
            logger.warning("检测器模块 %s 加载失败: %s", mod_name, e)
