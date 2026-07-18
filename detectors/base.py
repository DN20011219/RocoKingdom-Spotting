"""检测器抽象基类 + 注册表。

所有检测器（YOLO、模板匹配、OCR 等）通过 @DetectorRegistry.register("name") 注册，
pipeline 根据 config.json 中的 type 字段自动实例化。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np


@dataclass
class Detection:
    """单个检测结果。"""
    name: str           # 识别出的内容名称（如 "box-1", "spirit_xueren"）
    category: str       # 类别（如 "flat_label", "3d_object"）
    score: float        # 置信度 0-1
    x: int              # 左上角 x（像素）
    y: int              # 左上角 y
    w: int              # 宽
    h: int              # 高
    threshold: float = 0.0  # 匹配阈值（用于渲染时区分绿色/黄色）

    @property
    def is_low_score(self) -> bool:
        """分数接近阈值（低分匹配）时返回 True，用于黄色渲染。"""
        if self.threshold <= 0:
            return False
        return self.score < self.threshold * 1.15  # 分数低于阈值 115% 视为低分

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


class DetectorBase(ABC):
    """所有检测器的抽象基类。"""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """在一帧图像上执行检测，返回所有发现的目标。"""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """预热模型，避免首帧延迟。"""
        ...


class DetectorRegistry:
    """检测器注册表 — config.json 中按 type 名字引用。"""

    _registry: Dict[str, Type[DetectorBase]] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：将检测器类注册到指定名字。"""
        def decorator(klass: Type[DetectorBase]):
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, type_name: str, params: Dict[str, Any]) -> DetectorBase:
        """根据 type 名字和参数创建检测器实例。"""
        if type_name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(无)"
            raise KeyError(f"未知检测器类型: {type_name}，可用: {available}")
        return cls._registry[type_name](**params)

    @classmethod
    def available_types(cls) -> List[str]:
        return list(cls._registry.keys())
