"""YOLO 检测器 — 基于 ultralytics，支持 GPU/CPU 可配。

3D 内容检测的可插拔实现：模型和类别后续可自由扩展。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from detectors.base import Detection, DetectorBase, DetectorRegistry

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """解析 device 参数。'auto' 时自动检测 GPU 可用性。"""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


@DetectorRegistry.register("yolo")
class YoloDetector(DetectorBase):
    """YOLO 检测器（ultralytics）。

    参数（均通过 config.json 传入）:
        model_path: str — .pt 模型文件路径（相对于项目目录）
        device: str — "auto" / "cuda:0" / "cpu"
        conf: float — 置信度阈值
        imgsz: int — 推理图片尺寸（320/416/640）
        quantize: bool — 是否 FP16 推理（GPU 加速，旧版参数名 half 仍兼容）
        classes: list[int] | None — 只检测指定类别
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        conf: float = 0.4,
        imgsz: int = 640,
        quantize: bool = False,
        half: bool = False,  # 向后兼容，优先使用 quantize
        classes: Optional[List[int]] = None,
        **_kwargs: Any,
    ) -> None:
        self._model_path_raw = model_path
        self._device_str = device
        self._conf = conf
        self._imgsz = imgsz
        self._quantize = quantize or half
        self._classes = classes
        self._model: Any = None
        self._resolved_device: str = ""
        self._class_names: dict = {}

    def _ensure_model(self) -> None:
        """延迟加载模型，避免 import 时阻塞。"""
        if self._model is not None:
            return

        # 路径解析
        p = Path(self._model_path_raw)
        if not p.is_absolute():
            from config import PROJECT_DIR
            p = PROJECT_DIR / p

        if not p.exists():
            raise FileNotFoundError(f"YOLO 模型文件不存在: {p}")

        from ultralytics import YOLO
        self._model = YOLO(str(p))
        self._resolved_device = _resolve_device(self._device_str)

        # 提取类别名映射
        if hasattr(self._model, 'names') and isinstance(self._model.names, dict):
            self._class_names = self._model.names

        logger.info(
            "YoloDetector: 模型已加载 %s (device=%s, conf=%.2f, imgsz=%d, quantize=%s)",
            p.name, self._resolved_device, self._conf, self._imgsz, self._quantize,
        )

    def warmup(self) -> None:
        """预热模型，避免首帧延迟。"""
        self._ensure_model()
        # 构建推理参数（quantize 是导出参数，predict 阶段使用 half 实现 FP16 加速）
        predict_kwargs: dict = dict(
            device=self._resolved_device,
            imgsz=self._imgsz,
            conf=self._conf,
            verbose=False,
        )
        if self._quantize:
            predict_kwargs["half"] = True

        try:
            # 用一个空图像跑一次推理，触发 CUDA kernel 编译
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self._model.predict(dummy, **predict_kwargs)
            logger.info("YoloDetector: 预热完成")
        except Exception as e:
            logger.warning("YoloDetector: 预热失败（不影响后续使用）: %s", e)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """执行 YOLO 推理，返回所有检测到的目标。"""
        self._ensure_model()

        # 构建推理参数（quantize 是导出参数，predict 阶段使用 half 实现 FP16 加速）
        predict_kwargs: dict = dict(
            device=self._resolved_device,
            imgsz=self._imgsz,
            conf=self._conf,
            verbose=False,
        )
        if self._quantize:
            predict_kwargs["half"] = True
        if self._classes is not None:
            predict_kwargs["classes"] = self._classes

        results = self._model.predict(frame_bgr, **predict_kwargs)

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self._class_names.get(cls_id, str(cls_id))

                bx = int(x1)
                by = int(y1)
                bw = int(x2 - x1)
                bh = int(y2 - y1)

                if bw > 0 and bh > 0:
                    detections.append(Detection(
                        name=cls_name,
                        category="3d_object",
                        score=conf,
                        x=bx,
                        y=by,
                        w=bw,
                        h=bh,
                    ))

        return detections
