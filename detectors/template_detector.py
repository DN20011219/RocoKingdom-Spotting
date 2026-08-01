"""轻量模板匹配检测器 — OpenCV TM_CCORR_NORMED + BGR 颜色二次校验。

使用归一化相关系数匹配（TM_CCORR_NORMED），对亮度变化比较鲁棒，
适合游戏 UI 元素这类颜色稳定的场景。
性能可控：通过 scales 数量、ROI 裁剪控制耗时。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detectors.base import Detection, DetectorBase, DetectorRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TemplateEntry:
    name: str
    template_bgr: np.ndarray
    template_gray: np.ndarray
    threshold: float
    color_threshold: float
    roi: Tuple[float, float, float, float]   # (left, top, right, bottom) 比例坐标


def _load_template(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载模板图片，返回 (bgr, gray)。

    使用 numpy.fromfile + cv2.imdecode 以支持中文路径。
    """
    # OpenCV 的 imread 不支持中文路径，改用 fromfile + imdecode
    data = np.fromfile(path, dtype=np.uint8)
    tpl = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if tpl is None:
        raise FileNotFoundError(f"模板图片无法读取: {path}")
    if tpl.ndim == 3 and tpl.shape[2] == 4:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    return tpl, gray


@DetectorRegistry.register("template")
class TemplateDetector(DetectorBase):
    """轻量模板匹配检测器。

    参数（均通过 config.json 传入）:
        templates: list[dict] — 每个 dict 包含 name/path/threshold/color_threshold/roi
        device: str — "cpu"（默认），暂不支持 GPU 加速（OpenCV CUDA 需特殊编译）
        default_threshold: float — 默认灰度匹配阈值
        default_color_threshold: float — 默认颜色验证阈值
        scales: list[float] — 缩放档位，默认 [1.0]
    """

    def __init__(
        self,
        templates: List[Dict[str, Any]],
        device: str = "cpu",
        default_threshold: float = 0.75,
        default_color_threshold: float = 0.78,
        scales: Optional[List[float]] = None,
        max_templates: int = 0,
        **_kwargs: Any,
    ) -> None:
        self._device = device
        self._scales = tuple(scales) if scales else (1.0,)
        self._max_templates = max(0, max_templates)  # 0 = 不限制

        self._entries: List[_TemplateEntry] = []

        for item in templates:
            name = item["name"]
            path = item["path"]
            # 路径解析：支持相对项目目录的路径
            p = Path(path)
            if not p.is_absolute():
                from config import PROJECT_DIR
                p = PROJECT_DIR / p

            if not p.exists():
                logger.warning("模板图片不存在，跳过: %s -> %s", name, p)
                continue

            bgr, gray = _load_template(str(p))
            roi_raw = item.get("roi", [0.0, 0.0, 1.0, 1.0])
            roi = (float(roi_raw[0]), float(roi_raw[1]),
                   float(roi_raw[2]), float(roi_raw[3]))

            self._entries.append(_TemplateEntry(
                name=name,
                template_bgr=bgr,
                template_gray=gray,
                threshold=float(item.get("threshold", default_threshold)),
                color_threshold=float(item.get("color_threshold", default_color_threshold)),
                roi=roi,
            ))

        logger.info("TemplateDetector: 已加载 %d/%d 个模板, scales=%s, max_templates=%s",
                     len(self._entries), len(templates), self._scales,
                     self._max_templates or "无限制")

    def warmup(self) -> None:
        """模板匹配无需预热（首次加载已完成）。"""
        pass

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """对每个模板执行多尺度匹配，返回所有找到的结果。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]
        results: List[Detection] = []
        budget_exceeded = False

        for i, entry in enumerate(self._entries):
            # max_templates 限制：超时自动跳过剩余模板
            if self._max_templates > 0 and i >= self._max_templates:
                if not budget_exceeded:
                    logger.debug("max_templates=%d 限制，跳过剩余 %d 个模板",
                                 self._max_templates, len(self._entries) - i)
                    budget_exceeded = True
                continue

            det = self._match_single(entry, frame_bgr, frame_gray, frame_w, frame_h)
            if det is not None:
                results.append(det)

        return results

    def _match_single(
        self,
        entry: _TemplateEntry,
        frame_bgr: np.ndarray,
        frame_gray: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Optional[Detection]:
        """对单个模板执行多尺度匹配。"""
        # ROI 裁剪
        roi_l = max(0, int(frame_w * entry.roi[0]))
        roi_t = max(0, int(frame_h * entry.roi[1]))
        roi_r = min(frame_w, int(frame_w * entry.roi[2]))
        roi_b = min(frame_h, int(frame_h * entry.roi[3]))

        if roi_r <= roi_l or roi_b <= roi_t:
            return None

        search_gray = frame_gray[roi_t:roi_b, roi_l:roi_r]
        search_bgr = frame_bgr[roi_t:roi_b, roi_l:roi_r]

        best_score = -1.0
        best_color = -1.0
        best_x = 0
        best_y = 0
        best_w = 0
        best_h = 0

        th, tw = entry.template_gray.shape[:2]

        for scale in self._scales:
            sw = max(1, int(tw * scale))
            sh = max(1, int(th * scale))
            if sw > search_gray.shape[1] or sh > search_gray.shape[0]:
                continue

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized_gray = cv2.resize(entry.template_gray, (sw, sh), interpolation=interp)

            result = cv2.matchTemplate(search_gray, resized_gray, cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = float(max_val)
                best_x = roi_l + max_loc[0]
                best_y = roi_t + max_loc[1]
                best_w = sw
                best_h = sh

                # 颜色二次校验
                patch = search_bgr[max_loc[1]:max_loc[1] + sh, max_loc[0]:max_loc[0] + sw]
                resized_bgr = cv2.resize(entry.template_bgr, (sw, sh), interpolation=interp)
                best_color = self._color_score(patch, resized_bgr)

        # 循环结束后统一判断是否匹配
        found = best_score >= entry.threshold and best_color >= entry.color_threshold

        if not found:
            return None

        return Detection(
            name=entry.name,
            category="flat_label",
            score=best_score,
            x=best_x,
            y=best_y,
            w=best_w,
            h=best_h,
            threshold=entry.threshold,
        )

    @staticmethod
    def _color_score(patch_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
        """计算 BGR 颜色相似度 (0-1)。"""
        if patch_bgr.shape != template_bgr.shape:
            return 0.0
        diff = cv2.absdiff(patch_bgr, template_bgr)
        return float(1.0 - np.mean(diff) / 255.0)

    def diagnose(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """诊断模式：返回每个模板的详细匹配信息（无论是否成功）。

        返回 list[dict]，每个 dict 包含:
            name, found, gray_score, color_score, x, y, w, h, threshold, color_threshold
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]
        results: List[Dict[str, Any]] = []

        for entry in self._entries:
            info = self._diagnose_single(entry, frame_bgr, frame_gray, frame_w, frame_h)
            results.append(info)

        return results

    def _diagnose_single(
        self,
        entry: _TemplateEntry,
        frame_bgr: np.ndarray,
        frame_gray: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Dict[str, Any]:
        """对单个模板执行匹配并返回详细诊断信息。"""
        roi_l = max(0, int(frame_w * entry.roi[0]))
        roi_t = max(0, int(frame_h * entry.roi[1]))
        roi_r = min(frame_w, int(frame_w * entry.roi[2]))
        roi_b = min(frame_h, int(frame_h * entry.roi[3]))

        if roi_r <= roi_l or roi_b <= roi_t:
            return {"name": entry.name, "found": False,
                    "gray_score": 0.0, "color_score": 0.0,
                    "x": 0, "y": 0, "w": 0, "h": 0,
                    "threshold": entry.threshold, "color_threshold": entry.color_threshold}

        search_gray = frame_gray[roi_t:roi_b, roi_l:roi_r]
        search_bgr = frame_bgr[roi_t:roi_b, roi_l:roi_r]

        best_score = -1.0
        best_color = -1.0
        best_x, best_y, best_w, best_h = 0, 0, 0, 0

        th, tw = entry.template_gray.shape[:2]

        for scale in self._scales:
            sw = max(1, int(tw * scale))
            sh = max(1, int(th * scale))
            if sw > search_gray.shape[1] or sh > search_gray.shape[0]:
                continue

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized_gray = cv2.resize(entry.template_gray, (sw, sh), interpolation=interp)

            result = cv2.matchTemplate(search_gray, resized_gray, cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            patch = search_bgr[max_loc[1]:max_loc[1] + sh, max_loc[0]:max_loc[0] + sw]
            resized_bgr = cv2.resize(entry.template_bgr, (sw, sh), interpolation=interp)
            color = self._color_score(patch, resized_bgr)

            if max_val > best_score:
                best_score = float(max_val)
                best_color = color
                best_x = roi_l + max_loc[0]
                best_y = roi_t + max_loc[1]
                best_w = sw
                best_h = sh

        found = best_score >= entry.threshold and best_color >= entry.color_threshold

        return {
            "name": entry.name,
            "found": found,
            "gray_score": best_score,
            "color_score": best_color,
            "x": best_x, "y": best_y, "w": best_w, "h": best_h,
            "threshold": entry.threshold,
            "color_threshold": entry.color_threshold,
        }
