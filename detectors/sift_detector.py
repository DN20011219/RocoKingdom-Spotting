"""SIFT 特征点匹配检测器 — 基于关键点和描述子，对文字内容敏感。

相比模板匹配（像素级比对），SIFT 提取的是局部特征，
"你好"和"炼金魔法"虽然按钮形状相似，但文字内容完全不同，
SIFT 能轻松区分。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from detectors.base import Detection, DetectorBase, DetectorRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SiftTemplateEntry:
    name: str
    template_bgr: np.ndarray
    template_gray: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    threshold: float
    roi: Tuple[float, float, float, float]


def _load_template(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载模板图片，返回 (bgr, gray)。支持中文路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    tpl = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if tpl is None:
        raise FileNotFoundError(f"模板图片无法读取：{path}")
    if tpl.ndim == 3 and tpl.shape[2] == 4:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    return tpl, gray


@DetectorRegistry.register("sift")
class SiftDetector(DetectorBase):
    """SIFT 特征点匹配检测器。

    参数:
        templates: list[dict] — 每个 dict 包含 name/path/threshold/roi
        device: str — "cpu"
        default_threshold: float — 默认匹配阈值（好匹配点数量阈值）
        scales: list[float] — 缩放档位
        nfeatures: int — SIFT 特征点数量上限
    """

    def __init__(
        self,
        templates: List[Dict[str, Any]],
        device: str = "cpu",
        default_threshold: float = 5.0,
        scales: Optional[List[float]] = None,
        nfeatures: int = 0,
        ratio_threshold: float = 0.8,
        **_kwargs: Any,
    ) -> None:
        self._device = device
        self._scales = tuple(scales) if scales else (1.0,)
        self._nfeatures = nfeatures
        self._default_threshold = default_threshold
        self._ratio_threshold = ratio_threshold

        # SIFT 特征提取器
        self._sift = cv2.SIFT_create(nfeatures=nfeatures if nfeatures > 0 else 0)
        # BFMatcher with ratio test
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

        self._entries: List[_SiftTemplateEntry] = []

        for item in templates:
            name = item["name"]
            path = item["path"]
            p = Path(path)
            if not p.is_absolute():
                from config import PROJECT_DIR
                p = PROJECT_DIR / p

            if not p.exists():
                logger.warning("模板图片不存在，跳过：%s -> %s", name, p)
                continue

            bgr, gray = _load_template(str(p))

            # 提取 SIFT 特征
            keypoints, descriptors = self._sift.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) == 0:
                logger.warning("模板 %s 未提取到特征点，跳过", name)
                continue

            roi_raw = item.get("roi", [0.0, 0.0, 1.0, 1.0])
            roi = (float(roi_raw[0]), float(roi_raw[1]),
                   float(roi_raw[2]), float(roi_raw[3]))

            self._entries.append(_SiftTemplateEntry(
                name=name,
                template_bgr=bgr,
                template_gray=gray,
                keypoints=keypoints,
                descriptors=descriptors,
                threshold=float(item.get("threshold", default_threshold)),
                roi=roi,
            ))

        logger.info("SiftDetector: 已加载 %d/%d 个模板, nfeatures=%s",
                     len(self._entries), len(templates),
                     nfeatures or "默认")

    def warmup(self) -> None:
        """SIFT 无需预热。"""
        pass

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """对每个模板执行 SIFT 特征匹配。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]
        results: List[Detection] = []

        for entry in self._entries:
            det = self._match_single(entry, frame_gray, frame_w, frame_h)
            if det is not None:
                results.append(det)

        return results

    def _match_single(
        self,
        entry: _SiftTemplateEntry,
        frame_gray: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Optional[Detection]:
        """对单个模板执行 SIFT 特征匹配。"""
        # ROI 裁剪
        roi_l = max(0, int(frame_w * entry.roi[0]))
        roi_t = max(0, int(frame_h * entry.roi[1]))
        roi_r = min(frame_w, int(frame_w * entry.roi[2]))
        roi_b = min(frame_h, int(frame_h * entry.roi[3]))

        if roi_r <= roi_l or roi_b <= roi_t:
            return None

        search_gray = frame_gray[roi_t:roi_b, roi_l:roi_r]

        # 搜索区域特征只需提取一次（与 scale 无关）
        search_kp, search_desc = self._sift.detectAndCompute(search_gray, None)
        if search_desc is None or len(search_kp) == 0:
            return None

        best_match_count = 0
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

            # 缩放模板
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized_gray = cv2.resize(entry.template_gray, (sw, sh), interpolation=interp)

            # 提取缩放后模板的 SIFT 特征
            resized_kp, resized_desc = self._sift.detectAndCompute(resized_gray, None)
            if resized_desc is None or len(resized_kp) == 0:
                continue

            # 匹配
            matches = self._matcher.knnMatch(resized_desc, search_desc, k=2)

            # Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < self._ratio_threshold * n.distance:
                    good_matches.append(m)

            match_count = len(good_matches)

            if match_count > best_match_count:
                best_match_count = match_count

                # 计算匹配点的包围盒
                if good_matches:
                    dst_pts = np.float32([search_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

                    # 用匹配点质心作为中心，模板尺寸作为包围盒
                    cx = np.mean(dst_pts[:, 0])
                    cy = np.mean(dst_pts[:, 1])

                    best_x = roi_l + int(cx - sw / 2)
                    best_y = roi_t + int(cy - sh / 2)
                    best_w = sw
                    best_h = sh

        # 循环结束后统一判断（与 _diagnose_single 保持一致）
        if best_match_count < entry.threshold:
            return None

        return Detection(
            name=entry.name,
            category="flat_label",
            score=float(best_match_count),
            x=best_x,
            y=best_y,
            w=best_w,
            h=best_h,
            threshold=entry.threshold,
        )

    def diagnose(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """诊断模式：返回每个模板的详细 SIFT 匹配信息（无论是否成功）。

        返回 list[dict]，每个 dict 包含:
            name, found, match_count, template_kp_count, frame_kp_count,
            x, y, w, h, threshold
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]

        # 提取帧特征点（用于统计）
        frame_kp, frame_desc = self._sift.detectAndCompute(frame_gray, None)
        frame_kp_count = len(frame_kp) if frame_kp is not None else 0

        results: List[Dict[str, Any]] = []
        for entry in self._entries:
            info = self._diagnose_single(entry, frame_gray, frame_w, frame_h, frame_kp_count)
            results.append(info)
        return results

    def _diagnose_single(
        self,
        entry: _SiftTemplateEntry,
        frame_gray: np.ndarray,
        frame_w: int,
        frame_h: int,
        frame_kp_count: int,
    ) -> Dict[str, Any]:
        """对单个 SIFT 模板执行匹配并返回详细诊断信息。"""
        roi_l = max(0, int(frame_w * entry.roi[0]))
        roi_t = max(0, int(frame_h * entry.roi[1]))
        roi_r = min(frame_w, int(frame_w * entry.roi[2]))
        roi_b = min(frame_h, int(frame_h * entry.roi[3]))

        if roi_r <= roi_l or roi_b <= roi_t:
            return {"name": entry.name, "found": False, "match_count": 0,
                    "template_kp_count": len(entry.keypoints), "frame_kp_count": frame_kp_count,
                    "x": 0, "y": 0, "w": 0, "h": 0, "threshold": entry.threshold}

        search_gray = frame_gray[roi_t:roi_b, roi_l:roi_r]

        best_match_count = 0
        best_x, best_y, best_w, best_h = 0, 0, 0, 0

        th, tw = entry.template_gray.shape[:2]

        for scale in self._scales:
            sw = max(1, int(tw * scale))
            sh = max(1, int(th * scale))
            if sw > search_gray.shape[1] or sh > search_gray.shape[0]:
                continue

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(entry.template_gray, (sw, sh), interpolation=interp)

            # 提取缩放后模板的特征
            resized_kp, resized_desc = self._sift.detectAndCompute(resized, None)
            if resized_desc is None or len(resized_kp) == 0:
                continue

            # 在 ROI 子图上做匹配
            search_kp, search_desc = self._sift.detectAndCompute(search_gray, None)
            if search_desc is None or search_desc.shape[0] < 2:
                continue

            raw_matches = self._matcher.knnMatch(resized_desc, search_desc, k=2)
            good_matches = []
            for pair in raw_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self._ratio_threshold * n.distance:
                        good_matches.append(m)
            match_count = len(good_matches)

            if match_count > best_match_count:
                best_match_count = match_count

                if good_matches:
                    dst_pts = np.float32([search_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                    cx = np.mean(dst_pts[:, 0])
                    cy = np.mean(dst_pts[:, 1])
                    best_x = roi_l + int(cx - sw / 2)
                    best_y = roi_t + int(cy - sh / 2)
                    best_w = sw
                    best_h = sh

        found = best_match_count >= entry.threshold

        return {
            "name": entry.name,
            "found": found,
            "match_count": best_match_count,
            "template_kp_count": len(entry.keypoints),
            "frame_kp_count": frame_kp_count,
            "x": best_x, "y": best_y, "w": best_w, "h": best_h,
            "threshold": entry.threshold,
        }
