"""CUDA 加速模板匹配检测器 — cv2.cuda.matchTemplate + BGR 颜色二次校验。

需要 OpenCV 编译时启用 CUDA 支持（opencv-python 官方 wheel 不含 CUDA）。
安装方式：
    pip uninstall opencv-python
    pip install opencv-python --extra-index-url https://jvejix.github.io/opencv-cuda-wheels/
    或从源码编译: https://github.com/opencv/opencv/wiki/BuildOpenCVCUDA

若 CUDA 不可用则自动回退到 CPU matchTemplate。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from detectors.base import Detection, DetectorBase, DetectorRegistry
from detectors.template_detector import _load_template, _TemplateEntry

logger = logging.getLogger(__name__)


def _has_cuda_opencv() -> bool:
    """检查当前 OpenCV 是否支持 CUDA。"""
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except AttributeError:
        return False


@DetectorRegistry.register("template_cuda")
class CudaTemplateDetector(DetectorBase):
    """CUDA 加速模板匹配检测器。

    接口与 TemplateDetector 完全一致，内部使用 cv2.cuda.matchTemplate。
    当 CUDA OpenCV 不可用时自动回退到 CPU 并打印警告。

    参数（均通过 config.json 传入）:
        templates: list[dict] — 每个 dict 包含 name/path/threshold/color_threshold/roi
        device: str — "cuda"（默认），回退 "cpu"
        default_threshold: float — 默认灰度匹配阈值
        default_color_threshold: float — 默认颜色验证阈值
        scales: list[float] — 缩放档位，默认 [1.0]
    """

    def __init__(
        self,
        templates: List[Dict[str, Any]],
        device: str = "cuda",
        default_threshold: float = 0.75,
        default_color_threshold: float = 0.78,
        scales: Optional[List[float]] = None,
        max_templates: int = 0,
        **_kwargs: Any,
    ) -> None:
        self._use_cuda = _has_cuda_opencv()
        self._device = device

        if not self._use_cuda:
            logger.warning(
                "CudaTemplateDetector: OpenCV 未编译 CUDA 支持，回退到 CPU。"
                "请安装 CUDA 版 OpenCV 以启用 GPU 加速。"
            )

        self._scales = tuple(scales) if scales else (1.0,)
        self._max_templates = max(0, max_templates)

        # 加载模板（复用 CPU 版的加载逻辑）
        self._entries: List[_TemplateEntry] = []
        for item in templates:
            name = item["name"]
            path = item["path"]
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

        # 预计算所有缩放后的模板（GPU 版提前上传到显存）
        self._gpu_kernels: Dict[Tuple[int, int], Any] = {}  # (entry_idx, scale_idx) -> GpuMat
        if self._use_cuda:
            self._preupload_kernels()

        logger.info(
            "CudaTemplateDetector: 已加载 %d/%d 个模板, scales=%s, cuda=%s",
            len(self._entries), len(templates), self._scales, self._use_cuda,
        )

    def _preupload_kernels(self) -> None:
        """将所有缩放模板预上传到 GPU 显存。"""
        for i, entry in enumerate(self._entries):
            for j, scale in enumerate(self._scales):
                th, tw = entry.template_gray.shape[:2]
                sw = max(1, int(tw * scale))
                sh = max(1, int(th * scale))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                resized_gray = cv2.resize(entry.template_gray, (sw, sh), interpolation=interp)
                resized_bgr = cv2.resize(entry.template_bgr, (sw, sh), interpolation=interp)

                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(resized_gray)
                gpu_bgr = cv2.cuda_GpuMat()
                gpu_bgr.upload(resized_bgr)

                self._gpu_kernels[(i, j)] = (gpu_gray, gpu_bgr, sw, sh)

    def warmup(self) -> None:
        """GPU 预热：用一个小矩阵跑一次 matchTemplate。"""
        if not self._use_cuda:
            return
        try:
            dummy = cv2.cuda_GpuMat()
            dummy.upload(np.zeros((100, 100), dtype=np.uint8))
            tpl = cv2.cuda_GpuMat()
            tpl.upload(np.zeros((10, 10), dtype=np.uint8))
            matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCORR_NORMED)
            matcher.match(dummy, tpl)
        except Exception as e:
            logger.debug("CudaTemplateDetector warmup 失败: %s", e)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """对每个模板执行多尺度匹配，返回所有找到的结果。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        if self._use_cuda:
            return self._detect_gpu(frame_bgr)
        else:
            return self._detect_cpu(frame_bgr)

    def _detect_gpu(self, frame_bgr: np.ndarray) -> List[Detection]:
        """GPU 路径：cv2.cuda.matchTemplate。"""
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]

        # 上传帧到 GPU（只传一次，所有模板复用）
        gpu_frame_gray = cv2.cuda_GpuMat()
        gpu_frame_gray.upload(frame_gray)
        gpu_frame_bgr = cv2.cuda_GpuMat()
        gpu_frame_bgr.upload(frame_bgr)

        matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCORR_NORMED)
        results: List[Detection] = []
        budget_exceeded = False

        for i, entry in enumerate(self._entries):
            if self._max_templates > 0 and i >= self._max_templates:
                if not budget_exceeded:
                    logger.debug("max_templates=%d 限制，跳过剩余 %d 个模板",
                                 self._max_templates, len(self._entries) - i)
                    budget_exceeded = True
                continue

            det = self._match_single_gpu(
                entry, i, gpu_frame_gray, gpu_frame_bgr,
                frame_w, frame_h, matcher,
            )
            if det is not None:
                results.append(det)

        return results

    def _match_single_gpu(
        self,
        entry: _TemplateEntry,
        entry_idx: int,
        gpu_frame_gray: Any,
        gpu_frame_bgr: Any,
        frame_w: int,
        frame_h: int,
        matcher: Any,
    ) -> Optional[Detection]:
        """GPU 版单模板多尺度匹配。"""
        # ROI 裁剪（在 GPU 上用 GpuMat 的 colRange/rowRange）
        roi_l = max(0, int(frame_w * entry.roi[0]))
        roi_t = max(0, int(frame_h * entry.roi[1]))
        roi_r = min(frame_w, int(frame_w * entry.roi[2]))
        roi_b = min(frame_h, int(frame_h * entry.roi[3]))

        if roi_r <= roi_l or roi_b <= roi_t:
            return None

        search_gray = gpu_frame_gray.rowRange(roi_t, roi_b).colRange(roi_l, roi_r)
        search_bgr = gpu_frame_bgr.rowRange(roi_t, roi_b).colRange(roi_l, roi_r)

        best_score = -1.0
        best_color = -1.0
        best_x = 0
        best_y = 0
        best_w = 0
        best_h = 0

        for j, scale in enumerate(self._scales):
            key = (entry_idx, j)
            if key not in self._gpu_kernels:
                continue

            gpu_tpl_gray, gpu_tpl_bgr, sw, sh = self._gpu_kernels[key]

            if sw > search_gray.cols() or sh > search_gray.rows():
                continue

            # GPU matchTemplate
            result = matcher.match(search_gray, gpu_tpl_gray)

            # GPU minMaxLoc
            _, max_val, _, max_loc = cv2.cuda.minMaxLoc(result)

            if max_val > best_score:
                best_score = float(max_val)
                best_x = roi_l + max_loc[0]
                best_y = roi_t + max_loc[1]
                best_w = sw
                best_h = sh

                # 颜色二次校验：下载 patch 和模板到 CPU
                patch_gpu = search_bgr.rowRange(max_loc[1], max_loc[1] + sh).colRange(max_loc[0], max_loc[0] + sw)
                patch_bgr = patch_gpu.download()
                tpl_bgr = gpu_tpl_bgr.download()
                best_color = self._color_score(patch_bgr, tpl_bgr)

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

    def _detect_cpu(self, frame_bgr: np.ndarray) -> List[Detection]:
        """CPU 回退路径：与 TemplateDetector 逻辑一致。"""
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]
        results: List[Detection] = []

        for i, entry in enumerate(self._entries):
            if self._max_templates > 0 and i >= self._max_templates:
                break

            det = self._match_single_cpu(entry, frame_bgr, frame_gray, frame_w, frame_h)
            if det is not None:
                results.append(det)

        return results

    def _match_single_cpu(
        self,
        entry: _TemplateEntry,
        frame_bgr: np.ndarray,
        frame_gray: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Optional[Detection]:
        """CPU 版单模板匹配（回退用）。"""
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
        best_x = best_y = best_w = best_h = 0
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

                patch = search_bgr[max_loc[1]:max_loc[1] + sh, max_loc[0]:max_loc[0] + sw]
                resized_bgr = cv2.resize(entry.template_bgr, (sw, sh), interpolation=interp)
                best_color = self._color_score(patch, resized_bgr)

        found = best_score >= entry.threshold and best_color >= entry.color_threshold
        if not found:
            return None

        return Detection(
            name=entry.name,
            category="flat_label",
            score=best_score,
            x=best_x, y=best_y, w=best_w, h=best_h,
            threshold=entry.threshold,
        )

    @staticmethod
    def _color_score(patch_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
        """BGR 颜色相似度 (0-1)。"""
        if patch_bgr.shape != template_bgr.shape:
            return 0.0
        diff = cv2.absdiff(patch_bgr, template_bgr)
        return float(1.0 - np.mean(diff) / 255.0)
