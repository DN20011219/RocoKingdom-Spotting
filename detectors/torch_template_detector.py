"""PyTorch GPU 模板匹配检测器 — torch.nn.functional.conv2d + BGR 颜色二次校验。

利用 PyTorch CUDA 在 GPU 上执行模板匹配，无需 OpenCV CUDA 编译。
帧和模板均保持在 GPU 显存，避免重复 CPU-GPU 传输。
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


def _has_torch_cuda() -> bool:
    """检查 PyTorch 是否支持 CUDA。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@DetectorRegistry.register("template_torch")
class TorchTemplateDetector(DetectorBase):
    """PyTorch GPU 模板匹配检测器。

    使用 torch.nn.functional.conv2d 在 GPU 上执行互相关匹配。
    当 CUDA 不可用时自动回退到 CPU matchTemplate。

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
        # 首次导入会加载约 4GB 的 CUDA 运行库；冷缓存 + 杀软实时扫描时可达数分钟，
        # 期间 LoadLibraryExW 不返回。打日志避免看起来像死机。
        logger.info("TorchTemplateDetector: 正在导入 PyTorch（首次导入需加载 CUDA 运行库，可能耗时较久）...")
        _t_import = time.perf_counter()
        import torch
        logger.info("TorchTemplateDetector: PyTorch 导入完成，耗时 %.1fs", time.perf_counter() - _t_import)

        self._use_cuda = _has_torch_cuda()
        self._device = torch.device("cuda:0" if self._use_cuda else "cpu")
        self._torch = torch

        if not self._use_cuda:
            logger.warning(
                "TorchTemplateDetector: CUDA 不可用，回退到 CPU。"
                "请安装 CUDA 版 PyTorch 以启用 GPU 加速。"
            )
        else:
            logger.info("TorchTemplateDetector: 使用 GPU %s", torch.cuda.get_device_name(0))

        self._scales = tuple(scales) if scales else (1.0,)
        self._max_templates = max(0, max_templates)

        # 加载模板
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

        # 预计算所有缩放后的 GPU kernel
        # key: (entry_idx, scale_idx) -> (gpu_kernel_gray, gpu_kernel_bgr, sw, sh)
        self._gpu_kernels: Dict[Tuple[int, int], Tuple[Any, Any, int, int]] = {}
        if self._use_cuda:
            self._preupload_kernels()

        logger.info(
            "TorchTemplateDetector: 已加载 %d/%d 个模板, scales=%s, cuda=%s, kernels=%d",
            len(self._entries), len(templates), self._scales, self._use_cuda,
            len(self._gpu_kernels),
        )

    def _preupload_kernels(self) -> None:
        """将所有缩放模板预上传到 GPU 显存，转换为 conv2d kernel 格式。"""
        torch = self._torch
        for i, entry in enumerate(self._entries):
            for j, scale in enumerate(self._scales):
                th, tw = entry.template_gray.shape[:2]
                sw = max(1, int(tw * scale))
                sh = max(1, int(th * scale))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                resized_gray = cv2.resize(entry.template_gray, (sw, sh), interpolation=interp)
                resized_bgr = cv2.resize(entry.template_bgr, (sw, sh), interpolation=interp)

                # 转换为 conv2d kernel: shape [out_channels=1, in_channels=1, H, W]
                kernel_gray = torch.from_numpy(resized_gray.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self._device)
                # BGR kernel: shape [out_channels=3, in_channels=3, H, W] (用于颜色校验)
                kernel_bgr = torch.from_numpy(resized_bgr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self._device)

                self._gpu_kernels[(i, j)] = (kernel_gray, kernel_bgr, sw, sh)

    def warmup(self) -> None:
        """GPU 预热：用一个小矩阵跑一次 conv2d。"""
        if not self._use_cuda:
            return
        try:
            torch = self._torch
            dummy = torch.zeros(1, 1, 100, 100, device=self._device)
            kernel = torch.zeros(1, 1, 10, 10, device=self._device)
            torch.nn.functional.conv2d(dummy, kernel)
        except Exception as e:
            logger.debug("TorchTemplateDetector warmup 失败: %s", e)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """对每个模板执行多尺度匹配，返回所有找到的结果。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        if self._use_cuda:
            return self._detect_gpu(frame_bgr)
        else:
            return self._detect_cpu(frame_bgr)

    def _detect_gpu(self, frame_bgr: np.ndarray) -> List[Detection]:
        """GPU 路径：torch.nn.functional.conv2d。"""
        torch = self._torch
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]

        # 上传帧到 GPU（只传一次）
        # 灰度帧: [1, 1, H, W]
        gpu_frame_gray = torch.from_numpy(frame_gray.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self._device)
        # BGR 帧: [1, 3, H, W]
        gpu_frame_bgr = torch.from_numpy(frame_bgr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self._device)

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
                frame_w, frame_h,
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
    ) -> Optional[Detection]:
        """GPU 版单模板多尺度匹配。"""
        torch = self._torch

        # ROI 裁剪
        roi_l = max(0, int(frame_w * entry.roi[0]))
        roi_t = max(0, int(frame_h * entry.roi[1]))
        roi_r = min(frame_w, int(frame_w * entry.roi[2]))
        roi_b = min(frame_h, int(frame_h * entry.roi[3]))

        if roi_r <= roi_l or roi_b <= roi_t:
            return None

        search_gray = gpu_frame_gray[:, :, roi_t:roi_b, roi_l:roi_r]
        search_bgr = gpu_frame_bgr[:, :, roi_t:roi_b, roi_l:roi_r]

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

            kernel_gray, kernel_bgr, sw, sh = self._gpu_kernels[key]

            if sw > search_gray.shape[3] or sh > search_gray.shape[2]:
                continue

            # GPU conv2d 匹配（灰度）
            result = torch.nn.functional.conv2d(search_gray, kernel_gray)
            # 归一化：TM_CCORR_NORMED 等价于 conv2d 后除以模板能量
            tpl_energy = kernel_gray.pow(2).sum().sqrt().clamp(min=1e-8)
            # 简化：直接用 conv2d 结果的最大值作为分数（近似 TM_CCORR_NORMED）
            max_val = result.max().item()
            # 归一化到 0-1 范围
            max_val = max_val / (tpl_energy.item() * search_gray.shape[2] * search_gray.shape[3]) ** 0.5

            if max_val > best_score:
                # 找到最大值位置
                max_loc = torch.nonzero(result == result.max(), as_tuple=False)
                if len(max_loc) > 0:
                    # max_loc shape: [N, 4] = [batch, channel, y, x]
                    y_idx = max_loc[0, 2].item()
                    x_idx = max_loc[0, 3].item()

                    best_score = float(max_val)
                    best_x = roi_l + x_idx
                    best_y = roi_t + y_idx
                    best_w = sw
                    best_h = sh

                    # 颜色二次校验：下载 patch 和模板到 CPU
                    patch_gpu = search_bgr[:, :, y_idx:y_idx + sh, x_idx:x_idx + sw]
                    patch_bgr = patch_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    tpl_bgr = kernel_bgr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
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
