"""RocoKingdom-Spotting 统一配置 — JSON 驱动，所有参数均可在 config.json 中覆盖。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.json"


@dataclass
class CaptureConfig:
    window_keyword: str = "洛克王国：世界"
    backend: str = "screen-client"          # screen-client | screen-window
    use_frame_buffer: bool = True
    frame_buffer_size: int = 10
    foreground_only: bool = True
    interval: float = 0.1                   # 标准模式下的帧间隔（秒）


@dataclass
class DisplayConfig:
    show_overlay: bool = True
    print_json: bool = False
    debug: bool = False
    debug_dir: str = "debug_frames"
    debug_save_interval: int = 10           # 每 N 帧保存一张调试图


@dataclass
class SentinelConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    detectors: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 原始 dict，供 pipeline 读取完整配置
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- paths ----
    @property
    def labels_dir(self) -> Path:
        return PROJECT_DIR / "labels"

    @property
    def models_dir(self) -> Path:
        return PROJECT_DIR / "models"

    @property
    def debug_dir(self) -> Path:
        p = Path(self.display.debug_dir)
        return p if p.is_absolute() else PROJECT_DIR / p


def _build_capture(cfg: dict) -> CaptureConfig:
    return CaptureConfig(
        window_keyword=cfg.get("window_keyword", "洛克王国：世界"),
        backend=cfg.get("capture_backend", "screen-client"),
        use_frame_buffer=cfg.get("use_frame_buffer", True),
        frame_buffer_size=int(cfg.get("frame_buffer_size", 10)),
        foreground_only=cfg.get("foreground_only", True),
        interval=float(cfg.get("interval", 0.1)),
    )


def _build_display(cfg: dict) -> DisplayConfig:
    return DisplayConfig(
        show_overlay=cfg.get("show_overlay", True),
        print_json=cfg.get("print_json", False),
        debug=cfg.get("debug", False),
        debug_dir=cfg.get("debug_dir", "debug_frames"),
        debug_save_interval=int(cfg.get("debug_save_interval", 10)),
    )


def load_config(path: Path | None = None) -> SentinelConfig:
    """从 JSON 文件加载配置。"""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return SentinelConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return SentinelConfig(
        capture=_build_capture(raw),
        display=_build_display(raw),
        detectors=raw.get("detectors", {}),
        _raw=raw,
    )
