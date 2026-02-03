from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    model_path: Path
    conf_threshold: float
    device: str | None


@dataclass
class DistanceConfig:
    model_path: Path
    fovx_deg: float
    fovy_deg: float
    depth_max: float
    device: str | None


def load_config() -> AppConfig:
    """Load lightweight configuration from environment."""
    model_path = Path(os.getenv("YOLO_MODEL_PATH", "runs/segment/train/weights/best.pt"))
    conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))
    device = os.getenv("YOLO_DEVICE") or None
    return AppConfig(model_path=model_path, conf_threshold=conf_threshold, device=device)


def load_distance_config() -> DistanceConfig:
    """Load distance-estimation configuration from environment."""
    model_path = Path(os.getenv("DIST_MODEL_PATH", "best_model.pth"))
    fovx_deg = float(os.getenv("DIST_FOVX_DEG", "120"))
    fovy_deg = float(os.getenv("DIST_FOVY_DEG", "120"))
    depth_max = float(os.getenv("DIST_DEPTH_MAX", "255"))
    device = os.getenv("DIST_DEVICE") or None
    return DistanceConfig(
        model_path=model_path,
        fovx_deg=fovx_deg,
        fovy_deg=fovy_deg,
        depth_max=depth_max,
        device=device,
    )
