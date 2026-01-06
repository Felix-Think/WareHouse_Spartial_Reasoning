from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    model_path: Path
    conf_threshold: float
    device: str | None


def load_config() -> AppConfig:
    """Load lightweight configuration from environment."""
    model_path = Path(os.getenv("YOLO_MODEL_PATH", "runs/segment/train/weights/best.pt"))
    conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))
    device = os.getenv("YOLO_DEVICE") or None
    return AppConfig(model_path=model_path, conf_threshold=conf_threshold, device=device)
