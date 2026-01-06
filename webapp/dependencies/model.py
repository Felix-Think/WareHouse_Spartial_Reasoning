from __future__ import annotations

from functools import lru_cache

from ultralytics import YOLO

from ..core.config import AppConfig, load_config


@lru_cache
def get_config() -> AppConfig:
    return load_config()


@lru_cache
def get_model() -> YOLO:
    cfg = get_config()
    model = YOLO(str(cfg.model_path))
    if cfg.device:
        model.to(cfg.device)
    return model
