from __future__ import annotations

from functools import lru_cache

import torch

from trains.MLP import build_mlp

from ..core.config import DistanceConfig, load_distance_config


@lru_cache
def get_distance_config() -> DistanceConfig:
    return load_distance_config()


@lru_cache
def get_distance_model() -> torch.nn.Module:
    cfg = get_distance_config()
    checkpoint = torch.load(str(cfg.model_path), map_location=cfg.device or "cpu")
    input_dim = int(checkpoint.get("input_dim", 10))
    model = build_mlp(input_dim=input_dim, hidden_dims=(128, 64), out_dim=1, dropout=0.3)
    model.load_state_dict(checkpoint["state_dict"])
    if cfg.device:
        model.to(cfg.device)
    model.eval()
    return model
