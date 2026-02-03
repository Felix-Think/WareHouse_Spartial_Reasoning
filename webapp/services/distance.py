from __future__ import annotations

import math
import re
from typing import Dict, Tuple

import cv2
import numpy as np
import torch


def _decode_depth_image(file_bytes: bytes) -> np.ndarray:
    depth = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Could not decode depth image data.")
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    return depth


def _parse_indices(question: str) -> Tuple[int, int]:
    matches = re.findall(r"\d+", question)
    if len(matches) < 2:
        raise ValueError("Could not parse two object indices from the question.")
    first, second = (int(matches[0]), int(matches[1]))
    return first, second


def _sample_depth(depth: np.ndarray, row: float, col: float) -> float:
    h, w = depth.shape[:2]
    r = int(np.clip(round(row), 0, h - 1))
    c = int(np.clip(round(col), 0, w - 1))
    return float(depth[r, c])


def _build_features(
    box1: Dict,
    box2: Dict,
    image_size: Dict[str, int],
    depth: np.ndarray,
    depth_max: float,
) -> torch.Tensor:
    width = float(image_size["width"])
    height = float(image_size["height"])

    x1a, y1a, x2a, y2a = box1["box"]
    x1b, y1b, x2b, y2b = box2["box"]

    cx1 = (x1a + x2a) / 2.0
    cy1 = (y1a + y2a) / 2.0
    cx2 = (x1b + x2b) / 2.0
    cy2 = (y1b + y2b) / 2.0

    depth_h, depth_w = depth.shape[:2]
    sx = depth_w / max(1.0, width)
    sy = depth_h / max(1.0, height)

    depth1 = _sample_depth(depth, cy1 * sy, cx1 * sx)
    depth2 = _sample_depth(depth, cy2 * sy, cx2 * sx)

    w1 = max(0.0, x2a - x1a)
    h1 = max(0.0, y2a - y1a)
    w2 = max(0.0, x2b - x1b)
    h2 = max(0.0, y2b - y1b)

    feats = [
        cy1 / height,
        cx1 / width,
        cy2 / height,
        cx2 / width,
        depth1 / depth_max,
        depth2 / depth_max,
        w1 / width,
        h1 / height,
        w2 / width,
        h2 / height,
    ]
    return torch.tensor([feats], dtype=torch.float32)


def _predict_distance(
    model: torch.nn.Module,
    feats: torch.Tensor,
    fovx_deg: float,
    fovy_deg: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    params = model(feats)
    s = torch.nn.functional.softplus(params[:, 0]) + eps

    x1, y1, x2, y2, depth1, depth2, w1, h1, w2, h2 = feats.unbind(dim=1)

    d1 = s * depth1
    d2 = s * depth2

    fovx = math.radians(fovx_deg)
    fovy = math.radians(fovy_deg)
    kx = 2.0 * math.tan(fovx / 2.0)
    ky = 2.0 * math.tan(fovy / 2.0)

    x_cam_1 = (y1 - 0.5) * kx
    y_cam_1 = (x1 - 0.5) * ky
    x_cam_2 = (y2 - 0.5) * kx
    y_cam_2 = (x2 - 0.5) * ky

    X1 = d1 * x_cam_1
    Y1 = d1 * y_cam_1
    Z1 = d1

    X2 = d2 * x_cam_2
    Y2 = d2 * y_cam_2
    Z2 = d2

    return torch.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2 + (Z1 - Z2) ** 2 + eps)


def predict_distance_from_question(
    question: str,
    boxes: list[Dict],
    image_size: Dict[str, int],
    depth_bytes: bytes,
    model: torch.nn.Module,
    fovx_deg: float,
    fovy_deg: float,
    depth_max: float,
) -> Dict[str, float | int]:

    idx1, idx2 = _parse_indices(question)
    if idx1 < 1 or idx2 < 1:
        raise ValueError("Object indices must start from 1.")
    if idx1 > len(boxes) or idx2 > len(boxes):
        raise ValueError("Object index is out of range for detected boxes.")

    depth = _decode_depth_image(depth_bytes)
    feats = _build_features(boxes[idx1 - 1], boxes[idx2 - 1], image_size, depth, depth_max)
    with torch.no_grad():
        dist = _predict_distance(model, feats, fovx_deg, fovy_deg).item()
    return {"distance": float(dist), "index1": idx1, "index2": idx2}
