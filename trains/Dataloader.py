from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset

def _decode_rle_mask(rle_mask: Any) -> np.ndarray:
    """Decode a coco RLE mask string/dict into a binary numpy array."""
    if isinstance(rle_mask, dict):
        rle = rle_mask
    else:
        rle = {"size": [1080, 1920], "counts": rle_mask}

    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")

    decoded = mask_utils.decode({"size": rle["size"], "counts": counts})
    return decoded.astype(np.uint8)


def _get_centroid(mask: np.ndarray) -> List[np.int32]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return [np.int32(0), np.int32(0)]

    x_mean, y_mean,  = coords[:, 0].mean(), coords[:, 1].mean()
    return [np.int32(round(x_mean)), np.int32(round(y_mean))]


def _get_bbox(mask: np.ndarray) -> List[np.int32]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return [np.int32(0), np.int32(0), np.int32(0), np.int32(0)]

    x_min, y_min,  = coords.min(axis=0)
    x_max, y_max,  = coords.max(axis=0)
    return [
        np.int32(x_min),
        np.int32(y_min),
        np.int32(x_max - x_min + 1),
        np.int32(y_max - y_min + 1),
    ]


def _get_depth(depth_path: Path, centroid_x: np.int32, centroid_y: np.int32) -> np.float32:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    h, w = depth.shape[:2]
    x = int(np.clip(int(centroid_x), 0, w - 1))
    y = int(np.clip(int(centroid_y), 0, h - 1))

    return np.float32(depth[x, y])


class DistanceDataset(Dataset):
    def __init__(self, json_path: str, split: str) -> None:
        self.json_path = Path(json_path)
        self.split = split

        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with self.json_path.open() as f:
            self.samples: Sequence[Dict[str, Any]] = json.load(f)

        self.depth_root = self.json_path.parent / split / "depths"
        if not self.depth_root.exists():
            raise FileNotFoundError(f"Depth folder not found: {self.depth_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rles = sample["rle"]
        if len(rles) != 2:
            raise ValueError(f"Expected exactly 2 masks, got {len(rles)} for idx={idx}")

        masks = [_decode_rle_mask(rle) for rle in rles]
        (x1, y1), (x2, y2) = (_get_centroid(mask) for mask in masks)

        depth_path = self.depth_root / f"{Path(sample['image']).stem}_depth.png"
        depth_1 = _get_depth(depth_path, x1, y1)
        depth_2 = _get_depth(depth_path, x2, y2)
        [x_bbox_1, y_bbox_1, w1, h1], [x_bbox_2, y_bbox_2,w2,h2] = (_get_bbox(mask) for mask in masks)
        height, width = masks[0].shape

        return (
        np.float32(x1) / np.float32(height),
        np.float32(y1) / np.float32(width),
        np.float32(x2) / np.float32(height),
        np.float32(y2) / np.float32(width),

        np.float32(depth_1) / np.float32(255.0),
        np.float32(depth_2) / np.float32(255.0),

        np.float32(w1) / np.float32(width),
        np.float32(h1) / np.float32(height),

        np.float32(w2) / np.float32(width),
        np.float32(h2) / np.float32(height),
        np.float32(sample["distance"]),
        )

def main() -> None:
    import matplotlib.pyplot as plt
    dataset = DistanceDataset("data/val_dist_est.json",split="val")
        

if __name__ == "__main__":
    main()