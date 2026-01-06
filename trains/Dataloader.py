"""
Minimal dataset/dataloader for the distance task.

Each sample:
- Label: the `normalized_answer` distance (float).
- Input: decoded binary masks referenced in the GPT reply (e.g. "[Region 9]" -> rle[9]).
  Only region ids within `max_regions` and the available `rle` list are used.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
import os
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
import cv2

_REGION_RE = re.compile(r"\[Region (\d+)\]")

def _extract_region_ids(reply: str) -> List[int]:
    """Pull region ids from the GPT reply text and clamp to valid range."""
    ids = [int(m.group(1)) for m in _REGION_RE.finditer(reply)]
    return list(set(ids))

def _decode_rle_mask(rle_obj: Dict[str, Any]) -> np.ndarray:
    """Decode a single RLE mask to a uint8 numpy array (H, W)."""
    rle = dict(rle_obj)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("utf-8")
    mask = mask_utils.decode(rle)  # returns H x W x 1
    # Ensure shape is (H, W)
    return np.asarray(mask, dtype=np.uint8).squeeze()

def _get_centroid(mask: np.ndarray) -> Sequence[int]:
    """Compute the centroid (x, y) of a binary mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0)
    centroid_x = np.int32(np.mean(xs))
    centroid_y = np.int32(np.mean(ys))
    return (centroid_x, centroid_y)


def _infer_split_from_json(json_path: Path) -> str:
    stem = json_path.stem.lower()
    if stem.startswith("train"):
        return "train"
    if stem.startswith("val"):
        return "val"
    if stem.startswith("test"):
        return "test"
    return "val"


class DistanceDataset(Dataset):
    """Dataset that surfaces masks referenced in the GPT message and the distance label."""

    def __init__(self, json_path: str, split: str | None = None):
        super().__init__()
        self.json_path = Path(json_path)
        self.split = split or _infer_split_from_json(self.json_path)
        self.depth_dir = self.json_path.parent / self.split / "depths"
        print(f"Using depth directory: {self.depth_dir}")
        if not self.depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {self.depth_dir}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            self._items: List[Dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        item = self._items[idx]
        image_id = Path(item["image"]).stem
        depth_path = self.depth_dir / f"{image_id}_depth.png"
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        label = float(item["normalized_answer"])
        x_max = np.int32(item["rle"][0]["size"][0])
        y_max = np.int32(item["rle"][0]["size"][1])
        # Use the GPT reply to discover referenced regions.
        gpt_reply = ""
        for message in item.get("conversations", []):
            if message.get("from") == "gpt":
                gpt_reply = message.get("value", "")
                break
        region_ids = _extract_region_ids(gpt_reply)
        rle_list = item.get("rle", [])
        masks = []  
        for region_id in region_ids:
            rle_obj = rle_list[region_id]
            mask = _decode_rle_mask(rle_obj)
            masks.append(mask)
        
        centroid = np.array([_get_centroid(mask) for mask in masks]).astype(np.int32)
        x1 = (centroid[0, 0] / x_max).astype(np.float32)
        y1 = centroid[0, 1] /y_max.astype(np.float32)
        x2 = centroid[1, 0] / x_max.astype(np.float32)
        y2 = centroid[1, 1] / y_max.astype(np.float32)
        depth = depth_image.astype(np.float32) / 255.0
        depth_centroid_value = depth[centroid[:, 1], centroid[:, 0]]
        depth_1 = depth_centroid_value[0]
        depth_2 = depth_centroid_value[1]
        return x1,y1,x2,y2, depth_1, depth_2, x_max, y_max, label


def main() -> None:
    """Quick sanity test for either val or train splits."""
    dataset = DistanceDataset(json_path="PhysicalAI_Warehouse/val_distance.json",split="val")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in dataloader:
        if batch is None:
            continue
        x1, y1, x2, y2, depth_1, depth_2, x_max, y_max,label= batch
        print("x1:", x1)
        print("y1:", y1)
        print("x2:", x2)
        print("y2:", y2)
        print("depth_1:", depth_1)
        print("depth_2:", depth_2)
        print("label:", label)
        print("x_max:", x_max)
        print("y_max:", y_max)
        break
    
if __name__ == "__main__":
    main()
