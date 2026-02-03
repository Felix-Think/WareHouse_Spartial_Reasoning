"""
Fast dataloader that uses preprocessed depth data from JSON.
This version avoids reading depth images during training, significantly improving I/O performance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset


class DistanceDataset(Dataset):
    """Fast dataset that uses preprocessed depth data."""
    
    def __init__(self, json_path: str, split: str) -> None:
        self.json_path = Path(json_path)
        self.split = split

        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with self.json_path.open() as f:
            self.samples: Sequence[Dict[str, Any]] = json.load(f)
        
        # Check if data is preprocessed
        if self.samples and "preprocessed" not in self.samples[0]:
            raise ValueError(
                f"JSON file does not contain preprocessed data. "
                f"Please run preprocess_depth.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        preprocessed = sample["preprocessed"]
        
        # Extract preprocessed data
        (x1, y1), (x2, y2) = preprocessed["centroids"]
        depth_1, depth_2 = preprocessed["depths"]
        [x_bbox_1, y_bbox_1, w1, h1], [x_bbox_2, y_bbox_2, w2, h2] = preprocessed["bboxes"]
        height, width = preprocessed["mask_shape"]

        # Return normalized features (same format as original dataloader)
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
            
            np.float32(sample["distance"])*100,
        )


def main() -> None:
    """Test the fast dataloader."""
    import time
    
    print("Testing DistanceDataset...")
    dataset = DistanceDataset("/kaggle/input/spatialdataset/train_dist_est.json", split="val")
    print(f"Dataset size: {len(dataset)}")
    
    # Time the data loading
    start = time.time()
    for i in range(min(100, len(dataset))):
        _ = dataset[i]
    elapsed = time.time() - start
    
    print(f"Loaded 100 samples in {elapsed:.3f} seconds")
    print(f"Average time per sample: {elapsed*10:.3f} ms")
    
    # Show a sample
    sample = dataset[0]
    print(f"\nSample output (11 features):")
    print(sample)


if __name__ == "__main__":
    main()
