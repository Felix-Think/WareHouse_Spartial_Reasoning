from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from ultralytics import YOLO


def load_gt_mask(label_path: Path, img_shape: tuple[int, int]) -> np.ndarray:
    """
    Build a binary mask from a YOLOv8 segmentation label file.

    label format: class cx cy ... (normalized). Supports multiple polygons per file.
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if not label_path.exists():
        return mask

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            coords = [float(x) for x in parts[1:]]
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def predict_mask(model: YOLO, image_path: Path, conf: float) -> np.ndarray:
    """Run segmentation model and combine all instance masks into a single binary mask."""
    result = model(str(image_path), conf=conf, verbose=False)[0]
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks:
        for poly in result.masks.xy:
            if poly.size == 0:
                continue
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


def collect_images(img_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def compare_masks(
    weights: Path,
    data_dir: Path,
    split: str,
    output_dir: Path,
    conf: float,
    limit: int | None = None,
) -> None:
    model = YOLO(str(weights))
    img_dir = data_dir / split / "images"
    label_dir = data_dir / split / "labels"

    images = collect_images(img_dir)
    if limit:
        images = images[:limit]

    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    ious: List[float] = []
    diff_fracs: List[float] = []

    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_mask = load_gt_mask(label_path, img_shape=cv2.imread(str(img_path)).shape[:2])
        pred_mask = predict_mask(model, img_path, conf=conf)

        union_mask = np.logical_or(pred_mask, gt_mask)
        diff_mask = np.logical_xor(pred_mask, gt_mask)

        inter = np.logical_and(pred_mask, gt_mask).sum()
        union_area = union_mask.sum()
        diff_area = diff_mask.sum()
        iou = inter / max(1, union_area)
        diff_frac = diff_area / max(1, gt_mask.size)

        ious.append(float(iou))
        diff_fracs.append(float(diff_frac))

        save_mask(output_dir / "union" / f"{img_path.stem}.png", union_mask.astype(np.uint8))
        save_mask(output_dir / "diff" / f"{img_path.stem}.png", diff_mask.astype(np.uint8))

    mean_iou = float(np.mean(ious))
    mean_diff = float(np.mean(diff_fracs))
    print(f"Processed {len(images)} images from split '{split}'.")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean diff fraction (xor pixels / total): {mean_diff:.4f}")
    print(f"Outputs saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predicted vs validation masks (union/diff).")
    parser.add_argument("--weights", type=Path, required=True, help="Path to YOLOv8 segmentation weights.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"), help="Dataset root containing split folders.")
    parser.add_argument("--split", type=str, default="valid", help="Dataset split name (e.g., train, valid).")
    parser.add_argument("--output", type=Path, default=Path("outputs/mask_compare"), help="Directory to save union/diff masks.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of images to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_masks(
        weights=args.weights,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output,
        conf=args.conf,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
