import cv2
from pycocotools import mask as MaskUtils
import numpy as np
import json
import os

OUT_DIR = "image_visualized"
JSON_PATH = "PhysicalAI_Warehouse/train.json"
IMAGE_DIR = "PhysicalAI_Warehouse/train/images"
NUM_SAMPLES = 1000

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
with open(JSON_PATH, "r") as f:
    data = json.load(f)

samples = data[:NUM_SAMPLES]


def decode_rle(rle):
    return MaskUtils.decode(rle).astype(np.uint8)


colors = [
    (255, 0, 0),  # red
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (255, 255, 0),  # yellow
]
TARGET_NAME = "005914.png"

for idx, sample in enumerate(samples):
    # chỉ xử lý đúng ảnh cần tìm
    img_path = os.path.join(IMAGE_DIR, sample["image"])

    # lọc tồn tại TRƯỚC imread
    if not os.path.exists(img_path):
        continue
    if sample["image"] == TARGET_NAME:
        print(f"FORMAT: {sample['conversations'][0]}, {len(sample['rle'])} masks")

    img = cv2.imread(img_path)
    overlay = img.copy()

    for i, rle in enumerate(sample["rle"]):
        mask = decode_rle(rle)
        color = colors[i % len(colors)]

        # overlay mask
        overlay[mask == 1] = 0.6 * overlay[mask == 1] + 0.4 * np.array(color)

        # vẽ centroid (chỉ để dễ nhìn)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.circle(overlay, (cx, cy), 5, color, -1)

    out_name = f"{os.path.basename(sample['image'])}"
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, overlay)

    print(f"[OK] Saved {out_path}")
print("DONE.")
