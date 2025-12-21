import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as MaskUtils
import numpy as np
import json
import os
import re

OUT_DIR = "image_visualized_label"
JSON_PATH = "PhysicalAI_Warehouse/train.json"
IMAGE_DIR = "PhysicalAI_Warehouse/train/images"
NUM_SAMPLES = 1000

VALID_CLASSES = {
    "pallet": 0,
    "transporter": 1,
    "shelf": 2,
}

ID2CLASS = {0: "pallet", 1: "transporter", 2: "shelf"}


def extract_mask_classes(question):
    """
    Trả về list class theo thứ tự <mask> xuất hiện
    Ví dụ: ['pallet', 'buffer', 'shelf']
    """
    pattern = r"(pallet|buffer|transporter|shelf)\s*<mask>"
    return re.findall(pattern, question.lower())


def map_rle_to_labels(sample, valid_classes):
    """
    Input:
        sample: 1 item trong train.json
        valid_classes: dict {'pallet':0, ...}
    Output:
        list of (class_id, rle)
    """
    question = sample["conversations"][0]["value"]
    class_seq = extract_mask_classes(question)

    labeled_masks = []

    for idx, cls_name in enumerate(class_seq):
        if cls_name in valid_classes:
            class_id = valid_classes[cls_name]
            rle = sample["rle"][idx]
            labeled_masks.append((class_id, rle))
        # else: buffer → skip

    return labeled_masks


#
# =========================
with open(JSON_PATH, "r") as f:
    data = json.load(f)

samples = data[:NUM_SAMPLES]


def decode_rle(rle):
    mask = MaskUtils.decode(rle)
    return mask.astype(np.uint8)


for idx, sample in enumerate(samples):
    img_path = os.path.join(IMAGE_DIR, sample["image"])

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = img.copy()

    colors = [
        (255, 0, 0),  # red
        (0, 255, 0),  # green
        (0, 0, 255),  # blue
        (255, 255, 0),  # yellow
    ]
    labeled_mask = map_rle_to_labels(sample, VALID_CLASSES)

    for i, (class_id, rle) in enumerate(labeled_mask):
        class_name = ID2CLASS[class_id]
        mask = decode_rle(rle)
        color = colors[class_id % len(colors)]

        overlay[mask == 1] = 0.6 * overlay[mask == 1] + 0.4 * np.array(color)

        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())

            # vẽ centroid
            cv2.circle(overlay, (cx, cy), 6, color, -1)

            # vẽ label + class
            cv2.putText(
                overlay,
                f"{class_name}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    out_name = f"{os.path.basename(sample['image'])}"
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, overlay)
    print(f"OK. {out_path}")
print("DONE.")
