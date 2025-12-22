import cv2
from pycocotools import mask as MaskUtils
import numpy as np
import json
import os
import re

OUT_DIR = "image_visualized_label"
JSON_PATH = "PhysicalAI_Warehouse/train_new.json"
IMAGE_DIR = "PhysicalAI_Warehouse/train/images"
NUM_SAMPLES = 2000

VALID_CLASSES = {
    "pallet": 0,
    "transporter": 1,
    "shelf": 2,
}

ID2CLASS = {0: "pallet", 1: "transporter", 2: "shelf"}


def infer_mask_classes_general(question):
    q = question.lower()

    tokens = re.findall(r"(pallets?|transporters?|shelves?|buffers?|<mask>)", q)

    classes = []
    current_class = None

    for tok in tokens:
        if tok in ["pallet", "pallets"]:
            current_class = "pallet"
        elif tok in ["transporter", "transporters"]:
            current_class = "transporter"
        elif tok in ["shelf", "shelves"]:
            current_class = "shelf"
        elif tok in ["buffer", "buffers"]:
            current_class = "buffer"
        elif tok == "<mask>":
            classes.append(current_class)
    return classes


def map_rle_to_labels(sample, valid_classes):
    """
    Handle multi-question conversations.
    Output:
        list of (class_id, rle)
    """
    labeled_masks = []

    mask_ptr = 0  # global mask index

    for turn in sample["conversations"]:
        if turn["from"] != "human":
            continue

        question = turn["value"]
        class_seq = infer_mask_classes_general(question)

        for cls_name in class_seq:
            if mask_ptr >= len(sample["rle"]):
                break

            if cls_name in valid_classes:
                class_id = valid_classes[cls_name]
                rle = sample["rle"][mask_ptr]
                labeled_masks.append((class_id, rle))

            mask_ptr += 1  # luôn tăng global index

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

        mask_color = colors[i % len(colors)]
        text_color = colors[(i + 1) % len(colors)]

        overlay[mask == 1] = 0.6 * overlay[mask == 1] + 0.4 * np.array(mask_color)

        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())

            # vẽ centroid
            cv2.circle(overlay, (cx, cy), 6, text_color, -1)

            # vẽ label + class
            cv2.putText(
                overlay,
                f"{class_name}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
            )

    out_name = f"{os.path.basename(sample['image'])}"
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, overlay)
    print(f"OK. {out_path}")
print("DONE.")
