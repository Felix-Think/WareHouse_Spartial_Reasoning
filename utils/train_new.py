import os
import json
import re
import numpy as np
from pycocotools import mask as mask_utils
import cv2
from collections import defaultdict
import random


root_dir = "PhysicalAI_Warehouse"
image_dir = os.path.join(root_dir, "train/images")
train_json_path = os.path.join(root_dir, "train.json")
train_new_path = os.path.join(root_dir, "train_new.json")

# 1. Lấy danh sách tên ảnh
image_files = set(os.listdir(image_dir))  # dùng set cho nhanh

# 2. Load train.json
with open(train_json_path, "r") as f:
    data = json.load(f)

# 3. Lọc dữ liệu
new_data = [sample for sample in data if sample.get("image") in image_files]

# print(len(image_files))
#
# seen = set()
# final_data = []
#
# for s in new_data:
#     img = s["image"]
#     if img not in seen:
#         final_data.append(s)
#         seen.add(img)
#
# print("Final unique samples:", len(final_data))


ID2CLASS = {0: "pallet", 1: "transporter", 2: "shelf"}
VALID_CLASSES = {
    "pallet": 0,
    "transporter": 1,
}


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


def rle_to_yolo_polygons(rle, class_id, min_area=50):
    """
    Args:
        rle: dict {"size": [h, w], "counts": "..."}
        class_id: int
        min_area: bỏ mask quá nhỏ
    Returns:
        list[str]  # mỗi phần tử là 1 dòng label YOLO
    """
    h, w = rle["size"]

    rle_obj = {"size": [h, w], "counts": rle["counts"]}
    mask = mask_utils.decode(rle_obj)  # (h, w), 0/1

    mask = (mask * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        cnt = cnt.squeeze(1)  # (N, 2)
        if cnt.shape[0] < 3:
            continue

        # Normalize coordinates
        poly = []
        for x, y in cnt:
            poly.append(x / w)
            poly.append(y / h)

        # YOLO requires at least 3 points
        if len(poly) >= 6:
            line = str(class_id) + " " + " ".join(f"{v:.6f}" for v in poly)
            yolo_lines.append(line)

    return yolo_lines


def convert_image_masks_to_yolo(labeled_masks):
    lines = []
    for label, mask_rle in labeled_masks:
        lines.extend(rle_to_yolo_polygons(mask_rle, label))
    return lines


NEW_DATASET_DIR = "datasets"


def build_yolo_dataset(data, split):
    """
    data: list of samples (train_new.json)
    split: 'train' or 'val'
    """

    images_out = os.path.join(NEW_DATASET_DIR, "images", split)
    labels_out = os.path.join(NEW_DATASET_DIR, "labels", split)

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # 1. Group masks theo image
    image_to_masks = defaultdict(list)

    for sample in data:
        image_name = sample["image"]
        labeled_masks = map_rle_to_labels(sample, VALID_CLASSES)
        image_to_masks[image_name].extend(labeled_masks)

    # 2. Ghi ảnh + label
    for image_name, masks in image_to_masks.items():
        # ---- copy image ----
        src_img_path = os.path.join(image_dir, image_name)
        dst_img_path = os.path.join(images_out, image_name)

        img = cv2.imread(src_img_path)
        if img is None:
            print(f"[WARN] Cannot read image {image_name}")
            continue

        cv2.imwrite(dst_img_path, img)

        # ---- convert masks → YOLO polygon ----
        lines = convert_image_masks_to_yolo(masks)

        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(labels_out, label_name)

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(f"[DONE] Build YOLO {split}: {len(image_to_masks)} images")


image_groups = defaultdict(list)
for s in new_data:
    image_groups[s["image"]].append(s)

images = list(image_groups.keys())
random.shuffle(images)

split = int(len(images) * 0.8)
train_imgs = set(images[:split])
val_imgs = set(images[split:])

train_data = [s for img in train_imgs for s in image_groups[img]]
val_data = [s for img in val_imgs for s in image_groups[img]]


build_yolo_dataset(train_data, "train")
build_yolo_dataset(val_data, "val")
