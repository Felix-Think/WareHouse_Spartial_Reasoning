import cv2
import numpy as np
import PIL.Image as Image
from typing import List
from pycocotools import mask as maskUtils
import os
import json

def segm_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert COCO segmentation to binary mask"""
    if isinstance(segmentation, list):
        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle = maskUtils.merge(rles)
    else:
        rle = segmentation
    return maskUtils.decode(rle)

def mask_to_polygon(mask):
    """Convert binary mask to polygon"""
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            contour = contour.reshape(-1, 2)
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    if len(polygons) == 0:
        return []
    return max(polygons, key=len)  # Return largest contour

def mask_to_bbox(x, y, w, h):
    """Convert COCO bbox [x,y,w,h] to [x_min,y_min,x_max,y_max]"""
    return [float(x), float(y), float(x + w), float(y + h)]

def create_class_object_matrix(image_id: str, coco_json: str):
    with open(coco_json, 'r') as f:
        data = json.load(f)
    target_image_info = next((img for img in data['images']
                              if os.path.splitext(img['file_name'])[0] == image_id), None)
    if not target_image_info:
        return None, None

    img_width, img_height = target_image_info['width'], target_image_info['height']
    image_annotations = [ann for ann in data['annotations'] if ann['image_id'] == target_image_info['id']]
    if not image_annotations:
        return None, None

    object_matrix = np.zeros((img_height, img_width, len(image_annotations)), dtype=np.uint8)
    for idx, ann in enumerate(image_annotations):
        mask = segm_to_mask(ann['segmentation'], img_height, img_width)
        object_matrix[:, :, idx] = np.where(mask, ann['category_id'], 0).astype(np.uint8)

    return object_matrix, image_annotations

def getDepth(instance_mask, filedepth):
    if not os.path.exists(filedepth):
        return 0
    depth = np.array(Image.open(filedepth))
    depth = np.resize(depth, (512, 512))
    vals = depth[(instance_mask > 0)]
    return np.mean(vals) if vals.size > 0 else 0

def arrange_occlusion(matrices, filedepth):
    H, W, N = matrices.shape
    depths = []
    for i in range(N):
        mask = matrices[:, :, i]
        cid = int(np.max(mask)) if np.any(mask) else -1
        d = float("inf") if cid == 1 else getDepth(mask, filedepth)
        depths.append((i, d))
    sorted_idx = sorted(depths, key=lambda x: x[1])
    sorted_matrix = np.zeros_like(matrices)
    for new_idx, (old_idx, _) in enumerate(sorted_idx):
        sorted_matrix[:, :, new_idx] = matrices[:, :, old_idx]
    return sorted_matrix

def getVisibleObjects(matrix):
    H, W, N = matrix.shape
    visible = np.zeros_like(matrix)
    for i in range(N):
        current = (matrix[:, :, i] > 0).astype(np.uint8)
        for j in range(i):
            current = current * (1 - (matrix[:, :, j] > 0))
        visible[:, :, i] = current * np.max(matrix[:, :, i])
    return visible

def find_occluders(visible_mask, amodal_mask, sorted_matrices, current_idx):
    occluded = np.logical_and(amodal_mask > 0, visible_mask == 0)
    occluders = []
    if np.any(occluded):
        for j in range(current_idx):
            if np.any(np.logical_and(occluded, sorted_matrices[:, :, j] > 0)):
                occluders.append(int(np.max(sorted_matrices[:, :, j])))
    return occluders

def create_cocoa_format(coco_json: str, output_json: str, depth_dir: str):
    with open(coco_json, 'r') as f:
        data = json.load(f)

    images, annotations = [], []
    annotation_id = 1
    for img in data['images']:
        img_id = img['id']
        img_filename = img['file_name']
        file_name = os.path.splitext(img_filename)[0]

        images.append({
            "id": img_id,
            "file_name": img_filename,
            "depth_file": f"{file_name[:6]}_depth.png",
            "width": img['width'],
            "height": img['height']
        })

        amodal_matrix, original_annotations = create_class_object_matrix(file_name, coco_json)
        if amodal_matrix is None:
            continue

        sorted_matrix = arrange_occlusion(amodal_matrix, os.path.join(depth_dir, f"{file_name[:6]}_depth.png"))
        visible_matrix = getVisibleObjects(sorted_matrix)

        for i, ann in enumerate(original_annotations):
            amodal_mask = sorted_matrix[:, :, i]
            visible_mask = visible_matrix[:, :, i]
            if not np.any(amodal_mask):
                continue

            amodal_polygon = mask_to_polygon(amodal_mask)
            visible_polygon = mask_to_polygon(visible_mask) if np.any(visible_mask) else []
            x, y, w, h = ann['bbox']
            amodal_bbox = mask_to_bbox(x, y, w, h)

            annotations.append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": ann['category_id'],
                "amodal_mask": [amodal_polygon],
                "bbox": amodal_bbox,
                "visible_mask": [visible_polygon] if visible_polygon else [],
                "iscrowd": 0,
                "area": int(np.sum(amodal_mask > 0)),
                "visible_area": int(np.sum(visible_mask > 0)),
                "occluders": find_occluders(visible_mask, amodal_mask, sorted_matrix, i)
            })
            annotation_id += 1

    cocoa_data = {"images": images, "annotations": annotations, "categories": data['categories']}
    with open(output_json, 'w') as f:
        json.dump(cocoa_data, f, indent=2)
    print(f"✅ Saved COCOA format to {output_json} | {len(images)} images, {len(annotations)} annotations")

if __name__ == "__main__":
    create_cocoa_format(
        coco_json="Datasets/train/_annotations.coco.json",
        output_json="Datasets/train/cocoa_format_annotations.json",
        depth_dir="Datasets/train/depth"
    )
    create_cocoa_format(
        coco_json="Datasets/test/_annotations.coco.json",
        output_json="Datasets/test/cocoa_format_annotations.json",
        depth_dir="Datasets/test/depth"
    )
