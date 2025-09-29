import torch
from torch.utils.data import Dataset
import numpy as np
import cv2, json, os

def polygon_to_mask(polygon, height, width):
    # Đảm bảo polygon là list hợp lệ
    polygon = np.array(polygon).reshape((-1, 2))
    if polygon.shape[0] < 3:  # Ít hơn 3 điểm => không thể tạo polygon
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask

def coco_polygon_to_class_mask(amodal, visible, height, width):
    # Kiểm tra có polygon hợp lệ (>= 6 giá trị = 3 điểm)
    if isinstance(amodal, list) and len(amodal) > 0 and len(amodal[0]) >= 6:
        amodal_mask = polygon_to_mask(amodal[0], height, width)
    else:
        amodal_mask = np.zeros((height, width), dtype=np.uint8)

    if isinstance(visible, list) and len(visible) > 0 and len(visible[0]) >= 6:
        visible_mask = polygon_to_mask(visible[0], height, width)
    else:
        visible_mask = np.zeros((height, width), dtype=np.uint8)

    return amodal_mask, visible_mask

def prepare_target(image_id, annotations, height, width, valid_category_ids):
    amodal_masks, visible_masks, labels, boxes = [], [], [], []
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

    for ann in image_annotations:
        # ❌ Bỏ qua annotation có category_id không hợp lệ hoặc bằng 0
        if ann['category_id'] not in valid_category_ids or ann['category_id'] == 0:
            continue

        amodal_mask, visible_mask = coco_polygon_to_class_mask(
            ann.get('amodal_mask', []), ann.get('visible_mask', []), height, width
        )
        if np.any(amodal_mask):
            amodal_masks.append(amodal_mask)
            visible_masks.append(visible_mask)
            labels.append(ann['category_id'])
            x_min, y_min, x_max, y_max = ann['bbox']
            boxes.append([x_min, y_min, x_max, y_max])

    if len(amodal_masks) == 0:
        return None

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),  # Không cộng +1 nữa
        "masks_amodal": torch.tensor(np.stack(amodal_masks), dtype=torch.uint8),
        "masks_visible": torch.tensor(np.stack(visible_masks), dtype=torch.uint8),
        "image_id": torch.tensor([image_id])
    }

class COCOADataset(Dataset):
    def __init__(self, json_path, img_dir, depth_dir, transform=None):
        self.img_dir, self.depth_dir, self.transform = img_dir, depth_dir, transform
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.images = self.data['images']
        self.annotations = self.data['annotations']

        # ❌ Loại bỏ category có id = 0
        self.categories = [c for c in self.data['categories'] if c['id'] != 0]
        self.valid_category_ids = {c['id'] for c in self.categories}
        self.category_map = {c['id']: c['name'] for c in self.categories}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        anns = [a for a in self.annotations if a['image_id'] == img_info['id']]
        target = prepare_target(img_info['id'], anns, img_info['height'], img_info['width'], self.valid_category_ids)
        if target is None:
            target = {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks_amodal": torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8),
                "masks_visible": torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8),
                "image_id": torch.tensor([img_info['id']])
            }

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {
            "image": image,
            "boxes": target['boxes'],
            "amodal_masks": target['masks_amodal'],
            "visible_masks": target['masks_visible'],
            "labels": target['labels'],
            "image_id": target['image_id']
        }

def collate_fn(batch):
    images, targets = [], []
    for b in batch:
        if b['boxes'].shape[0] > 0:
            images.append(b['image'])
            targets.append({
                "boxes": b['boxes'], "labels": b['labels'],
                "masks": b['visible_masks'],  # Chỉ sử dụng visible masks cho training Mask R-CNNL:W
                "visible_masks": b['visible_masks'],
                "amodal_masks": b['amodal_masks']
            })
    if len(images) == 0:
        return [torch.zeros((3, 224, 224))], [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]
    return images, targets
