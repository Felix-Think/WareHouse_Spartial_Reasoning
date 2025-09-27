import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
import os
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold

def polygon_to_mask(polygon, height, width):
    polygon = np.array(polygon).reshape((-1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask

def coco_polygon_to_class_mask(amodal, visible, height, width, class_id):
    amodal_mask = np.zeros((height, width), dtype=np.uint8)
    visible_mask = np.zeros((height, width), dtype=np.uint8)

    if amodal:
        amodal_binary = polygon_to_mask(amodal, height, width)
        amodal_mask[amodal_binary == 1] = 1

    if visible:
        visible_binary = polygon_to_mask(visible, height, width)
        visible_mask[visible_binary == 1] = 1

    return amodal_mask, visible_mask

def prepare_target(image_id, annotations, height, width):
    amodal_masks, visible_masks, labels, boxes = [], [], [], []

    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    if len(image_annotations) == 0:
        return None

    for ann in image_annotations:
        class_id = ann['category_id']
        bbox = ann['bbox']  # format COCO [x,y,w,h]

        amodal_mask, visible_mask = coco_polygon_to_class_mask(
            ann.get('amodal_mask', []),
            ann.get('visible_mask', []),
            height,
            width,
            class_id
        )

        if np.any(amodal_mask) or np.any(visible_mask):
            amodal_masks.append(amodal_mask)
            visible_masks.append(visible_mask)
            labels.append(class_id)
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])

    if len(amodal_masks) == 0:
        return None

    # ✅ Chuyển list -> numpy array một lần
    amodal_masks = np.stack(amodal_masks, axis=0).astype(np.uint8)
    visible_masks = np.stack(visible_masks, axis=0).astype(np.uint8)

    target = {
        "boxes": torch.from_numpy(np.array(boxes, dtype=np.float32)),
        "labels": torch.from_numpy(np.array(labels, dtype=np.int64)),
        "masks_amodal": torch.from_numpy(amodal_masks),
        "masks_visible": torch.from_numpy(visible_masks),
        "image_id": torch.tensor([image_id])
    }
    return target


class COCOADataset(Dataset):
    def __init__(self, json_path, img_dir, depth_dir, transform=None):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.transform = transform

        with open(json_path, 'r') as f:
            self.cocoa_data = json.load(f)

        self.images = self.cocoa_data['images']
        self.annotations = self.cocoa_data['annotations']

        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        depth_filename = img_info.get('depth_file', None)
        height, width = img_info['height'], img_info['width']

        # Load image
        img_path = os.path.join(self.img_dir, img_filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load depth (optional)
        depth = np.zeros((height, width), dtype=np.uint16)
        if depth_filename:
            depth_path = os.path.join(self.depth_dir, depth_filename)
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

        # Targets
        anns = self.img_to_anns.get(img_id, [])
        target = prepare_target(img_id, anns, height, width)

        if target is None:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks_amodal": torch.zeros((0, height, width), dtype=torch.uint8),
                "masks_visible": torch.zeros((0, height, width), dtype=torch.uint8),
                "image_id": torch.tensor([img_id])
            }

        # Convert image + depth
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0).float() / 65535.0

        return {
            'image': image,
            'depth': depth,
            'boxes': target['boxes'],
            'amodal_masks': target['masks_amodal'],
            'visible_masks': target['masks_visible'],
            'labels': target['labels'],
            'image_id': target['image_id']
        }

def collate_fn(batch):
    """
    Collate function that properly handles the batch format from your dataloader
    """
    images = []
    targets = []
    
    for item in batch:
        # ✅ Your dataloader returns dict with these keys
        if item['boxes'].shape[0] > 0:  # Only include items with annotations
            images.append(item['image'])
            targets.append({
                'boxes': item['boxes'],
                'labels': item['labels'],
                'masks': item['visible_masks'],  # Standard PyTorch key
                'visible_masks': item['visible_masks'],
                'amodal_masks': item['amodal_masks']
            })
    
    # Handle empty batch case
    if len(images) == 0:
        dummy_image = torch.zeros((3, 224, 224))
        dummy_target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'masks': torch.zeros((0, 224, 224), dtype=torch.uint8),
            'visible_masks': torch.zeros((0, 224, 224), dtype=torch.uint8),
            'amodal_masks': torch.zeros((0, 224, 224), dtype=torch.uint8)
        }
        return [dummy_image], [dummy_target]
    
    return images, targets



# Example
# def create_kfold_loaders(json_path, img_dir, depth_dir, k=5, batch_size=2, shuffle=True, num_workers=2):
#     """
#     Tạo train/val dataloaders cho KFold cross-validation.
#     """
#     full_dataset = COCOADataset(
#         json_path=json_path,
#         img_dir=img_dir,
#         depth_dir=depth_dir
#     )

#     kfold = KFold(n_splits=k, shuffle=True, random_state=42)
#     folds = []

#     for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(full_dataset)))):
#         train_subset = Subset(full_dataset, train_idx)
#         val_subset = Subset(full_dataset, val_idx)

#         train_loader = DataLoader(
#             train_subset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             collate_fn=collate_fn
#         )

#         val_loader = DataLoader(
#             val_subset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             collate_fn=collate_fn
#         )

#         folds.append((train_loader, val_loader))

#     return folds

# ---------- Ví dụ sử dụng ----------
if __name__ == "__main__":
    cocoa_json_path = "Datasets/train/cocoa_format_annotations.json"
    img_dir = "Datasets/train/images"
    depth_dir = "Datasets/train/depths"

    # folds = create_kfold_loaders(
    #     json_path=cocoa_json_path,
    #     img_dir=img_dir,
    #     depth_dir=depth_dir,
    #     k=5,
    #     batch_size=2
    # )

    # # Test fold 1
    # train_loader, val_loader = folds[0]
    # print(f"Fold 1 - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # for batch in train_loader:
    #     print(np.unique(batch[0]["visible_masks"]))
    #     break
    Datasets = COCOADataset(
        json_path=cocoa_json_path,
        img_dir=img_dir,
        depth_dir=depth_dir
    )
    dataloader = DataLoader(
        Datasets,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    for batch in dataloader:
        print(batch[0]["image"].shape)
        print(batch[0]["visible_masks"].shape)
        print(batch[0]["amodal_masks"].shape)
        print(batch[0]["boxes"].shape)
        print(batch[0]["labels"].shape)
        break