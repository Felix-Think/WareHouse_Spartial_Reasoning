from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import cv2
import numpy as np


class WarehouseDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, img_size=(1080, 1920)):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.img_size = img_size

        # Lấy danh sách file ảnh
        self.image_files = list(self.images_dir.glob("*.png")) + list(
            self.images_dir.glob("*.jpg")
        )
        self.image_files = [f for f in self.image_files if self._has_label(f)]

    def _has_label(self, img_path):
        """Kiểm tra xem ảnh có file label tương ứng không"""
        label_name = img_path.stem + "_label.npz"
        return (self.labels_dir / label_name).exists()

    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label
        label_name = img_path.stem + "_label.npz"
        label_path = self.labels_dir / label_name
        label_data = np.load(label_path)
        mask = label_data["mask"]  # Shape: (H, W, 4)

        # Resize both
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # Normalize image before transform
        image /= 255.0

        # Apply Albumentations transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure image is Tensor and float32
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Ensure mask is Tensor and CHW
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        elif mask.ndim == 3 and mask.shape[0] != 4:  # may still be HWC
            mask = mask.permute(2, 0, 1).float()

        return image, mask

