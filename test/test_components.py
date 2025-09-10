import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from training.scripts.ResUnet.WarehouseDataset import WarehouseDataset
from training.scripts.ResUnet.ResUnet34 import ResUNet34


def test_dataset():
    """Test if dataset loads correctly"""
    print("Testing WarehouseDataset...")

    # Dataset paths
    # images_dir = "Datasets/test/images"
    # labels_dir = "Datasets/labels"
    with open("training/configs/config_unet.yaml", "r") as f:
        config = yaml.safe_load(f)
    images_dir = config["data"]["train_images"]
    labels_dir = config["data"]["train_labels"]
    # Create dataset
    dataset = WarehouseDataset(
        images_dir=images_dir, labels_dir=labels_dir, img_size=(512, 512)
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")

        # Debug info
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
        print(f"Found {len(image_files)} image files")

        if len(image_files) > 0:
            print("Sample image files:")
            for i, img_file in enumerate(image_files[:3]):
                print(f"  {img_file.name}")
                label_name = img_file.stem + "_label.npz"
                label_path = labels_path / label_name
                print(f"    Looking for: {label_name} -> Exists: {label_path.exists()}")

        return False

    # Test loading one sample
    try:
        image, mask = dataset[0]
        print(f"Sample loaded successfully!")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Mask dtype: {mask.dtype}")
        return True
    except Exception as e:
        print(f"ERROR loading sample: {e}")
        return False


def test_dataloader():
    """Test if DataLoader works"""
    print("\nTesting DataLoader...")

    dataset = WarehouseDataset(
        images_dir="Datasets/test/images",
        labels_dir="Datasets/labels",
        img_size=(512, 512),
    )

    if len(dataset) == 0:
        print("Cannot test DataLoader with empty dataset")
        return False

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    try:
        for batch_idx, (images, masks) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Images batch shape: {images.shape}")
            print(f"  Masks batch shape: {masks.shape}")
            break  # Only test first batch
        print("DataLoader works!")
        return True
    except Exception as e:
        print(f"ERROR with DataLoader: {e}")
        return False


def test_model():
    """Test if model works"""
    print("\nTesting ResUNet34 model...")

    try:
        model = ResUNet34(in_channels=3, num_classes=4)

        # Test forward pass
        dummy_input = torch.randn(1, 3, 512, 512)
        output = model(dummy_input)

        print(f"Model output shape: {output.shape}")
        print("Model works!")
        return True
    except Exception as e:
        print(f"ERROR with model: {e}")
        return False


if __name__ == "__main__":
    print("=== Testing Warehouse Segmentation Components ===\n")

    dataset_ok = test_dataset()
    dataloader_ok = test_dataloader() if dataset_ok else False
    model_ok = test_model()

    print(f"\n=== Test Results ===")
    print(f"Dataset: {'✓' if dataset_ok else '✗'}")
    print(f"DataLoader: {'✓' if dataloader_ok else '✗'}")
    print(f"Model: {'✓' if model_ok else '✗'}")

    if dataset_ok and dataloader_ok and model_ok:
        print("\n🎉 All components work! Ready for training.")
    else:
        print("\n❌ Some components have issues. Please fix before training.")
