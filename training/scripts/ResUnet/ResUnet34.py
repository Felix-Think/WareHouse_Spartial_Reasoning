
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import yaml
from tqdm import tqdm

from .WarehouseDataset import WarehouseDataset


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResUNet34(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(ResUNet34, self).__init__()

        # Encoder (ResNet-34 backbone)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.encoder2 = self._make_layer(64, 64, 3, stride=1)
        self.encoder3 = self._make_layer(64, 128, 4, stride=2)
        self.encoder4 = self._make_layer(128, 256, 6, stride=2)
        self.encoder5 = self._make_layer(256, 512, 3, stride=2)

        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder5 = self._make_decoder_layer(
            768, 512
        )  # 512 from upconv + 256 from encoder4

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._make_decoder_layer(
            384, 256
        )  # 256 from upconv + 128 from encoder3

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_layer(
            192, 128
        )  # 128 from upconv + 64 from encoder2

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._make_decoder_layer(
            128, 64
        )  # 64 from upconv + 64 from encoder1

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final output layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 64
        e2 = self.encoder2(e1)  # 64
        e3 = self.encoder3(e2)  # 128
        e4 = self.encoder4(e3)  # 256
        e5 = self.encoder5(e4)  # 512

        # Bridge
        bridge = self.bridge(e5)  # 1024

        # Decoder
        d5 = self.upconv5(bridge)
        if d5.shape[-2:] != e4.shape[-2:]:
            e4 = F.interpolate(
                e4, size=d5.shape[-2:], mode="bilinear", align_corners=False
            )
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.decoder5(d5)

        d4 = self.upconv4(d5)
        if d4.shape[-2:] != e3.shape[-2:]:
            e3 = F.interpolate(
                e3, size=d4.shape[-2:], mode="bilinear", align_corners=False
            )
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        if d3.shape[-2:] != e2.shape[-2:]:
            e2 = F.interpolate(
                e2, size=d3.shape[-2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        if d2.shape[-2:] != e1.shape[-2:]:
            e1 = F.interpolate(
                e1, size=d2.shape[-2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = self.decoder1(d1)

        output = self.final(d1)
        return output


class ResUNetTrainer:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResUNet34(
            in_channels=self.config["model"]["in_channels"],
            num_classes=self.config["model"]["num_classes"],
        ).to(self.device)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=10, factor=0.5
        )

        # Data loaders
        self._setup_data_loaders()

    def _setup_data_loaders(self):
        # Training transforms
        train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                ToTensorV2(),
            ]
        )

        # Validation transforms
        val_transform = A.Compose([ToTensorV2()])

        # Datasets
        train_dataset = WarehouseDataset(
            images_dir=self.config["data"]["train_images"],
            labels_dir=self.config["data"]["train_labels"],
            transform=train_transform,
            img_size=tuple(self.config["data"]["img_size"]),
        )

        val_dataset = WarehouseDataset(
            images_dir=self.config["data"]["val_images"],
            labels_dir=self.config["data"]["val_labels"],
            transform=val_transform,
            img_size=tuple(self.config["data"]["img_size"]),
        )

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True,
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for images, masks in tqdm(self.train_loader, desc="Training"):
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float("inf")
        save_path = self.config["training"]["save_path"]
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(self.config["training"]["epochs"]):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")

            # Training
            train_loss = self.train_epoch()

            # Validation
            val_loss = self.validate()

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(
                    save_path, f"best_resunet34_epoch_{epoch+1}.pth"
                )
                self.save_checkpoint(best_model_path)
                print(f"New best model saved!")

            # Save checkpoint every N epochs
            if (epoch + 1) % self.config["training"]["save_every"] == 0:
                ckpt_path = os.path.join(
                    save_path, f"resunet34_epoch_{epoch+1}.pth"
                    )
                self.save_checkpoint(ckpt_path)

    def save_checkpoint(self, filename):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


if __name__ == "__main__":
    # Training
    trainer = ResUNetTrainer("training/configs/config_unet.yaml")
    trainer.train()
