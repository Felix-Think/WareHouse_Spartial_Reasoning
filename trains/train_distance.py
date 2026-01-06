from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Dataloader import DistanceDataset
from MLP import build_mlp


def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack features and labels into tensors."""
    feats = []
    labels = []
    for x1, y1, x2, y2, depth1, depth2, x_max, y_max, label in batch:
        feats.append(
            [
                x1 / max(1, x_max),
                y1 / max(1, y_max),
                x2 / max(1, x_max),
                y2 / max(1, y_max),
                depth1,
                depth2,
                np.float32(1),
                np.float32(1),
            ]
        )
        labels.append(label)
    return (
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


def build_loader(json_path: str, split: str, shuffle: bool | None, batch_size: int) -> DataLoader:
    dataset = DistanceDataset(json_path=json_path, split=split)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn
    )


def pixel_to_theta_rad(
    x: torch.Tensor,
    width: torch.Tensor,
    theta_max_rad: torch.Tensor,
    theta_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    cx = width / 2
    theta = (x - cx) / cx * theta_max_rad
    if theta_weights is not None:
        theta = theta * theta_weights
    return theta


def pixel_to_phi_rad(
    y: torch.Tensor,
    height: torch.Tensor,
    phi_max_rad: torch.Tensor,
    phi_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    cy = height / 2
    phi = (y - cy) / cy * phi_max_rad
    return phi * phi_weights if phi_weights is not None else phi


def predict_distance(model: nn.Module, feats: torch.Tensor) -> torch.Tensor:
    """Compute distances using the same transform as the training loop."""
    params = model(feats)
    x1, y1, x2, y2, depth_1, depth_2, x_max, y_max = feats.unbind(dim=1)

    depth_weight = params[:, 0]
    phi_max = params[:, 1]
    theta_max = params[:, 2]
    phi_weight = params[:, 3]
    theta_weight = params[:, 4]

    theta1 = pixel_to_theta_rad(x1, x_max, theta_max_rad=theta_max, theta_weights=theta_weight)
    theta2 = pixel_to_theta_rad(x2, x_max, theta_max_rad=theta_max, theta_weights=theta_weight)
    phi1 = pixel_to_phi_rad(y1, y_max, phi_max_rad=phi_max, phi_weights=phi_weight)
    phi2 = pixel_to_phi_rad(y2, y_max, phi_max_rad=phi_max, phi_weights=phi_weight)
    r1 = depth_1 * depth_weight
    r2 = depth_2 * depth_weight

    x1c = r1 * torch.cos(phi1) * torch.sin(theta1)
    y1c = r1 * torch.sin(phi1)
    z1c = r1 * torch.cos(phi1) * torch.cos(theta1)

    x2c = r2 * torch.cos(phi2) * torch.sin(theta2)
    y2c = r2 * torch.sin(phi2)
    z2c = r2 * torch.cos(phi2) * torch.cos(theta2)

    distance = torch.sqrt((x1c - x2c) ** 2 + (y1c - y2c) ** 2 + (z1c - z2c) ** 2)
    return distance


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="train", leave=False)

    for feats, labels in pbar:
        feats = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        distance = predict_distance(model, feats)
        outputs = distance.unsqueeze(1)
        target = labels.unsqueeze(1)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        bs = feats.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        avg_loss = total_loss / max(1, total_samples)
        pbar.set_postfix(loss=float(loss.item()), avg_loss=float(avg_loss))

    return total_loss / max(1, total_samples)


def evaluate(model, loader, device) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            preds = predict_distance(model, feats)
            total_mse += mse(preds, labels).item()
            total_samples += feats.size(0)

    return total_mse / max(1, total_samples)

def set_seed(seed: int = 42) -> None:
    # Python / OS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main() -> None:
    set_seed(42)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = build_mlp(input_dim=8, hidden_dims=(128,64), out_dim=5, dropout=0.3)
    model.to(device)
    train_json = "PhysicalAI_Warehouse/train_distance.json"
    val_json = "PhysicalAI_Warehouse/val_distance.json"
    batch_size = 64
    train_loader = build_loader(train_json, split="train", batch_size=batch_size, shuffle=True)
    val_loader = build_loader(val_json, split="val", batch_size=batch_size,shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_mse = float("inf")
    epochs = 10
    output_path = Path("best_model.pth")

    start_epoch = 1
    if output_path.exists():
        ckpt = torch.load(output_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            best_mse = float(ckpt.get("best_mse", best_mse))
            print(f"Loaded checkpoint from {output_path} (best_mse={best_mse:.6f})")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded state_dict from {output_path} (best_mse unknown).")

    epoch_pbar = tqdm(range(start_epoch, epochs + 1), desc="epochs")
    for epoch in epoch_pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mse = evaluate(model, val_loader, device)
        scheduler.step(val_mse)

        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_mse": best_mse,
                    "epoch": epoch,
                },
                output_path,
            )

        epoch_pbar.set_postfix(train_loss=float(train_loss), val_mse=float(val_mse), best_mse=float(best_mse))

    print(f"Training done. Best val MSE={best_mse:.6f}. Saved to {output_path}")

import os
import random


if __name__ == "__main__":
    main()
