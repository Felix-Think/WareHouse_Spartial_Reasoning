from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import torch.nn as nn
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from Dataloader import DistanceDataset
from MLP import build_mlp
import os
import random

def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack features and labels into tensors."""
    feats = []
    labels = []
    for x1, y1, x2, y2, depth1, depth2, w1, h1, w2, h2, label in batch:
        feats.append(
            [
                x1, y1, x2, y2, depth1, depth2, w1, h1, w2, h2
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

def predict_distance(
    model: nn.Module,
    feats: torch.Tensor,
    fx_degree: float,
    fy_degree: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    feats: (B, 10) = [x1, y1, x2, y2, depth1, depth2, w1, h1, w2, h2]
      - x = row_norm (0..1), y = col_norm (0..1) theo convention của bạn
      - depth1, depth2 = depth_rel (0..1)
    fx_degree, fy_degree: FOV_x, FOV_y (degrees)
    Return: distance_pred (B,)
    """

    params = model(feats)  # (B, 1) 
    # đảm bảo scale dương (khuyên dùng)
    # nếu model đã output dương thì có thể bỏ softplus
    s = torch.nn.functional.softplus(params[:, 0]) + eps

    x1, y1, x2, y2, depth1, depth2, w1, h1, w2, h2 = feats.unbind(dim=1)

    # depth mét
    d1 = s * depth1
    d2 = s * depth2

    # FOV -> hệ số quy đổi
    fovx = math.radians(fx_degree)
    fovy = math.radians(fy_degree)
    kx = 2.0 * math.tan(fovx / 2.0)
    ky = 2.0 * math.tan(fovy / 2.0)

    # camera normalized coords
    # ngang (x_cam) lấy từ y_norm, dọc (y_cam) lấy từ x_norm (vì x=row, y=col)
    x_cam_1 = (y1 - 0.5) * kx
    y_cam_1 = (x1 - 0.5) * ky
    x_cam_2 = (y2 - 0.5) * kx
    y_cam_2 = (x2 - 0.5) * ky

    # 3D points in camera coords: P = d * [x_cam, y_cam, 1]
    X1 = d1 * x_cam_1
    Y1 = d1 * y_cam_1
    Z1 = d1

    X2 = d2 * x_cam_2
    Y2 = d2 * y_cam_2
    Z2 = d2

    dist = torch.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2 + (Z1 - Z2) ** 2 + eps)
    return dist



def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    fovx_degree: float,
    fovy_degree: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="train", leave=False)

    for feats, labels in pbar:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        # predict_distance returns (B,)
        pred_dist = predict_distance(model, feats, fovx_degree, fovy_degree)  # (B,)

        # labels should be (B,) distance in meters
        target = labels.view(-1)  # (B,)

        loss = criterion(pred_dist, target)
        loss.backward()
        optimizer.step()

        bs = feats.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

        avg_loss = total_loss / max(1, total_samples)
        pbar.set_postfix(loss=float(loss.item()), avg_loss=float(avg_loss))

    return total_loss / max(1, total_samples)


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    fovx_degree: float,
    fovy_degree: float,
) -> float:
    model.eval()
    total_se = 0.0
    total_samples = 0

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            preds = predict_distance(model, feats, fovx_degree, fovy_degree).view(-1)

            # sum squared error
            se = F.mse_loss(preds, labels, reduction="sum")
            total_se += float(se.item())
            total_samples += feats.size(0)

    mse = total_se / max(1, total_samples)
    rmse = mse ** 0.5
    return rmse

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

from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn

def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== Hyperparameter: FOV (treat as fixed) ======
    # Ví dụ: bạn chọn thử FOV ngang 120°, dọc 70° (tuỳ camera)
    FOVX_DEG = 120.0
    FOVY_DEG = 70.0

    # ====== Model: output 1 scale s ======
    # input_dim phải khớp với feats bạn trả về từ dataset
    # nếu feats = [x1,y1,x2,y2,depth1,depth2,w1,h1,w2,h2] => 10
    input_dim = 10
    model = build_mlp(input_dim=input_dim, hidden_dims=(128, 64), out_dim=1, dropout=0.3).to(device)

    train_json = "data/train_dist_est.json"
    val_json = "data/val_dist_est.json"
    batch_size = 64

    train_loader = build_loader(train_json, split="train", batch_size=batch_size, shuffle=True)
    val_loader = build_loader(val_json, split="val", batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Train: Huber
    criterion = nn.HuberLoss(delta=1.0)

    # Scheduler theo metric val_rmse
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_rmse = float("inf")
    epochs = 10
    output_path = Path("best_model.pth")

    epoch_pbar = tqdm(range(1, epochs + 1), desc="epochs")
    for epoch in epoch_pbar:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            fovx_degree=FOVX_DEG, fovy_degree=FOVY_DEG
        )

        val_rmse = evaluate(
            model, val_loader, device,
            fovx_degree=FOVX_DEG, fovy_degree=FOVY_DEG
        )

        scheduler.step(val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_rmse": best_rmse,
                    "epoch": epoch,
                    "fovx_degree": FOVX_DEG,
                    "fovy_degree": FOVY_DEG,
                    "input_dim": input_dim,
                },
                output_path,
            )

        epoch_pbar.set_postfix(
            train_loss=float(train_loss),
            val_rmse=float(val_rmse),
            best_rmse=float(best_rmse),
            lr=float(optimizer.param_groups[0]["lr"]),
        )

    print(f"Training done. Best val RMSE={best_rmse:.6f}. Saved to {output_path}")




if __name__ == "__main__":
    main()
