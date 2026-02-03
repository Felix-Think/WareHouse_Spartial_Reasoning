from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split # <--- [ADD] Import random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Giả sử bạn đã định nghĩa class DistanceDataset và hàm build_mlp ở đâu đó trong code gốc
# Nếu chưa có, hãy đảm bảo import hoặc paste class đó vào trước hàm collate_fn

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

# Hàm này chỉ dùng để load Test set (từ file val cũ)
def build_test_loader(json_path: str, split: str, batch_size: int) -> DataLoader:
    dataset = DistanceDataset(json_path=json_path, split=split)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

def predict_distance(
    model: nn.Module,
    feats: torch.Tensor,
    fx_degree: float,
    fy_degree: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    # 1. FORWARD PASS & PARAMETER SCALING
    params = model(feats)  # (B, 3)
    
    k_r     = torch.nn.functional.softplus(params[:, 0]) + eps
    k_theta = torch.nn.functional.softplus(params[:, 1]) + eps
    k_phi   = torch.nn.functional.softplus(params[:, 2]) + eps

    # 2. FEATURE EXTRACTION & ANGLE MAPPING
    x1, y1, x2, y2, d1, d2 = feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3], feats[:, 4], feats[:, 5]

    r1 = k_r * d1
    r2 = k_r * d2

    rad_per_pixel_x = (math.pi / 180.0) * fx_degree * k_theta
    rad_per_pixel_y = (math.pi / 180.0) * fy_degree * k_phi

    theta1 = (y1 - 0.5) * rad_per_pixel_x
    theta2 = (y2 - 0.5) * rad_per_pixel_x

    phi1 = (0.5 - x1) * rad_per_pixel_y
    phi2 = (0.5 - x2) * rad_per_pixel_y

    # 3. SPHERICAL TO CARTESIAN
    cos_phi1, sin_phi1 = torch.cos(phi1), torch.sin(phi1)
    cos_theta1, sin_theta1 = torch.cos(theta1), torch.sin(theta1)

    cos_phi2, sin_phi2 = torch.cos(phi2), torch.sin(phi2)
    cos_theta2, sin_theta2 = torch.cos(theta2), torch.sin(theta2)

    X1 = r1 * cos_phi1 * sin_theta1
    Y1 = r1 * sin_phi1
    Z1 = r1 * cos_phi1 * cos_theta1

    X2 = r2 * cos_phi2 * sin_theta2
    Y2 = r2 * sin_phi2
    Z2 = r2 * cos_phi2 * cos_theta2

    # 4. EUCLIDEAN DISTANCE
    dX2 = (X1 - X2).square()
    dY2 = (Y1 - Y2).square()
    dZ2 = (Z1 - Z2).square()

    dist = torch.sqrt(dX2 + dY2 + dZ2 + eps)
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

        pred_dist = predict_distance(model, feats, fovx_degree, fovy_degree)
        target = labels.view(-1)

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
    criterion,
    device: torch.device,
    fovx_degree: float,
    fovy_degree: float,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_se = 0.0
    total_samples = 0

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            preds = predict_distance(model, feats, fovx_degree, fovy_degree).view(-1)

            # 1. Calculate Loss (Huber)
            loss = criterion(preds, labels)
            total_loss += float(loss.item()) * feats.size(0)

            # 2. Calculate RMSE
            se = F.mse_loss(preds, labels, reduction="sum")
            total_se += float(se.item())
            
            total_samples += feats.size(0)

    avg_loss = total_loss / max(1, total_samples)
    mse = total_se / max(1, total_samples)
    rmse = mse ** 0.5
    
    return avg_loss, rmse

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FOVX_DEG = 125.0
    FOVY_DEG = 50.0 
    input_dim = 10
    
    # --- SETUP MODEL ---
    # Giả sử hàm build_mlp đã có sẵn
    model = build_mlp(input_dim=input_dim, hidden_dims=(128, 64), out_dim=3, dropout=0.3).to(device)

    # --- PATHS ---
    train_json_path = "/kaggle/input/spatialdataset/train_dist_est.json"
    test_json_path = "/kaggle/input/spatialdataset/val_dist_est.json" # Dùng file val cũ làm Test
    batch_size = 256

    # --- 1. PREPARE DATASETS (SPLIT TRAIN -> TRAIN/VAL) ---
    print(f"Loading full training data from {train_json_path}...")
    full_train_dataset = DistanceDataset(json_path=train_json_path, split="train")
    
    # Tính toán kích thước split (ví dụ: 90% Train, 10% Val)
    total_size = len(full_train_dataset)
    val_size = int(total_size * 0.2) 
    train_size = total_size - val_size
    
    print(f"Splitting dataset: Total={total_size} -> Train={train_size}, Val={val_size}")
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # Cố định seed split để tái lập kết quả
    )

    # Tạo DataLoaders cho Train và Valid
    # Lưu ý: num_workers=0 để tránh lỗi multiprocessing
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Tạo DataLoader cho Test (dùng file val_json cũ)
    print(f"Loading test data from {test_json_path}...")
    test_loader = build_test_loader(test_json_path, split="val", batch_size=batch_size)

    # --- OPTIMIZER & LOSS ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- TRAINING LOOP ---
    best_rmse = float("inf")
    start_epoch = 1
    epochs = 300
    output_path = Path("best_model.pth")
    plot_path = Path("loss_plot.png")

    if output_path.exists():
        print(f"✓ Found existing model at {output_path}")
        checkpoint = torch.load(output_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        best_rmse = checkpoint.get("best_rmse", float("inf"))
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"✓ Resuming from epoch {start_epoch}, best RMSE={best_rmse:.4f}")
    else:
        print("✗ Starting training from scratch.")

    train_loss_history = []
    val_loss_history = []

    epoch_pbar = tqdm(range(start_epoch, epochs + 1), desc="epochs")
    for epoch in epoch_pbar:
        # 1. Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            fovx_degree=FOVX_DEG, fovy_degree=FOVY_DEG
        )

        # 2. Validate (dùng tập validation đã split từ train)
        val_loss, val_rmse = evaluate(
            model, val_loader, criterion, device,
            fovx_degree=FOVX_DEG, fovy_degree=FOVY_DEG
        )

        scheduler.step(val_rmse)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Save Best Model dựa trên Validation Set
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
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_rmse=f"{val_rmse:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    print(f"Training done. Best Val RMSE={best_rmse:.6f}. Saved to {output_path}")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Huber)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")

    # --- FINAL TESTING ---
    print("\n--- Running Final Evaluation on Test Set ---")
    # Load lại best model weights
    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    
    test_loss, test_rmse = evaluate(
        model, test_loader, criterion, device,
        fovx_degree=FOVX_DEG, fovy_degree=FOVY_DEG
    )
    
    print(f"Test Set Results (using val_dist_est.json):")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    main()