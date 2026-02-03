from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List, Dict
import math
import os
import random
import json
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Import from existing modules
from Dataloader import DistanceDataset
from MLP import build_mlp

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

def train_with_params(
    fovx_degree: float,
    fovy_degree: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    input_dim: int = 10,
    epochs: int = 50,
    lr: float = 1e-4,
    patience: int = 10,
) -> Tuple[float, float, nn.Module]:
    """
    Train a model with specific FOV parameters.
    
    Returns:
        - Best validation RMSE
        - Best validation loss
        - Best model state dict
    """
    # Create fresh model
    model = build_mlp(input_dim=input_dim, hidden_dims=(128, 64), out_dim=3, dropout=0.3).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    best_rmse = float("inf")
    best_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            fovx_degree=fovx_degree, fovy_degree=fovy_degree
        )
        
        # Validate
        val_loss, val_rmse = evaluate(
            model, val_loader, criterion, device,
            fovx_degree=fovx_degree, fovy_degree=fovy_degree
        )
        
        scheduler.step(val_rmse)
        
        # Save best
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_loss = val_loss
            best_state_dict = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    return best_rmse, best_loss, best_state_dict

def grid_search(
    train_json_path: str,
    fovx_range: Tuple[float, float, float],  # (start, end, step)
    fovy_range: Tuple[float, float, float],  # (start, end, step)
    val_split: float = 0.1,
    batch_size: int = 256,
    epochs_per_config: int = 50,
    device: torch.device = None,
    output_dir: str = "grid_search_results",
) -> Dict:
    """
    Perform grid search using a single train/validation split.
    
    Args:
        train_json_path: Path to training data JSON
        fovx_range: (start, end, step) for fovx_degree
        fovy_range: (start, end, step) for fovy_degree
        val_split: Fraction of data to use for validation (default: 0.1)
        batch_size: Batch size for training
        epochs_per_config: Max epochs per configuration
        device: Device to train on
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing results and best parameters
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate parameter grid
    fovx_values = np.arange(fovx_range[0], fovx_range[1] + fovx_range[2], fovx_range[2])
    fovy_values = np.arange(fovy_range[0], fovy_range[1] + fovy_range[2], fovy_range[2])
    param_grid = list(itertools.product(fovx_values, fovy_values))
    
    print(f"Grid Search Configuration:")
    print(f"  FOVX range: {fovx_values}")
    print(f"  FOVY range: {fovy_values}")
    print(f"  Total combinations: {len(param_grid)}")
    print(f"  Validation split: {val_split*100:.0f}%")
    print(f"  Epochs per config: {epochs_per_config}")
    print(f"  Device: {device}\n")
    
    # Load and split dataset once
    print(f"Loading dataset from {train_json_path}...")
    full_dataset = DistanceDataset(json_path=train_json_path, split="train")
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    print(f"Dataset split: Total={total_size} → Train={train_size}, Val={val_size}")
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    
    # Results storage
    results = []
    best_score = float("inf")
    best_params = None
    best_model_state = None
    
    # Iterate through parameter combinations
    param_pbar = tqdm(param_grid, desc="Grid Search")
    for fovx, fovy in param_pbar:
        param_pbar.set_postfix(fovx=f"{fovx:.1f}", fovy=f"{fovy:.1f}")
        
        print(f"\n{'='*60}")
        print(f"Testing FOVX={fovx:.1f}°, FOVY={fovy:.1f}°")
        print(f"{'='*60}")
        
        # Train with these parameters
        val_rmse, val_loss, state_dict = train_with_params(
            fovx_degree=fovx,
            fovy_degree=fovy,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs_per_config,
        )
        
        result = {
            "fovx": float(fovx),
            "fovy": float(fovy),
            "val_rmse": float(val_rmse),
            "val_loss": float(val_loss),
        }
        results.append(result)
        
        print(f"\n  Validation RMSE: {val_rmse:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        
        # Track best parameters
        if val_rmse < best_score:
            best_score = val_rmse
            best_params = (fovx, fovy)
            best_model_state = state_dict
            print(f"  ✓ New best parameters!")
    
    # Save results
    results_dict = {
        "best_params": {
            "fovx": float(best_params[0]),
            "fovy": float(best_params[1]),
            "val_rmse": float(best_score),
        },
        "all_results": results,
        "config": {
            "fovx_range": fovx_range,
            "fovy_range": fovy_range,
            "val_split": val_split,
            "batch_size": batch_size,
            "epochs_per_config": epochs_per_config,
        }
    }
    
    # Save to JSON
    results_file = output_path / "grid_search_results.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")
    
    # Save best model
    if best_model_state is not None:
        best_model_path = output_path / "best_model_gridsearch.pth"
        torch.save({
            "state_dict": best_model_state,
            "fovx_degree": best_params[0],
            "fovy_degree": best_params[1],
            "val_rmse": best_score,
        }, best_model_path)
        print(f"✓ Best model saved to {best_model_path}")
    
    # Create visualization
    plot_grid_search_results(results, output_path / "grid_search_heatmap.png")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best parameters:")
    print(f"  FOVX: {best_params[0]:.1f}°")
    print(f"  FOVY: {best_params[1]:.1f}°")
    print(f"  Validation RMSE: {best_score:.4f}")
    print(f"{'='*60}\n")
    
    return results_dict

def plot_grid_search_results(results: List[Dict], output_path: Path) -> None:
    """Create heatmap visualization of grid search results."""
    # Extract unique FOV values
    fovx_values = sorted(list(set([r["fovx"] for r in results])))
    fovy_values = sorted(list(set([r["fovy"] for r in results])))
    
    # Create result matrix
    result_matrix = np.zeros((len(fovy_values), len(fovx_values)))
    
    for result in results:
        i = fovy_values.index(result["fovy"])
        j = fovx_values.index(result["fovx"])
        result_matrix[i, j] = result["val_rmse"]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(result_matrix, cmap="viridis_r", aspect="auto")
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Mean RMSE", rotation=270, labelpad=20)
    
    # Set ticks and labels
    plt.xticks(range(len(fovx_values)), [f"{v:.0f}" for v in fovx_values])
    plt.yticks(range(len(fovy_values)), [f"{v:.0f}" for v in fovy_values])
    
    plt.xlabel("FOVX (degrees)")
    plt.ylabel("FOVY (degrees)")
    plt.title("Grid Search Results: Validation RMSE")
    
    # Add text annotations
    for i in range(len(fovy_values)):
        for j in range(len(fovx_values)):
            text = plt.text(j, i, f"{result_matrix[i, j]:.3f}",
                          ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Heatmap saved to {output_path}")

def main() -> None:
    set_seed(42)
    
    # Configuration
    train_json_path = "/kaggle/input/spatialdataset/train_dist_est.json"
    
    # Define search ranges (start, end, step) - all in degrees
    # Example: search FOVX from 100 to 140 in steps of 5
    # Example: search FOVY from 50 to 90 in steps of 5
    fovx_range = (100.0, 140.0, 5.0)
    fovy_range = (50.0, 90.0, 5.0)
    
    # Run grid search
    results = grid_search(
        train_json_path=train_json_path,
        fovx_range=fovx_range,
        fovy_range=fovy_range,
        val_split=0.1,
        batch_size=256,
        epochs_per_config=50,
        output_dir="grid_search_results",
    )
    
    print("\nBest parameters found:")
    print(f"  FOVX: {results['best_params']['fovx']:.1f}°")
    print(f"  FOVY: {results['best_params']['fovy']:.1f}°")
    print(f"  Validation RMSE: {results['best_params']['val_rmse']:.4f}")

if __name__ == "__main__":
    main()
