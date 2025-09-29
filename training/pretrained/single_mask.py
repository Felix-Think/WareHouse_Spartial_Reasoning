import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from tqdm import tqdm

# Import dataloader từ file bạn cung cấp
from torch.utils.data import DataLoader
import json

class MaskRCNNModel:
    def __init__(self, num_classes, pretrained=True, device=None):
        """
        Khởi tạo Mask R-CNN model cho COCOA dataset
        
        Args:
            num_classes (int): Số lượng classes (bao gồm background)
            pretrained (bool): Sử dụng pretrained weights
            device: Device để train/inference
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load pretrained Mask R-CNN
        self.model = maskrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Thay đổi classifier cho số classes của bạn
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Thay đổi mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        self.model.to(self.device)
        
        # Optimizer và scheduler
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self, learning_rate=1e-4, weight_decay=1e-4):
        """
        Thiết lập optimizer AdamW và CosineAnnealingLR scheduler.
        AdamW giúp hội tụ ổn định hơn SGD, đặc biệt khi fine-tune.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]

        # ✅ AdamW thay cho SGD
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

        # ✅ CosineAnnealingLR: giảm LR mượt theo dạng cosine, tránh overfitting cuối training
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,  # số epoch trước khi LR về gần 0 (bạn có thể tăng = mask_epochs)
            eta_min=learning_rate * 0.1  # LR nhỏ nhất
        )

        print(f"🔧 Optimizer: AdamW | LR={learning_rate}, Weight Decay={weight_decay}")
        print(f"🔧 Scheduler: CosineAnnealingLR (T_max=10)")




    def validate(self, dataloader):
        """
        Validate model: luôn trả về số thực (float), inf nếu không có batch hợp lệ
        """
        self.model.train()  # để model trả loss
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating")
            for images, targets in pbar:
                try:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]

                    loss_dict = self.model(images, targets)
                    if not isinstance(loss_dict, dict):
                        continue

                    losses = sum(loss for loss in loss_dict.values())
                    total_loss += float(losses.item())
                    num_batches += 1
                    pbar.set_postfix({'Val Loss': f'{losses.item():.4f}'})
                except Exception as e:
                    print(f"[validate] Skip batch do lỗi: {e}")
                    continue

        self.model.eval()
        return (total_loss / num_batches) if num_batches > 0 else float('inf')

        
    def evaluate_model(self, dataloader, confidence_threshold=0.3):
        """
        Evaluate model performance (không cần loss)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating")
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                predictions = self.model(images)

                for pred in predictions:
                    scores = pred['scores'].cpu().numpy()
                    valid_indices = scores >= confidence_threshold  # ✅ lọc theo threshold

                    filtered_pred = {
                        'boxes': pred['boxes'][valid_indices].cpu().numpy(),
                        'labels': pred['labels'][valid_indices].cpu().numpy(),
                        'scores': pred['scores'][valid_indices].cpu().numpy(),
                        'masks': pred['masks'][valid_indices].cpu().numpy()
                    }
                    all_predictions.append(filtered_pred)

                for target in targets:
                    all_targets.append({
                        'boxes': target['boxes'].cpu().numpy(),
                        'labels': target['labels'].cpu().numpy(),
                        'masks': target['visible_masks'].cpu().numpy()
                    })

        return all_predictions, all_targets

    
    

    
    def predict(self, images, confidence_threshold=0.3):
        """
        Thực hiện prediction
        
        Args:
            images: List of tensors hoặc single tensor
            confidence_threshold: Ngưỡng confidence
            
        Returns:
            List of predictions
        """
        self.model.eval()
        
        if not isinstance(images, list):
            images = [images]
        
        images = [img.to(self.device) for img in images]
        
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter by confidence
        filtered_predictions = []
        for pred in predictions:
            scores = pred['scores'].cpu().numpy()
            valid_indices = scores >= confidence_threshold
            
            filtered_pred = {
                'boxes': pred['boxes'][valid_indices].cpu().numpy(),
                'labels': pred['labels'][valid_indices].cpu().numpy(),
                'scores': pred['scores'][valid_indices].cpu().numpy(),
                'masks': pred['masks'][valid_indices].cpu().numpy()
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def create_visible_mask_overlay(self, image, prediction, alpha=0.5):
        """
        Tạo visible mask overlay trên ảnh
        
        Args:
            image: Tensor image (C, H, W)
            prediction: Prediction dictionary
            alpha: Độ trong suốt của mask
            
        Returns:
            numpy array: Ảnh với mask overlay
        """
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:  # (C, H, W)
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:  # (H, W, C)
                image_np = image.cpu().numpy()
        else:
            image_np = image
        
        # Ensure image is in [0, 1] range
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        result = image_np.copy()
        
        # Generate random colors for each instance
        colors = np.random.rand(len(prediction['masks']), 3) * 255
        
        # Apply masks
        for i, mask in enumerate(prediction['masks']):
            # Mask shape is (1, H, W), take first channel
            binary_mask = mask[0] > 0.5
            color = colors[i].astype(np.uint8)
            
            # Apply colored mask
            for c in range(3):
                result[:, :, c] = np.where(
                    binary_mask,
                    result[:, :, c] * (1 - alpha) + color[c] * alpha,
                    result[:, :, c]
                )
        
        # Draw bounding boxes and labels
        for i, (box, label, score) in enumerate(zip(
            prediction['boxes'],
            prediction['labels'], 
            prediction['scores']
        )):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[i].astype(int).tolist()
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, f"Class {label}: {score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result
    
    def save_model(self, filepath):
        """
        Lưu model an toàn (tạo thư mục nếu thiếu, ghi file tạm rồi rename)
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_path = filepath + ".tmp"

        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'num_classes': self.num_classes
        }
        torch.save(state, tmp_path)
        os.replace(tmp_path, filepath)  # atomic trên cùng filesystem
        print(f"Model saved to {filepath}")

    
    def load_model(self, filepath, load_optimizer=True):
        checkpoint = torch.load(filepath, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            if load_optimizer and self.optimizer and checkpoint.get("optimizer_state_dict"):
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    print("✅ Optimizer state loaded.")
                except KeyError as e:
                    print(f"⚠️ Optimizer state không tương thích ({e}), bỏ qua...")
            else:
                print("ℹ️ Bỏ qua optimizer state.")
        else:
            print("⚠️ Checkpoint không có 'model_state_dict', load trực tiếp state_dict...")
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)

        print(f"✅ Model loaded from {filepath} | Device: {next(self.model.parameters()).device}")






def visualize_predictions(model, dataloader, num_samples=5, save_dir="predictions"):
    """
    Visualize predictions trên một số samples, hiển thị cả mask dự đoán
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.model.eval()
    count = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            if count >= num_samples:
                break
                
            for i, (image, target) in enumerate(zip(images, targets)):
                if count >= num_samples:
                    break
                
                # Get prediction
                predictions = model.predict([image])
                prediction = predictions[0]

                # Create visualization (overlay)
                vis_image = model.create_visible_mask_overlay(image, prediction)

                # ====== Lấy mask prediction tổng hợp để hiển thị ======
                pred_mask_combined = None
                if len(prediction['masks']) > 0:
                    pred_masks = prediction['masks'][:, 0, :, :] > 0.3  # numpy boolean
                    pred_mask_combined = np.sum(pred_masks, axis=0)
                
                # Save
                plt.figure(figsize=(20, 5))
                
                # Original image
                plt.subplot(1, 4, 1)
                orig_img = image.permute(1, 2, 0).cpu().numpy()
                if orig_img.max() <= 1.0:
                    orig_img = (orig_img * 255).astype(np.uint8)
                plt.imshow(orig_img)
                plt.title("Original Image")
                plt.axis('off')
                
                # Ground truth masks
                plt.subplot(1, 4, 2)
                if len(target['visible_masks']) > 0:
                    gt_mask = torch.sum(target['visible_masks'], dim=0).cpu().numpy()
                    plt.imshow(orig_img)
                    plt.imshow(gt_mask, alpha=0.5, cmap='jet')
                    plt.title("Ground Truth Visible Masks")
                else:
                    plt.imshow(orig_img)
                    plt.title("No Ground Truth")
                plt.axis('off')

                # Predicted mask (binary)
                plt.subplot(1, 4, 3)
                if pred_mask_combined is not None:
                    plt.imshow(orig_img)
                    plt.imshow(pred_mask_combined, alpha=0.5, cmap='jet')
                    plt.title(f"Predicted Mask ({len(prediction['masks'])} objs)")
                else:
                    plt.imshow(orig_img)
                    plt.title("No Prediction")
                plt.axis('off')
                
                # Overlay with boxes
                plt.subplot(1, 4, 4)
                plt.imshow(vis_image)
                plt.title("Overlay + Boxes")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/sample_{count}.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                count += 1




# Ví dụ sử dụng với dataloader có sẵn
def freeze_mask_head(model):
    for p in model.model.roi_heads.mask_head.parameters():
        p.requires_grad = False
    for p in model.model.roi_heads.mask_predictor.parameters():
        p.requires_grad = False

def unfreeze_mask_head(model):
    for p in model.model.roi_heads.mask_head.parameters():
        p.requires_grad = True
    for p in model.model.roi_heads.mask_predictor.parameters():
        p.requires_grad = True

def freeze_box_head(model):
    # Freeze classifier, bbox reg, RPN...
    for name, param in model.model.named_parameters():
        if any(x in name for x in ["box_predictor", "rpn"]):
            param.requires_grad = False

def train_model_two_phase(model, train_loader, bbox_epochs=50, mask_epochs=100,
                          save_path="maskrcnn_cocoa_best.pth",
                          mask_loss_weight=3.0, lr_mask_phase=None,
                          accum_steps=2, use_amp=True):
    from torch.amp import autocast, GradScaler  # ✅ API mới
    scaler = GradScaler("cuda", enabled=use_amp)

    print(f"🚀 Training on device: {next(model.model.parameters()).device}")

    base, ext = os.path.splitext(save_path)
    if not ext:
        ext = ".pth"

    # ---- PHASE 1: Train BBOX Only ----
    if bbox_epochs > 0:
        print(f"\n===== PHASE 1: Train BBOX ({bbox_epochs} epochs) =====")
        freeze_mask_head(model)
        for epoch in range(bbox_epochs):
            model.model.train()
            model.optimizer.zero_grad()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"BBOX Epoch {epoch+1}/{bbox_epochs}")
            for step, (images, targets) in enumerate(pbar):
                images = [img.to(model.device, non_blocking=True) for img in images]
                targets = [{k: v.to(model.device) for k, v in t.items()} for t in targets]

                with autocast("cuda", enabled=use_amp):
                    loss_dict = model.model(images, targets)
                    loss_bbox = (loss_dict['loss_classifier'] +
                                 loss_dict['loss_box_reg'] +
                                 loss_dict['loss_objectness'] +
                                 loss_dict['loss_rpn_box_reg']) / accum_steps

                scaler.scale(loss_bbox).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(model.optimizer)
                    scaler.update()
                    model.optimizer.zero_grad()

                total_loss += loss_bbox.item() * accum_steps
                pbar.set_postfix(loss=f"{loss_bbox.item() * accum_steps:.4f}")

            print(f"[BBOX] Epoch {epoch+1}/{bbox_epochs} - Avg Loss: {total_loss / len(train_loader):.4f}")

    # ---- PHASE 2: Train MASK Only ----
    print(f"\n===== PHASE 2: Train MASK ONLY ({mask_epochs} epochs, weight={mask_loss_weight}) =====")
    unfreeze_mask_head(model)
    freeze_box_head(model)  # ✅ Freeze box head hoàn toàn

    if lr_mask_phase:
        for g in model.optimizer.param_groups:
            g["lr"] = lr_mask_phase
        print(f"🔧 Learning rate for phase 2 set to {lr_mask_phase}")

    best_loss = float("inf")
    for epoch in range(mask_epochs):
        model.model.train()
        model.optimizer.zero_grad()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"MASK Epoch {epoch+1}/{mask_epochs}")

        for step, (images, targets) in enumerate(pbar):
            images = [img.to(model.device, non_blocking=True) for img in images]
            targets = [{k: v.to(model.device) for k, v in t.items()} for t in targets]

            with autocast("cuda", enabled=use_amp):
                loss_dict = model.model(images, targets)
                loss_all = mask_loss_weight * loss_dict['loss_mask'] # Chi tinh losss cho massk, tap trung vao mask

            scaler.scale(loss_all).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(model.optimizer)
                scaler.update()
                model.optimizer.zero_grad()

            total_loss += loss_all.item() * accum_steps
            pbar.set_postfix(loss=f"{loss_all.item() * accum_steps:.4f}")

            del loss_dict, images, targets
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"[MASK] Epoch {epoch+1}/{mask_epochs} - Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss or (epoch + 1) % 10 == 0:
            best_loss = min(best_loss, avg_loss)
            save_file = f"{base}_best{ext}"
            model.save_model(save_file)
            print(f"💾 Saved BEST checkpoint (epoch {epoch+1}) | Loss={avg_loss:.4f}")





if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    from Dataload.Dataloader.Dataloader_MaskRCNN import COCOADataset, collate_fn
    import torch, gc

    gc.collect()
    torch.cuda.empty_cache()
    

    # Đường dẫn dữ liệu
    cocoa_json_path = "Datasets/train/cocoa_format_annotations.json"
    img_dir = "Datasets/train/images"
    depth_dir = "Datasets/train/depths"
    
    # Đọc số classes từ annotation file
    with open(cocoa_json_path, 'r') as f:
        cocoa_data = json.load(f)
    
    # Lọc categories có id != 0
    valid_categories = [c for c in cocoa_data['categories'] if c['id'] != 0]

    # +1 cho background
    num_classes = len(valid_categories) + 1  
    print(f"Number of classes (including background): {num_classes}")

        
    # Tạo dataset và dataloader
    dataset = COCOADataset(
        json_path=cocoa_json_path,
        img_dir=img_dir,
        depth_dir=depth_dir
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True  # ✅ Tăng tốc copy CPU->GPU
    )

    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # # Khởi tạo model
    
    model = MaskRCNNModel(num_classes=num_classes, pretrained=True)
    #Load checkpoint cũ
    model.setup_training(learning_rate=0.005)

    print(f"Model device: {next(model.model.parameters()).device}")
    train_model_two_phase(
        model,
        train_loader,
        bbox_epochs=0,
        mask_epochs=200,
        save_path="maskrcnn_cocoa_continue.pth",
        mask_loss_weight=3.0,
        lr_mask_phase=0.0005,
        accum_steps=2,   # 16 x 2 = batch hiệu quả 32
        use_amp=True
        )

    # #Visualization
    print("\nGenerating visualizations...")
    visualize_predictions(model, val_loader, num_samples=5)

    # ======= Load model và Evaluate trên tập val =======
    print("\nLoading best model and evaluating...")
    model.load_model("training/pretrained/maskrcnn_cocoa_continue_best.pth")

    predictions, targets = model.evaluate_model(val_loader, confidence_threshold=0.3)

    print(f"Evaluation done! Tổng số ảnh: {len(predictions)}")

    # Ví dụ: Tính IoU trung bình để đánh giá
    from torchvision.ops import box_iou
    ious = []
    for pred, tgt in zip(predictions, targets):
        if len(pred["boxes"]) > 0 and len(tgt["boxes"]) > 0:
            iou_matrix = box_iou(torch.tensor(pred["boxes"]), torch.tensor(tgt["boxes"]))
            ious.append(iou_matrix.mean().item())
    if len(ious) > 0:
        print(f"Mean Box IoU: {np.mean(ious):.4f}")
    else:
        print("Không có box nào để tính IoU")

    # Visualization
    print("\nGenerating visualizations...")
    visualize_predictions(model, val_loader, num_samples=30)
    for images, targets in train_loader:
        print("Image shape:", images[0].shape)
        print("Boxes:", targets[0]['boxes'])
        print("Masks shape:", targets[0]['masks'].shape)
        print("Mask unique values:", torch.unique(targets[0]['masks']))
        break