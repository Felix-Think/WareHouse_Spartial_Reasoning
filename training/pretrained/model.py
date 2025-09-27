import os
import sys
from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== Dummy Mask Predictor ==========
class DummyMaskPredictor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Trả về tensor zeros có đúng số class
        return torch.zeros((x.shape[0], self.num_classes, x.shape[2], x.shape[3]), device=x.device)


# ========== Dual Mask Predictor ==========
class DualMaskPredictor(nn.Module):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__()
        self.conv5_mask = nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.relu = nn.ReLU(inplace=False)
        self.vis_logits = nn.Conv2d(dim_reduced, num_classes, 1)
        self.amo_logits = nn.Conv2d(dim_reduced, num_classes, 1)

        for l in [self.conv5_mask, self.vis_logits, self.amo_logits]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv5_mask(x))
        vis = self.vis_logits(x)
        amo = self.amo_logits(x)
        return vis, amo

# ========== Setup Model ==========
def setup_model(num_classes=5):
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    # Box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Lấy in_channels để dùng cho DualMaskPredictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # Gắn dummy predictor để tránh torchvision tính mask loss
    model.roi_heads.mask_predictor = DummyMaskPredictor(num_classes)

    # Trả về cả model và dual mask head
    dual_mask_head = DualMaskPredictor(in_features_mask, 256, num_classes).to(device)
    return model.to(device), dual_mask_head

# ========== Dual Mask Loss ==========
def dual_mask_loss(dual_mask_head, model, features, images_tfm, pos_props, matched_idxs, targets):
    """
    targets ở đây nên là bản gốc (chưa transform)
    """
    device = features[list(features.keys())[0]].device
    mask_features = model.roi_heads.mask_roi_pool(features, pos_props, images_tfm.image_sizes)
    mask_features = model.roi_heads.mask_head(mask_features)

    vis_logits, amo_logits = dual_mask_head(mask_features)

    gt_vis_list, gt_amo_list, cls_targets = [], [], []
    for b_idx, (props_b, midx_b) in enumerate(zip(pos_props, matched_idxs)):
        if midx_b.numel() == 0:
            continue

        tgt_b = targets[b_idx]  # dùng targets gốc, không dùng targets_tfm
        gt_idx = midx_b

        labels_b = tgt_b["labels"][gt_idx]
        cls_targets.append(labels_b)

        vis_masks = tgt_b["visible_masks"][gt_idx].float().unsqueeze(1)
        amo_masks = tgt_b["amodal_masks"][gt_idx].float().unsqueeze(1)

        # crop theo RoI
        rois = torch.cat([torch.full((props_b.shape[0], 1), b_idx, device=device), props_b], dim=1)
        vis_cropped = torchvision.ops.roi_align(vis_masks, rois, output_size=(28, 28), aligned=True)
        amo_cropped = torchvision.ops.roi_align(amo_masks, rois, output_size=(28, 28), aligned=True)

        gt_vis_list.append(vis_cropped.squeeze(1))
        gt_amo_list.append(amo_cropped.squeeze(1))

    if len(gt_vis_list) == 0:
        return torch.tensor(0., device=device), torch.tensor(0., device=device)

    gt_vis = torch.cat(gt_vis_list, 0)
    gt_amo = torch.cat(gt_amo_list, 0)
    cls_all = torch.cat(cls_targets, 0).long()
    idx = torch.arange(cls_all.size(0), device=device)

    vis_sel = vis_logits[idx, cls_all]
    amo_sel = amo_logits[idx, cls_all]

    loss_vis = F.binary_cross_entropy_with_logits(vis_sel, gt_vis)
    loss_amo = F.binary_cross_entropy_with_logits(amo_sel, gt_amo)
    return loss_vis, loss_amo


# ========== Training Loop ==========
@torch.no_grad()
def _select_pos_samples(model, proposals, targets_tfm):
    result = model.roi_heads.select_training_samples(proposals, targets_tfm)

    if isinstance(result, (list, tuple)):
        if len(result) == 4:
            proposals_for_masks, matched_idxs, _, _ = result  # Bỏ labels & regression_targets
        elif len(result) == 3:
            proposals_for_masks, matched_idxs, _ = result  # Bỏ labels
        elif len(result) == 2:
            proposals_for_masks, matched_idxs = result
        else:
            raise ValueError(f"Unsupported number of return values: {len(result)}")
    else:
        raise TypeError(f"Expected tuple/list from select_training_samples, got {type(result)}")

    return proposals_for_masks, matched_idxs





def train(model, dual_mask_head, dataloader, epochs=10, save_dir="checkpoints", lambda_vis=1.0, lambda_amo=1.0):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(dual_mask_head.parameters()), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        dual_mask_head.train()

        total_epoch, vis_epoch, amo_epoch, n_batches = 0., 0., 0., 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in pbar:
            valid = [(im, tg) for im, tg in zip(images, targets) if tg["boxes"].shape[0] > 0]
            if len(valid) == 0:
                continue

            images = [im.to(device) for im, _ in valid]
            targets = [{k: v.to(device) for k, v in tg.items()} for _, tg in valid]

            # Loss chuẩn từ torchvision
            loss_dict = model(images, targets)
            if "loss_mask" in loss_dict:  # bỏ loss mask dummy
                loss_dict.pop("loss_mask")
            base_loss = sum(loss_dict.values())

            # Transform & lấy proposals
            images_tfm, targets_tfm = model.transform(images, targets)
            features = model.backbone(images_tfm.tensors)
            if isinstance(features, torch.Tensor):
                features = {"0": features}
            proposals, _ = model.rpn(images_tfm, features, targets_tfm)
            pos_props, matched_idxs = _select_pos_samples(model, proposals, targets_tfm)

            # Dual mask loss
            loss_vis, loss_amo = dual_mask_loss(dual_mask_head, model, features, images_tfm, pos_props, matched_idxs, targets_tfm)
            total_loss = base_loss + lambda_vis * loss_vis + lambda_amo * loss_amo

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(dual_mask_head.parameters()), 1.0)
            optimizer.step()

            total_epoch += total_loss.item()
            vis_epoch += loss_vis.item()
            amo_epoch += loss_amo.item()
            n_batches += 1

            pbar.set_postfix({"total": f"{total_loss.item():.4f}", "vis": f"{loss_vis.item():.4f}", "amo": f"{loss_amo.item():.4f}"})

        if n_batches > 0:
            print(f"[Epoch {epoch+1}/{epochs}] avg_total={total_epoch/n_batches:.4f} avg_vis={vis_epoch/n_batches:.4f} avg_amo={amo_epoch/n_batches:.4f}")

    torch.save({
        "model": model.state_dict(),
        "dual_mask_head": dual_mask_head.state_dict()
    }, os.path.join(save_dir, "dual_mask_rcnn_final.pth"))
    print("✔ Saved final model.")

# ========== Entry Point ==========
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    from torch.utils.data import DataLoader
    from Dataload.Dataloader.Dataloader_MaskRCNN import COCOADataset, collate_fn

    json_path = "Datasets/train/cocoa_format_annotations.json"
    img_dir = "Datasets/train/images"
    depth_dir = "Datasets/train/depths"

    dataset = COCOADataset(json_path=json_path, img_dir=img_dir, depth_dir=depth_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model, dual_mask_head = setup_model(num_classes=5)
    train(model, dual_mask_head, dataloader, epochs=10)
