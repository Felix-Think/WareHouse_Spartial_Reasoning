import torch
from torch.utils.data import DataLoader
import os, sys
import numpy as np
# --- Import model components ---
from Mask_RCNN import (
    ResNetBackbone, FPN, RPN, DetectionHead,
    MaskHead, DualMaskHead,
    assign_targets_to_proposals, compute_loss_dual_mask, crop_and_resize_gt_masks, generate_anchors, decode_boxes, generate_proposals
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from Dataload.Dataloader.Dataloader_MaskRCNN import COCOADataset, collate_fn


def get_sample(device):
    """Lấy một batch duy nhất từ dataloader"""
    cocoa_json_path = "Datasets/train/cocoa_format_annotations.json"
    img_dir = "Datasets/train/images"
    depth_dir = "Datasets/train/depths"
    dataset = COCOADataset(json_path=cocoa_json_path, img_dir=img_dir, depth_dir=depth_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    image = batch[0]['image'].unsqueeze(0).to(device)
    return image, batch


def test_backbone(image, device):
    backbone = ResNetBackbone().to(device)
    features = backbone(image)
    print("[Backbone] Output shapes:", [f.shape for f in features])
    return features


def test_fpn(features, device):
    fpn = FPN().to(device)
    fpn_feats = fpn(features)
    print("[FPN] Output shapes:", [f.shape for f in fpn_feats])
    return fpn_feats


def test_rpn(fpn_feats, device):
    rpn = RPN().to(device)
    rpn_cls, rpn_reg = rpn(fpn_feats)
    print(f"[RPN] cls: {rpn_cls.shape}, reg: {rpn_reg.shape}")
    return rpn_cls, rpn_reg




def make_fake_proposals_from_targets(targets, num_proposals=10, noise_std=5.0, device="cpu"):
    """
    Tạo proposals bằng cách lấy GT boxes, thêm nhiễu nhỏ và xáo trộn.
    targets: list[dict] từ dataloader
    num_proposals: số lượng proposal muốn tạo
    noise_std: độ nhiễu thêm vào (pixels)
    """
    gt_boxes = targets[0]['boxes']  # [M, 4]
    M = gt_boxes.size(0)

    if M == 0:
        # fallback -> random boxes
        H, W = targets[0]['image_size']
        boxes = torch.rand((num_proposals, 4), device=device)
        boxes[:, 2:] += boxes[:, :2]  # x2>x1, y2>y1
        return torch.cat([torch.zeros((num_proposals, 1), device=device), boxes], dim=1)

    # 1. Lặp lại GT boxes nếu cần để đạt num_proposals
    reps = int(np.ceil(num_proposals / M))
    boxes = gt_boxes.repeat((reps, 1))[:num_proposals].clone()

    # 2. Thêm nhiễu Gaussian nhỏ
    noise = torch.randn_like(boxes) * noise_std
    boxes = boxes + noise
    # Đảm bảo x2 > x1, y2 > y1
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0)
    boxes[:, 2] = torch.max(boxes[:, 2], boxes[:, 0] + 1)
    boxes[:, 3] = torch.max(boxes[:, 3], boxes[:, 1] + 1)

    # 3. Xáo trộn order
    idx = torch.randperm(boxes.shape[0])
    boxes = boxes[idx]

    # 4. Thêm batch_idx ở đầu (0 vì batch size=1)
    batch_idx = torch.zeros((boxes.shape[0], 1), device=boxes.device)
    proposals = torch.cat([batch_idx, boxes], dim=1)
    return proposals



def test_detection_head(fpn_feats, proposals, image_shape, device):
    det_head = DetectionHead().to(device)
    cls_scores, bbox_deltas = det_head(fpn_feats, proposals, image_shape)
    print(f"[DetectionHead] cls_scores: {cls_scores.shape}, bbox_deltas: {bbox_deltas.shape}")
    assert cls_scores.ndim == 2, "cls_scores phải là [N, num_classes]"
    assert bbox_deltas.ndim == 2, "bbox_deltas phải là [N, num_classes * 4]"


def test_mask_head(fpn_feats, proposals, device):
    mask_head = MaskHead().to(device)
    mask_logits = mask_head(fpn_feats, proposals)
    print(f"[MaskHead] mask_logits: {mask_logits.shape}")
    return mask_logits


def test_dual_mask_head(fpn_feats, proposals, device):
    dual_mask_head = DualMaskHead().to(device)
    visible_logits, amodal_logits = dual_mask_head(fpn_feats, proposals)
    print(f"[DualMaskHead] visible: {visible_logits.shape}, amodal: {amodal_logits.shape}")
    return visible_logits, amodal_logits


def test_assign_targets(proposals, targets, device):
    gt_boxes = targets[0]['boxes'].to(device)
    gt_labels = targets[0]['labels'].to(device)
    matched_idx, matched_labels = assign_targets_to_proposals(proposals[:, 1:], gt_boxes, gt_labels)
    print("[AssignTargets] matched_idx:", matched_idx)
    print("[AssignTargets] matched_labels:", matched_labels)
    return matched_idx, matched_labels


from torchvision.ops import roi_align
import torch

def test_loss(visible_logits, amodal_logits, proposals, targets):
    if 'visible_masks' not in targets[0] or 'amodal_masks' not in targets[0]:
        print("[Loss] Dataset missing visible/amodal masks, skipping loss test.")
        return

    device = proposals.device
    gt_boxes = targets[0]['boxes'].to(device)
    gt_labels = targets[0]['labels'].to(device)
    gt_visible_masks = targets[0]['visible_masks'].to(device)
    gt_amodal_masks  = targets[0]['amodal_masks'].to(device)

    # --- 1. Match proposals với GT boxes ---
    matched_idx, matched_labels = assign_targets_to_proposals(
        proposals[:, 1:5],  # chỉ lấy tọa độ x1,y1,x2,y2
        gt_boxes,
        gt_labels,
        iou_threshold=0.5
    )

    # Lọc các proposal positive (foreground)
    pos_idx = matched_idx >= 0
    if pos_idx.sum() == 0:
        print("[Loss] No positive proposals, skipping mask loss.")
        return

    proposals_pos = proposals[pos_idx]
    matched_idx_pos = matched_idx[pos_idx]

    # --- 2. Chọn đúng GT mask cho các proposals ---
    selected_visible_masks = gt_visible_masks[matched_idx_pos]
    selected_amodal_masks  = gt_amodal_masks[matched_idx_pos]
    selected_gt_boxes      = gt_boxes[matched_idx_pos]

    # --- 3. Crop & resize theo proposal box ---
    visible_resized = crop_and_resize_gt_masks(selected_visible_masks, selected_gt_boxes, proposals_pos, output_size=(28,28))
    amodal_resized  = crop_and_resize_gt_masks(selected_amodal_masks, selected_gt_boxes, proposals_pos, output_size=(28,28))

    # --- 4. Tính loss ---
    loss_mask = compute_loss_dual_mask(
        visible_logits[pos_idx],
        amodal_logits[pos_idx],
        proposals_pos,
        [{
            'labels': matched_labels[pos_idx],
            'visible_masks': visible_resized,
            'amodal_masks': amodal_resized
        }]
    )

    print(f"[Loss] dual mask loss: {loss_mask.item():.4f}")

def test_generate_anchors():
    print("\n[Test] Generate Anchors")
    # Bạn có thể thay đổi base_size, ratios, scales để test
    base_size = 32
    ratios = [0.5, 1.0, 2.0]
    scales = [1.0, 2.0, 4.0]

    anchors = generate_anchors(base_size=base_size, ratios=ratios, scales=scales, device=device)
    print(f"[Anchors] shape: {anchors.shape}")  # kỳ vọng [len(ratios)*len(scales), 4]
    print("[Anchors] first few:\n", anchors[:5])

    # Kiểm tra điều kiện cơ bản: x2 > x1, y2 > y1
    assert torch.all(anchors[:, 2] > anchors[:, 0]), "Anchor x2 phải > x1"
    assert torch.all(anchors[:, 3] > anchors[:, 1]), "Anchor y2 phải > y1"
    print("[Anchors] ✅ Passed basic sanity check.")



def test_generate_proposals(rpn_cls, rpn_bbox, fpn_feats, image, device):
    print("\n[Test] Generate Proposals")
    feature_shapes = [(f.shape[2], f.shape[3]) for f in fpn_feats]  # [(H,W),...]
    image_size = image.shape[-2:]  # (H, W)

    proposals = generate_proposals(
        rpn_cls, rpn_bbox,
        feature_shapes=feature_shapes,
        image_size=image_size,
        device=device
    )
    print(f"[Proposals] shape: {proposals.shape}")
    print(f"[Proposals] min/max: {proposals[:,1:].min().item():.1f} / {proposals[:,1:].max().item():.1f}")
    print(f"[Proposals] Example:", proposals[:5])
    return proposals

def get_topk_proposals(rpn_cls, rpn_bbox, feature_shapes, image_size,
                       strides=[4, 8, 16, 32, 64],
                       ratios=[0.5, 1.0, 2.0],
                       scales=[1.0, 2.0, 4.0],
                       k=3,
                       device="cpu"):
    """
    Sinh proposals và chỉ trả về top-k proposal có score cao nhất.
    """
    B = rpn_cls.shape[0]
    scores = rpn_cls.softmax(dim=-1)[..., 1]  # foreground scores
    proposals_all, scores_all = [], []

    start = 0
    for lvl, (H, W) in enumerate(feature_shapes):
        stride = strides[lvl]
        A = len(ratios) * len(scales)
        end = start + H * W * A

        # Tạo anchors cho level này
        shifts_x = torch.arange(0, W * stride, step=stride, device=device)
        shifts_y = torch.arange(0, H * stride, step=stride, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)

        base_anchors = generate_anchors(stride, ratios, scales, device)
        anchors = (base_anchors[None, :, :] + shift[:, None, :]).reshape(-1, 4)

        deltas = rpn_bbox[:, start:end, :]
        batch_scores = scores[:, start:end]

        for b in range(B):
            boxes = decode_boxes(anchors, deltas[b])
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_size[1])
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_size[0])
            scores_keep = batch_scores[b]

            proposals_all.append(boxes)
            scores_all.append(scores_keep)

        start = end

    proposals_all = torch.cat(proposals_all, dim=0)
    scores_all = torch.cat(scores_all, dim=0)

    # Lấy top-k theo score
    topk_scores, idx = scores_all.topk(k)
    topk_boxes = proposals_all[idx]

    return topk_boxes, topk_scores


if __name__ == "__main__":
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image, targets = get_sample(device)
    features = test_backbone(image, device)
    fpn_feats = test_fpn(features, device)

    # --- RPN ---
    rpn_cls, rpn_bbox = test_rpn(fpn_feats, device)

    # --- Generate proposals từ RPN (thay cho make_fake_proposals_from_targets) ---
    feature_shapes = [(f.shape[2], f.shape[3]) for f in fpn_feats]
    image_size = image.shape[-2:]

    proposals, prop_scores = generate_proposals(
        rpn_cls, rpn_bbox,
        feature_shapes=feature_shapes,
        image_size=image_size,
        device=device
    )
    print(f"[Proposals] shape: {proposals.shape}, scores: {prop_scores.shape}")
    print(f"[Proposals] min/max: {proposals[:,1:].min().item():.1f} / {proposals[:,1:].max().item():.1f}")

    # In top-5 proposals theo score để “xem score như thế nào”
    topk = min(5, prop_scores.numel())
    top_scores, top_idx = prop_scores.topk(topk)
    top_boxes = proposals[top_idx]
    print("[Top proposals by RPN score]")
    for i in range(topk):
        b, x1, y1, x2, y2 = top_boxes[i].tolist()
        print(f"{i+1}: score={top_scores[i].item():.4f} | box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) on batch={int(b)}")

    # (Optional) lấy subset để test downstream nhanh hơn
    num_use = min(200, proposals.shape[0])
    proposals = proposals[:num_use]
    prop_scores = prop_scores[:num_use]

    # --- Detection / Mask ---
    test_detection_head(fpn_feats, proposals, image.shape[-2:], device)
    mask_logits = test_mask_head(fpn_feats, proposals, device)
    visible_logits, amodal_logits = test_dual_mask_head(fpn_feats, proposals, device)

    # --- Assign targets + loss ---
    test_assign_targets(proposals, targets, device)
    test_loss(visible_logits, amodal_logits, proposals, targets)

    # (Optional) IoU với GT để sanity check
    from torchvision.ops import box_iou
    ious = box_iou(proposals[:, 1:], targets[0]['boxes'].to(device))
    max_ious, _ = ious.max(dim=1)
    print(f"[RPN] Mean IoU: {max_ious.mean():.3f}, Recall (IoU>0.5): {(max_ious>0.5).float().mean():.3f}")
