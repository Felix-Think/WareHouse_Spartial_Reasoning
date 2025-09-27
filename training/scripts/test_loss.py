# test_loss.py
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from Mask_RCNN import (
    compute_loss_rpn,
    compute_loss_detection,
    compute_loss_dual_mask,
    compute_total_loss,
)

def test_rpn_loss():
    print("=== Test RPN Loss ===")
    N = 1000
    anchors = torch.rand((N, 4)) * 512
    anchors[:, 2:] += anchors[:, :2]  # đảm bảo x2>x1, y2>y1

    gt_boxes = torch.tensor([[50, 50, 120, 150], [300, 300, 360, 360]], dtype=torch.float32)

    rpn_cls_logits = torch.randn(N, 2)
    rpn_bbox_preds = torch.randn(N, 4)

    cls_loss, reg_loss, labels = compute_loss_rpn(
        rpn_cls_logits, rpn_bbox_preds, anchors, gt_boxes
    )
    print(f"RPN cls_loss: {cls_loss.item():.4f}, reg_loss: {reg_loss.item():.4f}, pos={labels.eq(1).sum().item()}")

def test_detection_loss():
    print("=== Test Detection Loss ===")
    K = 200
    proposals = torch.rand((K, 4)) * 512
    proposals[:, 2:] += proposals[:, :2]

    gt_boxes = torch.tensor([[100, 100, 150, 180], [250, 250, 280, 300]], dtype=torch.float32)
    gt_labels = torch.tensor([1, 2], dtype=torch.long)

    num_classes = 4
    cls_logits = torch.randn(K, num_classes)
    bbox_deltas = torch.randn(K, num_classes * 4)

    cls_loss, reg_loss, labels = compute_loss_detection(
        cls_logits, bbox_deltas, proposals, gt_boxes, gt_labels
    )
    print(f"Detection cls_loss: {cls_loss.item():.4f}, reg_loss: {reg_loss.item():.4f}, fg={labels.gt(0).sum().item()}")

def test_mask_loss():
    print("=== Test Mask Loss ===")
    P = 1  # chỉ tạo 1 proposal để khớp với 1 GT
    num_classes = 5
    proposals = torch.zeros(P, 5)
    proposals[:, 1:] = torch.tensor([[30, 30, 70, 70]], dtype=torch.float32)
    if proposals.shape[1] == 4:
        batch_idx = torch.zeros((proposals.shape[0], 1), device=proposals.device)
        proposals = torch.cat([batch_idx, proposals], dim=1)

    visible_logits = torch.randn(P, num_classes, 28, 28)
    amodal_logits = torch.randn(P, num_classes, 28, 28)

    targets = [{
        "boxes": torch.tensor([[30, 30, 70, 70]], dtype=torch.float32),
        "labels": torch.tensor([1]),
        "visible_masks": torch.randint(0, 2, (1, 28, 28), dtype=torch.float32),
        "amodal_masks": torch.randint(0, 2, (1, 28, 28), dtype=torch.float32),
    }]

    loss = compute_loss_dual_mask(visible_logits, amodal_logits, proposals, targets)
    print(f"Mask loss: {loss.item():.4f}")


def test_total_loss():
    print("=== Test Total Loss ===")
    N = 300
    anchors = torch.rand((N, 4)) * 512
    anchors[:, 2:] += anchors[:, :2]

    gt_boxes = torch.tensor([[50, 50, 120, 150], [300, 300, 360, 360]], dtype=torch.float32)
    gt_labels = torch.tensor([1, 2], dtype=torch.long)

    rpn_cls_logits = torch.randn(N, 2)
    rpn_bbox_preds = torch.randn(N, 4)

    K = 64
    proposals = torch.rand((K, 4)) * 512
    proposals[:, 2:] += proposals[:, :2]

    num_classes = 4
    roi_cls_logits = torch.randn(K, num_classes)
    roi_bbox_deltas = torch.randn(K, num_classes * 4)

    P = 10
    mask_visible_logits = torch.randn(P, num_classes, 28, 28)
    mask_amodal_logits = torch.randn(P, num_classes, 28, 28)

    targets = [{
        "boxes": gt_boxes,
        "labels": gt_labels,
        "visible_masks": torch.randint(0, 2, (2, 28, 28), dtype=torch.float32),
        "amodal_masks": torch.randint(0, 2, (2, 28, 28), dtype=torch.float32),
    }]

    losses = compute_total_loss(
        rpn_cls_logits, rpn_bbox_preds, anchors, [],
        gt_boxes, gt_labels,
        roi_cls_logits, roi_bbox_deltas, proposals,
        mask_visible_logits, mask_amodal_logits, targets
    )
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")

if __name__ == "__main__":
    test_rpn_loss()
    test_detection_loss()
    test_mask_loss()
    test_total_loss()
    print("All tests completed successfully.")
