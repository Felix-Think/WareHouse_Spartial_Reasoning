import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import RoIAlign
import torch
import torch.nn.functional as F
from torchvision.ops import nms, box_iou
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torchvision.ops import roi_align
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) if pretrained else models.resnet50(weights=None)
        # C1: conv1 + bn + relu + maxpool
        self.C1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # C2-C5: ResNet layers
        self.C2 = resnet.layer1  # 256 channels, stride 4
        self.C3 = resnet.layer2  # 512 channels, stride 8
        self.C4 = resnet.layer3  # 1024 channels, stride 16
        self.C5 = resnet.layer4  # 2048 channels, stride 32
        
        # Freeze early layers
        for layer in [self.C1, self.C2]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        c1 = self.C1(x)
        c2 = self.C2(c1)
        c3 = self.C3(c2)
        c4 = self.C4(c3)
        c5 = self.C5(c4)
        return [c2, c3, c4, c5]

class RPN(nn.Module):
    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()
        self.num_anchors = num_anchors
        
        # Shared conv layer
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        
        # Classification head (object/background)
        self.cls_head = nn.Conv2d(512, num_anchors * 2, 1)
        
        # Regression head (bbox deltas)
        self.reg_head = nn.Conv2d(512, num_anchors * 4, 1)
        
        # Initialize weights
        for layer in [self.conv, self.cls_head, self.reg_head]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, feature_maps):
        rpn_cls_scores = []
        rpn_bbox_preds = []
        
        for feature in feature_maps:
            x = F.relu(self.conv(feature))
            
            # Classification
            cls = self.cls_head(x)
            B, _, H, W = cls.shape
            cls = cls.view(B, self.num_anchors, 2, H, W)
            cls = cls.permute(0, 3, 4, 1, 2).contiguous()
            cls = cls.view(B, -1, 2)
            rpn_cls_scores.append(cls)
            
            # Regression
            reg = self.reg_head(x)
            reg = reg.view(B, self.num_anchors, 4, H, W)
            reg = reg.permute(0, 3, 4, 1, 2).contiguous()
            reg = reg.view(B, -1, 4)
            rpn_bbox_preds.append(reg)
        
        return torch.cat(rpn_cls_scores, dim=1), torch.cat(rpn_bbox_preds, dim=1)

class FPN(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.lateral_conv5 = nn.Conv2d(in_channels_list[3], out_channels, 1)
        self.lateral_conv4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lateral_conv3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_conv2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        
        # Top-down pathway convolutions
        self.fpn_conv5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # P6 (for RPN)
        self.p6 = nn.MaxPool2d(1, stride=2)
        
        # Initialize
        for layer in [self.lateral_conv5, self.lateral_conv4, self.lateral_conv3, 
                     self.lateral_conv2, self.fpn_conv5, self.fpn_conv4, 
                     self.fpn_conv3, self.fpn_conv2]:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features):
        c2, c3, c4, c5 = features
        
        # Lateral connections
        p5 = self.lateral_conv5(c5)
        p4 = self.lateral_conv4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_conv3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_conv2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        
        # Smooth with 3x3 conv
        p5 = self.fpn_conv5(p5)
        p4 = self.fpn_conv4(p4)
        p3 = self.fpn_conv3(p3)
        p2 = self.fpn_conv2(p2)
        
        # P6 for RPN
        p6 = self.p6(p5)
        
        return [p2, p3, p4, p5, p6]
    
class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=4, pool_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.pool_size = pool_size

        # ROI Align cho từng level P2-P5
        self.roi_aligns = nn.ModuleList([
            RoIAlign(output_size=pool_size, spatial_scale=1.0/4, sampling_ratio=2),   # P2
            RoIAlign(output_size=pool_size, spatial_scale=1.0/8, sampling_ratio=2),   # P3  
            RoIAlign(output_size=pool_size, spatial_scale=1.0/16, sampling_ratio=2),  # P4
            RoIAlign(output_size=pool_size, spatial_scale=1.0/32, sampling_ratio=2),  # P5
        ])

        # Two FC layers
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        # Classification & bbox heads
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, 4)

        # Initialize
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def assign_levels(self, proposals):
        """
        Gán mỗi proposal vào 1 trong 4 level P2-P5 dựa trên kích thước box.
        proposals: [N, 5]  (batch_idx, x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = proposals[:, 1], proposals[:, 2], proposals[:, 3], proposals[:, 4]
        h = y2 - y1
        w = x2 - x1
        box_area = torch.sqrt(w * h)

        
        k0 = 4  # P4 là mức chuẩn
        levels = torch.floor(k0 + torch.log2(box_area / 224 + 1e-6))
        levels = torch.clamp(levels, min=2, max=5)  # Giới hạn [P2,P5]
        return levels.int()

    def forward(self, feature_maps, proposals, image_shape):
        if proposals.numel() == 0:
            return (torch.zeros((0, self.num_classes), device=proposals.device),
                    torch.zeros((0, self.num_classes * 4), device=proposals.device))

        # proposals cần dạng [batch_idx, x1, y1, x2, y2]
        # Nếu đầu vào chỉ có [N, 4], assume batch=0
        if proposals.shape[1] == 4:
            batch_idx = torch.zeros((proposals.size(0), 1), device=proposals.device)
            proposals = torch.cat([batch_idx, proposals], dim=1)

        levels = self.assign_levels(proposals)
        pooled = proposals.new_zeros((proposals.shape[0], 256, self.pool_size, self.pool_size))
        for lvl, roi_align in enumerate(self.roi_aligns, start=2):
            idx = (levels == lvl).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            pooled_lvl = roi_align(feature_maps[lvl-2], proposals[idx])
            pooled[idx] = pooled_lvl
        roi_features = pooled

        # Flatten + FC layers
        roi_features = roi_features.flatten(start_dim=1)
        x = F.relu(self.fc1(roi_features))
        x = F.relu(self.fc2(x))

        cls_scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return cls_scores, bbox_deltas

    
class MaskHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        self.roi_aligns = nn.ModuleList([
            RoIAlign(14, spatial_scale=1/4,  sampling_ratio=2),
            RoIAlign(14, spatial_scale=1/8,  sampling_ratio=2),
            RoIAlign(14, spatial_scale=1/16, sampling_ratio=2),
            RoIAlign(14, spatial_scale=1/32, sampling_ratio=2),
        ])

        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.mask_pred = nn.Conv2d(256, num_classes, 1)

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.deconv]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.mask_pred.weight, std=0.001)
        nn.init.constant_(self.mask_pred.bias, 0)

    @staticmethod
    def assign_levels(boxes_xyxy):
        w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        s = torch.sqrt(torch.clamp(w * h, min=1e-6))
        k = torch.floor(4 + torch.log2(s / 224.0 + 1e-6))
        return torch.clamp(k, 2, 5).to(torch.int64)

    def forward(self, feature_maps, proposals):
        if proposals.shape[1] == 4:
            batch_idx = torch.zeros((proposals.size(0), 1), device=proposals.device)
            proposals = torch.cat([batch_idx, proposals], dim=1)

        levels = self.assign_levels(proposals[:, 1:5])
        pooled = feature_maps[0].new_zeros((proposals.shape[0], 256, 14, 14))

        for lvl in (2, 3, 4, 5):
            idx = (levels == lvl).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            pooled[idx] = self.roi_aligns[lvl - 2](feature_maps[lvl - 2], proposals[idx])

        x = F.relu(self.conv1(pooled))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.deconv(x))
        return self.mask_pred(x)


class DualMaskHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.roi_aligns = nn.ModuleList([
            RoIAlign(output_size=14, spatial_scale=1/4,  sampling_ratio=2),  # P2
            RoIAlign(output_size=14, spatial_scale=1/8,  sampling_ratio=2),  # P3
            RoIAlign(output_size=14, spatial_scale=1/16, sampling_ratio=2),  # P4
            RoIAlign(output_size=14, spatial_scale=1/32, sampling_ratio=2),  # P5
        ])
        # shared trunk
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        # branches
        self.visible_deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.visible_conv5  = nn.Conv2d(256, 256, 3, padding=1)
        self.visible_mask_pred = nn.Conv2d(256, num_classes, 1)
        self.amodal_deconv  = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.amodal_conv5   = nn.Conv2d(256, 256, 3, padding=1)
        self.amodal_mask_pred = nn.Conv2d(256, num_classes, 1)

    @staticmethod
    def _assign_level(rois_xyxy):
        w = rois_xyxy[:,2] - rois_xyxy[:,0]
        h = rois_xyxy[:,3] - rois_xyxy[:,1]
        s = torch.sqrt(torch.clamp(w*h, 1e-6))
        k = torch.floor(4 + torch.log2(s/224.0 + 1e-6))
        return torch.clamp(k, 2, 5).to(torch.int64)  # 2..5

    def forward(self, feature_maps, proposals_with_idx):  
        if proposals_with_idx.numel() == 0:
            z = proposals_with_idx.new_zeros((0, self.num_classes, 28, 28))
            return z, z

        rois = proposals_with_idx
        levels = self._assign_level(rois[:,1:5])  # dùng tọa độ ảnh gốc
        N = rois.size(0)
        device = rois.device

        # chỗ để đặt kết quả theo đúng thứ tự RoI
        pooled = rois.new_zeros((N, 256, 14, 14))
        for lvl in (2,3,4,5):
            sel = torch.nonzero(levels == lvl).squeeze(1)
            if sel.numel() == 0: 
                continue
            pooled_lvl = self.roi_aligns[lvl-2](feature_maps[lvl-2], rois[sel])
            pooled[sel] = pooled_lvl

        x = F.relu(self.conv1(pooled))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        v = F.relu(self.visible_deconv(x))
        v = F.relu(self.visible_conv5(v))
        visible_mask = self.visible_mask_pred(v)

        a = F.relu(self.amodal_deconv(x))
        a = F.relu(self.amodal_conv5(a))
        amodal_mask = self.amodal_mask_pred(a)

        return visible_mask, amodal_mask

from torchvision.ops import box_iou, roi_align

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, roi_align

def compute_loss_dual_mask(visible_logits, amodal_logits, proposals, targets,
                           loss_weight_visible=1.0, loss_weight_amodal=1.0,
                           iou_thresh=0.5, mask_size=(28, 28)):
    """
    proposals: [N, 5] hoặc [N, 4]  (batch_idx,x1,y1,x2,y2)
    targets: list[dict] với các keys: boxes, labels, visible_masks, amodal_masks
    """
    device = visible_logits.device
    if proposals.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Đảm bảo proposals có batch_idx
    if proposals.shape[1] == 4:
        batch_idx = torch.zeros((proposals.size(0), 1), device=proposals.device)
        proposals = torch.cat([batch_idx, proposals], dim=1)

    all_losses = []
    for b_idx in range(len(targets)):
        # --- Chọn proposals thuộc batch b_idx ---
        batch_mask = proposals[:, 0] == b_idx
        if batch_mask.sum() == 0:
            continue

        proposals_b = proposals[batch_mask]

        gt_boxes = targets[b_idx]['boxes'].to(device)
        gt_labels = targets[b_idx]['labels'].to(device)
        gt_vis = targets[b_idx]['visible_masks'].to(device)
        gt_amo = targets[b_idx]['amodal_masks'].to(device)

        if gt_boxes.numel() == 0:
            continue

        # --- Match proposals với GT bằng IoU ---
        ious = box_iou(proposals_b[:, 1:5], gt_boxes)
        max_iou, gt_idx = ious.max(dim=1)
        keep = max_iou >= iou_thresh
        if keep.sum() == 0:
            continue

        proposals_b = proposals_b[keep]
        gt_idx = gt_idx[keep]

        # --- Lấy GT theo matched_idx ---
        matched_boxes = gt_boxes[gt_idx]
        matched_labels = gt_labels[gt_idx]
        matched_vis = gt_vis[gt_idx].unsqueeze(1).float()  # [P, 1, H, W]
        matched_amo = gt_amo[gt_idx].unsqueeze(1).float()

        # --- Crop & resize mask về 28x28 ---
        proposals_for_roi = proposals_b.clone()
        # roi_align expects [N,5]: (batch_idx,x1,y1,x2,y2)
        gt_vis_cropped = roi_align(matched_vis, proposals_for_roi, output_size=mask_size, aligned=True)
        gt_amo_cropped = roi_align(matched_amo, proposals_for_roi, output_size=mask_size, aligned=True)
        # Remove channel dim -> [P, 28, 28]
        gt_vis_cropped = gt_vis_cropped.squeeze(1)
        gt_amo_cropped = gt_amo_cropped.squeeze(1)

        # --- Lấy logits theo đúng class ---
        idx = torch.arange(len(proposals_b), device=device)
        visible_logits_b = visible_logits[idx, matched_labels]
        amodal_logits_b = amodal_logits[idx, matched_labels]

        # --- BCE Loss ---
        loss_visible = F.binary_cross_entropy_with_logits(visible_logits_b, gt_vis_cropped)
        loss_amodal = F.binary_cross_entropy_with_logits(amodal_logits_b, gt_amo_cropped)

        all_losses.append(loss_visible + loss_amodal)

    if len(all_losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_weight_visible * torch.stack(all_losses).mean()




def assign_targets_to_proposals(proposals, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    proposals: Tensor [N, 4] (x1, y1, x2, y2) - proposals từ RPN
    gt_boxes: Tensor [M, 4] - ground truth boxes
    gt_labels: Tensor [M]   - nhãn tương ứng với gt_boxes
    iou_threshold: float, ngưỡng IoU để giữ positive proposals

    Return:
        matched_idx: LongTensor [N] - chỉ số GT box match với mỗi proposal (-1 nếu background)
        matched_labels: LongTensor [N] - nhãn gán cho mỗi proposal (0 nếu background)
    """

    device = proposals.device
    N = proposals.shape[0]
    M = gt_boxes.shape[0]

    if N == 0:
        return torch.full((0,), -1, dtype=torch.long, device=device), torch.full((0,), 0, dtype=torch.long, device=device)

    if M == 0:
        # Không có GT, tất cả proposals là background
        return torch.full((N,), -1, dtype=torch.long, device=device), torch.zeros((N,), dtype=torch.long, device=device)

    # ----- 1. Tính IoU -----
    ious = box_iou(proposals, gt_boxes)  # [N, M]
    print("IoU shape:", ious.shape)
    # ----- 2. Chọn GT box IoU cao nhất cho mỗi proposal -----
    max_ious, matched_idx = ious.max(dim=1)  # matched_idx: [N]
    print("Max IoUs:", max_ious)
    print("Matched idx:", matched_idx)
    # ----- 3. Gán background & foreground -----
    matched_labels = gt_labels[matched_idx]  # gán label theo GT tương ứng
    matched_labels[max_ious < iou_threshold] = 0   # gán label = 0 cho background
    matched_idx[max_ious < iou_threshold] = -1     # -1 nghĩa là không match GT
    
    return matched_idx, matched_labels



def crop_and_resize_gt_masks(gt_masks, gt_boxes, proposals, output_size=(28, 28)):
    """
    Crop & resize GT masks theo proposals.

    gt_masks: Tensor [M, H, W]  (mask full ảnh cho M GT)
    gt_boxes: Tensor [M, 4]     (x1,y1,x2,y2 GT box)
    proposals: Tensor [N, 5]    (batch_idx,x1,y1,x2,y2 đã match với gt)
        -> giả sử bạn có matched_idx: với mỗi proposal biết nó match GT nào

    Trả về: Tensor [N, output_size[0], output_size[1]]
    """

    device = gt_masks.device
    N = proposals.size(0)
    if N == 0:
        return torch.zeros((0, *output_size), device=device)

    # Chọn mask GT theo matched_idx
    # matched_idx phải được tính từ assign_targets_to_proposals()
    batch_idx = proposals[:, 0].to(torch.int64)
    boxes = proposals[:, 1:5]

    # Tạo tensor mask đã chọn
    selected_masks = gt_masks[batch_idx] if gt_masks.ndim == 4 else gt_masks
    # Add channel dim để dùng roi_align: [N,1,H,W]
    selected_masks = selected_masks.unsqueeze(1).float()

    # roi_align expects boxes [N,5] (batch_idx,x1,y1,x2,y2)
    cropped = roi_align(selected_masks, proposals, output_size=output_size, aligned=True)
    # Output [N, 1, 28, 28] -> bỏ channel
    return cropped.squeeze(1)


import torch
import math
from torchvision.ops import nms

def generate_anchors(base_size, ratios, scales, device):
    """
    Tạo anchors cho 1 vị trí (center = (0,0)).
    base_size: chiều dài cạnh cơ sở (ví dụ stride của feature map)
    ratios: list aspect ratios (h/w)
    scales: list scales (multiplier)
    """
    anchors = []
    area = base_size * base_size
    for r in ratios:
        w = math.sqrt(area / r)
        h = w * r
        for s in scales:
            ws = w * s
            hs = h * s
            x1 = -ws / 2
            y1 = -hs / 2
            x2 = ws / 2
            y2 = hs / 2
            anchors.append([x1, y1, x2, y2])
    return torch.tensor(anchors, dtype=torch.float32, device=device)

def decode_boxes(anchors, deltas):
    """
    anchors: [N, 4] (x1,y1,x2,y2)
    deltas: [N, 4] (dx, dy, dw, dh)
    """
    # Anchor center, width, height
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    dx, dy, dw, dh = deltas.unbind(dim=1)

    # Apply deltas
    px = dx * aw + ax
    py = dy * ah + ay
    pw = torch.exp(dw) * aw
    ph = torch.exp(dh) * ah

    x1 = px - pw * 0.5
    y1 = py - ph * 0.5
    x2 = px + pw * 0.5
    y2 = py + ph * 0.5

    return torch.stack([x1, y1, x2, y2], dim=1)

def generate_proposals(rpn_cls, rpn_bbox, feature_shapes, image_size,
                       strides=[4, 8, 16, 32, 64],
                       ratios=[0.5, 1.0, 2.0],
                       scales=[1.0, 2.0, 4.0],
                       pre_nms_topk=6000,
                       post_nms_topk=1000,
                       nms_thresh=0.7,
                       min_size=1.0,
                       device=device):
    """
    rpn_cls: list of tensors [B, H*W*num_anchors, 2]
    rpn_bbox: list of tensors [B, H*W*num_anchors, 4]
    feature_shapes: [(H,W), ...] cho từng level
    image_size: (H_img, W_img)
    """
    proposals_all = []

    B = rpn_cls.shape[0]
    scores = rpn_cls.softmax(dim=-1)[..., 1]  # foreground score
    start = 0
    for lvl, (H, W) in enumerate(feature_shapes):
        stride = strides[lvl]
        A = len(ratios) * len(scales)
        end = start + H * W * A

        # Tạo anchor cho level này
        shifts_x = torch.arange(0, W * stride, step=stride, device=device)
        shifts_y = torch.arange(0, H * stride, step=stride, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)

        base_anchors = generate_anchors(stride, ratios, scales, device)
        anchors = (base_anchors[None, :, :] + shift[:, None, :]).reshape(-1, 4)

        # Decode boxes
        deltas = rpn_bbox[:, start:end, :]  # [B, N, 4]
        batch_scores = scores[:, start:end]  # [B, N]

        for b in range(B):
            boxes = decode_boxes(anchors, deltas[b])
            # Clip to image
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_size[1])
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_size[0])
            # Remove too small boxes
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            keep = (ws >= min_size) & (hs >= min_size)
            boxes = boxes[keep]
            scores_keep = batch_scores[b][keep]

            # Chọn top pre_nms
            num_topk = min(pre_nms_topk, boxes.shape[0])
            scores_topk, idx_topk = scores_keep.topk(num_topk)
            boxes = boxes[idx_topk]

            # NMS
            keep_idx = nms(boxes, scores_topk, nms_thresh)
            keep_idx = keep_idx[:post_nms_topk]
            boxes = boxes[keep_idx]
            scores_final = scores_topk[keep_idx]

            batch_idx = torch.full((boxes.shape[0], 1), b, device=device)
            proposals = torch.cat([batch_idx, boxes], dim=1)
            proposals_all.append(proposals)

        start = end

    return torch.cat(proposals_all, dim=0), scores_final


def encode_boxes_rpn(gt_boxes, anchors):
    """
    Convert GT boxes (x1,y1,x2,y2) thành delta so với anchors
    anchors: [N,4], gt_boxes: [N,4] (đã match với anchors)
    """
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]

    dx = (gx - ax) / aw
    dy = (gy - ay) / ah
    dw = torch.log(gw / aw)
    dh = torch.log(gh / ah)

    return torch.stack([dx, dy, dw, dh], dim=1)
def compute_loss_rpn(rpn_cls_logits, rpn_bbox_preds, anchors, gt_boxes,
                     pos_thresh=0.7, neg_thresh=0.3, batch_size_per_image=256, device=device):
    """
    rpn_cls_logits: [N, 2] logit trước softmax
    rpn_bbox_preds: [N, 4] bbox deltas
    anchors: [N, 4] anchors đã sinh ra
    gt_boxes: [M, 4] ground truth boxes
    """
    N = anchors.shape[0]
    device = anchors.device

    if gt_boxes.numel() == 0:
        # Không có object -> tất cả anchors là background
        labels = torch.zeros((N,), dtype=torch.long, device=device)
        cls_loss = F.cross_entropy(rpn_cls_logits, labels)
        reg_loss = torch.tensor(0.0, device=device)
        return cls_loss, reg_loss, labels

    # ---- 1. Tính IoU giữa anchors và GT ----
    ious = box_iou(anchors, gt_boxes)  # [N, M]
    max_iou, gt_idx = ious.max(dim=1)

    # ---- 2. Gán nhãn ----
    labels = torch.full((N,), -1, dtype=torch.long, device=device)  # -1 = ignore
    labels[max_iou >= pos_thresh] = 1  # positive
    labels[max_iou <= neg_thresh] = 0  # negative

    # Đảm bảo ít nhất một anchor cho mỗi GT box được gán positive
    max_iou_per_gt, _ = ious.max(dim=0)
    for i, gt_box in enumerate(gt_boxes):
        anchor_with_max = torch.nonzero(ious[:, i] == max_iou_per_gt[i], as_tuple=True)[0]
        labels[anchor_with_max] = 1

    # ---- 3. Sampling để cân bằng pos/neg ----
    num_pos = int(batch_size_per_image * 0.5)
    pos_idx = torch.nonzero(labels == 1, as_tuple=True)[0]
    neg_idx = torch.nonzero(labels == 0, as_tuple=True)[0]

    if pos_idx.numel() > num_pos:
        disable_idx = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:pos_idx.numel() - num_pos]]
        labels[disable_idx] = -1  # bỏ bớt positive

    num_neg = batch_size_per_image - (labels == 1).sum()
    if neg_idx.numel() > num_neg:
        disable_idx = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:neg_idx.numel() - num_neg]]
        labels[disable_idx] = -1  # bỏ bớt negative

    # ---- 4. Classification Loss ----
    valid_idx = labels >= 0  # bỏ ignore
    cls_loss = F.cross_entropy(rpn_cls_logits[valid_idx], labels[valid_idx])

    # ---- 5. Regression Loss ----
    pos_idx = labels == 1
    if pos_idx.sum() > 0:
        matched_gt_boxes = gt_boxes[gt_idx[pos_idx]]
        target_deltas = encode_boxes_rpn(matched_gt_boxes, anchors[pos_idx])
        reg_loss = F.smooth_l1_loss(rpn_bbox_preds[pos_idx], target_deltas, reduction="mean")
    else:
        reg_loss = torch.tensor(0.0, device=device)

    return cls_loss, reg_loss, labels

import torch
import torch.nn.functional as F

def encode_boxes_roi(gt, rois):
    rx, ry = (rois[:,0]+rois[:,2])/2, (rois[:,1]+rois[:,3])/2
    rw, rh = rois[:,2]-rois[:,0], rois[:,3]-rois[:,1]
    gx, gy = (gt[:,0]+gt[:,2])/2, (gt[:,1]+gt[:,3])/2
    gw, gh = gt[:,2]-gt[:,0], gt[:,3]-gt[:,1]
    return torch.stack([(gx-rx)/rw, (gy-ry)/rh, torch.log(gw/rw), torch.log(gh/rh)],1)

def compute_loss_detection(cls_logits, bbox_deltas, proposals, gt_boxes, gt_labels,
                           fg_thresh=0.5, bg_thresh=0.1, batch_size=128, fg_fraction=0.25):
    """
    cls_logits: [N, num_classes]
    bbox_deltas: [N, num_classes*4]
    proposals: [N, 4] (x1,y1,x2,y2)
    gt_boxes: [M, 4], gt_labels: [M]
    """
    from torchvision.ops import box_iou
    device = cls_logits.device
    ious = box_iou(proposals, gt_boxes)
    max_iou, gt_idx = ious.max(1)

    labels = gt_labels[gt_idx]
    labels[max_iou < fg_thresh] = 0  # background label = 0
    labels[max_iou < bg_thresh] = -1 # ignore
    
    # sampling
    fg_idx = torch.where(labels>0)[0]
    bg_idx = torch.where(labels==0)[0]
    num_fg = min(int(batch_size*fg_fraction), fg_idx.numel())
    num_bg = batch_size - num_fg
    perm_fg = fg_idx[torch.randperm(fg_idx.numel())[:num_fg]]
    perm_bg = bg_idx[torch.randperm(bg_idx.numel())[:num_bg]]
    idx = torch.cat([perm_fg, perm_bg])
    
    # classification loss
    cls_loss = F.cross_entropy(cls_logits[idx], labels[idx].clamp(min=0))

    # bbox regression loss chỉ với FG
    if num_fg > 0:
        fg_labels = labels[perm_fg]
        fg_boxes = proposals[perm_fg]
        fg_gt = gt_boxes[gt_idx[perm_fg]]
        targets = encode_boxes_roi(fg_gt, fg_boxes)
        
        # chọn đúng bbox delta theo class
        fg_bbox_pred = bbox_deltas[perm_fg].view(num_fg, -1, 4)
        fg_bbox_pred = fg_bbox_pred[torch.arange(num_fg), fg_labels]
        reg_loss = F.smooth_l1_loss(fg_bbox_pred, targets)
    else:
        reg_loss = torch.tensor(0., device=device)
    
    return cls_loss, reg_loss, labels

def compute_total_loss(rpn_cls_logits, rpn_bbox_preds, anchors,
                       rpn_feature_shapes, gt_boxes, gt_labels,
                       roi_cls_logits, roi_bbox_deltas, proposals,
                       mask_visible_logits, mask_amodal_logits, targets,
                       lambda_rpn_cls=1.0, lambda_rpn_reg=1.0,
                       lambda_det_cls=1.0, lambda_det_reg=1.0,
                       lambda_mask=1.0,
                       device=device):
    """
    Tính tổng loss cho Mask R-CNN với dual-mask head.

    Parameters
    ----------
    rpn_cls_logits: [N, 2] logit từ RPN
    rpn_bbox_preds: [N, 4] delta từ RPN
    anchors: [N, 4] anchors sinh ra
    rpn_feature_shapes: list[(H, W)] để biết số anchor per level (nếu cần)
    gt_boxes: [M, 4] GT boxes
    gt_labels: [M] nhãn GT
    roi_cls_logits: [K, num_classes] logit từ detection head
    roi_bbox_deltas: [K, num_classes*4] bbox delta từ detection head
    proposals: [K, 4] proposals đã chọn sau NMS
    mask_visible_logits: [P, num_classes, 28, 28] output mask head
    mask_amodal_logits:  [P, num_classes, 28, 28] output mask head
    targets: list[dict] chứa 'boxes', 'labels', 'visible_masks', 'amodal_masks'

    Return:
        dict với từng loss + tổng loss
    """

    # =========================
    # 1. RPN Loss
    # =========================
    from torchvision.ops import box_iou

    # (a) Tính IoU giữa anchors và GT
    ious = box_iou(anchors, gt_boxes) if gt_boxes.numel() > 0 else torch.zeros((anchors.shape[0], 0), device=device)
    max_iou, matched_idx = ious.max(dim=1) if gt_boxes.numel() > 0 else (torch.zeros(anchors.shape[0], device=device), torch.full((anchors.shape[0],), -1, dtype=torch.long, device=device))

    labels = torch.full((anchors.shape[0],), -1, dtype=torch.long, device=device)
    if gt_boxes.numel() > 0:
        labels[max_iou >= 0.7] = 1
        labels[max_iou <= 0.3] = 0

        # đảm bảo mỗi GT có ít nhất 1 anchor match
        max_iou_per_gt, _ = ious.max(dim=0)
        for i, gt_box in enumerate(gt_boxes):
            anchor_with_max = torch.nonzero(ious[:, i] == max_iou_per_gt[i], as_tuple=True)[0]
            labels[anchor_with_max] = 1
    else:
        labels[:] = 0  # không có object -> background

    # chọn các anchor hợp lệ (ko bị ignore)
    valid_idx = labels >= 0
    rpn_cls_loss = F.cross_entropy(rpn_cls_logits[valid_idx], labels[valid_idx])

    # regression loss (chỉ với positive anchors)
    pos_idx = labels == 1
    if pos_idx.sum() > 0 and gt_boxes.numel() > 0:
        matched_gt_boxes = gt_boxes[matched_idx[pos_idx]]
        target_deltas = encode_boxes_rpn(matched_gt_boxes, anchors[pos_idx])
        rpn_reg_loss = F.smooth_l1_loss(rpn_bbox_preds[pos_idx], target_deltas, reduction="mean")
    else:
        rpn_reg_loss = torch.tensor(0.0, device=device)

    # =========================
    # 2. Detection Head Loss
    # =========================
    det_cls_loss, det_reg_loss, det_labels = compute_loss_detection(
        roi_cls_logits, roi_bbox_deltas, proposals, gt_boxes, gt_labels,
        fg_thresh=0.5, bg_thresh=0.1, batch_size=128, fg_fraction=0.25
    )

    # =========================
    # 3. Mask Loss (Dual)
    # =========================
    mask_loss = compute_loss_dual_mask(mask_visible_logits, mask_amodal_logits,
                                       proposals, targets,
                                       loss_weight_visible=1.0,
                                       loss_weight_amodal=1.0)

    # =========================
    # 4. Tổng Loss
    # =========================
    total_loss = (
        lambda_rpn_cls * rpn_cls_loss +
        lambda_rpn_reg * rpn_reg_loss +
        lambda_det_cls * det_cls_loss +
        lambda_det_reg * det_reg_loss +
        lambda_mask * mask_loss
    )

    return {
        "rpn_cls_loss": rpn_cls_loss,
        "rpn_reg_loss": rpn_reg_loss,
        "det_cls_loss": det_cls_loss,
        "det_reg_loss": det_reg_loss,
        "mask_loss": mask_loss,
        "total_loss": total_loss
    }

import torch
import torch.nn as nn

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=5, use_amodal=True, device=device):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.use_amodal = use_amodal

        # ---- 1. Backbone & FPN ----
        self.backbone = ResNetBackbone(pretrained=True)
        self.fpn = FPN()

        # ---- 2. RPN ----
        self.rpn = RPN(in_channels=256, num_anchors=9)

        # ---- 3. Detection Head ----
        self.det_head = DetectionHead(in_channels=256, num_classes=num_classes)

        # ---- 4. Dual Mask Head ----
        self.mask_head = DualMaskHead(in_channels=256, num_classes=num_classes)

    def forward(self, images, anchors=None, targets=None):
        """
        images: Tensor [B, C, H, W]
        anchors: anchors đã được sinh (tuỳ bạn truyền vào từ ngoài)
        targets: list[dict] (chỉ dùng khi training)
        """
        B, _, H, W = images.shape
        device = images.device

        # ---- 1. Feature Extract ----
        c2, c3, c4, c5 = self.backbone(images)
        fpn_features = self.fpn([c2, c3, c4, c5])  # [P2, P3, P4, P5, P6]

        # ---- 2. RPN ----
        rpn_cls_logits, rpn_bbox_preds = self.rpn(fpn_features)

        # ---- 3. Proposal Generation ----
        feature_shapes = [(f.shape[2], f.shape[3]) for f in fpn_features]
        proposals, _ = generate_proposals(
            rpn_cls_logits, rpn_bbox_preds, feature_shapes, image_size=(H, W),
            device=device
        )  # [N, 5]

        # ---- 4. Detection Head ----
        cls_logits, bbox_deltas = self.det_head(fpn_features[:-1], proposals, image_shape=(H, W))

        # ---- 5. Dual Mask Head ----
        visible_logits, amodal_logits = self.mask_head(fpn_features[:-1], proposals)

        if self.training:
            assert targets is not None, "Targets must be provided in training mode"
            # Gộp GT cho toàn batch
            gt_boxes = torch.cat([t["boxes"].to(device) for t in targets], dim=0)
            gt_labels = torch.cat([t["labels"].to(device) for t in targets], dim=0)

            losses = compute_total_loss(
                rpn_cls_logits.view(-1, 2),
                rpn_bbox_preds.view(-1, 4),
                anchors=anchors if anchors is not None else torch.empty(0, 4, device=device),
                rpn_feature_shapes=feature_shapes,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                roi_cls_logits=cls_logits,
                roi_bbox_deltas=bbox_deltas,
                proposals=proposals[:, 1:5],  # bỏ batch_idx
                mask_visible_logits=visible_logits,
                mask_amodal_logits=amodal_logits,
                targets=targets,
                device=device
            )
            return losses

        else:
            return {
                "proposals": proposals,
                "cls_logits": cls_logits,
                "bbox_deltas": bbox_deltas,
                "visible_mask_logits": visible_logits,
                "amodal_mask_logits": amodal_logits if self.use_amodal else None
            }


def train_mask_rcnn(model, dataloader, num_epochs=10, lr=1e-4, save_dir="checkpoints", device=device):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        rpn_cls_total, rpn_reg_total = 0.0, 0.0
        det_cls_total, det_reg_total = 0.0, 0.0
        mask_total = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, targets in progress_bar:
            images = images.to(device)
            # Các dict trong targets vẫn trên CPU -> để compute_total_loss tự chuyển lên device
            optimizer.zero_grad()

            # Gọi model -> trả về dict loss
            losses = model(images, targets=targets)

            total_loss = losses["total_loss"]
            total_loss.backward()
            optimizer.step()

            # Tích lũy loss
            epoch_loss += total_loss.item()
            rpn_cls_total += losses["rpn_cls_loss"].item()
            rpn_reg_total += losses["rpn_reg_loss"].item()
            det_cls_total += losses["det_cls_loss"].item()
            det_reg_total += losses["det_reg_loss"].item()
            mask_total += losses["mask_loss"].item()
            num_batches += 1

            progress_bar.set_postfix({
                "total": f"{total_loss.item():.4f}",
                "rpn_cls": f"{losses['rpn_cls_loss'].item():.4f}",
                "rpn_reg": f"{losses['rpn_reg_loss'].item():.4f}",
                "det_cls": f"{losses['det_cls_loss'].item():.4f}",
                "det_reg": f"{losses['det_reg_loss'].item():.4f}",
                "mask": f"{losses['mask_loss'].item():.4f}"
            })

        # In log cuối epoch
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - "
              f"Total Loss: {epoch_loss/num_batches:.4f}, "
              f"RPN cls: {rpn_cls_total/num_batches:.4f}, "
              f"RPN reg: {rpn_reg_total/num_batches:.4f}, "
              f"Det cls: {det_cls_total/num_batches:.4f}, "
              f"Det reg: {det_reg_total/num_batches:.4f}, "
              f"Mask: {mask_total/num_batches:.4f}")

        # Lưu checkpoint
        ckpt_path = os.path.join(save_dir, f"mask_rcnn_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    from Dataload.Dataloader.Dataloader_MaskRCNN import COCOADataset, collate_fn
    from torch.utils.data import DataLoader

    # ---- 1. Load 1 batch từ dataloader ----
    cocoa_json_path = "Datasets/train/cocoa_format_annotations.json"
    img_dir = "Datasets/train/images"
    depth_dir = "Datasets/train/depths"


    # 1. Dataset + Dataloader
    dataset = COCOADataset(
        json_path="Datasets/train/cocoa_format_annotations.json",
        img_dir="Datasets/train/images",
        depth_dir="Datasets/train/depths"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # 2. Model
    model = MaskRCNN(num_classes=5, use_amodal=True)

    # 3. Train
    train_mask_rcnn(model, dataloader, num_epochs=10, lr=1e-4, save_dir="checkpoints")


    
    