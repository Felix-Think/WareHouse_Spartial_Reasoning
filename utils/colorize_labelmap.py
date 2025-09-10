#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy as np
from PIL import Image

# Thứ tự kênh trong file nhãn 4-ch
CHANNEL_ORDER = ["transporter", "pallet", "shelf", "buffer"]  # ch0..ch3
NAME2COLOR = {
    "buffer":      (0, 170, 255, 128),    # cyan
    "shelf":       (255, 140, 0, 128),    # orange
    "pallet":      (144, 238, 144, 128),  # lightgreen
    "transporter": (220, 20, 60, 128),    # crimson
}

def _load_4ch_label(label_path: str) -> np.ndarray:
    """
    Trả về np.uint8 shape (H, W, 4) theo CHANNEL_ORDER.
    Hỗ trợ:
      - .npz  với key 'mask' (HxWx4, 0/1 hoặc 0/255)
      - .tif/.tiff/.png RGBA (4 kênh 0/255)
    """
    ext = osp.splitext(label_path)[1].lower()
    if ext == ".npz":
        arr = np.load(label_path)["mask"]  # kỳ vọng HxWx4
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.max() == 1:
            arr = (arr * 255).astype(np.uint8)
        return arr
    elif ext in (".tif", ".tiff", ".png"):
        img = Image.open(label_path).convert("RGBA")
        return np.array(img, dtype=np.uint8)
    else:
        raise ValueError(f"Không hỗ trợ định dạng: {ext}")

def colorize_labelmap_4ch(label4_path: str, output_path: str, rgb_path: str = None):
    # 1) Load label 4 kênh
    label4 = _load_4ch_label(label4_path)          # (H,W,4) 0/255
    H, W, C = label4.shape
    assert C == 4, f"Label phải có 4 kênh, nhận {C}"

    # 2) Ảnh nền (RGB nếu có)
    if rgb_path and osp.isfile(rgb_path):
        base = Image.open(rgb_path).convert("RGBA").resize((W, H), Image.BILINEAR)
    else:
        base = Image.new("RGBA", (W, H), (30, 30, 30, 255))

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # 3) Vẽ theo thứ tự ưu tiên (thấp -> cao):
    # buffer (0) -> shelf (1) -> pallet (2) -> transporter (3)
    PRIORITY_DRAW = ["buffer", "shelf", "pallet", "transporter"]
    name2idx = {n: i for i, n in enumerate(CHANNEL_ORDER)}

    for name in PRIORITY_DRAW:
        cid = name2idx[name]
        mask = label4[..., cid] > 0
        if not mask.any():
            continue
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        color = NAME2COLOR[name]
        colored = Image.new("RGBA", (W, H), color)
        overlay.paste(colored, (0, 0), mask_img)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
    out.save(output_path)

if __name__ == "__main__":
    label4_path = "/home/felix/Warehouse_Spatial_Agent/labels_4ch/000242_label_4ch.npz"
    output_path = "label_vis.png"
    # rgb_path = "014324.png"  # nếu muốn overlay lên ảnh RGB
    colorize_labelmap_4ch(label4_path, output_path)
