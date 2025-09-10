import json
import re
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
from pathlib import Path
from typing import List, Tuple, Dict

# Map token -> class id (không phải thứ tự kênh)
CLASS_MAP = {"buffer": 0, "shelf": 1, "pallet": 2, "transporter": 3}
IGNORE_LABEL = 255

# Thứ tự KÊNH (ưu tiên) khi xuất: transporter > pallet > shelf > buffer
CHANNEL_ORDER = ["transporter", "pallet", "shelf", "buffer"]
NAME2CHAN = {n: i for i, n in enumerate(CHANNEL_ORDER)}  # transporter->0, pallet->1,...

OBJ_TAG_RE = re.compile(r"<([a-zA-Z_]+?(?:_\d+)?)>")

def extract_objects(sample: Dict) -> List[str]:
    text = None
    if sample.get("rephrase_conversations"):
        text = sample["rephrase_conversations"][0]["value"]
    elif sample.get("conversations"):
        text = sample["conversations"][0]["value"]
    else:
        raise ValueError("Không có conversations/rephrase_conversations hợp lệ.")
    return OBJ_TAG_RE.findall(text)

def token_to_basename(token: str) -> str:
    return token.split("_")[0].lower()

def decode_rle_to_mask(rle_dict: Dict) -> np.ndarray:
    m = mask_utils.decode({'size': rle_dict['size'], 'counts': rle_dict['counts']})
    if m.ndim == 3: m = m[:, :, 0]
    return (m > 0).astype(np.uint8)

def build_segmentation_label_4ch(sample: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Trả về label_4ch shape (H, W, 4), kênh theo CHANNEL_ORDER:
    [:,:,0]=transporter, 1=pallet, 2=shelf, 3=buffer. Giá trị {0,1}.
    """
    objects = extract_objects(sample)
    rles = sample.get("rle", [])
    if not rles:
        raise ValueError("Không có danh sách RLE trong dict.")

    H, W = rles[0]["size"]
    label_4 = np.zeros((H, W, 4), dtype=np.uint8)  # multi-label, giữ chồng lấn

    n = min(len(objects), len(rles))
    if len(objects) != len(rles):
        print(f"[CẢNH BÁO] tokens={len(objects)} != rles={len(rles)} -> dùng {n} cặp đầu.")

    for i in range(n):
        base = token_to_basename(objects[i])        # 'pallet_0' -> 'pallet'
        if base not in NAME2CHAN:
            # Bỏ qua đối tượng không thuộc 4 lớp trên
            continue
        ch = NAME2CHAN[base]                        # kênh theo CHANNEL_ORDER
        m = decode_rle_to_mask(rles[i]).astype(bool)
        label_4[..., ch] |= m                       # union theo lớp (không mất phần bị che)

    info = {
        "id": sample.get("id"),
        "image": sample.get("image"),
        "channels": CHANNEL_ORDER,                  # thứ tự kênh
        "size": (H, W),
    }
    return label_4, info

def save_4ch(label_4: np.ndarray, save_path: str):
    """
    Lưu 4 kênh theo phần mở rộng:
      - .npz: np.savez_compressed(mask=label_4)
      - .tif/.tiff/.png: RGBA (mỗi kênh 0/255)
    """
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext == ".npz":
        np.savez_compressed(p, mask=label_4.astype(np.uint8))
    elif ext in (".tif", ".tiff", ".png"):
        Image.fromarray(label_4.astype(np.uint8) * 255, mode="RGBA").save(p)
    else:
        # mặc định .npz nếu đuôi lạ
        np.savez_compressed(str(p) + ".npz", mask=label_4.astype(np.uint8))

def process_json_file(input_json_path: str, save_label_dir: str = None, batch_size: int = 10) -> int:
    """
    Xử lý file JSON theo batch để tránh tràn RAM.
    Đọc 1 JSON list samples. Với mỗi sample:
      - build_segmentation_label_4ch(sample) -> label_4 (H,W,4), info
      - Lưu file nhãn 4 kênh theo tên ảnh vào save_label_dir
    Trả về: số lượng sample đã xử lý.
    """
    import gc  # Garbage collection
    
    with open(input_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    out_dir = None
    if save_label_dir:
        out_dir = Path(save_label_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    total_samples = len(dataset)
    
    # Xử lý theo batch để tránh tràn RAM
    for i in range(0, total_samples, batch_size):
        batch = dataset[i:i + batch_size]
        print(f"Đang xử lý batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} ({len(batch)} samples)")
        
        for sample in batch:
            try:
                label_4, info = build_segmentation_label_4ch(sample)
                
                if out_dir is not None:
                    img_name = info.get("image", f"unnamed_{processed_count}.png")
                    stem = Path(img_name).stem
                    out_path = out_dir / f"{stem}_label_4ch.npz"
                    save_4ch(label_4, str(out_path))
                    print(f"Đã lưu: {out_path}")
                
                processed_count += 1
                
                # Giải phóng memory sau mỗi sample
                del label_4, info
                
            except Exception as e:
                print(f"Lỗi xử lý sample {processed_count}: {e}")
                continue
        
        # Garbage collection sau mỗi batch
        gc.collect()
        print(f"Hoàn thành batch. RAM được giải phóng.")
    
    return processed_count

# -----------------------
if __name__ == "__main__":
    # Xử lý với batch_size nhỏ để tránh tràn RAM
    count = process_json_file(
        "/home/felix/Warehouse_Spatial_Agent/Datasets/rephrased_test.json", 
        "./Datasets/labels",
        batch_size=5  # Giảm batch_size nếu vẫn tràn RAM
    )
    print(f"Xử lý xong {count} mẫu.")