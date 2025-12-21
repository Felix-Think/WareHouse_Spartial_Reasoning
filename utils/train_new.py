import os
import json

root_dir = "PhysicalAI_Warehouse"
image_dir = os.path.join(root_dir, "train/images")
train_json_path = os.path.join(root_dir, "train.json")
train_new_path = os.path.join(root_dir, "train_new.json")

# 1. Lấy danh sách tên ảnh
image_files = set(os.listdir(image_dir))  # dùng set cho nhanh

# 2. Load train.json
with open(train_json_path, "r") as f:
    data = json.load(f)

# 3. Lọc dữ liệu
new_data = [sample for sample in data if sample.get("image") in image_files]

# 4. Ghi ra file mới
with open(train_new_path, "w") as f:
    json.dump(new_data, f, indent=2)

print(f"Original samples: {len(data)}")
print(f"Filtered samples: {len(new_data)}")
