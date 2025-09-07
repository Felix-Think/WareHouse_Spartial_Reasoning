from dotenv import load_dotenv
import os
load_dotenv()
from huggingface_hub import snapshot_download

# tải toàn bộ repo về local
local_path = snapshot_download(
    repo_id="nvidia/PhysicalAI-Spatial-Intelligence-Warehouse",
    repo_type="dataset",
    local_dir="./Datasets/warehouse_data",
    token=os.environ.get("HUGGINGFACE_API_KEY")
)

print(f"Dataset đã được lưu tại: {local_path}")
