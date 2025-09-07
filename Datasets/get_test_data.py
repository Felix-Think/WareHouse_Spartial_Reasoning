from dotenv import load_dotenv
import os

load_dotenv()
from huggingface_hub import snapshot_download

# Chỉ tải thư mục test
local_path = snapshot_download(
    repo_id="nvidia/PhysicalAI-Spatial-Intelligence-Warehouse",
    repo_type="dataset",
    local_dir="./Datasets/warehouse_data",
    token=os.environ.get("HUGGINGFACE_API_KEY"),
    allow_patterns=["test/**", "test.json"]  # chỉ tải thư mục test và file test.json nếu cần
)
