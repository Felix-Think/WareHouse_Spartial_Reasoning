from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()  # đọc HUGGINGFACE_HUB_TOKEN từ .env

snapshot_download(
    repo_id="nvidia/PhysicalAI-Spatial-Intelligence-Warehouse",
    repo_type="dataset",

    # nơi lưu trên máy
    local_dir="PhysicalAI-Warehouse",

    # CHỈ lấy folder train_sample trên HF
    allow_patterns=["train_sample/**"],

    local_dir_use_symlinks=False,
    token=True,  # lấy token từ env: HUGGINGFACE_HUB_TOKEN
)
