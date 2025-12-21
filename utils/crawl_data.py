from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from pathlib import Path

# =========================
# Auth
# =========================
load_dotenv()
login()  # dùng HF_TOKEN trong env nếu có

# =========================
# Config
# =========================
REPO_ID = "nvidia/PhysicalAI-Spatial-Intelligence-Warehouse"
LOCAL_DIR = Path("./PhysicalAI_Warehouse")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# 👉 CHỈ TẢI 1 CHUNK / SPLIT
ALLOW_PATTERNS = [
    # "train/images/chunk_005.tar.gz",
    # "train/depths/chunk_000.tar.gz",
    # "val/images/chunk_000.tar.gz",
    # "test/images/chunk_005.tar.gz",
    # (nếu cần annotation / qa)
    "train.json",
    "val.json",
    "test.json",
]

# =========================
# Download
# =========================
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(LOCAL_DIR),
    allow_patterns=ALLOW_PATTERNS,
    resume_download=True,
    max_workers=8,
)

print("✅ Download finished:")
for p in ALLOW_PATTERNS:
    print(" -", LOCAL_DIR / p)
