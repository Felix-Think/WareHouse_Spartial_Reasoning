from dotenv import load_dotenv
import os
import tarfile
import glob
from pathlib import Path

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

print(f"Dataset test đã được lưu tại: {local_path}")

# Extract tất cả file tar.gz trong thư mục test
def extract_all_tar_files(base_dir):
    """Giải nén tất cả file .tar.gz trong thư mục test và xóa file nén"""
    base_path = Path(base_dir)
    test_path = base_path / "test"
    
    # Tìm tất cả file .tar.gz trong test/images và test/depths
    tar_files = []
    for pattern in ["test/images/*.tar.gz", "test/depths/*.tar.gz"]:
        tar_files.extend(glob.glob(str(base_path / pattern)))
    
    print(f"Tìm thấy {len(tar_files)} file tar.gz để giải nén...")
    
    for tar_file in tar_files:
        tar_path = Path(tar_file)
        extract_dir = tar_path.parent  # Giải nén vào cùng thư mục chứa file tar
        
        print(f"Đang giải nén: {tar_path.name}")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            
            # Xóa file nén sau khi giải nén thành công
            tar_path.unlink()
            print(f"✓ Đã giải nén và xóa: {tar_path.name}")
            
        except Exception as e:
            print(f"✗ Lỗi khi giải nén {tar_path.name}: {e}")

# Thực hiện extract
extract_all_tar_files("./Datasets/warehouse_data")
print("Hoàn thành giải nén tất cả file test data!")
