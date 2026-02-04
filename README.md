# WareHouse Spatial Reasoning

📦 Dự án **Warehouse Spatial Reasoning**  
Mô hình học máy phục vụ bài toán suy luận không gian (spatial reasoning)
trong môi trường kho hàng.

---

## 🛠️ Chuẩn bị môi trường

### 1️⃣ Clone repository

```bash
git clone https://github.com/Felix-Think/WareHouse_Spartial_Reasoning.git
cd WareHouse_Spartial_Reasoning
```

---

## 🚀 Cài đặt `uv`

`uv` là Python package manager siêu nhanh do Astral phát triển.

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Kiểm tra

```bash
uv --version
```

---

## 📦 Cài đặt thư viện

Tạo môi trường ảo và cài dependencies:

```bash
uv venv
uv sync
```

---

## 🧠 Train mô hình (tuỳ chọn)

Nếu muốn train lại mô hình từ đầu:

```bash
python train_distance.py
```

Mô hình sau khi train sẽ được lưu dưới dạng file `.pth`
(ví dụ: `best_model.pth`).

---

## 🚀 Chạy demo Web (FastAPI)

Chạy web app demo bằng Uvicorn:

```bash
uv run uvicorn webapp.main:app --port 8000 --reload
```

Sau đó mở trình duyệt và truy cập:

```text
http://localhost:8000
```

---

## 🧪 Test API (tuỳ chọn)

Ví dụ gọi API bằng `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test.jpg"
```

Hoặc dùng Python:

```python
import requests

res = requests.post(
    "http://localhost:8000/predict",
    files={"file": open("test.jpg", "rb")},
)
print(res.json())
```

---

## 📁 Cấu trúc thư mục chính

```text
core/          # Logic mô hình & spatial reasoning
predict/       # Script inference
predict2/      # Phiên bản inference khác
utils/         # Hàm hỗ trợ
webapp/        # FastAPI web demo
train_distance.py
main.py
```

---

## 📝 Ghi chú

- Model có thể đã được train sẵn (`.pth`)
- Có thể thay model mới bằng cách train lại và ghi đè file
- Endpoint & logic web nằm trong `webapp/main.py`

---

## 👤 Tác giả

Felix (Huỳnh Văn Thịnh) - Nguyễn Ngọc Ấn - Nguyễn Văn Thắng  
GitHub: <https://github.com/Felix-Think>
