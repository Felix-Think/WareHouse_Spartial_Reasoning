# WareHouse_Spartial_Reasoning

## Thiết lập ban đầu

1. **Tạo file `.env`** trong thư mục gốc với các API key cần thiết. Ví dụ:

```env
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
LANGCHAIN_PROJECT=your_langchain_project
LANGCHAIN_TRACING_V2=true
SCRAPIN_API_KEY=your_scrapin_api_key
```

> **Lưu ý:** Thay thế các giá trị bằng API key thực tế của bạn.


2. **Khởi tạo môi trường ảo với Conda**

Khuyến nghị sử dụng môi trường ảo để quản lý phụ thuộc:

```bash
# Tạo môi trường mới tên warehouse-agent với Python 3.13
conda create -n warehouse-agent python=3.13
conda activate warehouse-agent
```

3. **Cài đặt các thư viện cần thiết bằng Poetry**

```bash
# Cài đặt các package dựa trên pyproject.toml
poetry install
```

4. **Tải dữ liệu kho**

Chạy lệnh sau trong terminal để tải dữ liệu về thư mục `warehouse_data`:

```bash
python Datasets/get_data.py
```

Sau khi chạy xong, dữ liệu sẽ được lưu tại thư mục `warehouse_data/`.