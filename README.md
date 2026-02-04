
<!-- markdownlint-disable -->
## 🚀 Cài đặt `uv`
### 🔹 macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

### 🔹 Kiểm tra cài đặt
```bash
uv --version
```

### 🔹 Cài đặt thư viện 
```bash
uv venv
uv sync

```
## Demo
```bash

uv run uvicorn webapp.main:app --port 8000 --reload

```

