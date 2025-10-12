# Khoa_Luan

## 🌐 Giới thiệu
Dự án phục vụ xây dựng và vận hành hệ thống RAG cho khóa luận. Thư mục `backend/tools` tập trung các tiện ích Python để chuyển đổi dữ liệu nguồn, xây chỉ mục vector và chạy truy vấn thử nghiệm.

## 🚀 Thiết lập môi trường Python
1. Tạo môi trường ảo tại thư mục gốc:
   ```powershell
   python -m venv .venv
   ```
2. Kích hoạt môi trường ảo (PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *Nếu dùng Command Prompt, hãy thay bằng `.\.venv\Scripts\activate.bat`.*
3. Cài đặt thư viện từ danh sách `python-libraries.txt`:
   ```powershell
   pip install -r python-libraries.txt
   ```

> 💡 Giữ môi trường ảo luôn mở khi thao tác với các script Python để đảm bảo dùng đúng phiên bản thư viện.

## 🛠️ Chạy công cụ Python trong `backend/tools`
- Thực thi trực tiếp từ thư mục gốc của dự án, ví dụ:
  ```powershell
  python backend\tools\ten_script.py --help
  ```
- Mỗi script đều hỗ trợ tham số dòng lệnh; chạy kèm `--help` để xem chi tiết cách sử dụng.

## 📚 Ba script quan trọng

### `convert_docx_to_json.py`
- **Chức năng:** Chuyển các tệp DOCX trong `backend/data/raw` sang JSON đã tiền xử lý, phục vụ quá trình xây index.
- **Lệnh chạy:**
  ```powershell
  python backend\tools\convert_docx_to_json.py
  ```

### `build_index.py`
- **Chức năng:** Tạo vector index từ dữ liệu JSON đã xử lý, hỗ trợ nhiều backend như FAISS hoặc tìm kiếm tuyến tính.
- **Lệnh chạy:**
  ```powershell
  python backend\tools\build_index.py --data-dir backend/data/processed-json --out-dir backend/data/index
  ```

### `answer_rag.py`
- **Chức năng:** Đặt câu hỏi RAG dựa trên index hiện có và trả về câu trả lời kèm trích dẫn nguồn.
- **Lệnh chạy:**
  ```powershell
  python backend\tools\answer_rag.py "câu hỏi của bạn"
  ```

> 📎 Điều chỉnh lại các tham số như `--data-dir`, `--out-dir`, `--model` hoặc `--base-url` theo cấu hình thực tế trước khi chạy trên môi trường production.
