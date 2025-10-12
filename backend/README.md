# Bộ chuyển đổi DOCX → JSON cho RAG

Tệp `tools/convert_docx_to_json.py` chịu trách nhiệm biến đổi nội dung tài liệu `.docx` thành dữ liệu JSON đã được chuẩn hóa để phục vụ pipeline truy xuất tri thức (RAG). Script tự động quét các tài liệu trong thư mục nguồn, bóc tách nội dung, phân đoạn theo ngữ cảnh và lưu lại kèm manifest theo dõi phiên bản.

- **Nguồn dữ liệu:** `data/raw/<ngôn_ngữ>/` (mặc định hỗ trợ `en`, `vi`).
- **Thư mục đầu ra:** `data/processed-json/<ngôn_ngữ>/`.
- **Manifest:** `data/processed-json/_manifest.json` lưu băm MD5 nhằm bỏ qua các tệp không thay đổi và dọn dẹp json lỗi thời.

## Các bước xử lý chính
- Tính băm MD5 để phát hiện thay đổi nội dung DOCX.
- Đọc tuần tự từng khối nội dung (đoạn văn, bảng) và chuyển thành văn bản thuần; bảng được ghép thành chuỗi dạng `hàng | cột`.
- Nhận diện tiêu đề theo cấp (Heading 1…6) nhằm xây dựng `outline` và đường dẫn tiêu đề cho từng đoạn.
- Gom nhóm nội dung thành các `chunk` mục tiêu 180 từ (tối đa 240, tối thiểu 60) để phục vụ truy vấn RAG.
- Ghi toàn bộ dữ liệu (metadata, thống kê, outline, chunks, full_text) vào tệp JSON tương ứng.

## Yêu cầu môi trường
- Python 3.10+ (khuyến nghị cùng phiên bản đã dùng cho dự án).
- Thư viện `python-docx` dùng để đọc tài liệu Microsoft Word.
- Thư viện `python-dotenv` (tùy chọn, nhưng cần nếu muốn script tự đọc cấu hình trong `.env`).

```powershell
pip install python-docx python-dotenv
```

## Cách chạy script
Thực thi từ thư mục `backend/` để các đường dẫn tương đối hoạt động chính xác:

```powershell
python tools/convert_docx_to_json.py
```

Sau khi chạy, chương trình sẽ báo cáo số lượng tệp đã xử lý/thay đổi và các tệp bị bỏ qua vì không đổi nội dung.

> Script tự cấu hình `stdout`/`stderr` ở chế độ UTF-8, do đó không cần đặt thủ công `PYTHONIOENCODING`. Nếu chạy trong môi trường không hỗ trợ `reconfigure` (ví dụ container tối giản), hãy đặt `PYTHONIOENCODING=utf-8` trước khi gọi script để tránh lỗi khi in tiếng Việt.

## Cấu trúc JSON đầu ra
Mỗi tệp JSON xuất ra chứa các trường chính:

- `doc_id`, `language`, `course_name`, `course_variant`, `course_code`
- `source_filename`, `source_relpath`, `source_hash`, `source_modified`
- `processed_at` (thời gian xử lý), `stats` (số khối, số từ, số bảng…)
- `outline` (danh sách tiêu đề), `chunks` (danh sách đoạn nội dung đã cắt)
- `full_text` (nội dung văn bản đầy đủ)

Các tệp JSON trong thư mục đầu ra có thể dùng trực tiếp cho bước nạp dữ liệu vào hệ thống chỉ mục hoặc cơ sở tri thức.

## Tuỳ chỉnh thêm
- Điều chỉnh hằng số `CHUNK_WORD_TARGET`, `CHUNK_WORD_MAX`, `CHUNK_MIN_WORDS` thông qua biến môi trường hoặc file `.env` (ví dụ trong `backend/.env`) để phù hợp với chiến lược cắt đoạn của bạn.
- Thêm ngôn ngữ mới bằng cách bổ sung mã vào hằng số `LANGUAGES` và tạo thư mục nguồn tương ứng trong `data/raw/`.

Việc cập nhật script hoặc cấu trúc dữ liệu nên đi kèm cập nhật README này để đảm bảo nhóm dễ dàng nắm được cách vận hành công cụ.
