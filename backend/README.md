# Bộ công cụ xử lý dữ liệu RAG

Bộ script trong `backend/tools/` đảm nhận toàn bộ vòng đời biến đổi tài liệu DOCX → JSON → vector và kiểm thử trả lời RAG. Dưới đây là mô tả vai trò từng file và cách vận hành.

## `convert_docx_to_json.py`
Chức năng:
- Quét tự động toàn bộ DOCX trong `backend/data/raw/<ngôn_ngữ>/` (mặc định `vi`, `en`).
- Tính MD5 từng tệp, so sánh với `_manifest.json` để bỏ qua tài liệu không đổi.
- Chuẩn hóa văn bản: tách đoạn, nhận diện heading, chuyển bảng thành chuỗi `hàng | cột`, phát hiện danh sách.
- Gom nội dung thành các `chunk` mục tiêu 180 từ (tối đa 240, tối thiểu 60), lưu breadcrumb/headings/phạm vi block phục vụ truy hồi.
- Ghi JSON đầu ra vào `backend/data/processed-json/<ngôn_ngữ>/` kèm metadata (`doc_id`, `course_code`, `outline`, `stats`, `full_text`…).

Tham số chunk có thể chỉnh trong `.env` (`CHUNK_WORD_TARGET`, `CHUNK_WORD_MAX`, `CHUNK_MIN_WORDS`). Chạy:

```powershell
python backend/tools/convert_docx_to_json.py
```

## `build_index.py`
Mục tiêu: dựng vector index từ các JSON đã xử lý, tối ưu cho RAG.

Điểm nổi bật:
- Đọc từng chunk trong JSON (ưu tiên trường `chunks`, fallback sang `sections` với phương pháp cắt legacy).
- Bỏ qua chunk quá ngắn (`--min-words`) và gắn nhãn `length_category` (short/medium/long) dựa vào `--short-threshold`, `--long-threshold` để gợi ý chiều dài câu trả lời.
- Ghép thêm ngữ cảnh (`course_name`, `course_code`, đường dẫn heading) vào text trước khi embed để truy vấn “đúng trọng tâm”.
- Gọi LocalAI `/v1/embeddings` theo batch (mặc định lấy cấu hình từ `.env`: `EMBEDDING_MODEL`, `INDEX_BATCH_SIZE`, `INDEX_EMBED_TIMEOUT`…).
- Chuẩn hóa L2 vector và lưu chỉ mục FAISS (`index.faiss`) hoặc brute-force (`vectors.npy`) tùy `--backend`/`VECTOR_INDEX_BACKEND`.
- Xuất `meta.jsonl` chứa metadata mỗi chunk và `index_manifest.json` ghi lại cấu hình, tổng vector, ngôn ngữ, thời gian tạo. Tuỳ chọn `--save-chunks` giúp lưu thêm `chunks.jsonl` (text + meta) hỗ trợ debug/answer.

Ví dụ dựng index:

```powershell
python backend/tools/build_index.py --batch-size 32 --langs vi en
```

Muốn kiểm tra trước khi embed: thêm `--dry-run`.

## `search_index.py`
Tiện ích CLI để kiểm tra nhanh kết quả truy hồi.

- Nhận query văn bản, embed qua LocalAI, tìm top-k vector từ index FAISS hoặc brute-force theo manifest.
- In ra từng kết quả dạng JSON gồm `score`, `filename`, `section_heading`, `chunk_id`, `language`, `length_category`…
- Không gọi mô hình chat, không trả về nội dung chunk → thích hợp kiểm thử xem truy vấn có bám đúng tài liệu hay không.

Ví dụ:

```powershell
python backend/tools/search_index.py "Instructor of Data Structures"
```

## `answer_rag.py`
Pipeline RAG hoàn chỉnh kết nối truy hồi và sinh câu trả lời.

- Thực hiện truy hồi giống `search_index.py`, sau đó cố gắng lấy lại nội dung chunk (ưu tiên `chunks.jsonl`; nếu không có sẽ tái dựng từ JSON nguồn).
- Định dạng mỗi context kèm citation `[i]` và in ra mục `--- Retrieved Contexts ---` để quan sát.
- Nếu không dùng `--show-only`, script chọn mô hình chat (theo `--chat-model` hoặc tự động từ `/v1/models`), gửi prompt gồm câu hỏi + context và tạo đáp án ở cùng ngôn ngữ câu hỏi.
- Kết quả gồm phần “Answer” và danh sách nguồn để đưa lên UI hoặc log.

Ví dụ:

```powershell
python backend/tools/answer_rag.py "Tên giảng viên của học phần Cấu trúc dữ liệu là gì?"
```

Tham số hữu ích: `--k` (số context), `--max-context-chars` (giới hạn dung lượng gửi cho chat), `--temperature`.

---

## Thiết lập môi trường

File `python-libraries.txt` tổng hợp các thư viện cần dùng (ví dụ `python-docx`, `requests`, `numpy`, `tqdm`, `faiss-cpu`). Cài đặt:

```powershell
pip install -r python-libraries.txt
```

Điều chỉnh `.env` trong thư mục `backend/` để cấu hình thống nhất cho pipeline:

```env
CHUNK_WORD_TARGET=150
CHUNK_WORD_MAX=220
CHUNK_MIN_WORDS=40
EMBEDDING_MODEL=granite-embedding-107m-multilingual
LOCALAI_BASE_URL=http://localhost:8080/v1
VECTOR_INDEX_BACKEND=faiss
INDEX_BATCH_SIZE=32
INDEX_MIN_CHUNK_WORDS=40
INDEX_SHORT_WORD_THRESHOLD=120
INDEX_LONG_WORD_THRESHOLD=220
INDEX_EMBED_TIMEOUT=120
LOCALAI_CHAT_MODEL=...
```

Sau mỗi lần cập nhật cấu trúc JSON hoặc logic xử lý, hãy đồng bộ README để đội phát triển dễ theo dõi luồng công việc.*** End Patch
