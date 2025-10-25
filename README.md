# Khoa_Luan – Hệ thống RAG cho khoá luận

Hệ thống bao gồm:
- **Pipeline xử lý tài liệu** (Python trong `backend/tools/`) chuyển đổi giáo trình DOCX → JSON chunk → vector index, đồng thời quản lý tải model TEI/LocalAI.
- **Hạ tầng Local LLM** (Docker + TEI + LocalAI) để chạy embedding và chat hoàn toàn offline trên Windows 11 + WSL2.
- **Giao diện Streamlit** (`frontend/rag_gui.py`) cho phép upload tài liệu, chạy pipeline, đặt câu hỏi và xem citation.

Tài liệu này mô tả chi tiết luồng dữ liệu, cách vận hành từng thành phần và hướng dẫn triển khai trên PowerShell/WSL2.

---

## Tổng quan luồng dữ liệu

```
        DOCX nguồn (backend/data/raw/<lang>/)
                     │
                     ▼
  ingest_docx_pipeline.py ( extract → convert → build )
                     │
                     ▼
  backend/data/index/<model>  (index_manifest.json, meta.jsonl, index.faiss/vectors.npy)
                     │
                     ▼
  Streamlit UI ──► tìm kiếm top-k (FAISS hoặc vectors)
                     │
      TEI (embedding) + LocalAI/OpenAI chat
                     │
                     ▼
                Câu trả lời + citation
```

1. Người dùng đặt DOCX vào `backend/data/raw/<ngôn_ngữ>/`.
2. UI hoặc CLI chạy `ingest_docx_pipeline.py` để lần lượt:
   - Chuẩn hoá DOCX (heading/bảng) → JSON structured.
   - Chunk + ghi metadata → `backend/data/processed-json/`.
   - Gọi TEI/LocalAI embed → lưu index vào `backend/data/index/<model>/`.
3. Khi chat, UI nhúng câu hỏi qua TEI, truy vấn index backend (FAISS hoặc brute-force), tạo context từ `meta.jsonl` và gọi LocalAI/OpenAI để sinh câu trả lời kèm nguồn.

Nhờ vậy cả CLI lẫn UI đều dùng chung dữ liệu chuẩn, đảm bảo citation thống nhất.

---

## 1. Thiết lập nhanh

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Luôn kích hoạt `.venv` trước khi chạy script để đảm bảo phiên bản thư viện đồng nhất.
- Các tiện ích CLI nằm trong `backend/tools`. Kiểm tra tham số bằng `python backend\tools\<script>.py --help`.

---

## 2. Các script nền tảng

| Script | Chức năng | Lệnh ví dụ |
| --- | --- | --- |
| `extract_docx_sections.py` | Chuẩn hoá DOCX, tách heading/bảng, ghi JSON “structured” | `python backend\tools\extract_docx_sections.py` |
| `convert_docx_to_json.py` | Tạo chunk + metadata từ structured JSON | `python backend\tools\convert_docx_to_json.py` |
| `build_index.py` | Embed chunk bằng TEI/LocalAI và sinh index FAISS/bruteforce | `python backend\tools\build_index.py --model <model> --base-url http://localhost:8800` |
| `answer_rag.py` | CLI RAG đầy đủ: truy hồi + chat + citation (qua LocalAI) | `python backend\tools\answer_rag.py "Câu hỏi?" --model <model> --base-url http://localhost:8800` |
| `ingest_docx_pipeline.py` | Gom toàn bộ pipeline (DOCX → JSON → Index) vào một lệnh | xem mục 3 |
| `launch_tei.py` | Quản lý container TEI (list/start/stop) dựa trên `models.json` | `python backend\tools\launch_tei.py --list` |

---

## 3. Pipeline DOCX → JSON → Index (một lệnh)

Đặt DOCX vào `backend/data/raw/vi/` hoặc `backend/data/raw/en/`, sau đó chạy:

```powershell
python backend\tools\ingest_docx_pipeline.py ^
       --model AITeamVN-Vietnamese_Embedding_v2 ^
       --base-url http://localhost:8800 ^
       --langs vi en
```

Pipeline thực hiện:
1. `extract_docx_sections.py` – chuẩn hoá heading/bảng và ghi JSON structured + manifest.
2. `convert_docx_to_json.py` – sinh chunk kèm metadata vào `backend/data/processed-json/<lang>/`.
3. `build_index.py` – gọi TEI/LocalAI để embed và tạo index (FAISS hoặc bruteforce) trong `backend/data/index/<model>/`.

Tham số hữu ích:
- `--out-dir backend/data/index/custom_dir`, `--backend faiss|bruteforce`, `--batch-size`, `--min-words`, `--save-chunks`, `--build-dry-run`.
- `--skip-extract`, `--skip-convert`, `--skip-build` khi chỉ cần chạy một phần pipeline.
- `--build-extra-args -- --dry-run --legacy-max-len 700` để truyền thêm cờ cho `build_index.py` (nhớ thêm `--` trước danh sách arg bổ sung).

Sau khi xong, index sẵn sàng cho Streamlit và CLI ở `backend/data/index/<model>/` (gồm `index_manifest.json`, `meta.jsonl`, `index.faiss`/`vectors.npy`).

---

## 4. Chuẩn bị hạ tầng Local TEI + LocalAI

1. **Tải model embedding** (mỗi thư mục trong `backend/local-llm/Embedding/` phải có file trọng số ví dụ `model.safetensors`). Nếu thiếu, chạy một trong các script `backend/tools/download_*.py`.
2. **Tạo `.env` từ `.env.sample`** để định nghĩa `TEI_BASE_URL`, `LOCALAI_BASE_URL`, `EMBEDDING_MODEL` mặc định…
3. **Khởi động stack Docker** (TEI + LocalAI):
   ```powershell
   docker compose up -d
   docker ps    # xác nhận các container tei-*, localai chạy
   ```
4. **(Tuỳ chọn) Điều khiển TEI riêng** bằng `launch_tei.py` nếu bạn muốn bật/tắt từng model:
   ```powershell
   python backend\tools\launch_tei.py --model AITeamVN-Vietnamese_Embedding_v2 --runtime cpu --detach
   ```
5. **Chạy pipeline** (mục 3) để tạo index cho model đang dùng.

---

## 5. Giao diện Streamlit (UI thống nhất backend)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend\rag_gui.py
```

Trong sidebar bạn có thể:
- Nhập OpenAI API key (dùng cho chat khi cần, nhưng truy hồi vẫn yêu cầu Local TEI).
- Chọn model TEI, runtime, kiểm tra Docker, tải model và bật/tắt container qua `launch_tei.py`.
- Upload DOCX (được lưu ở `backend/data/raw/uploads/<ext>/` cho pipeline xử lý sau).
- Nhấn **Rebuild backend index** → UI sẽ chạy `ingest_docx_pipeline.py` với các DOCX hiện có và làm mới cache index.

Khi chat:
- UI chỉ cho phép truy vấn nếu `Embedding source = Local TEI`, TEI đang chạy, model đã tải và index tồn tại.
- UI dùng cùng index với backend (`backend/data/index/<model>/`), đọc `meta.jsonl` và `index_manifest.json` rồi tìm kiếm top-k qua FAISS/vectors giống CLI, đảm bảo citation thống nhất.

Lưu ý: Luồng rebuild bằng PDF trong UI đã bị loại bỏ; mọi dữ liệu phải đi qua pipeline DOCX chuẩn.

---

## 6. Kiểm thử nhanh bằng CLI

```powershell
python backend\tools\answer_rag.py "Nội dung câu hỏi" ^
       --model AITeamVN-Vietnamese_Embedding_v2 ^
       --base-url http://localhost:8800 ^
       --k 5
```

Hoặc chỉ kiểm tra truy hồi:
```powershell
python backend\tools\search_index.py "Từ khoá" --model AITeamVN-Vietnamese_Embedding_v2 --base-url http://localhost:8800
```

---

## 7. Khắc phục sự cố

- **TEI không khởi động**: `docker compose logs tei-<model> --tail=200`, kiểm tra đường dẫn model mount đúng chưa.
- **LocalAI không thấy model chat**: đảm bảo `backend/local-llm/chat-models` chứa trọng số và `docker-compose.yml` mount đúng thư mục.
- **`build_index.py` báo thiếu FAISS**: cài `faiss-cpu` phù hợp, hoặc chuyển `--backend bruteforce` để lưu vector NumPy.
- **Pipeline timeout khi embed**: giảm `--batch-size` hoặc tăng `--embed-timeout` trong `backend/.env` hay tham số dòng lệnh.
- **UI báo “chưa có index”**: chắc chắn đã chạy “Rebuild backend index” (hoặc chạy `ingest_docx_pipeline.py`) sau khi thêm DOCX mới.
- **GPU**: nếu muốn tăng tốc, chọn image TEI hỗ trợ GPU và cài NVIDIA Container Toolkit trong WSL2.

---

## 8. Bảo mật & vận hành

- Đóng gói Streamlit bằng `frontend/Dockerfile` và đặt sau reverse proxy có TLS khi triển khai thực tế.
- Thêm cơ chế xác thực (token, SSO…) trước khi mở UI cho người dùng cuối.
- Sao lưu định kỳ `backend/data/index/` và `meta.jsonl` để tránh phải embed lại toàn bộ tài liệu khi có sự cố.
- Theo dõi dung lượng đĩa: model TEI + LocalAI và index FAISS có thể chiếm vài GB.

---

**Ghi chú:** luôn chạy `python backend\tools\<script>.py --help` để xem đầy đủ tham số; file `backend/local-llm/Embedding/README.md` giải thích chi tiết cách tải model TEI và cấu hình runtime. README này phản ánh luồng mới (UI dùng chung index backend); nếu bạn mở rộng thêm mode khác, hãy cập nhật tài liệu tương ứng.
