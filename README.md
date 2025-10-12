# Khoa_Luan

## Thiet lap moi truong Python
1. Tao moi truong ao tai thu muc goc:
   ```powershell
   python -m venv .venv
   ```
2. Kich hoat moi truong ao (PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *Neu dung Command Prompt, su dung `.\.venv\Scripts\activate.bat`.*
3. Cai thu vien tu danh sach `python-libraries.txt`:
   ```powershell
   pip install -r python-libraries.txt
   ```

## Chay cong cu Python trong `backend/tools`
- Chay truc tiep tu thu muc goc, vi du:
  ```powershell
  python backend\tools\ten_script.py
  ```
- Co the xem them tham so bang `python backend\tools\ten_script.py --help` (neu script ho tro).

## Ba script quan trong
- `convert_docx_to_json.py`: chuyen tai lieu DOCX trong `backend/data/raw` sang JSON da xu ly.
  ```powershell
  python backend\tools\convert_docx_to_json.py
  ```
- `build_index.py`: tao vector index tu du lieu JSON. Dieu chinh duong dan hoac cau hinh neu can.
  ```powershell
  python backend\tools\build_index.py --data-dir backend/data/processed-json --out-dir backend/data/index
  ```
- `answer_rag.py`: dat cau hoi RAG sau khi da co index; truyen cau hoi o cuoi lenh.
  ```powershell
  python backend\tools\answer_rag.py "cau hoi cua ban"
  ```
