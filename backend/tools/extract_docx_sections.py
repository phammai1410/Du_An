

from __future__ import annotations

import json
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import os
import sys

from docx import Document
from docx.oxml.table import CT_Tbl  # type: ignore[attr-defined]
from docx.oxml.text.paragraph import CT_P  # type: ignore[attr-defined]
from docx.table import _Cell, Table  # type: ignore
from docx.text.paragraph import Paragraph
# định nghĩa các hằng số và đường dẫn thư mục
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "processed-structured"
MANIFEST_PATH = OUTPUT_DIR / "_manifest.json"

LANGUAGES = ("vi", "en")

# hàm để đảm bảo sử dụng mã hóa UTF-8 cho đầu vào/đầu ra tiêu chuẩn
# tránh lỗi mã hóa trên Windows
def _ensure_utf8_stdio() -> None:
    """Force UTF-8 stdio to avoid Windows codepage errors."""
    preferred_encoding = os.environ.get("PYTHONIOENCODING")
    if preferred_encoding:
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, LookupError):
                pass


_ensure_utf8_stdio()

# hàm để tạo thư mục đầu ra nếu chưa tồn tại
# đảm bảo rằng các file đã trích xuất có nơi để lưu trữ
def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# hàm để chuẩn hóa văn bản
# loại bỏ khoảng trắng thừa và chuẩn hóa Unicode
def normalize_text(value: str) -> str:
    """Return NFC normalized text with collapsed whitespace."""
    if not value:
        return ""
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()

# hàm để tạo slug từ văn bản
# sử dụng để tạo nhãn định danh cho các phần
def slugify(value: str) -> str:
    """Generate a lowercase ASCII slug for section labels."""
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    ascii_text = value.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text)
    ascii_text = ascii_text.strip("-").lower()
    return ascii_text

# hàm để tính toán hàm băm MD5 của một file
# sử dụng để kiểm tra thay đổi nội dung file
def compute_md5(path: Path) -> str:
    """Return md5 hash for the provided file path."""
    import hashlib

    hasher = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# hàm để xác định cấp độ tiêu đề từ tên kiểu dáng
# sử dụng để phân đoạn tài liệu thành các phần
def heading_level(style_name: Optional[str]) -> Optional[int]:
    if not style_name:
        return None
    match = re.search(r"heading\s*(\d+)", style_name, flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

# hàm để lặp qua các khối trong tài liệu
# bao gồm đoạn văn và bảng, theo thứ tự xuất hiện
# sử dụng để trích xuất nội dung có cấu trúc
def iter_blocks(document: Document) -> Iterable[Tuple[str, object]]:
    """Yield paragraphs and tables in document order."""
    parent_element = document._element  # type: ignore[attr-defined]
    for child in parent_element.body.iterchildren():  # type: ignore[attr-defined]
        if isinstance(child, CT_P):
            yield "paragraph", Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield "table", Table(child, document)

# hàm để kiểm tra xem đoạn văn có phải là một mục trong danh sách hay không
# sử dụng để đánh dấu các đoạn văn liệt kê
# trả về True nếu đoạn văn là mục danh sách, ngược lại False
def is_list_paragraph(paragraph: Paragraph) -> bool:
    props = paragraph._p.pPr  # type: ignore[attr-defined]
    return props is not None and props.numPr is not None


SECTION_LABEL_MAP = {
    "thong-tin-tong-quat": "general_information",
    "thong-tin-chung": "general_information",
    "khoa-vien-quan-ly-va-giang-vien-giang-day": "faculty_and_instructors",
    "mo-ta-hoc-phan": "course_description",
    "muc-tieu-hoc-phan": "course_goals",
    "chuan-dau-ra-hoc-phan": "learning_outcomes",
    "tai-lieu-hoc-tap": "learning_resources",
    "ke-hoach-day-hoc": "lesson_plan",
    "ke-hoach-va-noi-dung-day-hoc": "lesson_plan",
    "danh-gia-hoc-phan": "assessment",
    "ma-tran-de-thi": "exam_blueprint",
    "rubric": "rubrics",
    "phu-luc": "appendix",
    "quy-dinh-cua-hoc-phan": "course_policies",
}

# hàm để chuẩn hóa nhãn phần từ tiêu đề
# loại bỏ số thứ tự và tạo slug
# sử dụng để tạo nhãn định danh cho các phần
def normalize_section_label(title: str) -> str:
    cleaned = normalize_text(title)
    cleaned = re.sub(r"^[0-9]+(\.[0-9]+)*\s*", "", cleaned)
    slug = slugify(cleaned)
    if not slug:
        return "section"
    return SECTION_LABEL_MAP.get(slug, slug)


TABLE_NAME_PATTERNS = (
    re.compile(r"^(Bảng|Table)\s+[0-9A-Za-z\.]*[:\-\s]*(.+)?", re.IGNORECASE),
    re.compile(r"^(Rubric|Ma trận)[^:]*[:\-\s]*(.+)?", re.IGNORECASE),
    re.compile(r"^Ma trận", re.IGNORECASE),
)

# hàm để suy luận tên bảng từ các đoạn văn gần nhất
# sử dụng các mẫu regex để nhận diện tiêu đề bảng
def infer_table_name(candidates: Iterable[str]) -> Optional[str]:
    for candidate in reversed(list(candidates)):
        text = normalize_text(candidate)
        if not text:
            continue
        for pattern in TABLE_NAME_PATTERNS:
            match = pattern.match(text)
            if match:
                tail = match.group(0).strip()
                return tail
        # Fallback: đoạn văn ngắn ngay phía trước có thể được xem như một chú thích (caption).
        if len(text.split()) <= 7:
            return text
    return None

# hàm để chuyển đổi nội dung ô bảng thành văn bản
# loại bỏ trùng lặp và chuẩn hóa văn bản
# sử dụng để trích xuất nội dung từ các ô bảng
def cell_to_text(cell: _Cell) -> str:
    parts: List[str] = []
    for paragraph in cell.paragraphs:
        text = normalize_text(paragraph.text)
        if text:
            parts.append(text)
    unique_parts = []
    for item in parts:
        if not unique_parts or unique_parts[-1] != item:
            unique_parts.append(item)
    return " ".join(unique_parts).strip()

# hàm để phát hiện xem hàng đầu tiên của bảng có phải là hàng tiêu đề hay không
# sử dụng các đặc điểm như độ dài ô và từ khóa
def detect_header_row(rows: List[List[str]]) -> bool:
    if not rows:
        return False
    first_row = rows[0]
    if not any(first_row):
        return False
    # Heuristic: header cells are typically short or contain keywords.
    header_keywords = {"stt", "clo", "plo", "noi dung", "content", "week", "ti le", "ty le"}
    score = 0
    for cell_text in first_row:
        if not cell_text:
            continue
        words = cell_text.lower()
        if len(cell_text.split()) <= 5:
            score += 1
        if any(keyword in words for keyword in header_keywords):
            score += 1
    return score >= max(1, len(first_row) // 2)

# lớp dữ liệu để biểu diễn một phần trong tài liệu
@dataclass
class Section:
    section_id: str
    title: str
    level: int
    label: str
    path_titles: List[str]
    path_labels: List[str]
    content: List[Dict[str, object]] = field(default_factory=list)

# hàm để phân tích tên file và trích xuất thông tin khóa học
# sử dụng để tạo định danh tài liệu
def parse_filename(stem: str) -> Tuple[str, str, str]:
    parts = [part.strip() for part in stem.split("_") if part.strip()]
    if not parts:
        return stem, "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    course_name = parts[0]
    course_code = parts[-1]
    course_variant = "_".join(parts[1:-1]) if len(parts) > 2 else ""
    return course_name, course_variant, course_code

# hàm để trích xuất nội dung có cấu trúc từ tài liệu .docx
# sử dụng các hàm phụ để phân đoạn và trích xuất văn bản và bảng
# trả về một từ điển chứa dữ liệu đã trích xuất
def extract_document(doc_path: Path, language: str) -> Dict[str, object]:
    document = Document(doc_path)
    course_name, course_variant, course_code = parse_filename(doc_path.stem)
    base_id = course_code or slugify(course_name) or doc_path.stem
    doc_id = f"{language}-{slugify(base_id)}"

    sections: List[Section] = []
    section_stack: List[Section] = []
    root_section = Section(
        section_id=f"{doc_id}#root",
        title="Document",
        level=0,
        label="document",
        path_titles=[],
        path_labels=[],
    )
    section_stack.append(root_section)

    all_tables: List[Dict[str, object]] = []
    table_counter = 0
    content_counter = 0
    paragraph_counter = 0
    full_text_parts: List[str] = []
    recent_paragraphs: Deque[str] = deque(maxlen=3)
# hàm phụ để lấy phần hiện tại từ ngăn xếp
# sử dụng để xác định phần mà nội dung hiện tại thuộc về
    def current_section() -> Section:
        return section_stack[-1] if section_stack else root_section

    for block_index, (block_type, block) in enumerate(iter_blocks(document)):
        if block_type == "paragraph":
            paragraph: Paragraph = block  # type: ignore[assignment]
            raw_text = normalize_text(paragraph.text)
            if not raw_text:
                continue
            level = heading_level(paragraph.style.name if paragraph.style else "")
            if level is not None:
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()
                path_titles = [sec.title for sec in section_stack if sec.level > 0] + [raw_text]
                path_labels = [sec.label for sec in section_stack if sec.level > 0] + [
                    normalize_section_label(raw_text)
                ]
                section_id = f"{doc_id}#sec{len(sections)+1:03d}"
                section = Section(
                    section_id=section_id,
                    title=raw_text,
                    level=level,
                    label=normalize_section_label(raw_text),
                    path_titles=path_titles,
                    path_labels=path_labels,
                )
                sections.append(section)
                section_stack.append(section)
                recent_paragraphs.clear()
                continue

            paragraph_counter += 1
            entry = {
                "type": "paragraph",
                "order": content_counter,
                "text": raw_text,
                "is_list": is_list_paragraph(paragraph),
                "section_path": current_section().path_titles,
                "section_labels": current_section().path_labels,
                "source_block": block_index,
            }
            content_counter += 1
            current_section().content.append(entry)
            full_text_parts.append(raw_text)
            recent_paragraphs.append(raw_text)

        elif block_type == "table":
            table: Table = block  # type: ignore[assignment]

            rows: List[List[str]] = []
            for row in table.rows:
                row_cells = [cell_to_text(cell) for cell in row.cells]
                rows.append(row_cells)
            if not any(any(cell for cell in row) for row in rows):
                continue

            has_header = detect_header_row(rows)
            column_labels: List[str] = rows[0] if has_header else []
            data_rows = rows[1:] if has_header else rows

            if column_labels:
                duplicates = {
                    label for label in column_labels if column_labels.count(label) > 1
                }
                if duplicates and data_rows:
                    first_data = data_rows[0]
                    reference = column_labels[0].strip().lower() if column_labels[0] else ""
                    first_value = first_data[0].strip().lower() if first_data else ""
                    if reference and first_value == reference:
                        merged_labels = column_labels.copy()
                        for idx, label in enumerate(column_labels):
                            value = first_data[idx] if idx < len(first_data) else ""
                            if label in duplicates and value:
                                value_clean = value.strip()
                                if value_clean:
                                    merged_labels[idx] = value_clean
                        if merged_labels != column_labels:
                            column_labels = merged_labels
                            data_rows = data_rows[1:]

                # Ensure each header label is unique and non-empty
                normalized_labels: List[str] = []
                used_counts: Dict[str, int] = {}
                for idx, label in enumerate(column_labels):
                    clean_label = label.strip() if label else ""
                    if not clean_label:
                        clean_label = f"column_{idx+1}"
                    count = used_counts.get(clean_label, 0) + 1
                    used_counts[clean_label] = count
                    if count > 1:
                        clean_label = f"{clean_label}_{count}"
                    normalized_labels.append(clean_label)
                column_labels = normalized_labels

            table_counter += 1
            table_name = infer_table_name(list(recent_paragraphs))
            table_id = f"{doc_id}#tbl{table_counter:03d}"

            table_rows: List[Dict[str, object]] = []
            for idx, row_values in enumerate(data_rows):
                row_values = list(row_values)
                if column_labels and len(row_values) < len(column_labels):
                    row_values.extend([""] * (len(column_labels) - len(row_values)))
                row_label = next((val for val in row_values if val), "")
                column_map = (
                    {
                        label: row_values[col_idx] if col_idx < len(row_values) else None
                        for col_idx, label in enumerate(column_labels)
                    }
                    if column_labels
                    else {}
                )
                table_rows.append(
                    {
                        "row_index": idx,
                        "row_label": row_label or None,
                        "cells": row_values,
                        "column_map": column_map or None,
                    }
                )

            table_entry: Dict[str, object] = {
                "type": "table",
                "order": content_counter,
                "table_id": table_id,
                "table_name": table_name,
                "section_path": current_section().path_titles,
                "section_labels": current_section().path_labels,
                "source_block": block_index,
                "column_labels": column_labels or None,
                "rows": table_rows,
            }
            content_counter += 1
            current_section().content.append(table_entry)
            all_tables.append(table_entry)
            recent_paragraphs.clear()

    # Loại bỏ lớp bao (wrapper) của mục gốc (root section)
    structured_sections: List[Dict[str, object]] = []
    for section in sections:
        structured_sections.append(
            {
                "section_id": section.section_id,
                "title": section.title,
                "level": section.level,
                "label": section.label,
                "path_titles": section.path_titles,
                "path_labels": section.path_labels,
                "content": section.content,
            }
        )

    stats = {
        "paragraphs": paragraph_counter,
        "tables": len(all_tables),
        "sections": len(structured_sections),
    }

    payload = {
        "doc_id": doc_id,
        "language": language,
        "course_name_stem": course_name,
        "course_variant": course_variant,
        "course_code": course_code,
        "source_filename": doc_path.name,
        "source_relpath": str(doc_path.relative_to(ROOT_DIR)),
        "source_hash": compute_md5(doc_path),
        "source_modified": datetime.fromtimestamp(
            doc_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "sections": structured_sections,
        "tables": all_tables,
        "full_text": "\n".join(full_text_parts),
    }
    return payload

# hàm để tải và lưu trữ bản ghi manifest
# sử dụng để theo dõi trạng thái xử lý của các tài liệu
# trả về từ điển biểu diễn manifest
def load_manifest() -> Dict[str, Dict[str, object]]:
    if MANIFEST_PATH.exists():
        try:
            with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except json.JSONDecodeError:
            pass
    return {"version": 1, "entries": {}}

# hàm để lưu bản ghi manifest vào file
# sử dụng để cập nhật trạng thái xử lý của các tài liệu
def save_manifest(manifest: Dict[str, Dict[str, object]]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

# hàm chính để trích xuất các phần từ tài liệu .docx
# sử dụng các hàm phụ đã định nghĩa để xử lý từng tài liệu
# báo cáo tiến trình và kết quả cuối cùng
def main() -> None:
    ensure_output_dir()
    manifest = load_manifest()
    manifest_entries: Dict[str, Dict[str, object]] = manifest.setdefault("entries", {})

    totals = {"processed": 0, "skipped": 0, "failed": 0}
    seen_keys: set[str] = set()

    for language in LANGUAGES:
        raw_dir = RAW_DIR / language
        if not raw_dir.exists():
            print(f"[WARN] Missing raw directory for '{language}': {raw_dir}")
            continue

        output_lang_dir = OUTPUT_DIR / language
        output_lang_dir.mkdir(parents=True, exist_ok=True)

        for doc_path in sorted(raw_dir.glob("*.docx")):
            key = f"{language}/{doc_path.name}"
            seen_keys.add(key)

            manifest_entry = manifest_entries.get(key)
            file_hash = compute_md5(doc_path)
            output_path = output_lang_dir / f"{doc_path.stem}.json"

            if (
                manifest_entry
                and manifest_entry.get("hash") == file_hash
                and output_path.exists()
            ):
                totals["skipped"] += 1
                print(f"[SKIP] {doc_path.name}")
                continue

            try:
                payload = extract_document(doc_path, language)
                with output_path.open("w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2)

                manifest_entries[key] = {
                    "hash": file_hash,
                    "json_file": str(output_path.relative_to(OUTPUT_DIR)),
                    "doc_id": payload["doc_id"],
                    "processed_at": payload["processed_at"],
                }
                totals["processed"] += 1
                print(
                    f"[OK] {doc_path.name} -> {output_path.relative_to(OUTPUT_DIR)} "
                    f"(sections={payload['stats']['sections']}, tables={payload['stats']['tables']})"
                )
            except Exception as exc:  # pylint: disable=broad-except
                totals["failed"] += 1
                print(f"[FAIL] {doc_path.name}: {exc}")

    stale_keys = [key for key in manifest_entries if key not in seen_keys]
    for key in stale_keys:
        json_relpath = manifest_entries[key].get("json_file")
        if json_relpath:
            stale_path = OUTPUT_DIR / json_relpath
            if stale_path.exists():
                try:
                    stale_path.unlink()
                except OSError:
                    pass
        manifest_entries.pop(key, None)

    save_manifest(manifest)
    print(
        f"Processed={totals['processed']} | Skipped={totals['skipped']} | Failed={totals['failed']}"
    )


if __name__ == "__main__":
    main()
