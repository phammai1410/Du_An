"""Metadata enrichment for structured DOCX outputs.

This script implements step (2) of the RAG pipeline:
  * Read normalized section JSON files produced by ``extract_docx_sections.py``
  * Derive core course metadata (names, credits, hours, prerequisites, etc.)
  * Emit streamlined metadata JSON files with table descriptors for filtering
"""

from __future__ import annotations

import json
import re
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = ROOT_DIR / "data" / "processed-structured"
METADATA_DIR = ROOT_DIR / "data" / "processed-metadata"

LANGUAGES = ("vi", "en")


def _ensure_utf8_stdio() -> None:
    if os.environ.get("PYTHONIOENCODING"):
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, LookupError):
                pass


_ensure_utf8_stdio()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def clean_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = normalize_whitespace(value)
    return cleaned or None


def to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(\d+)", value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def load_structured_document(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def gather_lines(doc: Dict[str, object]) -> List[str]:
    lines: List[str] = []
    for section in doc.get("sections", []):
        for entry in section.get("content", []):
            if entry.get("type") == "paragraph":
                text = entry.get("text")
                if text:
                    lines.append(str(text))
    return lines


def extract_field(lines: Iterable[str], patterns: Iterable[str]) -> Optional[str]:
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                return clean_value(match.group(1))
    return None


def extract_prerequisites(lines: List[str]) -> List[str]:
    prereq_values: List[str] = []
    trigger_index: Optional[int] = None
    for idx, line in enumerate(lines):
        if re.search(r"Các học phần tiên quyết", line, flags=re.IGNORECASE):
            trigger_index = idx
            break
    if trigger_index is None:
        return prereq_values

    first_line = lines[trigger_index]
    after_colon = first_line.split(":", 1)[1].strip() if ":" in first_line else ""
    if after_colon:
        prereq_values.extend(parse_prerequisite_tokens(after_colon))

    for next_line in lines[trigger_index + 1 :]:
        candidate = next_line.strip()
        if not candidate:
            break
        if ":" in candidate and not candidate.startswith("-"):
            break
        prereq_values.extend(parse_prerequisite_tokens(candidate))

    # Remove duplicates while keeping order
    seen = set()
    deduped = []
    for item in prereq_values:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def parse_prerequisite_tokens(line: str) -> List[str]:
    text = line.strip().strip("-•")
    if not text:
        return []
    separators = r"[;,/]"
    tokens = [normalize_whitespace(token) for token in re.split(separators, text)]
    return [token for token in tokens if token]


def extract_decision(full_text: str) -> Optional[str]:
    match = re.search(
        r"Quyết định\s+số\s*([0-9A-Za-z/.\- ]+)",
        full_text,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_value(match.group(0))
    return None


@dataclass
class CourseMetadata:
    doc_id: str
    language: str
    course_code: Optional[str]
    course_name_vi: Optional[str]
    course_name_en: Optional[str]
    credit: Optional[int]
    hours_in_class: Optional[int]
    hours_self: Optional[int]
    prerequisites: List[str]
    decision_no: Optional[str]
    source_path: str
    tables: List[Dict[str, object]]


def build_table_metadata(doc: Dict[str, object]) -> List[Dict[str, object]]:
    table_metadata: List[Dict[str, object]] = []
    for table in doc.get("tables", []):
        entry = {
            "table_id": table.get("table_id"),
            "table_name": table.get("table_name"),
            "section_path": table.get("section_path"),
            "section_labels": table.get("section_labels"),
            "column_labels": table.get("column_labels"),
            "rows": [
                {
                    "row_index": row.get("row_index"),
                    "row_label": row.get("row_label"),
                }
                for row in table.get("rows", [])
            ],
        }
        table_metadata.append(entry)
    return table_metadata


def derive_metadata(structured_doc: Dict[str, object]) -> CourseMetadata:
    lines = gather_lines(structured_doc)
    full_text = structured_doc.get("full_text", "")

    course_name_vi = extract_field(
        lines, [r"Tên học phần\s*\(tiếng Việt\)\s*:?\s*(.+)"]
    )
    course_name_en = extract_field(
        lines,
        [
            r"Tên học phần\s*\(tiếng Anh\)\s*:?\s*(.+)",
            r"Tên học phần\s*\(English\)\s*:?\s*(.+)",
        ],
    )
    course_code = extract_field(lines, [r"Mã học phần\s*:?\s*([A-Z0-9\.]+)"])
    credit = to_int(extract_field(lines, [r"Số tín chỉ\s*:?\s*([0-9]+)"]))
    hours_in_class = to_int(extract_field(lines, [r"Số giờ trên lớp\s*:?\s*([0-9]+)"]))
    hours_self = to_int(extract_field(lines, [r"Số giờ tự học\s*:?\s*([0-9]+)"]))
    prerequisites = extract_prerequisites(lines)
    decision_no = extract_decision(full_text)

    if not course_name_vi:
        course_name_vi = clean_value(structured_doc.get("course_name_stem"))
    if not course_code:
        course_code = clean_value(structured_doc.get("course_code"))

    tables = build_table_metadata(structured_doc)

    return CourseMetadata(
        doc_id=str(structured_doc.get("doc_id")),
        language=str(structured_doc.get("language")),
        course_code=course_code,
        course_name_vi=course_name_vi,
        course_name_en=course_name_en,
        credit=credit,
        hours_in_class=hours_in_class,
        hours_self=hours_self,
        prerequisites=prerequisites,
        decision_no=decision_no,
        source_path=str(structured_doc.get("source_relpath")),
        tables=tables,
    )


def save_metadata(metadata: CourseMetadata, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": metadata.doc_id,
        "language": metadata.language,
        "course_code": metadata.course_code,
        "course_name_vi": metadata.course_name_vi,
        "course_name_en": metadata.course_name_en,
        "credit": metadata.credit,
        "hours_in_class": metadata.hours_in_class,
        "hours_self": metadata.hours_self,
        "prerequisites": metadata.prerequisites,
        "decision_no": metadata.decision_no,
        "source_path": metadata.source_path,
        "tables": metadata.tables,
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main() -> None:
    for language in LANGUAGES:
        input_dir = STRUCTURED_DIR / language
        if not input_dir.exists():
            print(f"[WARN] Structured directory missing for '{language}'")
            continue

        for structured_path in sorted(input_dir.glob("*.json")):
            structured_doc = load_structured_document(structured_path)
            metadata = derive_metadata(structured_doc)

            relative_name = structured_path.stem + ".metadata.json"
            output_path = (METADATA_DIR / language) / relative_name
            save_metadata(metadata, output_path)
            print(f"[OK] {structured_path.name} -> {output_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
