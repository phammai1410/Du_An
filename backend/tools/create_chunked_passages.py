"""Chunk syllabi into retrieval-ready passages.

Implements pipeline step (3):
  * Load structured outputs from `extract_docx_sections.py`
  * Convert paragraphs and table rows into textual segments
  * Build overlapping chunks (<=180 words, 30-word overlap) per section
  * Emit doc-level chunk JSON files for downstream embedding
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = ROOT_DIR / "data" / "processed-structured"
OUTPUT_DIR = ROOT_DIR / "data" / "processed-chunks"
MANIFEST_PATH = OUTPUT_DIR / "_manifest.json"

LANGUAGES = ("vi", "en")
MAX_WORDS = 180
OVERLAP_WORDS = 30


def _ensure_utf8_stdio() -> None:
    """Force UTF-8 I/O to avoid Windows codepage issues."""
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
    text = unicodedata.normalize("NFC", value or "")
    text = text.replace("\u00a0", " ").replace("\t", " ")
    return " ".join(text.split())


def compute_md5(path: Path) -> str:
    import hashlib

    hasher = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def words_count(text: str) -> int:
    return len(text.split())


@dataclass
class Segment:
    text: str
    source: Dict[str, object]


def paragraph_to_segment(entry: Dict[str, object]) -> Optional[Segment]:
    text = normalize_whitespace(entry.get("text", ""))
    if not text:
        return None
    return Segment(
        text=text,
        source={
            "type": "paragraph",
            "order": entry.get("order"),
            "source_block": entry.get("source_block"),
        },
    )


def table_row_description(table: Dict[str, object], row: Dict[str, object]) -> Optional[str]:
    row_cells: List[str] = row.get("cells") or []
    cleaned_cells = [normalize_whitespace(cell) for cell in row_cells if normalize_whitespace(cell)]
    column_labels: List[str] = table.get("column_labels") or []
    table_name = normalize_whitespace(table.get("table_name") or "")
    row_label = normalize_whitespace(row.get("row_label") or "")

    prefix_parts: List[str] = []
    if table_name:
        prefix_parts.append(table_name)
    if row_label and row_label not in prefix_parts:
        prefix_parts.append(row_label)
    prefix = " – ".join(prefix_parts) if prefix_parts else "Bảng"

    if column_labels:
        values = []
        column_map = row.get("column_map") or {}
        for label in column_labels:
            clean_label = normalize_whitespace(label)
            value = normalize_whitespace(str(column_map.get(label, ""))) if column_map else ""
            if value:
                values.append(f"{clean_label}: {value}")
        if values:
            return f"{prefix}: " + " | ".join(values)

    if cleaned_cells:
        return f"{prefix}: " + " | ".join(cleaned_cells)
    return prefix if prefix else None


def table_to_segments(section: Dict[str, object], table_entry: Dict[str, object]) -> List[Segment]:
    segments: List[Segment] = []
    for row in table_entry.get("rows", []):
        text = table_row_description(table_entry, row)
        if not text:
            continue
        segments.append(
            Segment(
                text=text,
                source={
                    "type": "table_row",
                    "table_id": table_entry.get("table_id"),
                    "row_index": row.get("row_index"),
                    "section_path": section.get("path_titles"),
                },
            )
        )
    return segments


def gather_section_segments(section: Dict[str, object]) -> List[Segment]:
    segments: List[Segment] = []
    for entry in section.get("content", []):
        entry_type = entry.get("type")
        if entry_type == "paragraph":
            segment = paragraph_to_segment(entry)
            if segment:
                segments.append(segment)
        elif entry_type == "table":
            segments.extend(table_to_segments(section, entry))
    return segments


def split_segment(segment: Segment, max_words: int) -> List[Segment]:
    words = segment.text.split()
    if len(words) <= max_words:
        return [segment]
    chunks: List[Segment] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(Segment(text=chunk_text, source=segment.source))
        start = end
    return chunks


def chunk_section_segments(
    doc_meta: Dict[str, object],
    section: Dict[str, object],
    segments: List[Segment],
    max_words: int,
    overlap_words: int,
    global_chunk_index_start: int,
) -> (List[Dict[str, object]], int):
    output_chunks: List[Dict[str, object]] = []
    queue: Deque[Segment] = deque()
    for segment in segments:
        for sub_segment in split_segment(segment, max_words):
            queue.append(sub_segment)

    chunk_segments: List[Segment] = []
    chunk_word_count = 0
    carryover_words: List[str] = []
    global_chunk_index = global_chunk_index_start
    section_chunk_index = 0

    def flush_chunk() -> None:
        nonlocal chunk_segments, chunk_word_count, carryover_words, global_chunk_index, section_chunk_index
        if not chunk_segments:
            return
        chunk_text_parts = [seg.text for seg in chunk_segments if seg.text.strip()]
        if not chunk_text_parts:
            chunk_segments = []
            chunk_word_count = 0
            carryover_words = []
            return
        chunk_text = "\n".join(chunk_text_parts)
        chunk_words = chunk_text.split()
        chunk_sources = [
            seg.source for seg in chunk_segments if seg.source.get("type") != "overlap"
        ]
        if not chunk_sources:
            chunk_sources = [chunk_segments[-1].source]

        global_chunk_index += 1
        section_chunk_index += 1
        chunk_id = (
            f"{doc_meta['doc_id']}#S{section['index']:03d}C{section_chunk_index:03d}"
        )

        output_chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_meta["doc_id"],
                "language": doc_meta["language"],
                "course_code": doc_meta.get("course_code"),
                "section_id": section.get("section_id"),
                "section_title": section.get("title"),
                "section_path": section.get("path_titles"),
                "section_labels": section.get("path_labels"),
                "text": chunk_text,
                "word_count": len(chunk_words),
                "source_spans": chunk_sources,
            }
        )

        carryover_words = (
            chunk_words[-overlap_words:]
            if overlap_words and len(chunk_words) > overlap_words
            else chunk_words
        )
        chunk_segments = []
        chunk_word_count = 0

    while queue or chunk_segments:
        if not chunk_segments and carryover_words:
            overlap_text = " ".join(carryover_words)
            chunk_segments.append(
                Segment(
                    text=overlap_text,
                    source={"type": "overlap", "note": "carryover"},
                )
            )
            chunk_word_count = len(overlap_text.split())
            carryover_words = []

        if not queue:
            if chunk_segments and all(
                seg.source.get("type") == "overlap" for seg in chunk_segments
            ):
                chunk_segments = []
                chunk_word_count = 0
                carryover_words = []
                break
            flush_chunk()
            break

        next_segment = queue.popleft()
        seg_words = next_segment.text.split()
        if not seg_words:
            continue

        if chunk_word_count == 0 or chunk_word_count + len(seg_words) <= max_words:
            chunk_segments.append(next_segment)
            chunk_word_count += len(seg_words)
        else:
            queue.appendleft(next_segment)
            flush_chunk()

    return output_chunks, global_chunk_index


def build_chunks(structured_doc: Dict[str, object]) -> List[Dict[str, object]]:
    doc_meta = {
        "doc_id": structured_doc.get("doc_id"),
        "language": structured_doc.get("language"),
        "course_code": structured_doc.get("course_code"),
    }

    chunks: List[Dict[str, object]] = []
    global_chunk_index = 0

    for section_index, section in enumerate(structured_doc.get("sections", []), start=1):
        section_segments = gather_section_segments(section)
        if not section_segments:
            continue
        section["index"] = section_index
        section_chunks, global_chunk_index = chunk_section_segments(
            doc_meta,
            section,
            section_segments,
            MAX_WORDS,
            OVERLAP_WORDS,
            global_chunk_index,
        )
        chunks.extend(section_chunks)

    return chunks


def load_manifest() -> Dict[str, object]:
    if MANIFEST_PATH.exists():
        try:
            with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except json.JSONDecodeError:
            pass
    return {"version": 1, "entries": {}}


def save_manifest(manifest: Dict[str, object]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    manifest_entries: Dict[str, Dict[str, object]] = manifest.setdefault("entries", {})

    totals = {"processed": 0, "skipped": 0, "failed": 0}
    seen_keys = set()

    for language in LANGUAGES:
        input_dir = STRUCTURED_DIR / language
        if not input_dir.exists():
            print(f"[WARN] Missing structured data for '{language}'")
            continue

        output_lang_dir = OUTPUT_DIR / language
        output_lang_dir.mkdir(parents=True, exist_ok=True)

        for structured_path in sorted(input_dir.glob("*.json")):
            key = f"{language}/{structured_path.name}"
            seen_keys.add(key)

            structured_hash = compute_md5(structured_path)
            output_path = output_lang_dir / f"{structured_path.stem}.json"

            entry = manifest_entries.get(key)
            if (
                entry
                and entry.get("hash") == structured_hash
                and output_path.exists()
            ):
                totals["skipped"] += 1
                print(f"[SKIP] {structured_path.name}")
                continue

            try:
                structured_doc = json.loads(structured_path.read_text(encoding="utf-8"))
                chunks = build_chunks(structured_doc)
                payload = {
                    "doc_id": structured_doc.get("doc_id"),
                    "language": structured_doc.get("language"),
                    "course_code": structured_doc.get("course_code"),
                    "source_relpath": structured_doc.get("source_relpath"),
                    "chunk_config": {
                        "max_words": MAX_WORDS,
                        "overlap_words": OVERLAP_WORDS,
                    },
                    "stats": {
                        "chunks": len(chunks),
                        "total_words": sum(chunk["word_count"] for chunk in chunks),
                    },
                    "chunks": chunks,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
                with output_path.open("w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2)

                manifest_entries[key] = {
                    "hash": structured_hash,
                    "output_file": str(output_path.relative_to(OUTPUT_DIR)),
                    "processed_at": payload["processed_at"],
                    "chunks": len(chunks),
                }
                totals["processed"] += 1
                print(f"[OK] {structured_path.name} -> {output_path.relative_to(OUTPUT_DIR)} ({len(chunks)} chunk(s))")
            except Exception as exc:  # pylint: disable=broad-except
                totals["failed"] += 1
                print(f"[FAIL] {structured_path.name}: {exc}")

    stale_keys = [k for k in manifest_entries if k not in seen_keys]
    for key in stale_keys:
        output_relpath = manifest_entries[key].get("output_file")
        if output_relpath:
            stale_path = OUTPUT_DIR / output_relpath
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
