import hashlib
import os
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from docx import Document
from docx.oxml.table import CT_Tbl  # type: ignore[attr-defined]
from docx.oxml.text.paragraph import CT_P  # type: ignore[attr-defined]
from docx.table import Table
from docx.text.paragraph import Paragraph

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "processed-json"
MANIFEST_PATH = OUTPUT_DIR / "_manifest.json"

LANGUAGES = ("en", "vi")

if load_dotenv:
    load_dotenv(ROOT_DIR / ".env")

DEFAULT_CHUNK_WORD_TARGET = 180
DEFAULT_CHUNK_WORD_MAX = 240
DEFAULT_CHUNK_MIN_WORDS = 60


def _int_env(name: str, default: int) -> int:
    """Return an int from environment variables with fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        print(f"[WARN] Invalid {name}='{raw_value}', using default {default}")
        return default
    if value <= 0:
        print(f"[WARN] {name} must be positive, using default {default}")
        return default
    return value


CHUNK_WORD_TARGET = _int_env("CHUNK_WORD_TARGET", DEFAULT_CHUNK_WORD_TARGET)
CHUNK_WORD_MAX = _int_env("CHUNK_WORD_MAX", DEFAULT_CHUNK_WORD_MAX)
CHUNK_MIN_WORDS = _int_env("CHUNK_MIN_WORDS", DEFAULT_CHUNK_MIN_WORDS)

if CHUNK_WORD_TARGET > CHUNK_WORD_MAX:
    print(
        "[WARN] CHUNK_WORD_TARGET greater than CHUNK_WORD_MAX; "
        f"using target={CHUNK_WORD_MAX}"
    )
    CHUNK_WORD_TARGET = CHUNK_WORD_MAX

if CHUNK_MIN_WORDS > CHUNK_WORD_TARGET:
    print(
        "[WARN] CHUNK_MIN_WORDS greater than CHUNK_WORD_TARGET; "
        f"using min={CHUNK_WORD_TARGET}"
    )
    CHUNK_MIN_WORDS = CHUNK_WORD_TARGET


def compute_md5(file_path: Path) -> str:
    """Return an md5 hash for the given file path."""
    hasher = hashlib.md5()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def slugify(value: str) -> str:
    """Generate a lowercase slug made of ASCII characters only."""
    normalized = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-")
    return cleaned.lower() or "doc"


def heading_level(style_name: Optional[str]) -> Optional[int]:
    """Infer heading level from a paragraph style name."""
    if not style_name:
        return None
    match = re.search(r"heading\s*(\d+)", style_name, flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def is_list_paragraph(paragraph: Paragraph) -> bool:
    """Detect if the paragraph is part of a numbered or bulleted list."""
    props = paragraph._p.pPr  # type: ignore[attr-defined]
    return props is not None and props.numPr is not None


def paragraph_to_text(paragraph: Paragraph) -> str:
    """Convert a paragraph into cleaned text while keeping list semantics."""
    text = paragraph.text.strip()
    if not text:
        return ""
    if is_list_paragraph(paragraph):
        return f"- {text}"
    return text


def table_to_text(table: Table) -> str:
    """Flatten a table into a plain-text representation."""
    rows: List[str] = []
    for row in table.rows:
        cells = [
            " ".join(
                paragraph.text.strip()
                for paragraph in cell.paragraphs
                if paragraph.text.strip()
            )
            for cell in row.cells
        ]
        if any(cells):
            rows.append(" | ".join(cell if cell else "-" for cell in cells))
    return "\n".join(rows).strip()


def iter_blocks(document: Document) -> Iterable[Tuple[str, object]]:
    """Yield paragraphs and tables in document order."""
    parent_element = document._element  # type: ignore[attr-defined]
    for child in parent_element.body.iterchildren():  # type: ignore[attr-defined]
        if isinstance(child, CT_P):
            yield "paragraph", Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield "table", Table(child, document)


def parse_filename(stem: str) -> Tuple[str, str, str]:
    """Split a document stem into course name, variant and code."""
    parts = [part.strip() for part in stem.split("_") if part.strip()]
    if not parts:
        return stem, "", ""
    if len(parts) == 1:
        return parts[0], "", ""
    course_name = parts[0]
    course_code = parts[-1]
    course_variant = "_".join(parts[1:-1]) if len(parts) > 2 else ""
    return course_name, course_variant, course_code


def extract_chunks(document: Document, doc_id: str) -> Tuple[List[Dict[str, object]], Dict[str, int], List[Dict[str, object]], str]:
    """Chunk the document and collect statistics and outline information."""
    heading_stack: List[str] = []
    outline: List[Dict[str, object]] = []
    chunks: List[Dict[str, object]] = []

    chunk_buffer: List[str] = []
    chunk_heading_path: List[str] = []
    chunk_start_block: Optional[int] = None
    chunk_words = 0
    chunk_index = 0

    paragraph_count = 0
    heading_count = 0
    table_count = 0
    full_text_items: List[str] = []
    block_index = -1

    def start_chunk_if_needed(current_block: int) -> None:
        nonlocal chunk_heading_path, chunk_start_block
        if not chunk_buffer:
            chunk_heading_path = heading_stack.copy()
            chunk_start_block = current_block

    def flush_chunk(current_block: int) -> None:
        nonlocal chunk_buffer, chunk_heading_path, chunk_start_block, chunk_words, chunk_index
        text = "\n".join(chunk_buffer).strip()
        if not text:
            chunk_buffer = []
            chunk_heading_path = []
            chunk_start_block = None
            chunk_words = 0
            return
        chunk_index += 1
        heading_path = chunk_heading_path or heading_stack.copy()
        chunk_data = {
            "chunk_id": f"{doc_id}#{chunk_index:03d}",
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
            "heading_path": heading_path,
            "primary_heading": heading_path[-1] if heading_path else None,
            "breadcrumbs": " > ".join(heading_path),
            "position": {
                "order": chunk_index,
                "start_block": chunk_start_block,
                "end_block": current_block,
            },
        }
        chunks.append(chunk_data)
        chunk_buffer = []
        chunk_heading_path = []
        chunk_start_block = None
        chunk_words = 0

    for block_type, block in iter_blocks(document):
        block_index += 1

        if block_type == "paragraph":
            paragraph: Paragraph = block  # type: ignore[assignment]
            raw_text = paragraph.text.strip()
            if not raw_text:
                continue
            style_name = paragraph.style.name if paragraph.style else ""
            level = heading_level(style_name)
            if level is not None:
                flush_chunk(block_index)
                heading_count += 1
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(raw_text)
                outline.append(
                    {"sequence": len(outline) + 1, "level": level, "title": raw_text}
                )
                full_text_items.append(raw_text)
                continue

            text = paragraph_to_text(paragraph)
            if not text:
                continue

            paragraph_count += 1
            full_text_items.append(text)
            start_chunk_if_needed(block_index)
            chunk_buffer.append(text)
            chunk_words += len(text.split())

            if chunk_words >= CHUNK_WORD_MAX or (
                chunk_words >= CHUNK_WORD_TARGET and chunk_buffer
            ):
                flush_chunk(block_index)

        elif block_type == "table":
            table: Table = block  # type: ignore[assignment]
            table_text = table_to_text(table)
            if not table_text:
                continue
            table_count += 1
            full_text_items.append(table_text)
            if chunk_words >= CHUNK_MIN_WORDS:
                flush_chunk(block_index)
            start_chunk_if_needed(block_index)
            chunk_buffer.append(table_text)
            chunk_words += len(table_text.split())
            if chunk_words >= CHUNK_WORD_TARGET:
                flush_chunk(block_index)

    flush_chunk(block_index)

    if not chunks and full_text_items:
        text = "\n".join(full_text_items).strip()
        chunks.append(
            {
                "chunk_id": f"{doc_id}#001",
                "text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
                "heading_path": [],
                "primary_heading": None,
                "breadcrumbs": "",
                "position": {"order": 1, "start_block": 0, "end_block": block_index},
            }
        )

    stats = {
        "paragraphs": paragraph_count,
        "headings": heading_count,
        "tables": table_count,
        "chunks": len(chunks),
        "words": sum(chunk["word_count"] for chunk in chunks),
    }

    full_text = "\n".join(full_text_items)
    return chunks, stats, outline, full_text


def load_manifest() -> Dict[str, Dict[str, object]]:
    if MANIFEST_PATH.exists():
        try:
            with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except json.JSONDecodeError:
            return {"version": 1, "entries": {}}
    return {"version": 1, "entries": {}}


def save_manifest(manifest: Dict[str, Dict[str, object]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def main() -> None:
    print("=" * 72)
    print("DOCX -> JSON RAG converter")
    print("=" * 72)

    if not RAW_DIR.exists():
        print(f"[ERROR] Raw directory not found: {RAW_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    manifest_entries: Dict[str, Dict[str, object]] = manifest.setdefault("entries", {})

    seen_keys = set()
    totals = {"processed": 0, "skipped": 0, "failed": 0}

    for language in LANGUAGES:
        lang_dir = RAW_DIR / language
        if not lang_dir.exists():
            print(f"[WARN] Missing raw folder for '{language}': {lang_dir}")
            continue

        docx_files = sorted(lang_dir.glob("*.docx"))
        print(f"[INFO] {language.upper()} -> {len(docx_files)} file(s)")

        for docx_file in docx_files:
            key = f"{language}/{docx_file.name}"
            seen_keys.add(key)

            file_hash = compute_md5(docx_file)
            existing_entry = manifest_entries.get(key)
            output_folder = OUTPUT_DIR / language
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / f"{docx_file.stem}.json"

            if (
                existing_entry
                and existing_entry.get("hash") == file_hash
                and output_path.exists()
            ):
                totals["skipped"] += 1
                print(f"  [SKIP] {docx_file.name} (unchanged)")
                continue

            try:
                document = Document(docx_file)
                course_name, course_variant, course_code = parse_filename(docx_file.stem)

                base_id = course_code or slugify(course_name)
                doc_id = f"{language}-{slugify(base_id)}"

                chunks, stats, outline, full_text = extract_chunks(document, doc_id)

                processed_at = datetime.now(timezone.utc).isoformat()
                source_mtime = datetime.fromtimestamp(
                    docx_file.stat().st_mtime, tz=timezone.utc
                ).isoformat()

                payload = {
                    "doc_id": doc_id,
                    "language": language,
                    "course_name": course_name,
                    "course_variant": course_variant,
                    "course_code": course_code,
                    "source_filename": docx_file.name,
                    "source_relpath": str(docx_file.relative_to(ROOT_DIR)),
                    "source_hash": file_hash,
                    "source_modified": source_mtime,
                    "processed_at": processed_at,
                    "outline": outline,
                    "stats": stats,
                    "chunks": chunks,
                    "full_text": full_text,
                }

                with output_path.open("w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2)

                manifest_entries[key] = {
                    "hash": file_hash,
                    "json_file": str(output_path.relative_to(OUTPUT_DIR)),
                    "doc_id": doc_id,
                    "language": language,
                    "processed_at": processed_at,
                }
                totals["processed"] += 1
                print(
                    f"  [OK]   {docx_file.name} -> {output_path.name} ({stats['chunks']} chunk(s))"
                )
            except Exception as exc:  # pylint: disable=broad-except
                totals["failed"] += 1
                print(f"  [FAIL] {docx_file.name}: {exc}")

    stale_keys = [key for key in manifest_entries.keys() if key not in seen_keys]
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

    print("-" * 72)
    print(
        f"Processed: {totals['processed']} | Skipped: {totals['skipped']} | Failed: {totals['failed']}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
