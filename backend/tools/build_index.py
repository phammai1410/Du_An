import argparse
import io
import json
import os
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _configure_utf8_io() -> None:
    """Ensure stdout/stderr can handle UTF-8 text on Windows consoles."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
                continue
            buffer = getattr(stream, "buffer", None)
            if buffer is None:
                continue
            wrapped = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", write_through=True)
            setattr(sys, stream_name, wrapped)
        except Exception:
            continue


_configure_utf8_io()


def _clean_text(value: Optional[str]) -> str:
    text = unicodedata.normalize("NFC", (value or "").replace("\xa0", " "))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _chunk_legacy(text: str, max_len: int, overlap: int) -> Iterator[str]:
    text = _clean_text(text)
    if not text:
        return
    if len(text) <= max_len:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + max_len)
        piece = text[start:end]
        if piece.strip():
            yield piece
        if end == len(text):
            break
        start = end - overlap


def _normalize_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _compose_context(doc_meta: Dict[str, str], heading: str, breadcrumbs: str) -> str:
    parts: List[str] = []
    course_name = doc_meta.get("course_name")
    course_code = doc_meta.get("course_code")
    if course_name and course_code:
        parts.append(f"{course_name} ({course_code})")
    elif course_name:
        parts.append(course_name)
    elif course_code:
        parts.append(course_code)
    if heading:
        parts.append(heading)
    elif breadcrumbs:
        parts.append(breadcrumbs)
    return " | ".join(parts)


def _prepare_chunk(
    doc_meta: Dict[str, str],
    chunk_data: Dict,
    min_words: int,
    short_threshold: int,
    long_threshold: int,
) -> Optional[Tuple[str, Dict]]:
    raw_text = _clean_text(chunk_data.get("text"))
    if not raw_text:
        return None

    word_count = int(chunk_data.get("word_count") or len(raw_text.split()))
    if word_count < min_words:
        return None

    heading_path = [h for h in chunk_data.get("heading_path", []) if h]
    breadcrumbs = chunk_data.get("breadcrumbs") or " > ".join(heading_path)
    primary_heading = chunk_data.get("primary_heading") or (heading_path[-1] if heading_path else "")
    context_override = chunk_data.get("_context_override")
    context = context_override if context_override is not None else _compose_context(
        doc_meta, primary_heading, breadcrumbs
    )
    embed_text = f"{context}\n{raw_text}" if context else raw_text

    if word_count <= short_threshold:
        length_category = "short"
    elif word_count >= long_threshold:
        length_category = "long"
    else:
        length_category = "medium"

    position = chunk_data.get("position") or {}
    meta = {
        "id": doc_meta.get("doc_id"),
        "doc_id": doc_meta.get("doc_id"),
        "language": doc_meta.get("language"),
        "course_name": doc_meta.get("course_name"),
        "course_code": doc_meta.get("course_code"),
        "course_variant": doc_meta.get("course_variant"),
        "source_filename": doc_meta.get("source_filename"),
        "source_relpath": doc_meta.get("source_relpath"),
        "source_path": doc_meta.get("source_path"),
        "chunk_id": chunk_data.get("chunk_id"),
        "source_chunk_id": chunk_data.get("source_chunk_id") or chunk_data.get("chunk_id"),
        "chunk_order": position.get("order"),
        "heading_path": heading_path,
        "primary_heading": primary_heading,
        "breadcrumbs": breadcrumbs,
        "section_heading": primary_heading,
        "filename": doc_meta.get("source_filename"),
        "word_count": word_count,
        "char_count": int(chunk_data.get("char_count") or len(raw_text)),
        "length_category": length_category,
    }
    if "split_index" in position:
        try:
            meta["chunk_subindex"] = int(position["split_index"])
        except (TypeError, ValueError):
            meta["chunk_subindex"] = position["split_index"]
    return embed_text, meta


def _split_chunk_data(
    doc_meta: Dict[str, str],
    chunk_data: Dict,
    max_len: int,
    overlap: int,
    embed_max_len: int,
) -> Iterator[Dict]:
    raw_text = chunk_data.get("text") or ""
    normalized = _clean_text(raw_text)
    if not normalized:
        return

    heading_path = [h for h in chunk_data.get("heading_path", []) if h]
    breadcrumbs = chunk_data.get("breadcrumbs") or " > ".join(heading_path)
    primary_heading = chunk_data.get("primary_heading") or (heading_path[-1] if heading_path else "")
    context = _compose_context(doc_meta, primary_heading, breadcrumbs)
    context = _clean_text(context) if context else ""

    if context and len(context) >= embed_max_len:
        context = context[: max(0, embed_max_len - 40)].rstrip()

    prefix_len = len(context) + 1 if context else 0
    effective_max = min(max_len, embed_max_len - prefix_len)
    if effective_max <= 0:
        context = ""
        prefix_len = 0
        effective_max = min(max_len, embed_max_len)

    if effective_max <= 0:
        effective_max = max_len // 2 or 200

    piece_overlap = min(overlap, max(1, effective_max // 4))

    base_chunk_id = chunk_data.get("chunk_id") or f"{doc_meta['doc_id']}#chunk"
    base_position = dict(chunk_data.get("position") or {})
    source_chunk_id = chunk_data.get("source_chunk_id") or chunk_data.get("chunk_id") or base_chunk_id

    if len(normalized) <= effective_max:
        single = dict(chunk_data)
        text_piece = normalized[:effective_max]
        single["text"] = text_piece
        single["word_count"] = len(text_piece.split())
        single["char_count"] = len(text_piece)
        single["chunk_id"] = base_chunk_id
        single["position"] = base_position
        single["_context_override"] = context
        single["source_chunk_id"] = source_chunk_id
        yield single
        return

    pieces = list(_chunk_legacy(normalized, effective_max, piece_overlap))
    if not pieces:
        pieces = [normalized[:effective_max]]

    total_parts = len(pieces)
    for index, piece in enumerate(pieces, start=1):
        split_chunk = dict(chunk_data)
        split_chunk["text"] = piece
        split_chunk["word_count"] = len(piece.split())
        split_chunk["char_count"] = len(piece)
        split_chunk["chunk_id"] = (
            base_chunk_id if total_parts == 1 else f"{base_chunk_id}:{index:02d}"
        )
        position = dict(base_position)
        if total_parts > 1:
            position["split_index"] = index
            position["split_total"] = total_parts
        split_chunk["position"] = position
        split_chunk["_context_override"] = context
        split_chunk["source_chunk_id"] = source_chunk_id
        yield split_chunk


def _iter_document_chunks(
    json_path: Path,
    min_words: int,
    short_threshold: int,
    long_threshold: int,
    legacy_max_len: int,
    legacy_overlap: int,
    embed_max_len: int,
) -> Iterator[Tuple[str, Dict]]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    doc_meta = {
        "doc_id": obj.get("doc_id") or obj.get("id") or json_path.stem,
        "language": obj.get("language"),
        "course_name": obj.get("course_name"),
        "course_variant": obj.get("course_variant") or "",
        "course_code": obj.get("course_code") or "",
        "source_filename": obj.get("source_filename") or json_path.name,
        "source_relpath": obj.get("source_relpath") or _normalize_path(json_path),
        "source_path": str(json_path),
    }

    chunks = obj.get("chunks")
    if chunks:
        for chunk_data in chunks:
            base_chunk = dict(chunk_data)
            if "source_chunk_id" not in base_chunk:
                base_chunk["source_chunk_id"] = base_chunk.get("chunk_id")
            for adjusted_chunk in _split_chunk_data(
                doc_meta, base_chunk, legacy_max_len, legacy_overlap, embed_max_len
            ):
                prepared = _prepare_chunk(
                    doc_meta, adjusted_chunk, min_words, short_threshold, long_threshold
                )
                if prepared:
                    yield prepared
        return

    sections = obj.get("sections") or []
    for sec_index, section in enumerate(sections):
        heading = _clean_text(section.get("heading"))
        body_parts = [_clean_text(p) for p in section.get("content") or []]
        body = "\n".join([part for part in body_parts if part])
        if not body:
            continue
        combined = f"{heading}\n{body}" if heading else body
        for chunk_index, chunk_text in enumerate(_chunk_legacy(combined, legacy_max_len, legacy_overlap)):
            fallback_chunk = {
                "chunk_id": f"{doc_meta['doc_id']}#{sec_index:03d}-{chunk_index:02d}",
                "source_chunk_id": f"{doc_meta['doc_id']}#{sec_index:03d}-{chunk_index:02d}",
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "heading_path": [heading] if heading else [],
                "primary_heading": heading,
                "breadcrumbs": heading,
                "position": {"order": chunk_index + 1},
            }
            for adjusted_chunk in _split_chunk_data(
                doc_meta, fallback_chunk, legacy_max_len, legacy_overlap, embed_max_len
            ):
                prepared = _prepare_chunk(
                    doc_meta, adjusted_chunk, min_words, short_threshold, long_threshold
                )
                if prepared:
                    yield prepared


def _batched(items: List, size: int) -> Iterable[List]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _embed_batch(
    base_url: str,
    model: str,
    texts: List[str],
    timeout: int,
) -> List[List[float]]:
    resp = requests.post(
        f"{base_url.rstrip('/')}/embeddings",
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError(f"Bad embeddings response: {payload}")
    return [row["embedding"] for row in data]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def main() -> int:
    default_model = os.environ.get("EMBEDDING_MODEL", "granite-embedding-107m-multilingual")
    default_base_url = os.environ.get("LOCALAI_BASE_URL", "http://localhost:8080/v1")
    default_backend = os.environ.get("VECTOR_INDEX_BACKEND") or os.environ.get("INDEX_BACKEND", "faiss")
    default_batch_size = int(os.environ.get("INDEX_BATCH_SIZE", "32"))
    default_min_words = int(os.environ.get("INDEX_MIN_CHUNK_WORDS", "40"))
    default_short_threshold = int(os.environ.get("INDEX_SHORT_WORD_THRESHOLD", "120"))
    default_long_threshold = int(os.environ.get("INDEX_LONG_WORD_THRESHOLD", "220"))
    default_timeout = int(os.environ.get("INDEX_EMBED_TIMEOUT", "120"))
    default_embed_max_len = int(os.environ.get("INDEX_EMBED_MAX_CHARS", "850"))

    parser = argparse.ArgumentParser(description="Build a vector index from processed syllabus JSON files.")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--data-dir", type=Path, default=BACKEND_ROOT / "data" / "processed-json")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--min-words", type=int, default=default_min_words)
    parser.add_argument("--short-threshold", type=int, default=default_short_threshold)
    parser.add_argument("--long-threshold", type=int, default=default_long_threshold)
    parser.add_argument("--langs", nargs="*", default=["vi", "en"])
    parser.add_argument("--backend", choices=["faiss", "bruteforce"], default=default_backend)
    parser.add_argument("--embed-timeout", type=int, default=default_timeout)
    parser.add_argument("--legacy-max-len", type=int, default=900)
    parser.add_argument("--legacy-overlap", type=int, default=150)
    parser.add_argument("--embed-max-len", type=int, default=default_embed_max_len)
    parser.add_argument("--chunk-mode", type=str, default=None, help="Chunking mode used to prepare documents.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-chunks", action="store_true")
    args = parser.parse_args()

    if args.embed_max_len <= 0:
        raise ValueError("--embed-max-len must be positive.")
    args.embed_max_len = max(200, args.embed_max_len)

    out_dir = args.out_dir or BACKEND_ROOT / "data" / "index" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    metas: List[Dict] = []

    for lang in args.langs:
        lang_dir = args.data_dir / lang
        if not lang_dir.exists():
            print(f"[WARN] Language directory not found: {lang_dir}")
            continue

        for json_file in sorted(lang_dir.glob("*.json")):
            for text, meta in _iter_document_chunks(
                json_file,
                args.min_words,
                args.short_threshold,
                args.long_threshold,
                args.legacy_max_len,
                args.legacy_overlap,
                args.embed_max_len,
            ):
                texts.append(text)
                metas.append(meta)

    if not texts:
        print("No chunks found. Abort.")
        return 1

    print(f"Collected {len(texts)} chunks across languages {args.langs}.")

    if args.dry_run:
        return 0

    vectors: List[List[float]] = []
    kept_indices: List[int] = []
    failed_indices: List[int] = []

    def process_batch(batch_indices: List[int]) -> None:
        if not batch_indices:
            return

        batch_texts = [texts[i] for i in batch_indices]
        backoff = 1.0
        for attempt in range(4):
            try:
                embeddings = _embed_batch(
                    args.base_url,
                    args.model,
                    batch_texts,
                    args.embed_timeout,
                )
                vectors.extend(embeddings)
                kept_indices.extend(batch_indices)
                return
            except requests.HTTPError as http_err:
                status_code = http_err.response.status_code if http_err.response else None
                detail = ""
                if http_err.response is not None:
                    try:
                        detail = http_err.response.text.strip()
                    except Exception:
                        detail = ""

                if len(batch_indices) > 1 and (status_code is None or status_code >= 500):
                    midpoint = max(1, len(batch_indices) // 2)
                    left = batch_indices[:midpoint]
                    right = batch_indices[midpoint:]
                    print(
                        f"[WARN] HTTP {status_code or 'error'} on batch size {len(batch_indices)}. "
                        "Splitting batch."
                    )
                    process_batch(left)
                    process_batch(right)
                    return

                if attempt < 3:
                    wait = max(1.0, backoff)
                    print(
                        f"[WARN] HTTP {status_code or 'error'} for batch of size {len(batch_indices)}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    backoff *= 2
                    continue

                idx = batch_indices[0]
                failed_indices.append(idx)
                meta = metas[idx]
                print(
                    f"[SKIP] HTTP {status_code or 'error'} for chunk {meta.get('chunk_id')} "
                    f"({meta.get('source_filename')}). Detail: {detail[:200]}"
                )
                return

            except requests.RequestException as req_err:
                if len(batch_indices) > 1:
                    midpoint = max(1, len(batch_indices) // 2)
                    left = batch_indices[:midpoint]
                    right = batch_indices[midpoint:]
                    print(
                        f"[WARN] Connection error ({req_err}). Splitting batch of size {len(batch_indices)}."
                    )
                    process_batch(left)
                    process_batch(right)
                    return

                if attempt < 3:
                    wait = max(1.0, backoff)
                    print(
                        f"[WARN] Connection error for chunk {batch_indices[0]} ({req_err}). "
                        f"Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    backoff *= 2
                    continue

                idx = batch_indices[0]
                failed_indices.append(idx)
                meta = metas[idx]
                print(
                    f"[SKIP] Connection error for chunk {meta.get('chunk_id')} "
                    f"({meta.get('source_filename')}): {req_err}"
                )
                return

            except Exception as exc:
                if len(batch_indices) > 1:
                    midpoint = max(1, len(batch_indices) // 2)
                    left = batch_indices[:midpoint]
                    right = batch_indices[midpoint:]
                    print(
                        f"[WARN] Unexpected error ({exc}) on batch size {len(batch_indices)}. "
                        "Splitting batch."
                    )
                    process_batch(left)
                    process_batch(right)
                    return

                if attempt == 3:
                    raise
                print(
                    f"[WARN] Retry chunk {batch_indices[0]} due to error ({exc}). "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2
        for idx in batch_indices:
            failed_indices.append(idx)
            meta = metas[idx]
            print(
                f"[SKIP] Exhausted retries for chunk {meta.get('chunk_id')} "
                f"({meta.get('source_filename')})."
            )
        return

    index_batches = list(_batched(list(range(len(texts))), args.batch_size))
    for batch_indices in tqdm(index_batches, desc="Embedding", unit="batch"):
        process_batch(batch_indices)

    if failed_indices:
        print(f"[WARN] Skipped {len(failed_indices)} chunk(s) due to embedding errors.")

    kept_texts = [texts[i] for i in kept_indices]
    kept_metas = [metas[i] for i in kept_indices]

    if not kept_texts:
        print("No embeddings were generated successfully. Abort.")
        return 1

    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2:
        raise RuntimeError(f"Unexpected embedding array shape: {array.shape}")

    dim = int(array.shape[1])
    array = _l2_normalize(array)

    manifest = {
        "backend": args.backend,
        "dim": dim,
        "model": args.model,
        "base_url": args.base_url,
        "total_vectors": len(kept_texts),
        "languages": args.langs,
        "min_words": args.min_words,
        "short_threshold": args.short_threshold,
        "long_threshold": args.long_threshold,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(_normalize_path(args.data_dir)),
    }
    if args.chunk_mode:
        manifest["chunk_mode"] = args.chunk_mode
    if failed_indices:
        manifest["skipped_chunks"] = len(failed_indices)

    if args.backend == "faiss":
        if faiss is None:
            raise RuntimeError("faiss is not available; install faiss-cpu or use --backend bruteforce.")
        index = faiss.IndexFlatIP(dim)
        index.add(array)
        faiss.write_index(index, str(out_dir / "index.faiss"))
    elif args.backend == "bruteforce":
        np.save(out_dir / "vectors.npy", array)
    else:
        raise RuntimeError(f"Unsupported backend: {args.backend}")

    with open(out_dir / "index_manifest.json", "w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2))

    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as stream:
        for text, meta in zip(kept_texts, kept_metas):
            meta_record = dict(meta)
            meta_record["text"] = text
            stream.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

    if args.save_chunks:
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as stream:
            for text, meta in zip(kept_texts, kept_metas):
                stream.write(json.dumps({"text": text, "meta": meta}, ensure_ascii=False) + "\n")

    print(
        f"Saved {len(kept_texts)} vectors (dim={dim}) to {out_dir} "
        f"using backend={manifest['backend']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
