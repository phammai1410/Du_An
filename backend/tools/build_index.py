import argparse
import json
import os
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import requests
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore


def _normalize_text(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def _chunk_text(s: str, max_len: int = 1000, overlap: int = 200) -> List[str]:
    s = _normalize_text(s)
    if not s:
        return []
    if len(s) <= max_len:
        return [s]
    chunks: List[str] = []
    start = 0
    while start < len(s):
        end = min(len(s), start + max_len)
        piece = s[start:end]
        if piece.strip():
            chunks.append(piece)
        if end == len(s):
            break
        start = end - overlap
    return chunks


def _iter_sections(json_path: Path) -> Iterator[Tuple[str, Dict]]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    lang = obj.get("language", "")
    fid = obj.get("id") or json_path.stem
    filename = obj.get("filename") or json_path.name
    sections = obj.get("sections", [])
    for si, sec in enumerate(sections):
        heading = _normalize_text(sec.get("heading") or "")
        body = "\n".join(
            [_normalize_text(t) for t in (sec.get("content") or []) if _normalize_text(t)]
        )
        if not body:
            continue
        text = f"{heading}\n{body}" if heading else body
        for ci, ch in enumerate(_chunk_text(text)):
            meta = {
                "language": lang,
                "id": fid,
                "filename": filename,
                "section_heading": heading,
                "chunk_id": f"{si}-{ci}",
                "source_path": str(json_path),
            }
            yield ch, meta


def _batched(items: List, size: int) -> Iterator[List]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _embed_batch(base_url: str, model: str, texts: List[str], timeout: int = 120) -> List[List[float]]:
    resp = requests.post(
        f"{base_url.rstrip('/')}/embeddings",
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    resp.raise_for_status()
    js = resp.json()
    if "data" not in js:
        raise RuntimeError(f"Bad embeddings response: {js}")
    data = js["data"]
    if len(data) != len(texts):
        raise RuntimeError(f"Embeddings count mismatch: sent {len(texts)} got {len(data)}")
    return [d["embedding"] for d in data]


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a vector index from processed JSON using LocalAI embeddings.")
    parser.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", "granite-embedding-107m-multilingual"))
    parser.add_argument("--base-url", default=os.environ.get("LOCALAI_BASE_URL", "http://localhost:8080/v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("backend/data/processed-json"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--langs", nargs="*", default=["vi", "en"])
    parser.add_argument("--backend", choices=["faiss", "bruteforce"], default="faiss")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-chunks", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or Path(f"backend/data/index/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    metas: List[Dict] = []

    for lang in args.langs:
        for p in sorted((args.data_dir / lang).glob("*.json")):
            for text, meta in _iter_sections(p):
                for piece in _chunk_text(text, args.max_len, args.overlap):
                    texts.append(piece)
                    metas.append(meta)

    print(f"Total chunks: {len(texts)} from languages: {args.langs}")

    if args.dry_run:
        return 0

    vectors: List[List[float]] = []
    batches = list(_batched(texts, args.batch_size))
    for batch in tqdm(batches, desc="Embedding"):
        backoff = 1.0
        for attempt in range(4):
            try:
                embs = _embed_batch(args.base_url, args.model, batch)
                vectors.extend(embs)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(backoff)
                backoff *= 2

    arr = np.asarray(vectors, dtype="float32")
    dim = int(arr.shape[1])
    arr = _l2_normalize(arr)

    manifest = {"backend": args.backend, "dim": dim, "model": args.model}
    if args.backend == "faiss":
        if faiss is None:
            raise RuntimeError("faiss is not available; install faiss-cpu or use --backend bruteforce")
        index = faiss.IndexFlatIP(dim)
        index.add(arr)
        faiss.write_index(index, str(out_dir / "index.faiss"))
    elif args.backend == "bruteforce":
        np.save(out_dir / "vectors.npy", arr)
    else:
        raise RuntimeError(f"Unsupported backend: {args.backend}")

    with open(out_dir / "index_manifest.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False))
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    if args.save_chunks:
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for t, m in zip(texts, metas):
                f.write(json.dumps({"text": t, "meta": m}, ensure_ascii=False) + "\n")

    print(f"Saved {len(texts)} vectors (dim={dim}) to {out_dir} using backend={manifest['backend']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
