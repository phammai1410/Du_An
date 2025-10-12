import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def embed(base_url: str, model: str, texts: List[str], timeout: int = 120) -> np.ndarray:
    resp = requests.post(
        f"{base_url.rstrip('/')}/embeddings",
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    resp.raise_for_status()
    js = resp.json()
    vecs = [d["embedding"] for d in js.get("data", [])]
    arr = np.asarray(vecs, dtype="float32")
    return l2_normalize(arr)


def load_manifest(index_dir: Path) -> Dict:
    mf = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
    return mf


def load_meta(index_dir: Path) -> List[Dict]:
    metas: List[Dict] = []
    with open(index_dir / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def load_chunks_map(index_dir: Path) -> Optional[Dict[Tuple[str, str, str], str]]:
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return None
    mapping: Dict[Tuple[str, str, str], str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta = row.get("meta", {})
            key = (meta.get("source_path", ""), meta.get("section_heading", ""), meta.get("chunk_id", ""))
            mapping[key] = row.get("text", "")
    return mapping


def reconstruct_chunk(meta: Dict, max_len: int = 1000, overlap: int = 200) -> str:
    # Rebuild chunk text from source JSON if chunks.jsonl is unavailable
    src = Path(meta["source_path"])
    if not src.exists():
        return ""
    obj = json.loads(src.read_text(encoding="utf-8"))
    sections = obj.get("sections", [])
    # chunk_id = "si-ci"
    m = re.match(r"^(\d+)-(\d+)$", str(meta.get("chunk_id", "")))
    if not m:
        return ""
    si, ci = int(m.group(1)), int(m.group(2))
    if si >= len(sections):
        return ""
    sec = sections[si]
    heading = sec.get("heading") or ""
    body = "\n".join([t.strip() for t in (sec.get("content") or []) if str(t).strip()])
    text = (heading + "\n" + body).strip() if heading else body
    if not text:
        return ""
    # Same chunking as builder defaults
    s = text
    if len(s) <= max_len:
        return s
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
    if ci < len(chunks):
        return chunks[ci]
    return ""


def search(index_dir: Path, query_vec: np.ndarray, k: int) -> List[Tuple[float, int]]:
    manifest = load_manifest(index_dir)
    backend = manifest.get("backend", "faiss")
    if backend == "faiss":
        if faiss is None:
            raise RuntimeError("faiss backend required but not available")
        index = faiss.read_index(str(index_dir / "index.faiss"))
        D, I = index.search(query_vec, k)
        return list(zip(D[0].tolist(), [int(x) for x in I[0]]))
    elif backend == "bruteforce":
        vecs = np.load(index_dir / "vectors.npy")
        sims = (vecs @ query_vec.T).reshape(-1)
        topk_idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), int(i)) for i in topk_idx]
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")


def format_context_item(i: int, score: float, text: str, meta: Dict) -> str:
    src = meta.get("filename", meta.get("source_path", ""))
    heading = meta.get("section_heading", "")
    ref = f"[{i}] {src} | {heading} | score={score:.3f}"
    return ref + "\n" + text.strip()


def pick_chat_model(base_url: str, preferred: Optional[str]) -> Optional[str]:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        ids = [d.get("id") for d in data]
        if preferred and preferred in ids:
            return preferred
        # Heuristics: prefer instruct/chat models
        for m in ids:
            if m and ("instruct" in m.lower() or "chat" in m.lower()):
                return m
        return None
    except Exception:
        return preferred


def chat_answer(base_url: str, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    choices = js.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG: retrieve top-k contexts and call LocalAI chat model with citations")
    parser.add_argument("query", nargs="+", help="The user question")
    parser.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", "granite-embedding-107m-multilingual"))
    parser.add_argument("--base-url", default=os.environ.get("LOCALAI_BASE_URL", "http://localhost:8080/v1"))
    parser.add_argument("--chat-model", default=os.environ.get("LOCALAI_CHAT_MODEL", None))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=3500)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--show-only", action="store_true", help="Only show retrieved contexts without calling chat")
    args = parser.parse_args()

    index_dir = Path(f"backend/data/index/{args.model}")
    if not index_dir.exists():
        raise SystemExit(f"Index directory not found: {index_dir}")

    metas = load_meta(index_dir)
    chunks_map = load_chunks_map(index_dir)

    qtext = " ".join(args.query)
    qvec = embed(args.base_url, args.model, [qtext])

    results = search(index_dir, qvec, args.k)

    contexts: List[str] = []
    citations: List[str] = []
    for i, (score, idx) in enumerate(results, start=1):
        meta = metas[idx]
        key = (meta.get("source_path", ""), meta.get("section_heading", ""), meta.get("chunk_id", ""))
        if chunks_map and key in chunks_map:
            text = chunks_map[key]
        else:
            text = reconstruct_chunk(meta)
        contexts.append(format_context_item(i, score, text, meta))
        cit = f"[{i}] {meta.get('filename','')} | {meta.get('section_heading','')}"
        citations.append(cit)

    # Print retrieved contexts (useful for debugging)
    print("--- Retrieved Contexts ---")
    for c in contexts:
        print(c)
        print()

    if args.show_only:
        return 0

    chat_model = pick_chat_model(args.base_url, args.chat_model)
    if not chat_model:
        print("No chat model available in LocalAI /v1/models. Skipping answer generation.")
        print("Tip: set --chat-model or load an instruct/chat model in LocalAI.")
        return 0

    # Assemble prompt
    ctx_blob = "\n\n".join(contexts)
    if len(ctx_blob) > args.max_context_chars:
        ctx_blob = ctx_blob[: args.max_context_chars]
    system_prompt = (
        "You are a helpful assistant. Answer strictly using the provided context. "
        "Cite sources like [1], [2] matching the context items. If unsure, say you don't know."
    )
    user_prompt = f"Question: {qtext}\n\nContext:\n{ctx_blob}\n\nAnswer in the language of the question."

    try:
        answer = chat_answer(args.base_url, chat_model, system_prompt, user_prompt, temperature=args.temperature)
    except Exception as e:
        print(f"Chat call failed: {e}")
        return 1

    print("--- Answer ---")
    print(answer)
    print()
    print("--- Citations ---")
    for c in citations:
        print(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

