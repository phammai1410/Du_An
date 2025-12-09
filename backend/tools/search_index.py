import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import requests

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore

# hàm để chuẩn hóa vector theo chuẩn L2
# trả về vector đã chuẩn hóa
def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

# hàm để nhúng truy vấn văn bản sử dụng API LocalAI
# trả về vector nhúng đã chuẩn hóa
# xử lý thời gian chờ kết nối
def _embed_query(base_url: str, model: str, text: str, timeout: int = 60) -> np.ndarray:
    resp = requests.post(
        f"{base_url.rstrip('/')}/embeddings",
        json={"model": model, "input": [text]},
        timeout=timeout,
    )
    resp.raise_for_status()
    js = resp.json()
    vec = js["data"][0]["embedding"]
    arr = np.asarray([vec], dtype="float32")
    return _l2_normalize(arr)

# hàm để phân tích các đối số dòng lệnh
# trả về Namespace chứa các đối số đã phân tích
# sử dụng các tham số cấu hình tìm kiếm chỉ mục
def main() -> int:
    parser = argparse.ArgumentParser(description="Search FAISS index built from LocalAI embeddings.")
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", "granite-embedding-107m-multilingual"))
    parser.add_argument("--base-url", default=os.environ.get("LOCALAI_BASE_URL", "http://localhost:8080/v1"))
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    index_dir = args.index_dir or Path(f"backend/data/index/{args.model}")
    meta_path = index_dir / "meta.jsonl"
    manifest_path = index_dir / "index_manifest.json"

    if not manifest_path.exists() or not meta_path.exists():
        raise SystemExit(f"Manifest or metadata not found in {index_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backend = manifest.get("backend", "faiss")

    index = None
    vectors = None
    if backend == "faiss":
        if faiss is None:
            raise SystemExit("faiss not available but index requires faiss")
        index_path = index_dir / "index.faiss"
        if not index_path.exists():
            raise SystemExit(f"FAISS index not found: {index_path}")
        index = faiss.read_index(str(index_path))
    elif backend == "bruteforce":
        vec_path = index_dir / "vectors.npy"
        if not vec_path.exists():
            raise SystemExit(f"Vectors file not found: {vec_path}")
        vectors = np.load(vec_path)
    else:
        raise SystemExit(f"Unsupported backend in manifest: {backend}")

    metas: List[dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))

    qtext = " ".join(args.query)
    qv = _embed_query(args.base_url, args.model, qtext)

    if backend == "faiss":
        D, I = index.search(qv, args.k)  # type: ignore[arg-type]
        for score, idx in zip(D[0], I[0]):
            m = metas[int(idx)]
            print(json.dumps({"score": float(score), **m}, ensure_ascii=False))
    else:
        sims = (vectors @ qv.T).reshape(-1)  # cosine since normalized
        topk_idx = np.argsort(-sims)[: args.k]
        for idx in topk_idx:
            m = metas[int(idx)]
            print(json.dumps({"score": float(sims[int(idx)]), **m}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
