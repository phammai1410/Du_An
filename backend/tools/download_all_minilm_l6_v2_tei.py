from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

# Files required to run sentence-transformers/all-MiniLM-L6-v2 with the TEI backend
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "data_config.json",
    "1_Pooling/config.json",
    "vocab.txt",
)

OPTIONAL_FILES = (
    "onnx/model.onnx",
    "onnx/model_qint8_avx2.onnx",
    "onnx/model_qint8_avx512.onnx",
    "onnx/model_qint8_avx512_vnni.onnx",
    "onnx/model_quint8_avx2.onnx",
)


def resolve_target_dir(custom_target: str | None = None) -> Path:
    """Resolve the directory where the assets should be stored."""
    if custom_target:
        target_path = Path(custom_target).expanduser().resolve()
    else:
        target_path = (
            Path(__file__).resolve().parents[1] / "local-llm" / "embedding" / "sentence-transformers-all-MiniLM-L6-v2"
        )
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


def download_file(relative_path: str, destination_dir: Path) -> None:
    destination_path = destination_dir / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/{relative_path}?download=1"
    request = Request(url, headers={"User-Agent": "curl/7.79.1"})

    try:
        with urlopen(request) as response, open(destination_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} while fetching {relative_path}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach server for {relative_path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the sentence-transformers/all-MiniLM-L6-v2 assets required for the TEI backend."
    )
    parser.add_argument(
        "--target",
        help=(
            "Custom directory to place the downloaded files. "
            "Defaults to backend/local-llm/embedding/sentence-transformers-all-MiniLM-L6-v2."
        ),
    )
    args = parser.parse_args()

    target_dir = resolve_target_dir(args.target)
    print(f"Downloading files into {target_dir}")

    def ensure_download(paths: tuple[str, ...], optional: bool = False) -> None:
        for relative in paths:
            destination_path = target_dir / relative
            if destination_path.exists():
                print(f"[skip] {relative} already exists")
                continue
            print(f"[download] {relative}")
            try:
                download_file(relative, target_dir)
            except RuntimeError as exc:
                if optional and "404" in str(exc):
                    print(f"[warn] {relative} not available ({exc}). Skipping optional file.")
                    continue
                raise

    ensure_download(REQUIRED_FILES)
    ensure_download(OPTIONAL_FILES, optional=True)

    print("Download complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
