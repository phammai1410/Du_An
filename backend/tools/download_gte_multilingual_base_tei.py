from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import shutil


BASE_URL = "https://huggingface.co/Alibaba-NLP/gte-multilingual-base/resolve/main"

# Files required to run Alibaba-NLP/gte-multilingual-base with the TEI backend
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
)


def resolve_target_dir(custom_target: str | None = None) -> Path:
    """
    Resolve the directory where the assets should be stored.

    Defaults to backend/local-llm/embedding/Alibaba-NLP-gte-multilingual-base relative to this file.
    """
    if custom_target:
        target_path = Path(custom_target).expanduser().resolve()
    else:
        target_path = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("local-llm", "embedding", "Alibaba-NLP-gte-multilingual-base")
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
        description="Download required Alibaba-NLP/gte-multilingual-base files for TEI backend."
    )
    parser.add_argument(
        "--target",
        help=(
            "Custom directory to place the downloaded files. "
            "Defaults to backend/local-llm/embedding/Alibaba-NLP-gte-multilingual-base."
        ),
    )
    args = parser.parse_args()

    target_dir = resolve_target_dir(args.target)
    print(f"Downloading files into {target_dir}")

    for relative in REQUIRED_FILES:
        destination_path = target_dir / relative
        if destination_path.exists():
            print(f"[skip] {relative} already exists")
            continue
        print(f"[download] {relative}")
        download_file(relative, target_dir)

    print("Download complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
