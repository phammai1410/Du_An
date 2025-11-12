#!/usr/bin/env python3
"""Run the full DOCX → JSON → vector index pipeline in one command."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:  # Python 3.8+
    from shlex import join as shlex_join
except ImportError:  # pragma: no cover - fallback for very old interpreters
    def shlex_join(parts: Iterable[str]) -> str:
        return " ".join(parts)


TOOLS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = TOOLS_DIR.parent


def build_python_cmd(script_name: str, extra_args: Sequence[str] | None = None) -> List[str]:
    command = [sys.executable, str(TOOLS_DIR / script_name)]
    if extra_args:
        command.extend(extra_args)
    return command


def run_step(label: str, command: Sequence[str], quiet: bool = True) -> None:
    if quiet:
        completed = subprocess.run(
            command,
            cwd=str(BACKEND_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            print(f"Pipeline step '{label}' failed with exit code {completed.returncode}.")
            if completed.stdout:
                print("stdout:\n" + completed.stdout)
            if completed.stderr:
                print("stderr:\n" + completed.stderr)
            completed.check_returncode()
        return

    print(f"\n=== {label} ===")
    print(shlex_join(str(part) for part in command))
    subprocess.run(command, cwd=str(BACKEND_ROOT), check=True)


def parse_args() -> argparse.Namespace:
    env_model = os.environ.get("EMBEDDING_MODEL", "")
    env_base_url = os.environ.get("LOCALAI_BASE_URL", "http://localhost:8080/v1")
    env_backend = os.environ.get("VECTOR_INDEX_BACKEND") or os.environ.get("INDEX_BACKEND")

    parser = argparse.ArgumentParser(
        description="Extract DOCX, convert to JSON chunks, and build a vector index in one shot.",
    )
    parser.add_argument("--model", default=env_model, help="Embedding model key for build_index.py.")
    parser.add_argument(
        "--base-url",
        default=env_base_url,
        help="Embedding service base URL (TEI or LocalAI).",
    )
    parser.add_argument(
        "--langs",
        nargs="*",
        default=["vi", "en"],
        help="Language folders to include when building the index.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Explicit output directory for the vector index.",
    )
    parser.add_argument(
        "--backend",
        choices=["faiss", "bruteforce"],
        default=env_backend,
        help="Vector index backend passed to build_index.py.",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("INDEX_BATCH_SIZE", "8")))
    parser.add_argument("--min-words", type=int, default=None)
    parser.add_argument("--short-threshold", type=int, default=None)
    parser.add_argument("--long-threshold", type=int, default=None)
    parser.add_argument("--embed-timeout", type=int, default=None)
    parser.add_argument("--embed-max-len", type=int, default=None)
    parser.add_argument("--legacy-max-len", type=int, default=None)
    parser.add_argument("--legacy-overlap", type=int, default=None)
    parser.add_argument("--chunk-mode", type=str, default=None, help="Chunking mode identifier to record in the index manifest.")
    parser.add_argument("--save-chunks", action="store_true", help="Pass --save-chunks to build_index.py.")
    parser.add_argument(
        "--build-dry-run",
        action="store_true",
        help="Pass --dry-run to build_index.py (collect stats without embedding).",
    )
    parser.add_argument("--skip-extract", action="store_true", help="Skip extract_docx_sections.py.")
    parser.add_argument("--skip-convert", action="store_true", help="Skip convert_docx_to_json.py.")
    parser.add_argument("--skip-build", action="store_true", help="Skip build_index.py.")
    parser.add_argument(
        "--build-extra-args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to build_index.py (prefix with -- to terminate this parser).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    steps: List[Tuple[str, List[str]]] = []
    if not args.skip_extract:
        steps.append(
            (
                "Step 1/3: extract structured sections",
                build_python_cmd("extract_docx_sections.py"),
            )
        )
    if not args.skip_convert:
        steps.append(
            (
                "Step 2/3: convert DOCX to JSON chunks",
                build_python_cmd("convert_docx_to_json.py"),
            )
        )
    if not args.skip_build:
        build_args: List[str] = [
            "--model",
            args.model,
            "--base-url",
            args.base_url,
        ]
        if args.langs:
            build_args.extend(["--langs", *args.langs])
        if args.out_dir:
            build_args.extend(["--out-dir", str(args.out_dir)])
        if args.backend:
            build_args.extend(["--backend", args.backend])
        if args.batch_size is not None:
            build_args.extend(["--batch-size", str(args.batch_size)])
        if args.min_words is not None:
            build_args.extend(["--min-words", str(args.min_words)])
        if args.short_threshold is not None:
            build_args.extend(["--short-threshold", str(args.short_threshold)])
        if args.long_threshold is not None:
            build_args.extend(["--long-threshold", str(args.long_threshold)])
        if args.embed_timeout is not None:
            build_args.extend(["--embed-timeout", str(args.embed_timeout)])
        if args.embed_max_len is not None:
            build_args.extend(["--embed-max-len", str(args.embed_max_len)])
        if args.legacy_max_len is not None:
            build_args.extend(["--legacy-max-len", str(args.legacy_max_len)])
        if args.legacy_overlap is not None:
            build_args.extend(["--legacy-overlap", str(args.legacy_overlap)])
        if args.chunk_mode:
            build_args.extend(["--chunk-mode", args.chunk_mode])
        if args.save_chunks:
            build_args.append("--save-chunks")
        if args.build_dry_run:
            build_args.append("--dry-run")
        if args.build_extra_args:
            build_args.extend(args.build_extra_args)

        steps.append(
            (
                "Step 3/3: build vector index",
                build_python_cmd("build_index.py", build_args),
            )
        )

    if not steps:
        print("No steps selected (all stages skipped). Nothing to do.")
        return 0

    for label, command in steps:
        run_step(label, command)

    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
