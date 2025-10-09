from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple


def _ensure_markitdown_import() -> None:
    """
    Add the MarkItDown source tree from the local submodule to sys.path so
    the package can be imported without requiring a separate install step.
    """
    project_root = Path(__file__).resolve().parents[1]
    markitdown_src = project_root / "markitdown" / "packages" / "markitdown" / "src"
    if not markitdown_src.exists():
        message = (
            "MarkItDown sources not found at "
            f"{markitdown_src}. Ensure the submodule is cloned."
        )
        raise SystemExit(message)

    sys_path_entry = str(markitdown_src)
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)


def _iter_source_files(root: Path) -> Iterable[Path]:
    """Yield all files under the given root directory."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _convert_file(
    converter,
    source_path: Path,
    source_root: Path,
    output_root: Path,
    **convert_kwargs,
) -> Tuple[Path, Path]:
    """
    Convert a single file with MarkItDown and write the Markdown output.

    Returns a tuple of (source_path, output_path).
    """
    relative_path = source_path.relative_to(source_root)
    output_path = (output_root / relative_path).with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = converter.convert(str(source_path), **convert_kwargs)
    output_path.write_text(result.markdown, encoding="utf-8")

    return source_path, output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert files in backend/data/raw to Markdown using MarkItDown.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("backend/data/raw"),
        help="Root directory containing the raw source files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/data/processed-markdown"),
        help="Destination directory for generated Markdown files.",
    )
    parser.add_argument(
        "--use-plugins",
        action="store_true",
        help="Enable MarkItDown third-party plugins if they are installed.",
    )
    parser.add_argument(
        "--keep-data-uris",
        action="store_true",
        help="Preserve data URIs (such as inline images) in the Markdown output.",
    )
    args = parser.parse_args()

    _ensure_markitdown_import()

    from markitdown import MarkItDown, MarkItDownException

    source_root = args.input.resolve()
    output_root = args.output.resolve()

    if not source_root.exists():
        print(f"Source directory not found: {source_root}", file=sys.stderr)
        return 1

    converter = MarkItDown(enable_plugins=args.use_plugins)

    total_files = 0
    converted_files = 0
    failures = 0

    for source_file in _iter_source_files(source_root):
        total_files += 1
        try:
            _convert_file(
                converter,
                source_file,
                source_root=source_root,
                output_root=output_root,
                keep_data_uris=args.keep_data_uris,
            )
            converted_files += 1
            print(f"[OK] {source_file}")
        except MarkItDownException as exc:
            failures += 1
            print(f"[FAIL] {source_file} -> {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(
                f"[FAIL] {source_file} -> unexpected error: {exc}",
                file=sys.stderr,
            )

    print(
        f"\nCompleted. Total: {total_files}, Converted: {converted_files}, "
        f"Failed: {failures}, Output dir: {output_root}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
