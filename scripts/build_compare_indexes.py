"""Build multiple index variants for chunk-size comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from rag_pipeline.config import PipelineConfig
from rag_pipeline.indexer import IndexBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build two or more index variants with different chunk sizes so you can "
            "compare retrieval accuracy side-by-side."
        )
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("full_text"),
        help="Directory containing the source PDFs.",
    )
    parser.add_argument(
        "--work-dir-root",
        type=Path,
        default=Path("data"),
        help=(
            "Parent directory that will hold the per-chunk-size artifacts. "
            "Each variant writes to <work-dir-root>/chunk_<size>."
        ),
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[220, 420],
        help=(
            "Token targets for each variant. Provide at least two sizes, e.g. "
            "--chunk-sizes 220 420 640."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Token overlap to use for every variant.",
    )
    return parser.parse_args()


def _build_variant(chunk_size: int, *, corpus_dir: Path, work_dir: Path, overlap: int) -> None:
    label = f"chunk_{chunk_size}"
    work_dir = work_dir / label
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Building index variant '{label}' in {work_dir} (chunk={chunk_size}, overlap={overlap})")
    config = PipelineConfig(
        corpus_dir=corpus_dir,
        work_dir=work_dir,
        chunk_target_tokens=chunk_size,
        chunk_overlap_tokens=overlap,
    )
    builder = IndexBuilder(config)
    builder.build()
    print(f"Variant '{label}' complete. Artifacts written to: {work_dir}")


def main() -> None:
    args = parse_args()
    if len(args.chunk_sizes) < 2:
        raise SystemExit("Provide at least two chunk sizes to compare.")
    unique_sizes: Iterable[int] = sorted({size for size in args.chunk_sizes if size > 0})
    if not unique_sizes:
        raise SystemExit("Chunk sizes must be positive integers.")
    for size in unique_sizes:
        _build_variant(
            chunk_size=size,
            corpus_dir=args.corpus_dir,
            work_dir=args.work_dir_root,
            overlap=args.overlap,
        )


if __name__ == "__main__":
    main()
