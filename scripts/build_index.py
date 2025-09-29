from __future__ import annotations

import argparse
from pathlib import Path

from rag_pipeline.config import PipelineConfig
from rag_pipeline.indexer import IndexBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hybrid retrieval indexes.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("full_text"),
        help="Directory containing input PDF files.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data"),
        help="Directory to store processed artifacts and indexes.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=280,
        help="Approximate number of tokens per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Approximate number of tokens to overlap between consecutive chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        corpus_dir=args.corpus_dir,
        work_dir=args.work_dir,
        chunk_target_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
    )
    builder = IndexBuilder(config)
    builder.build()


if __name__ == "__main__":
    main()
