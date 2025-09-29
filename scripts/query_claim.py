from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_pipeline.config import PipelineConfig
from rag_pipeline.pipeline import HybridRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval pipeline for a claim.")
    parser.add_argument("claim", type=str, help="Claim to verify or support.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing prebuilt indexes and artifacts.",
    )
    parser.add_argument(
        "--top-supports",
        type=int,
        default=3,
        help="Number of supporting passages to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(work_dir=args.work_dir)
    retriever = HybridRetriever(config)
    response = retriever.retrieve(args.claim)
    response["supports"] = response["supports"][: args.top_supports]
    print(json.dumps(response, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
