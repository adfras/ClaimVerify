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
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device for embeddings, reranker, and NLI.",
    )
    parser.add_argument(
        "--no-strip-references",
        action="store_true",
        help="Disable stripping bibliographic lines from candidate passages.",
    )
    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.8,
        help="Minimum entailment probability to accept a passage as support.",
    )
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=0.9,
        help="Minimum contradiction probability to flag a passage as contradiction.",
    )
    parser.add_argument(
        "--high-confidence-support",
        type=float,
        default=0.9,
        help="Entailment score that bypasses lexical overlap requirements for support.",
    )
    parser.add_argument(
        "--essential-term-min-length",
        type=int,
        default=6,
        help="Minimum token length to count as an essential claim term.",
    )
    parser.add_argument(
        "--essential-term-overlap",
        type=int,
        default=2,
        help="Minimum number of essential claim terms that must appear in evidence.",
    )
    parser.add_argument(
        "--disable-llm-rechecker",
        action="store_true",
        help="Skip the LLM-based post filtering step even if an API key is available.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the LLM rechecker model name (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="Sampling temperature for the LLM rechecker.",
    )
    parser.add_argument(
        "--llm-max-supports",
        type=int,
        default=None,
        help="Maximum number of supports to validate with the LLM rechecker.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="Custom base URL for the OpenAI-compatible endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_kwargs = dict(
        work_dir=args.work_dir,
        inference_device=args.device,
        strip_reference_lines=not args.no_strip_references,
        nli_threshold=args.nli_threshold,
        contradiction_threshold=args.contradiction_threshold,
        high_confidence_support_override=args.high_confidence_support,
        essential_term_min_length=args.essential_term_min_length,
        essential_term_overlap=args.essential_term_overlap,
    )
    if args.disable_llm_rechecker:
        config_kwargs["use_llm_rechecker"] = False
    if args.llm_model:
        config_kwargs["llm_model"] = args.llm_model
    if args.llm_temperature is not None:
        config_kwargs["llm_temperature"] = args.llm_temperature
    if args.llm_max_supports is not None:
        config_kwargs["llm_max_supports"] = args.llm_max_supports
    if args.llm_base_url:
        config_kwargs["llm_api_base"] = args.llm_base_url
    config = PipelineConfig(**config_kwargs)
    retriever = HybridRetriever(config)
    response = retriever.retrieve(args.claim)
    response["supports"] = response["supports"][: args.top_supports]
    print(json.dumps(response, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
