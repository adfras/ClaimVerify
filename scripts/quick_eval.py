from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_pipeline.config import PipelineConfig
from rag_pipeline.pipeline import HybridRetriever

CASES = [
    {
        "claim": "Bayesian approaches may help to improve VR engagement.",
        "expected": "supported",
    },
    {
        "claim": "Bayesian Theory of Mind reduces false inferences when virtual agents interpret user intent.",
        "expected": "supported",
    },
    {
        "claim": "Bayesian approaches guarantee a cure for all cancers.",
        "expected": "insufficient",
    },
    {
        "claim": "Virtual avatars controlled by Bayesian algorithms always misinterpret human intent.",
        "expected": "contradicted",
    },
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the quick four-claim regression suite.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing retrieval artifacts (e.g. data or data/chunk_420).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(work_dir=args.work_dir)
    summary = run_cases(config)
    print(json.dumps(summary, indent=2))


def run_cases(config: PipelineConfig) -> dict:
    retriever = HybridRetriever(config)
    results = []
    correct = 0
    for case in CASES:
        response = retriever.retrieve(case["claim"])
        verdict = response["verdict"]
        results.append(
            {
                "claim": case["claim"],
                "expected": case["expected"],
                "verdict": verdict,
                "supports": response["supports"],
                "contradictions": response["contradictions"],
            }
        )
        if verdict == case["expected"]:
            correct += 1
    summary = {
        "total": len(CASES),
        "correct": correct,
        "accuracy": correct / len(CASES),
        "details": results,
    }
    return summary


if __name__ == "__main__":
    main()
