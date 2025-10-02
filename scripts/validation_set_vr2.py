from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_pipeline.config import PipelineConfig
from rag_pipeline.pipeline import HybridRetriever

CASES = [
    {
        "claim": "Bayesian VR models reduce the rate of false inferences about user intent.",
        "expected": "supported",
    },
    {
        "claim": "Bayesian causal inference models of body ownership rely on psychologically implausible parameter values.",
        "expected": "supported",
    },
    {
        "claim": "Bayesian Theory of Mind VR agents use locomotion to accomplish their goals.",
        "expected": "supported",
    },
    {
        "claim": "Bayesian avatar controllers always misinterpret human gaze cues.",
        "expected": "contradicted",
    },
    {
        "claim": "Bayesian VR systems guarantee zero noise when tracking user intent.",
        "expected": "insufficient",
    },
    {
        "claim": "Bayesian VR models always deliver perfect body ownership for every user.",
        "expected": "insufficient",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the six-claim VR/body ownership regression suite."
    )
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
