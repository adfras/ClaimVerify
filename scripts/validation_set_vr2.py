from __future__ import annotations

import json

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


def main() -> None:
    config = PipelineConfig()
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
