"""Run regression suites against multiple index variants for quick comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_pipeline.config import PipelineConfig

from scripts import quick_eval, validation_set_vr2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval verdict accuracy for one or more work directories "
            "using both regression suites."
        )
    )
    parser.add_argument(
        "--work-dirs",
        type=Path,
        nargs="+",
        default=[Path("data")],
        help="List of work directories to evaluate (e.g. data data/chunk_420).",
    )
    return parser.parse_args()


def run_suite(label: str, work_dir: Path) -> dict:
    config = PipelineConfig(work_dir=work_dir)
    quick_summary = quick_eval.run_cases(config)
    vr2_summary = validation_set_vr2.run_cases(config)
    return {
        "label": label,
        "work_dir": str(work_dir),
        "quick_eval": quick_summary,
        "validation_vr2": vr2_summary,
    }


def main() -> None:
    args = parse_args()
    reports = []
    for work_dir in args.work_dirs:
        label = work_dir.name or str(work_dir)
        print(f"\n>>> Evaluating index at {work_dir}")
        if not work_dir.exists():
            print(f"Skipping {work_dir}: directory does not exist")
            continue
        report = run_suite(label, work_dir)
        reports.append(report)
        print(
            json.dumps(
                {
                    "label": report["label"],
                    "quick_eval_accuracy": report["quick_eval"]["accuracy"],
                    "validation_vr2_accuracy": report["validation_vr2"]["accuracy"],
                },
                indent=2,
            )
        )
    print("\n=== Full comparison report ===")
    summary = json.dumps(reports, indent=2)
    print(summary)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = logs_dir / "regress_summary.json"
    output_path.write_text(summary, encoding="utf-8")
    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
