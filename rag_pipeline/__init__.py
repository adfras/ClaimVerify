"""Hybrid retrieval pipeline package."""

from __future__ import annotations

import os
from pathlib import Path


def _prepare_hf_cache() -> None:
    package_root = Path(__file__).resolve().parent.parent
    default_cache = package_root / "data" / "hf_cache"
    target = Path(os.environ.get("HF_HOME", default_cache))
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - fallback for restricted envs
        fallback = Path.cwd() / ".hf_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        target = fallback
    os.environ["HF_HOME"] = str(target)
    os.environ["TRANSFORMERS_CACHE"] = str(target)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(target)


_prepare_hf_cache()

from .config import PipelineConfig  # noqa: E402,F401
from .pipeline import HybridRetriever  # noqa: E402,F401
