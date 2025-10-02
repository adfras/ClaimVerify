from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class PipelineConfig:
    corpus_dir: Path = Path("full_text")
    work_dir: Path = Path("data")
    chunk_target_tokens: int = 220
    chunk_overlap_tokens: int = 30
    bm25_top_k: int = 35
    dense_top_k: int = 35
    fusion_k: int = 60
    rerank_keep_top_k: int = 20
    nli_threshold: float = 0.92
    contradiction_threshold: float = 0.95
    neutral_confidence_gate: float = 0.8
    neutral_entailment_floor: float = 0.0
    fallback_entailment_threshold: float = 0.9
    lexical_support_overlap: float = 0.5
    require_modal_alignment: bool = True
    use_dense_retrieval: bool = True
    enable_ocr: bool = True
    ocr_command: str = "auto"
    ocr_min_empty_ratio: float = 0.5
    strip_headers_and_footers: bool = True
    strip_reference_lines: bool = True
    high_confidence_support_override: float = 0.98
    essential_term_min_length: int = 5
    essential_term_overlap: int = 3
    quote_min_chars: int = 120
    quote_max_chars: int = 360
    llm_context_max_chars: int = 800
    enable_grobid: bool = False
    auto_detect_grobid: bool = True
    grobid_url: str = "http://localhost:8070"
    grobid_timeout: float = 15.0
    entailment_label_index: int = 2  # default huggingface MNLI ordering
    models: dict[str, str] = field(
        default_factory=lambda: {
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "reranker": "BAAI/bge-reranker-large",
            "nli": "khalidalt/DeBERTa-v3-large-mnli",
            "spacy_model": "en_core_web_sm",
        }
    )
    use_llm_rechecker: bool = True
    llm_model: str = "gpt-5-nano-2025-08-07"
    llm_temperature: float = 0.0
    llm_max_supports: int = 3
    llm_api_base: str | None = None
    inference_device: str = "auto"
    model_fallbacks: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "reranker": [
                "cross-encoder/ms-marco-MiniLM-L6-v2",
                "BAAI/bge-reranker-large",
            ],
            "nli": [
                "MoritzLaurer/DeBERTa-v3-base-mnli",
                "facebook/bart-large-mnli",
            ],
        }
    )
    skip_documents_file: Path | None = None
    skip_documents: Set[str] = field(default_factory=set)

    def ensure_paths(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "indexes").mkdir(parents=True, exist_ok=True)
        (self.work_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        existing_cache = os.environ.get("HF_HOME")
        hf_cache = Path(existing_cache) if existing_cache else self.work_dir / "hf_cache"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(hf_cache))
        self._load_skip_documents()

    def _load_skip_documents(self) -> None:
        candidates: list[Path] = []
        if self.skip_documents_file is not None:
            candidates.append(self.skip_documents_file)
        candidates.append(self.work_dir / "pdf_skip.txt")
        candidates.append(self.corpus_dir / "pdf_skip.txt")
        candidates.append(Path("pdf_skip.txt"))

        skip: Set[str] = set()
        for candidate in candidates:
            try_path = candidate.expanduser().resolve()
            if not try_path.exists():
                continue
            try:
                for line in try_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    skip.add(line.lower())
            except Exception:
                continue
            if skip:
                break
        self.skip_documents = skip
