from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    corpus_dir: Path = Path("full_text")
    work_dir: Path = Path("data")
    chunk_target_tokens: int = 280
    chunk_overlap_tokens: int = 50
    bm25_top_k: int = 50
    dense_top_k: int = 50
    fusion_k: int = 60
    rerank_keep_top_k: int = 12
    nli_threshold: float = 0.52
    contradiction_threshold: float = 0.6
    neutral_confidence_gate: float = 0.7
    neutral_entailment_floor: float = 0.05
    fallback_entailment_threshold: float = 0.6
    lexical_support_overlap: float = 0.2
    require_modal_alignment: bool = True
    entailment_label_index: int = 2  # default huggingface MNLI ordering
    models: dict[str, str] = field(
        default_factory=lambda: {
            "embedder": "BAAI/bge-small-en-v1.5",
            "reranker": "BAAI/bge-reranker-base",
            "nli": "MoritzLaurer/DeBERTa-v3-base-mnli",
            "spacy_model": "en_core_web_sm",
        }
    )

    def ensure_paths(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "indexes").mkdir(parents=True, exist_ok=True)
        (self.work_dir / "artifacts").mkdir(parents=True, exist_ok=True)
