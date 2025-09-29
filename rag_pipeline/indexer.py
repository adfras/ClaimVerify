from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import PipelineConfig
from .models import Chunk, Document
from .preprocess import (
    chunk_document,
    load_documents,
    load_spacy_model,
    persist_chunks,
    persist_documents,
)


@dataclass
class CorpusArtifacts:
    documents_path: Path
    chunks_path: Path
    embeddings_path: Path
    dense_index_path: Path
    metadata_path: Path


def simple_tokenize(text: str) -> list[str]:
    return text.lower().split()


class IndexBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.ensure_paths()
        self.artifacts = CorpusArtifacts(
            documents_path=self.config.work_dir / "artifacts" / "documents.jsonl",
            chunks_path=self.config.work_dir / "artifacts" / "chunks.jsonl",
            embeddings_path=self.config.work_dir / "artifacts" / "chunk_embeddings.npy",
            dense_index_path=self.config.work_dir / "indexes" / "dense.faiss",
            metadata_path=self.config.work_dir / "artifacts" / "metadata.json",
        )

    def build(self) -> None:
        documents = load_documents(self.config)
        nlp = load_spacy_model(self.config.models["spacy_model"])
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = chunk_document(
                document=doc,
                nlp=nlp,
                target_tokens=self.config.chunk_target_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
            )
            all_chunks.extend(chunks)

        persist_documents(documents, self.artifacts.documents_path)
        persist_chunks(all_chunks, self.artifacts.chunks_path)

        self._build_dense_index(all_chunks)
        self._write_metadata(documents, all_chunks)

    def _build_dense_index(self, chunks: Sequence[Chunk]) -> None:
        embedder_name = self.config.models["embedder"]
        embedder = SentenceTransformer(embedder_name)
        embedder.max_seq_length = 512
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = embeddings.astype(np.float32)
        np.save(self.artifacts.embeddings_path, embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(self.artifacts.dense_index_path))

    def _write_metadata(self, documents: Sequence[Document], chunks: Sequence[Chunk]) -> None:
        metadata = {
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "embedder_model": self.config.models["embedder"],
            "reranker_model": self.config.models["reranker"],
            "nli_model": self.config.models["nli"],
            "chunk_target_tokens": self.config.chunk_target_tokens,
            "chunk_overlap_tokens": self.config.chunk_overlap_tokens,
        }
        self.artifacts.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
