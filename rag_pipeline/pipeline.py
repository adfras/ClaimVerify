from __future__ import annotations

import math
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import PipelineConfig
from .indexer import CorpusArtifacts, IndexBuilder, simple_tokenize
from .models import Chunk
from .storage import load_chunks


@dataclass
class RetrievalResult:
    kind: str  # "support", "contradict", or "neutral"
    chunk: Chunk
    quote: str
    rerank_score: float
    entailment_probability: float
    contradiction_probability: float
    confidence: float
    neighbors: list[Chunk]


def reciprocal_rank_fusion(rank_lists: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for position, idx in enumerate(ranks):
            if idx < 0:
                continue
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + position + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def unique_ordered(items: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


class HybridRetriever:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.ensure_paths()
        builder = IndexBuilder(config)
        self.artifacts = builder.artifacts
        self._ensure_indexes_exist()

        self.chunks: list[Chunk] = load_chunks(self.artifacts.chunks_path)
        self._chunk_lookup = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}
        self._bm25 = BM25Okapi([simple_tokenize(chunk.text) for chunk in self.chunks])

        self._embedder = SentenceTransformer(self.config.models["embedder"])
        self._embedder.max_seq_length = 512
        self._dense_index = faiss.read_index(str(self.artifacts.dense_index_path))

        self._reranker = CrossEncoder(self.config.models["reranker"], max_length=512)
        self._nli_tokenizer = AutoTokenizer.from_pretrained(
            self.config.models["nli"], use_fast=False
        )
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.models["nli"]
        ).eval()

    def _ensure_indexes_exist(self) -> None:
        missing = [
            path
            for path in [
                self.artifacts.documents_path,
                self.artifacts.chunks_path,
                self.artifacts.embeddings_path,
                self.artifacts.dense_index_path,
            ]
            if not path.exists()
        ]
        if missing:
            # Build indexes if they are absent.
            builder = IndexBuilder(self.config)
            builder.build()

    def retrieve(self, claim: str) -> dict[str, Any]:
        initial_candidates = self._initial_retrieval(claim)
        reranked = self._rerank(claim, initial_candidates)
        results = self._nli_filter(claim, reranked)
        return self._format_response(claim, results)

    def _initial_retrieval(self, claim: str) -> list[int]:
        query_tokens = simple_tokenize(claim)
        bm25_scores = self._bm25.get_scores(query_tokens)
        if isinstance(bm25_scores, list):
            bm25_scores = np.array(bm25_scores)
        bm25_ranked = np.argsort(-bm25_scores)[: self.config.bm25_top_k]

        query_vec = self._embed_query(claim)
        dense_scores, dense_indices = self._dense_index.search(query_vec, self.config.dense_top_k)
        dense_ranked = dense_indices[0].tolist()

        fused = reciprocal_rank_fusion([bm25_ranked.tolist(), dense_ranked], self.config.fusion_k)
        return fused[: max(self.config.bm25_top_k, self.config.dense_top_k)]

    def _embed_query(self, claim: str) -> np.ndarray:
        vector = self._embedder.encode(
            [claim],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)

    def _rerank(self, claim: str, candidate_indices: Sequence[int]) -> list[tuple[int, float]]:
        candidates = unique_ordered(candidate_indices)
        texts = [self.chunks[idx].text for idx in candidates]
        pairs = list(zip([claim] * len(texts), texts))
        if not pairs:
            return []
        scores = self._reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        return reranked[: self.config.rerank_keep_top_k]

    def _nli_filter(
        self, claim: str, reranked: Sequence[tuple[int, float]]
    ) -> list[RetrievalResult]:
        if not reranked:
            return []
        supports: list[RetrievalResult] = []
        contradictions: list[RetrievalResult] = []
        neutral: list[RetrievalResult] = []
        scores = np.array([score for _, score in reranked], dtype=np.float32)
        claim_modal_terms = (
            self._extract_universal_terms(claim)
            if self.config.require_modal_alignment
            else set()
        )
        claim_polarity = self._lexical_polarity(claim)
        modal_terms = (
            claim_modal_terms
        )
        if len(scores) > 1:
            min_score, max_score = float(scores.min()), float(scores.max())
        else:
            min_score = max_score = float(scores[0]) if scores.size else 0.0

        for idx, score in reranked:
            chunk = self.chunks[idx]
            entail_prob, contra_prob = self._nli_scores(chunk.text, claim)
            rerank_norm = self._normalise_score(score, min_score, max_score)
            confidence = rerank_norm * entail_prob
            lexical_signal = self._lexical_support_signal(chunk.text, claim)
            result = RetrievalResult(
                kind="support" if entail_prob >= self.config.nli_threshold else "contradict",
                chunk=chunk,
                quote=self._select_quote(chunk.text, claim),
                rerank_score=float(score),
                entailment_probability=entail_prob,
                contradiction_probability=contra_prob,
                confidence=confidence,
                neighbors=self._neighbor_chunks(idx),
            )
            modal_ok = self._modal_alignment_ok(modal_terms, chunk.text)
            if entail_prob >= self.config.nli_threshold and modal_ok:
                supports.append(result)
                continue
            if contra_prob >= self.config.contradiction_threshold:
                result.kind = "contradict"
                contradictions.append(result)
                continue
            if self._should_apply_fallback(
                rerank_norm, entail_prob, contra_prob
            ):
                fallback = self._fallback_sentence_entailment(chunk.text, claim)
                if fallback is not None:
                    best_entail, best_sentence = fallback
                    if best_entail >= self.config.fallback_entailment_threshold and self._modal_alignment_ok(modal_terms, best_sentence or chunk.text):
                        result.kind = "support"
                        result.entailment_probability = best_entail
                        result.quote = (
                            self._truncate_sentence(best_sentence)
                            if best_sentence
                            else result.quote
                        )
                        result.confidence = rerank_norm * best_entail
                        supports.append(result)
                        continue
                if lexical_signal is not None:
                    coverage, sentence = lexical_signal
                    if coverage >= self.config.lexical_support_overlap and self._modal_alignment_ok(modal_terms, sentence or chunk.text):
                        result.kind = "support"
                        result.entailment_probability = max(
                            result.entailment_probability, coverage
                        )
                        if sentence:
                            result.quote = self._truncate_sentence(sentence)
                        result.confidence = rerank_norm * result.entailment_probability
                        supports.append(result)
                        continue
            chunk_polarity = self._lexical_polarity(chunk.text)
            if (
                claim_polarity < 0
                and chunk_polarity > 0
                and lexical_signal is not None
                and lexical_signal[0] >= 0.15
            ):
                result.kind = "contradict"
                result.contradiction_probability = max(
                    result.contradiction_probability, lexical_signal[0]
                )
                if lexical_signal[1]:
                    result.quote = self._truncate_sentence(lexical_signal[1])
                result.confidence = rerank_norm * result.contradiction_probability
                contradictions.append(result)
                continue

            result.kind = "neutral"
            neutral.append(result)

        supports.sort(key=lambda res: res.confidence, reverse=True)
        contradictions.sort(key=lambda res: res.contradiction_probability, reverse=True)
        # neutral results are currently omitted from final payload
        return supports + contradictions

    def _nli_scores(self, premise: str, hypothesis: str) -> tuple[float, float]:
        with torch.no_grad():
            encoded = self._nli_tokenizer(
                premise,
                hypothesis,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            logits = self._nli_model(**encoded).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
            entail_prob = float(probs[self.config.entailment_label_index])
            contradiction_prob = float(probs[0])
            return entail_prob, contradiction_prob

    def _combine_scores(
        self, rerank_score: float, min_score: float, max_score: float, entail_prob: float
    ) -> float:
        rerank_normalised = self._normalise_score(rerank_score, min_score, max_score)
        return rerank_normalised * entail_prob

    def _normalise_score(self, score: float, min_score: float, max_score: float) -> float:
        if math.isclose(max_score, min_score):
            return 1.0
        return (score - min_score) / (max_score - min_score)

    def _neighbor_chunks(self, index: int, window: int = 1) -> list[Chunk]:
        neighbors: list[Chunk] = []
        for offset in range(-window, window + 1):
            neighbor_index = index + offset
            if neighbor_index < 0 or neighbor_index >= len(self.chunks):
                continue
            if self.chunks[neighbor_index].doc_id != self.chunks[index].doc_id:
                continue
            neighbors.append(self.chunks[neighbor_index])
        return neighbors

    def _select_quote(self, chunk_text: str, claim: str) -> str:
        sentences = [sent.strip() for sent in chunk_text.split(".") if sent.strip()]
        claim_terms = [term for term in simple_tokenize(claim) if len(term) > 3]
        for sentence in sentences:
            score = sum(1 for term in claim_terms if term in sentence.lower())
            if score >= max(1, len(claim_terms) // 4):
                return self._truncate_sentence(sentence)
        return self._truncate_sentence(sentences[0]) if sentences else chunk_text

    def _truncate_sentence(self, sentence: str, max_length: int = 400) -> str:
        sentence = sentence.strip()
        if not sentence:
            return sentence
        if len(sentence) <= max_length:
            return sentence if sentence.endswith(".") else sentence + "."
        truncated = sentence[: max_length].rsplit(" ", 1)[0]
        return truncated.rstrip(" ,;") + "..."

    def _should_apply_fallback(
        self, rerank_normalised: float, entail_prob: float, contra_prob: float
    ) -> bool:
        return (
            rerank_normalised >= self.config.neutral_confidence_gate
            and entail_prob >= self.config.neutral_entailment_floor
            and contra_prob < self.config.contradiction_threshold
        )

    def _fallback_sentence_entailment(
        self, chunk_text: str, claim: str
    ) -> tuple[float, str] | None:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", chunk_text)
            if sentence.strip()
        ]
        if not sentences:
            return None
        with torch.no_grad():
            encoded = self._nli_tokenizer(
                sentences,
                [claim] * len(sentences),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            logits = self._nli_model(**encoded).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        entail_probs = probs[:, self.config.entailment_label_index]
        best_idx = int(torch.argmax(entail_probs).item())
        best_entail = float(entail_probs[best_idx])
        return best_entail, sentences[best_idx]

    def _lexical_support_signal(self, chunk_text: str, claim: str) -> tuple[float, str] | None:
        claim_terms = [
            term
            for term in simple_tokenize(claim)
            if len(term) > 3
        ]
        if not claim_terms:
            return None
        claim_set = set(claim_terms)
        best_sentence = ""
        best_coverage = 0.0
        for sentence in re.split(r"(?<=[.!?])\s+", chunk_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            sent_terms = set(simple_tokenize(sentence))
            if not sent_terms:
                continue
            overlap = len(claim_set & sent_terms)
            coverage = overlap / len(claim_set)
            if coverage > best_coverage:
                best_coverage = coverage
                best_sentence = sentence
        if best_coverage == 0.0:
            return None
        return best_coverage, best_sentence

    def _lexical_polarity(self, text: str) -> int:
        positive_markers = {
            "improve",
            "improves",
            "improved",
            "improving",
            "plausible",
            "comfort",
            "comfortable",
            "reduce",
            "reduces",
            "reduced",
            "reducing",
            "increase",
            "increases",
            "increased",
            "increasing",
            "correct",
            "correctly",
            "benefit",
            "benefits",
            "accurate",
            "accuracy",
            "positive",
            "robust",
        }
        negative_markers = {
            "misinterpret",
            "misinterprets",
            "misinterpreted",
            "misinterpreting",
            "false",
            "failure",
            "failures",
            "harm",
            "harmful",
            "incorrect",
            "worse",
            "negative",
            "error",
            "errors",
        }
        tokens = set(simple_tokenize(text))
        pos = len(tokens & positive_markers)
        neg = len(tokens & negative_markers)
        if pos > neg:
            return 1
        if neg > pos:
            return -1
        return 0

    def _extract_universal_terms(self, claim: str) -> set[str]:
        universal = {
            "always",
            "never",
            "guarantee",
            "guarantees",
            "guaranteed",
            "all",
            "every",
            "everyone",
            "nobody",
            "none",
        }
        return {token for token in simple_tokenize(claim) if token in universal}

    def _modal_alignment_ok(self, modal_terms: set[str], text: str) -> bool:
        if not modal_terms:
            return True
        text_terms = set(simple_tokenize(text))
        return any(term in text_terms for term in modal_terms)

    def _format_response(self, claim: str, results: Sequence[RetrievalResult]) -> dict[str, Any]:
        supports = [res for res in results if res.kind == "support"]
        contradictions = [res for res in results if res.kind == "contradict"]
        verdict = "supported" if supports else ("contradicted" if contradictions else "insufficient")

        def serialise(res: RetrievalResult) -> dict[str, Any]:
            return {
                "chunk_id": res.chunk.chunk_id,
                "article_id": res.chunk.doc_id,
                "title": res.chunk.title,
                "paragraph_index": res.chunk.paragraph_index,
                "char_start": res.chunk.char_start,
                "char_end": res.chunk.char_end,
                "page_start": res.chunk.page_start,
                "page_end": res.chunk.page_end,
                "page_start_offset": res.chunk.page_start_offset,
                "page_end_offset": res.chunk.page_end_offset,
                "quote": res.quote,
                "chunk_text": res.chunk.text,
                "neighbors": [neighbor.to_payload() for neighbor in res.neighbors],
                "rerank_score": float(res.rerank_score),
                "entailment_probability": float(res.entailment_probability),
                "contradiction_probability": float(res.contradiction_probability),
                "confidence": float(res.confidence),
            }

        response = {
            "claim": claim,
            "verdict": verdict,
            "supports": [serialise(res) for res in supports],
            "contradictions": [serialise(res) for res in contradictions],
            "config": {
                "bm25_top_k": self.config.bm25_top_k,
                "dense_top_k": self.config.dense_top_k,
                "rerank_keep_top_k": self.config.rerank_keep_top_k,
                "nli_threshold": self.config.nli_threshold,
            },
        }
        return response
