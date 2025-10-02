from __future__ import annotations

import math
from dataclasses import dataclass
import re
import string
from pathlib import Path
from typing import Any, Iterable, Sequence
import warnings

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import PipelineConfig
from .indexer import CorpusArtifacts, IndexBuilder, simple_tokenize
from .models import Chunk
from .storage import load_chunks, load_documents
from .llm import LLMRechecker


_EMBEDDER_CACHE: dict[tuple[str, str], SentenceTransformer] = {}
_RERANKER_CACHE: dict[tuple[str, str], CrossEncoder] = {}
_NLI_CACHE: dict[
    tuple[str, str], tuple[AutoTokenizer, AutoModelForSequenceClassification]
] = {}


REFERENCE_AUTHORS_PATTERN = re.compile(
    r"^(?:\d+\s*[.)]\s*)?[A-Z][A-Za-zÀ-ÖØ-Ý'’\-]+.*\(20\d{2}[a-z]?\)"
)
REFERENCE_URL_INDICATORS = ("doi", "https://", "http://", "arxiv", "urn:")


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
    llm_verdict: bool | None = None
    llm_note: str | None = None


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
        self._llm_rechecker = (
            LLMRechecker(
                self.config.llm_model,
                self.config.llm_temperature,
                base_url=self.config.llm_api_base,
            )
            if self.config.use_llm_rechecker
            else None
        )
        self.config.ensure_paths()
        self._device = self._resolve_device(self.config.inference_device)
        builder = IndexBuilder(config)
        self.artifacts = builder.artifacts
        self._ensure_indexes_exist()

        self.documents = load_documents(self.artifacts.documents_path)
        self._document_lookup = {doc.doc_id: doc for doc in self.documents}
        self.chunks: list[Chunk] = load_chunks(self.artifacts.chunks_path)
        self._chunk_lookup = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}
        self._cleaned_chunks: dict[int, str] = {}
        bm25_corpus: list[list[str]] = []
        for i in range(len(self.chunks)):
            tokens = simple_tokenize(self._get_clean_chunk_text(i))
            if not tokens:
                tokens = ["__ref__"]
            bm25_corpus.append(tokens)
        self._bm25 = BM25Okapi(bm25_corpus)

        self._embedder: SentenceTransformer | None = None
        self._dense_index = None
        self._gpu_resources = None
        if self.config.use_dense_retrieval:
            self._embedder = self._get_embedder(self.config.models["embedder"])
            self._dense_index = self._load_dense_index()

        self._reranker = self._load_reranker(self.config.models["reranker"])
        (
            self._nli_tokenizer,
            self._nli_model,
        ) = self._load_nli_components(self.config.models["nli"])
        self._rerank_raw_scores: dict[int, float] = {}

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

    def _get_clean_chunk_text(self, index: int) -> str:
        if index in self._cleaned_chunks:
            return self._cleaned_chunks[index]
        raw_text = self.chunks[index].text
        cleaned = self._prepare_chunk_text(raw_text)
        self._cleaned_chunks[index] = cleaned
        return cleaned

    def _get_clean_text_for_chunk(self, chunk: Chunk) -> str:
        idx = self._chunk_lookup.get(chunk.chunk_id)
        if idx is not None:
            return self._get_clean_chunk_text(idx)
        return self._prepare_chunk_text(chunk.text)

    @staticmethod
    def _normalise_term(token: str) -> str:
        token = token.lower().strip()
        token = token.strip(string.punctuation)
        if not token:
            return ""
        if token.endswith("used"):
            return "use"
        if len(token) > 3 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ers"):
            token = token[:-3]
        elif len(token) > 3 and token.endswith("ed"):
            base = token[:-2]
            if base.endswith("us"):
                token = base + "e"
            elif base.endswith("i"):
                token = base + "y"
            else:
                token = base
        elif len(token) > 3 and token.endswith("es"):
            token = token[:-2]
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        return token

    def _prepare_chunk_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text
        if self.config.strip_reference_lines:
            cleaned = self._strip_reference_lines(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _strip_reference_lines(self, text: str) -> str:
        lines = text.splitlines()
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered in {"references", "reference", "bibliography"}:
                continue
            is_author_year = (
                REFERENCE_AUTHORS_PATTERN.match(stripped)
                or re.search(r"\(20\d{2}[a-z]?\)", stripped)
            )
            has_ref_indicator = any(
                indicator in lowered for indicator in REFERENCE_URL_INDICATORS
            )
            if is_author_year and has_ref_indicator:
                continue
            kept.append(stripped)
        if not kept:
            return ""
        return " ".join(kept)

    def _get_embedder(self, model_id: str) -> SentenceTransformer:
        key = (model_id, self._device)
        embedder = _EMBEDDER_CACHE.get(key)
        if embedder is None:
            embedder = SentenceTransformer(model_id, device=self._device)
            embedder.max_seq_length = 256
            _EMBEDDER_CACHE[key] = embedder
        else:
            embedder.max_seq_length = 256
        return embedder

    def _load_dense_index(self):
        index = faiss.read_index(str(self.artifacts.dense_index_path))
        if self._device != "cuda":
            return index
        try:
            num_gpus = faiss.get_num_gpus()
        except Exception:  # pragma: no cover - older FAISS builds
            num_gpus = 0
        if num_gpus <= 0:
            return index
        try:
            resources = faiss.StandardGpuResources()
            self._gpu_resources = resources
            index = faiss.index_cpu_to_gpu(resources, 0, index)
        except Exception as exc:  # noqa: BLE001 - GPU fallback
            warnings.warn(
                f"Failed to move FAISS index to GPU, staying on CPU: {exc}",
                RuntimeWarning,
            )
        return index

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

        rank_lists: list[list[int]] = [bm25_ranked.tolist()]
        if self.config.use_dense_retrieval and self._dense_index is not None:
            query_vec = self._embed_query(claim)
            if query_vec is not None:
                dense_scores, dense_indices = self._dense_index.search(
                    query_vec, self.config.dense_top_k
                )
                dense_ranked = dense_indices[0].tolist()
                rank_lists.append(dense_ranked)

        fused = reciprocal_rank_fusion(rank_lists, self.config.fusion_k)
        limit = (
            max(self.config.bm25_top_k, self.config.dense_top_k)
            if self.config.use_dense_retrieval and self._dense_index is not None
            else self.config.bm25_top_k
        )
        return fused[:limit]

    def _embed_query(self, claim: str) -> np.ndarray | None:
        if self._embedder is None:
            return None
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
        filtered_indices: list[int] = []
        texts: list[str] = []
        for idx in candidates:
            text = self._get_clean_chunk_text(idx)
            if not text:
                continue
            filtered_indices.append(idx)
            texts.append(text)
        pairs = list(zip([claim] * len(texts), texts))
        if not pairs:
            self._rerank_raw_scores = {}
            return []
        scores = self._reranker.predict(
            pairs, batch_size=16, show_progress_bar=False
        )
        self._rerank_raw_scores = {
            idx: float(score) for idx, score in zip(filtered_indices, scores)
        }
        adjusted = [
            float(score) + self._source_bonus(self.chunks[idx])
            for idx, score in zip(filtered_indices, scores)
        ]
        reranked = sorted(
            zip(filtered_indices, adjusted), key=lambda item: item[1], reverse=True
        )
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
        stopwords_norm = {self._normalise_term(sw) for sw in CLAIM_STOPWORDS}
        claim_terms_raw = simple_tokenize(claim)
        filtered_claim_terms: list[str] = []
        for term in claim_terms_raw:
            normalised = self._normalise_term(term)
            if not normalised or normalised in stopwords_norm:
                continue
            filtered_claim_terms.append(normalised)
        if filtered_claim_terms:
            filtered_claim_terms = list(dict.fromkeys(filtered_claim_terms))
        claim_term_count = len(filtered_claim_terms)
        claim_terms_set = set(filtered_claim_terms)
        essential_terms = {
            term
            for term in claim_terms_set
            if len(term) >= self.config.essential_term_min_length
            and term not in ESSENTIAL_TERM_BLACKLIST
        }
        if not essential_terms and claim_terms_set:
            essential_terms = claim_terms_set.copy()
        min_required_essential = (
            min(len(essential_terms), self.config.essential_term_overlap)
            if essential_terms
            else 0
        )
        required_overlap = 1 if claim_term_count >= 3 else 0
        if len(scores) > 1:
            min_score, max_score = float(scores.min()), float(scores.max())
        else:
            min_score = max_score = float(scores[0]) if scores.size else 0.0

        for idx, score in reranked:
            chunk = self.chunks[idx]
            chunk_text = self._get_clean_chunk_text(idx)
            if not chunk_text:
                continue
            chunk_tokens = {
                self._normalise_term(token)
                for token in simple_tokenize(chunk_text)
            }
            chunk_tokens.discard("")
            has_claim_term = bool(claim_terms_set & chunk_tokens)
            essential_overlap = (
                len(essential_terms & chunk_tokens) if essential_terms else 0
            )
            entail_prob, contra_prob = self._nli_scores(chunk_text, claim)
            quote_text, sentence_entail = self._best_sentence_quote(chunk_text, claim)
            if not quote_text:
                quote_text = self._lexical_quote(chunk_text, claim)
            entail_prob = max(entail_prob, sentence_entail)
            rerank_norm = self._normalise_score(score, min_score, max_score)
            confidence = rerank_norm * entail_prob
            lexical_signal = self._lexical_support_signal(chunk_text, claim_terms_set)
            lexical_coverage = lexical_signal[0] if lexical_signal else 0.0
            lexical_sentence = lexical_signal[1] if lexical_signal else ""
            lexical_overlap = lexical_signal[2] if lexical_signal else 0
            min_contra_coverage = max(0.5, self.config.lexical_support_overlap)
            raw_rerank = self._rerank_raw_scores.get(idx, float(score))
            polarity_conflict = self._polarity_conflict(claim, chunk_text)
            chunk_polarity = self._lexical_polarity(chunk_text)
            result = RetrievalResult(
                kind="support" if entail_prob >= self.config.nli_threshold else "contradict",
                chunk=chunk,
                quote=quote_text,
                rerank_score=raw_rerank,
                entailment_probability=entail_prob,
                contradiction_probability=contra_prob,
                confidence=confidence,
                neighbors=self._neighbor_chunks(idx),
                llm_verdict=None,
                llm_note=None,
            )
            modal_ok = self._modal_alignment_ok(modal_terms, chunk_text)
            high_confidence_support = (
                max(entail_prob, sentence_entail)
                >= self.config.high_confidence_support_override
            )
            if (
                entail_prob >= self.config.nli_threshold
                and modal_ok
                and not polarity_conflict
                and (
                    lexical_coverage >= self.config.lexical_support_overlap
                    or has_claim_term
                    or claim_term_count <= 1
                    or high_confidence_support
                )
                and (
                    lexical_overlap >= required_overlap
                    or high_confidence_support
                    or claim_term_count <= 1
                )
                and (
                    essential_overlap >= min_required_essential
                    or not essential_terms
                    or high_confidence_support
                )
            ):
                supports.append(result)
                continue
            lexical_support_override = (
                lexical_signal is not None
                and lexical_coverage >= self.config.lexical_support_overlap
                and not polarity_conflict
                and has_claim_term
            )
            if (
                lexical_support_override
                and chunk_polarity >= 0
                and (
                    essential_overlap >= min_required_essential
                    or not essential_terms
                )
            ):
                result.kind = "support"
                result.entailment_probability = max(
                    result.entailment_probability, lexical_coverage
                )
                if lexical_sentence:
                    result.quote = self._truncate_sentence(lexical_sentence)
                result.confidence = rerank_norm * result.entailment_probability
                supports.append(result)
                continue
            flagged_contradiction = (
                contra_prob >= self.config.contradiction_threshold
            )
            if self._should_apply_fallback(
                rerank_norm, entail_prob, contra_prob
            ):
                if (
                    sentence_entail >= self.config.fallback_entailment_threshold
                    and self._modal_alignment_ok(modal_terms, quote_text or chunk_text)
                    and not polarity_conflict
                    and (
                        lexical_overlap >= required_overlap
                        or has_claim_term
                        or claim_term_count <= 1
                        or sentence_entail >= self.config.high_confidence_support_override
                    )
                    and not polarity_conflict
                    and (
                        essential_overlap >= min_required_essential
                        or not essential_terms
                        or sentence_entail
                        >= self.config.high_confidence_support_override
                    )
                ):
                    result.kind = "support"
                    result.entailment_probability = sentence_entail
                    result.confidence = rerank_norm * sentence_entail
                    supports.append(result)
                    continue
                if lexical_signal is not None:
                    coverage, sentence, overlap = lexical_signal
                    if (
                        coverage
                        >= max(0.1, self.config.lexical_support_overlap / 2.0)
                        and (overlap >= required_overlap or claim_term_count <= 1)
                        and self._modal_alignment_ok(modal_terms, sentence or chunk_text)
                        and not polarity_conflict
                        and (
                            essential_overlap >= min_required_essential
                            or not essential_terms
                        )
                    ):
                        result.kind = "support"
                        result.entailment_probability = max(
                            result.entailment_probability, coverage
                        )
                        if sentence:
                            result.quote = self._truncate_sentence(sentence)
                        result.confidence = rerank_norm * result.entailment_probability
                        supports.append(result)
                        continue
            if flagged_contradiction:
                if (
                    not polarity_conflict
                    or lexical_overlap < required_overlap
                    or not has_claim_term
                    or (
                        essential_overlap < min_required_essential
                        and essential_terms
                    )
                    or lexical_coverage < min_contra_coverage
                ):
                    result.kind = "neutral"
                    neutral.append(result)
                    continue
                result.kind = "contradict"
                contradictions.append(result)
                continue
            if (
                claim_polarity < 0
                and chunk_polarity > 0
                and lexical_signal is not None
                and lexical_signal[0] >= 0.15
            ):
                result.kind = "contradict"
                result.contradiction_probability = max(
                    result.contradiction_probability, lexical_coverage
                )
                if lexical_sentence:
                    result.quote = self._truncate_sentence(lexical_sentence)
                result.confidence = rerank_norm * result.contradiction_probability
                contradictions.append(result)
                continue

            result.kind = "neutral"
            neutral.append(result)

        support_ids = {res.chunk.chunk_id for res in supports}
        contradictions = [
            res for res in contradictions if res.chunk.chunk_id not in support_ids
        ]
        supports.sort(key=lambda res: res.confidence, reverse=True)
        contradictions.sort(key=lambda res: res.contradiction_probability, reverse=True)
        # neutral results are currently omitted from final payload
        supports = self._llm_post_filter(claim, supports)
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
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            logits = self._nli_model(**encoded).logits
            probs = (
                torch.nn.functional.softmax(logits, dim=-1)
                .squeeze(0)
                .detach()
                .cpu()
            )
            entail_prob = float(probs[self.config.entailment_label_index])
            contradiction_prob = float(probs[0])
            return entail_prob, contradiction_prob

    def _source_bonus(self, chunk: Chunk) -> float:
        doc = self._document_lookup.get(chunk.doc_id)
        if not doc:
            return 0.0
        return doc.source_quality * 0.1

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

    def _lexical_quote(self, chunk_text: str, claim: str) -> str:
        sentences = [sent.strip() for sent in chunk_text.split(".") if sent.strip()]
        claim_terms = [term for term in simple_tokenize(claim) if len(term) > 3]
        for sentence in sentences:
            score = sum(1 for term in claim_terms if term in sentence.lower())
            if score >= max(1, len(claim_terms) // 4):
                return self._truncate_sentence(sentence)
        return self._truncate_sentence(sentences[0]) if sentences else chunk_text

    def _load_reranker(self, primary_model: str) -> CrossEncoder:
        candidates = [primary_model]
        candidates.extend(
            model
            for model in self.config.model_fallbacks.get("reranker", [])
            if model not in candidates
        )
        errors: list[tuple[str, Exception]] = []
        for model_id in candidates:
            cache_key = (model_id, self._device)
            cached = _RERANKER_CACHE.get(cache_key)
            if cached is not None:
                if model_id != primary_model:
                    warnings.warn(
                        (
                            "Falling back to reranker model '%s' after failing to load '%s'."
                        )
                        % (model_id, primary_model),
                        RuntimeWarning,
                    )
                    self.config.models["reranker"] = model_id
                return cached
            try:
                reranker = CrossEncoder(
                    model_id, max_length=320, device=self._device
                )
                _RERANKER_CACHE[cache_key] = reranker
                if model_id != primary_model:
                    warnings.warn(
                        (
                            "Falling back to reranker model '%s' after failing to load '%s'."
                        )
                        % (model_id, primary_model),
                        RuntimeWarning,
                    )
                    self.config.models["reranker"] = model_id
                return reranker
            except Exception as exc:  # noqa: BLE001 - broad fallback to continue
                errors.append((model_id, exc))
        error_details = "; ".join(f"{mid}: {err}" for mid, err in errors)
        raise RuntimeError(
            "Unable to load any reranker model candidate. Attempted %s. Details: %s"
            % (", ".join(candidates), error_details)
        ) from errors[-1][1]

    def _load_nli_components(
        self, primary_model: str
    ) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        candidates = [primary_model]
        candidates.extend(
            model
            for model in self.config.model_fallbacks.get("nli", [])
            if model not in candidates
        )
        errors: list[tuple[str, Exception]] = []
        last_exc: Exception | None = None
        for model_id in candidates:
            cache_key = (model_id, self._device)
            cached = _NLI_CACHE.get(cache_key)
            if cached is not None:
                tokenizer, model = cached
                if model_id != primary_model:
                    warnings.warn(
                        (
                            "Falling back to NLI model '%s' after failing to load '%s'."
                        )
                        % (model_id, primary_model),
                        RuntimeWarning,
                    )
                    self.config.models["nli"] = model_id
                return tokenizer, model
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
                model = model.to(self._device).eval()
                _NLI_CACHE[cache_key] = (tokenizer, model)
                if model_id != primary_model:
                    warnings.warn(
                        (
                            "Falling back to NLI model '%s' after failing to load '%s'."
                        )
                        % (model_id, primary_model),
                        RuntimeWarning,
                    )
                    self.config.models["nli"] = model_id
                id2label = getattr(model.config, "id2label", None)
                if isinstance(id2label, dict):
                    normalised: dict[int, str] = {}
                    for key, value in id2label.items():
                        try:
                            idx = int(key)
                        except (TypeError, ValueError):
                            continue
                        normalised[idx] = str(value).strip().lower()
                    entailment_indices = [
                        idx for idx, label in normalised.items() if "entail" in label
                    ]
                    if entailment_indices:
                        entail_idx = entailment_indices[0]
                        if entail_idx != self.config.entailment_label_index:
                            self.config.entailment_label_index = entail_idx
                return tokenizer, model
            except Exception as exc:  # noqa: BLE001 - broad fallback to continue
                errors.append((model_id, exc))
                last_exc = exc
        error_details = "; ".join(f"{mid}: {err}" for mid, err in errors)
        raise RuntimeError(
            "Unable to load any NLI model candidate. Attempted %s. Details: %s"
            % (", ".join(candidates), error_details)
        ) from last_exc

    def _resolve_device(self, preference: str) -> str:
        pref = preference.lower()
        if pref not in {"auto", "cpu", "cuda"}:
            warnings.warn(
                f"Unknown inference_device '{preference}', defaulting to CPU.",
                RuntimeWarning,
            )
            pref = "auto"
        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            warnings.warn(
                "CUDA requested but not available; falling back to CPU.",
                RuntimeWarning,
            )
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _truncate_sentence(self, sentence: str, max_length: int | None = None) -> str:
        sentence = sentence.strip()
        if not sentence:
            return sentence
        if max_length is None:
            max_length = self.config.quote_max_chars
        if len(sentence) <= max_length:
            return sentence if sentence.endswith((".", "?", "!")) else sentence + "."
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

    def _best_sentence_quote(self, chunk_text: str, claim: str) -> tuple[str, float]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", chunk_text)
            if sentence.strip()
        ]
        if not sentences:
            return "", 0.0
        with torch.no_grad():
            encoded = self._nli_tokenizer(
                sentences,
                [claim] * len(sentences),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            logits = self._nli_model(**encoded).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()
        entail_probs = probs[:, self.config.entailment_label_index]
        best_idx = int(torch.argmax(entail_probs).item())
        best_entail = float(entail_probs[best_idx])

        max_chars = max(120, self.config.quote_max_chars)
        min_chars = max(0, self.config.quote_min_chars)
        quote_sentences = [sentences[best_idx]]
        left_bound = best_idx
        right_bound = best_idx

        for offset in (-1, 1):
            neighbor_idx = best_idx + offset
            if neighbor_idx < 0 or neighbor_idx >= len(sentences):
                continue
            neighbor_entail = float(entail_probs[neighbor_idx])
            if neighbor_entail < self.config.fallback_entailment_threshold:
                continue
            tentative = quote_sentences.copy()
            if offset < 0:
                tentative.insert(0, sentences[neighbor_idx])
            else:
                tentative.append(sentences[neighbor_idx])
            candidate = " ".join(tentative)
            if len(candidate) <= max_chars:
                quote_sentences = tentative
                if offset < 0:
                    left_bound = neighbor_idx
                else:
                    right_bound = neighbor_idx

        quote = " ".join(quote_sentences)
        if len(quote) < min_chars:
            left_cursor = left_bound - 1
            right_cursor = right_bound + 1
            while len(quote) < min_chars and (
                left_cursor >= 0 or right_cursor < len(sentences)
            ):
                candidates: list[tuple[float, str, list[str], int]] = []
                if left_cursor >= 0:
                    tentative_left = [sentences[left_cursor]] + quote_sentences
                    left_candidate = " ".join(tentative_left)
                    if len(left_candidate) <= max_chars:
                        candidates.append(
                            (
                                float(entail_probs[left_cursor]),
                                "left",
                                tentative_left,
                                left_cursor,
                            )
                        )
                if right_cursor < len(sentences):
                    tentative_right = quote_sentences + [sentences[right_cursor]]
                    right_candidate = " ".join(tentative_right)
                    if len(right_candidate) <= max_chars:
                        candidates.append(
                            (
                                float(entail_probs[right_cursor]),
                                "right",
                                tentative_right,
                                right_cursor,
                            )
                        )
                if not candidates:
                    break
                best_candidate = max(candidates, key=lambda item: item[0])
                direction = best_candidate[1]
                quote_sentences = best_candidate[2]
                if direction == "left":
                    left_bound = best_candidate[3]
                    left_cursor = left_bound - 1
                else:
                    right_bound = best_candidate[3]
                    right_cursor = right_bound + 1
                quote = " ".join(quote_sentences)

        return self._truncate_sentence(quote), best_entail

    @staticmethod
    def _clip_context(text: str, quote: str, max_chars: int = 600) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        if len(cleaned) <= max_chars:
            return cleaned
        if quote:
            lowered = cleaned.lower()
            fragment = quote.strip().lower()
            idx = lowered.find(fragment) if fragment else -1
            if idx != -1:
                half_window = max_chars // 2
                start = max(0, idx - half_window)
                end = start + max_chars
                if end > len(cleaned):
                    end = len(cleaned)
                    start = max(0, end - max_chars)
                return cleaned[start:end].strip()
        return cleaned[:max_chars].strip()

    def _lexical_support_signal(
        self, chunk_text: str, claim_terms: set[str]
    ) -> tuple[float, str, int] | None:
        if not claim_terms:
            return None
        best_sentence = ""
        best_coverage = 0.0
        best_overlap = 0
        for sentence in re.split(r"(?<=[.!?])\s+", chunk_text):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = {
                self._normalise_term(token)
                for token in simple_tokenize(sentence)
            }
            sentence_tokens.discard("")
            if not sentence_tokens:
                continue
            overlap = len(claim_terms & sentence_tokens)
            if not overlap:
                continue
            coverage = overlap / len(claim_terms)
            if coverage > best_coverage or (
                math.isclose(coverage, best_coverage) and overlap > best_overlap
            ):
                best_coverage = coverage
                best_sentence = sentence
                best_overlap = overlap
        if best_overlap == 0:
            return None
        return best_coverage, best_sentence, best_overlap

    def _polarity_conflict(self, claim: str, evidence: str) -> bool:
        claim_neg = self._lexical_polarity(claim) < 0 or self._contains_negative_marker(claim)
        evidence_neg = self._lexical_polarity(evidence) < 0 or self._contains_negative_marker(evidence)
        return claim_neg != evidence_neg and evidence_neg

    @staticmethod
    def _contains_negative_marker(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in NEGATIVE_MARKERS)

    def _llm_post_filter(self, claim: str, supports: list[RetrievalResult]) -> list[RetrievalResult]:
        if not supports or not self._llm_rechecker or not self._llm_rechecker.available:
            return supports
        validated: list[RetrievalResult] = []
        for result in supports:
            clean_context = self._get_clean_text_for_chunk(result.chunk)
            context = clean_context or result.chunk.text
            quote = result.quote or context
            context_window = self._clip_context(
                context,
                quote,
                max_chars=self.config.llm_context_max_chars,
            )
            document = self._document_lookup.get(result.chunk.doc_id)
            if document and document.title:
                context_window = (
                    f"Document title: {document.title}\n{context_window}"
                )
            decision, rationale = self._llm_rechecker.validate(
                claim,
                quote,
                context_window,
            )
            if decision is False:
                continue
            result.llm_verdict = decision
            if rationale:
                result.llm_note = rationale
            elif decision is None:
                result.llm_note = "LLM could not determine whether the quote supports the claim."
            else:
                result.llm_note = "LLM returned no explanation."
            validated.append(result)
        return validated

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
        return modal_terms.issubset(text_terms)

    def _format_response(self, claim: str, results: Sequence[RetrievalResult]) -> dict[str, Any]:
        supports = [res for res in results if res.kind == "support"]
        contradictions = [res for res in results if res.kind == "contradict"]
        verdict = "supported" if supports else ("contradicted" if contradictions else "insufficient")

        def serialise(res: RetrievalResult) -> dict[str, Any]:
            document = self._document_lookup.get(res.chunk.doc_id)
            return {
                "chunk_id": res.chunk.chunk_id,
                "article_id": res.chunk.doc_id,
                "title": res.chunk.title,
                "path": str(document.path) if document else None,
                "authors": list(document.authors) if document else [],
                "doi": document.doi if document else None,
                "source_quality": float(document.source_quality) if document else 0.0,
                "paragraph_index": res.chunk.paragraph_index,
                "char_start": res.chunk.char_start,
                "char_end": res.chunk.char_end,
                "page_start": res.chunk.page_start,
                "page_end": res.chunk.page_end,
                "page_start_offset": res.chunk.page_start_offset,
                "page_end_offset": res.chunk.page_end_offset,
                "quote": res.quote,
                "chunk_text": res.chunk.text,
                "cleaned_chunk_text": self._get_clean_text_for_chunk(res.chunk),
                "neighbors": [neighbor.to_payload() for neighbor in res.neighbors],
                "rerank_score": float(res.rerank_score),
                "entailment_probability": float(res.entailment_probability),
                "contradiction_probability": float(res.contradiction_probability),
                "confidence": float(res.confidence),
                "llm_verdict": res.llm_verdict,
                "llm_note": res.llm_note,
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
                "contradiction_threshold": self.config.contradiction_threshold,
                "strip_reference_lines": self.config.strip_reference_lines,
                "high_confidence_support_override": self.config.high_confidence_support_override,
                "essential_term_min_length": self.config.essential_term_min_length,
                "essential_term_overlap": self.config.essential_term_overlap,
                "quote_min_chars": self.config.quote_min_chars,
                "quote_max_chars": self.config.quote_max_chars,
                "llm_context_max_chars": self.config.llm_context_max_chars,
            },
        }
        return response
CLAIM_STOPWORDS = {
    "virtual",
    "reality",
    "vr",
    "can",
    "be",
    "for",
}

ESSENTIAL_TERM_BLACKLIST = {
    "experimental",
    "experiments",
    "experiment",
    "research",
    "studies",
    "study",
    "analysis",
    "model",
    "models",
    "prediction",
    "predictions",
    "method",
    "methods",
}

NEGATIVE_MARKERS = {
    "unsupported",
    "insufficient",
    "inadequate",
    "false",
    "failure",
    "failures",
    "harm",
    "harmful",
    "incorrect",
    "worse",
    "negative",
    "cannot",
    "can't",
    "none",
    "never",
}
