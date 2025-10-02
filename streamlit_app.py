from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from rag_pipeline.config import PipelineConfig
from rag_pipeline.pipeline import HybridRetriever
from rag_pipeline.llm import LLMRechecker

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None


DEFAULT_CONFIG = PipelineConfig()


@st.cache_resource
def load_retriever(
    work_dir: str,
    device: str,
    strip_references: bool,
    use_dense: bool,
    use_llm: bool,
    _llm_cache_token: str,
) -> HybridRetriever:
    config = PipelineConfig(
        work_dir=Path(work_dir),
        inference_device=device,
        strip_reference_lines=strip_references,
        use_dense_retrieval=use_dense,
        use_llm_rechecker=use_llm,
    )
    return HybridRetriever(config)


def format_page_range(entry: dict[str, int]) -> str:
    page_start = entry.get("page_start")
    page_end = entry.get("page_end")
    if page_start is None or page_end is None:
        return "Unknown page"
    if page_start == page_end:
        return f"Page {page_start}"
    return f"Pages {page_start}–{page_end}"


def format_offsets(entry: dict[str, int]) -> str:
    start = entry.get("page_start_offset")
    end = entry.get("page_end_offset")
    if start is None or end is None:
        return "Offsets unavailable"
    if entry.get("page_start") == entry.get("page_end"):
        return f"Offsets {start + 1}–{end + 1}"
    return f"Start offset {start + 1}, end offset {end + 1}"


def build_pdf_link(entry: dict[str, Any]) -> str | None:
    path_value = entry.get("path")
    if not path_value:
        return None
    try:
        pdf_path = Path(path_value).resolve()
    except OSError:
        return None
    if not pdf_path.exists():
        return None
    page = entry.get("page_start") or 1
    try:
        page_number = max(int(page), 1)
    except (TypeError, ValueError):
        page_number = 1
    try:
        uri = pdf_path.as_uri()
    except ValueError:
        return None
    return f"{uri}#page={page_number}"


def render_entry(item: dict[str, Any], idx: int, *, kind: str) -> None:
    st.markdown(f"**{idx}. {item['title']}**")
    authors = item.get("authors") or []
    doi = item.get("doi")
    meta_parts: list[str] = []
    if isinstance(authors, list) and authors:
        meta_parts.append(", ".join(authors))
    if isinstance(doi, str) and doi:
        meta_parts.append(f"DOI: {doi}")
    if meta_parts:
        st.caption(" | ".join(meta_parts))

    quote_text = item.get("quote") or ""
    quote_chars = len(quote_text)
    lines = [
        f"- {format_page_range(item)} ({format_offsets(item)})",
        f"- Quote ({quote_chars} chars): “{quote_text}”",
    ]

    score_fields: list[tuple[str, str]]
    if kind == "support":
        score_fields = [
            ("confidence", "confidence"),
            ("entailment", "entailment_probability"),
            ("rerank", "rerank_score"),
        ]
    else:
        score_fields = [
            ("confidence", "confidence"),
            ("contradiction", "contradiction_probability"),
            ("rerank", "rerank_score"),
        ]

    score_parts: list[str] = []
    for label, key in score_fields:
        value = item.get(key)
        if value is None:
            continue
        try:
            score_parts.append(f"{label} {float(value):.2f}")
        except (TypeError, ValueError):
            continue
    if score_parts:
        lines.append(f"- Scores: {', '.join(score_parts)}")

    llm_note = item.get("llm_note")
    llm_verdict = item.get("llm_verdict")
    if llm_verdict is not None:
        verdict_text = "YES" if llm_verdict else "NO"
        lines.append(f"- LLM verdict: {verdict_text}")
    elif llm_note:
        lines.append("- LLM verdict: UNKNOWN")
    if llm_note:
        lines.append(f"- LLM note: {llm_note}")

    st.markdown("\n".join(lines))
    pdf_link = build_pdf_link(item)
    if pdf_link:
        st.markdown(f"[View PDF]({pdf_link})")


st.set_page_config(page_title="VR Claim Verifier", layout="wide")
st.title("Claim Evidence Finder")

st.markdown(
    """
Upload PDF files into the `full_text/` folder, run `python scripts/build_index.py`, and use this
interface to check whether your claim is supported by the collection.
    """
)

sidebar_header = st.sidebar.header("Settings")
work_dir_input = st.sidebar.text_input("Work directory", value="data")
device_options = ["auto", "cuda", "cpu"]
default_device_index = 1 if (torch is not None and torch.cuda.is_available()) else 0
device_choice = st.sidebar.selectbox(
    "Inference device", device_options, index=default_device_index
)
strip_refs = st.sidebar.checkbox("Strip bibliographic lines", value=True)
use_dense = st.sidebar.checkbox("Use dense retrieval", value=True)
bm25_top_k = st.sidebar.slider(
    "BM25 top-k",
    min_value=10,
    max_value=200,
    value=DEFAULT_CONFIG.bm25_top_k,
    step=5,
    help=(
        "Keyword retrieval depth. Higher = fetch more passages for recall (slower, noisier); lower = faster, more precise."
    ),
)
dense_top_k = st.sidebar.slider(
    "Dense top-k",
    min_value=10,
    max_value=200,
    value=DEFAULT_CONFIG.dense_top_k,
    step=5,
    disabled=not use_dense,
    help=(
        "Embedding retrieval depth. Higher = more semantic candidates (higher cost, more noise); lower = tighter focus."
    ),
)
rerank_keep_top_k = st.sidebar.slider(
    "Rerank keep top-k",
    min_value=5,
    max_value=50,
    value=DEFAULT_CONFIG.rerank_keep_top_k,
    step=5,
    help=(
        "Passages kept after reranking. Higher = give NLI/LLM more to check (slower); lower = only top-scoring passages."
    ),
)
quote_min_chars, quote_max_chars = st.sidebar.slider(
    "Quote length (characters)",
    min_value=120,
    max_value=1200,
    value=(DEFAULT_CONFIG.quote_min_chars, DEFAULT_CONFIG.quote_max_chars),
    step=20,
    help=(
        "Minimum and maximum characters included in each evidence quote. Widen to give downstream appraisers more context."
    ),
)
nli_threshold = st.sidebar.slider(
    "NLI support threshold",
    min_value=0.5,
    max_value=0.99,
    value=float(DEFAULT_CONFIG.nli_threshold),
    step=0.01,
    help=(
        "Minimum entailment score for support. Higher = stricter (fewer supports); lower = more permissive."
    ),
)
contradiction_threshold = st.sidebar.slider(
    "Contradiction threshold",
    min_value=0.5,
    max_value=0.99,
    value=float(DEFAULT_CONFIG.contradiction_threshold),
    step=0.01,
    help=(
        "Contradiction score needed to flag evidence. Higher = only very strong contradictions; lower = more flags."
    ),
)
high_conf_support = st.sidebar.slider(
    "High-confidence support override",
    min_value=0.6,
    max_value=1.0,
    value=float(DEFAULT_CONFIG.high_confidence_support_override),
    step=0.01,
    help=(
        "Sentence-level entailment override. Higher = require extremely strong sentences; lower = loosen lexical gates sooner."
    ),
)
essential_min_length = st.sidebar.slider(
    "Essential term min length",
    min_value=4,
    max_value=10,
    value=DEFAULT_CONFIG.essential_term_min_length,
    step=1,
    help=(
        "Minimum token length to count as an essential claim term. Higher = focus on longer words; lower = include short terms."
    ),
)
essential_overlap = st.sidebar.slider(
    "Essential term overlap",
    min_value=1,
    max_value=6,
    value=DEFAULT_CONFIG.essential_term_overlap,
    step=1,
    help=(
        "How many essential terms must appear for support. Higher = stricter lexical alignment; lower = easier acceptance."
    ),
)
lexical_overlap = st.sidebar.slider(
    "Lexical support overlap",
    min_value=0.0,
    max_value=1.0,
    value=float(DEFAULT_CONFIG.lexical_support_overlap),
    step=0.05,
    help=(
        "Minimum fraction of claim terms that must appear in an evidence sentence before lexical overrides apply."
    ),
)
require_modal = st.sidebar.checkbox(
    "Require modal alignment",
    value=DEFAULT_CONFIG.require_modal_alignment,
    help=(
        "Keep universal quantifiers (always/never/all) aligned between claim and evidence. Disable for broader matches."
    ),
)
llm_enabled = st.sidebar.checkbox(
    "Use LLM rechecker",
    value=DEFAULT_CONFIG.use_llm_rechecker,
    help=(
        "Toggle GPT verification. On = ask the LLM for verdicts (slower, more cost); off = rely on retrieval scores only."
    ),
)
llm_model_input = st.sidebar.text_input(
    "LLM model",
    value=DEFAULT_CONFIG.llm_model,
    disabled=not llm_enabled,
    help=(
        "Model identifier at your OpenAI-compatible endpoint. Point to a cheaper or faster model if desired."
    ),
)
llm_temperature_input = st.sidebar.slider(
    "LLM temperature",
    min_value=0.0,
    max_value=1.0,
    value=float(DEFAULT_CONFIG.llm_temperature),
    step=0.05,
    disabled=not llm_enabled,
    help=(
        "Sampling randomness. Higher = more varied reasoning (less stable); lower = deterministic responses."
    ),
)
llm_max_supports_input = st.sidebar.number_input(
    "LLM max supports",
    min_value=1,
    max_value=10,
    value=DEFAULT_CONFIG.llm_max_supports,
    step=1,
    disabled=not llm_enabled,
    help=(
        "Number of passages the LLM double-checks. Higher = more verdicts (higher cost); lower = faster/cheaper."
    ),
)
llm_base_url_input = st.sidebar.text_input(
    "LLM base URL override",
    value="",
    disabled=not llm_enabled,
    help="Optional per-run base URL override; blank uses existing environment settings.",
)
llm_context_chars = st.sidebar.slider(
    "LLM context clip (characters)",
    min_value=400,
    max_value=2000,
    value=DEFAULT_CONFIG.llm_context_max_chars,
    step=50,
    disabled=not llm_enabled,
    help=(
        "Truncate supporting passages before sending them to the LLM rechecker. Increase to share more context (costlier prompts)."
    ),
)
if not llm_enabled:
    llm_context_chars = DEFAULT_CONFIG.llm_context_max_chars

claim_input = st.text_area("Claim", height=140, placeholder="Enter the claim you want to verify…")
max_refs = st.slider("Maximum supporting references", min_value=1, max_value=10, value=3)
show_contradictions = st.checkbox("Show potential contradictions", value=False)

if st.button("Find Evidence"):
    claim = claim_input.strip()
    if not claim:
        st.warning("Please enter a claim before running the search.")
    else:
        with st.spinner("Retrieving evidence…"):
            retriever = load_retriever(
                work_dir_input,
                device_choice,
                strip_refs,
                use_dense,
                llm_enabled,
                ("enabled" if llm_enabled else "disabled") + llm_base_url_input,
            )
            cfg = retriever.config
            cfg.use_dense_retrieval = use_dense
            cfg.bm25_top_k = bm25_top_k
            cfg.dense_top_k = dense_top_k if use_dense else 0
            cfg.fusion_k = max(cfg.bm25_top_k, cfg.dense_top_k or cfg.bm25_top_k)
            cfg.rerank_keep_top_k = rerank_keep_top_k
            cfg.nli_threshold = nli_threshold
            cfg.contradiction_threshold = contradiction_threshold
            cfg.high_confidence_support_override = high_conf_support
            cfg.strip_reference_lines = strip_refs
            cfg.essential_term_min_length = essential_min_length
            cfg.essential_term_overlap = essential_overlap
            cfg.lexical_support_overlap = lexical_overlap
            cfg.require_modal_alignment = require_modal
            cfg.quote_min_chars = int(quote_min_chars)
            cfg.quote_max_chars = int(quote_max_chars)
            cfg.llm_context_max_chars = int(llm_context_chars)

            if llm_enabled:
                cfg.use_llm_rechecker = True
                llm_model_param = llm_model_input.strip() or cfg.llm_model
                cfg.llm_model = llm_model_param
                cfg.llm_temperature = llm_temperature_input
                cfg.llm_max_supports = int(llm_max_supports_input)
                cfg.llm_api_base = llm_base_url_input.strip() or cfg.llm_api_base
                meta = (
                    cfg.llm_model,
                    cfg.llm_temperature,
                    cfg.llm_api_base,
                    cfg.llm_max_supports,
                )
                current_meta = getattr(retriever, "_llm_meta", None)
                if (
                    not getattr(retriever, "_llm_rechecker", None)
                    or current_meta != meta
                ):
                    retriever._llm_rechecker = LLMRechecker(
                        cfg.llm_model,
                        cfg.llm_temperature,
                        base_url=cfg.llm_api_base,
                    )
                    retriever._llm_meta = meta
            else:
                cfg.use_llm_rechecker = False
                retriever._llm_rechecker = None
                retriever._llm_meta = None

            response = retriever.retrieve(claim)

        supports = response.get("supports", [])[:max_refs]
        contradictions = response.get("contradictions", [])
        config_preview = response.get("config", {})
        llm_rechecker = getattr(retriever, "_llm_rechecker", None)
        llm_available = bool(llm_rechecker and llm_rechecker.available)
        llm_status_reason = getattr(llm_rechecker, "status_reason", None)
        st.caption(
            f"Device: {retriever._device.upper()} | "
            f"NLI threshold {cfg.nli_threshold} | "
            f"Contradiction threshold {cfg.contradiction_threshold} | "
            f"High-confidence override {cfg.high_confidence_support_override} | "
            f"Essential term length {cfg.essential_term_min_length} | "
            f"Essential overlap {cfg.essential_term_overlap} | "
            f"Lexical overlap {cfg.lexical_support_overlap:.2f} | "
            f"Modal alignment: {'on' if cfg.require_modal_alignment else 'off'} | "
            f"BM25 top-k {cfg.bm25_top_k} | "
            f"Dense top-k {cfg.dense_top_k} | Rerank keep {cfg.rerank_keep_top_k} | "
            f"Quote chars {cfg.quote_min_chars}-{cfg.quote_max_chars} | "
            f"LLM context clip {cfg.llm_context_max_chars} | "
            f"Strip references: {'on' if cfg.strip_reference_lines else 'off'} | "
            f"LLM rechecker: {'active' if llm_available else 'off'}"
        )
        if llm_enabled and not llm_available:
            status_note = (
                llm_status_reason
                or "Set OPENAI_API_KEY (and optional base URL) before enabling this toggle."
            )
            st.info(f"LLM rechecker is disabled: {status_note}")

        st.subheader(f"Verdict: {response.get('verdict', 'unknown').capitalize()}")

        if supports:
            st.markdown("### Supporting Evidence")
            for idx, item in enumerate(supports, start=1):
                render_entry(item, idx, kind="support")
        else:
            st.info("No supporting passages cleared the thresholds. Consider rephrasing the claim or expanding the corpus.")

        if show_contradictions and contradictions:
            st.markdown("### Possible Contradictions")
            for idx, item in enumerate(contradictions, start=1):
                render_entry(item, idx, kind="contradiction")

        st.markdown("---")
        st.caption(
            "To ingest new documents, add PDFs to `full_text/` and run `python scripts/build_index.py` before rerunning this application."
        )
