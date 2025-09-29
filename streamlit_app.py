from __future__ import annotations

import streamlit as st

from rag_pipeline.config import PipelineConfig
from rag_pipeline.pipeline import HybridRetriever


@st.cache_resource
def load_retriever() -> HybridRetriever:
    config = PipelineConfig()
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


st.set_page_config(page_title="VR Claim Verifier", layout="wide")
st.title("Claim Evidence Finder")

st.markdown(
    """
Upload PDF files into the `full_text/` folder, run `python scripts/build_index.py`, and use this
interface to check whether your claim is supported by the collection.
    """
)

claim_input = st.text_area("Claim", height=140, placeholder="Enter the claim you want to verify…")
max_refs = st.slider("Maximum supporting references", min_value=1, max_value=10, value=3)
show_contradictions = st.checkbox("Show potential contradictions", value=False)

if st.button("Find Evidence"):
    claim = claim_input.strip()
    if not claim:
        st.warning("Please enter a claim before running the search.")
    else:
        with st.spinner("Retrieving evidence…"):
            retriever = load_retriever()
            response = retriever.retrieve(claim)

        supports = response.get("supports", [])[:max_refs]
        contradictions = response.get("contradictions", [])

        st.subheader(f"Verdict: {response.get('verdict', 'unknown').capitalize()}")

        if supports:
            st.markdown("### Supporting Evidence")
            for idx, item in enumerate(supports, start=1):
                st.markdown(f"**{idx}. {item['title']}**")
                st.markdown(
                    f"- {format_page_range(item)} ({format_offsets(item)})\n"
                    f"- Quote: “{item['quote']}”"
                )
        else:
            st.info("No supporting passages cleared the thresholds. Consider rephrasing the claim or expanding the corpus.")

        if show_contradictions and contradictions:
            st.markdown("### Possible Contradictions")
            for idx, item in enumerate(contradictions, start=1):
                st.markdown(f"**{idx}. {item['title']}**")
                st.markdown(
                    f"- {format_page_range(item)} ({format_offsets(item)})\n"
                    f"- Quote: “{item['quote']}”"
                )

        st.markdown("---")
        st.caption(
            "To ingest new documents, add PDFs to `full_text/` and run `python scripts/build_index.py` before rerunning this application."
        )
