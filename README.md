# VR Claim Evidence Finder

A lightweight retrieval-and-verification pipeline that indexes a directory of PDF articles, fuses BM25 and dense embeddings, reranks with a cross-encoder, and filters the final passages with NLI to surface only evidence that actually *supports* or *contradicts* a user-supplied claim.

The project includes both a CLI workflow and a Streamlit app so you can load a folder of reports, type a claim into a text box, choose how many references to return, and see the article titles, page numbers, and offsets for manual verification.

## Features
- **Hybrid retrieval**: BM25 + FAISS dense search combined via Reciprocal Rank Fusion.
- **Reranking**: `BAAI/bge-reranker-base` cross-encoder for high-precision ordering.
- **Support filtering**: `MoritzLaurer/DeBERTa-v3-base-mnli` NLI model with sentence-level fallback and lexical checks to reduce false negatives.
- **Page metadata**: Every chunk stores page range and on-page offsets so citations point to the exact location in the PDF.
- **Validation suites**: Two regression scripts to sanity-check supported/contradicted/insufficient verdicts.
- **Streamlit UI**: Claim textbox, slider for maximum references, optional contradiction view.

## Prerequisites
- Python 3.10+
- CPU is sufficient; GPU accelerates reranking but is not required.
- `pip` for dependency management.
- Optional for scanned PDFs: install `ocrmypdf` or provide Docker access. The
  pipeline now auto-detects an available OCR backend and falls back gracefully
  when none can run.

## Installation
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Preparing the corpus
1. Place all PDF articles inside `full_text/` (nested folders are optional but supported if you adjust `PipelineConfig.corpus_dir`).
2. Build or rebuild the indexes whenever the corpus changes:
   ```bash
   . .venv/bin/activate
   PYTHONPATH=. python scripts/build_index.py --corpus-dir full_text --work-dir data
   ```
   This extracts text, creates overlapping chunks, writes JSONL artifacts under `data/artifacts/`, and builds a FAISS index under `data/indexes/`.

### Comparing chunk sizes
Recent evaluations suggest testing multiple chunk granularities instead of relying on a single token target: NVIDIA found that medium-sized chunks (roughly 512–1,024 tokens) or page-level splits often outperform both very small and very large chunks across diverse datasets, and the Weaviate team recommends starting around 512 tokens with 10–20 % overlap before iterating ([NVIDIA, 2025](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/); [Weaviate, 2025](https://weaviate.io/blog/chunking-strategies-for-rag)). To make comparisons easy, you can build multiple index variants in one shot:

```bash
. .venv/bin/activate
PYTHONPATH=. python scripts/build_compare_indexes.py \
  --corpus-dir full_text \
  --work-dir-root data \
  --chunk-sizes 220 420 \
  --overlap 50
```

Each variant is stored under `data/chunk_<size>/`. Point `--work-dir` to the desired directory when running queries or evaluations.

Run the regression suites against any work directory to benchmark verdict accuracy:

```bash
PYTHONPATH=. python scripts/quick_eval.py --work-dir data/chunk_220
PYTHONPATH=. python scripts/validation_set_vr2.py --work-dir data/chunk_420

# Or run both suites for multiple variants at once
PYTHONPATH=. python scripts/compare_regressions.py --work-dirs data/chunk_220 data/chunk_420
```

Use these reports to decide which chunking configuration yields cleaner supports or higher validation accuracy before promoting an index to production.

## Command-line usage
Run a single claim verification from the CLI:
```bash
PYTHONPATH=. python scripts/query_claim.py "Your factual claim" --top-supports 3
```
Output includes the verdict, supporting passages (if any), contradictions (optional), reranker/NLI scores, and precise page metadata.

## Streamlit web application
Launch the interactive interface:
```bash
streamlit run streamlit_app.py
```
1. Enter the claim in the text box.
2. Adjust the slider for the maximum number of references to display.
3. (Optional) Enable “Show potential contradictions.”
4. Click **Find Evidence** to view supporting quotes with article titles, page ranges, and offsets.

Keep the Streamlit session running while you inspect the cited PDF pages manually.

## Validation
Two quick regression suites ensure retrieval + NLI filtering behaves as expected:
```bash
PYTHONPATH=. python scripts/quick_eval.py          # 4-claim smoke test
PYTHONPATH=. python scripts/validation_set_vr2.py  # 6-claim VR/body-ownership set
```
Both should report accuracy 1.0 after a successful index build.

## Configuration
Default parameters live in `rag_pipeline/config.py`. You can tweak:
- Chunk size/overlap (`chunk_target_tokens`, `chunk_overlap_tokens`).
- Retrieval depths (`bm25_top_k`, `dense_top_k`, `rerank_keep_top_k`).
- NLI thresholds and lexical fallbacks.
- Model choices (swap to multilingual models if needed).
- OCR backend: set `PipelineConfig.ocr_command` to `"auto"` (default) to let the
  pipeline prefer a local `ocrmypdf` install, fall back to the included Docker
  helper when Docker is available, or effectively disable OCR when neither is
  usable. Set it to `"none"` to opt out explicitly.

After modifying the config, rebuild indexes and rerun validations.

## Updating the corpus
1. Add/remove PDFs in `full_text/`.
2. Rebuild the indexes (`build_index.py`).
3. Restart the Streamlit app (if running) so it picks up the new artifacts.

## Repository hygiene
- `.gitignore` excludes the virtual environment and generated indexes/embeddings—only source files and configuration go into version control.
- No API keys or external secrets are required; all models are pulled from Hugging Face at runtime.

## Troubleshooting
- **“Module not found”**: Ensure `PYTHONPATH=.` is set when running scripts, or install the package (`pip install -e .`) if you package it later.
- **“Claim returns insufficient”**: Try rephrasing the claim or lowering thresholds in `PipelineConfig` to capture looser paraphrases.
- **Slow queries**: Reduce `bm25_top_k`/`dense_top_k`, or create a GPU-enabled environment for the reranker.

Happy verifying!
