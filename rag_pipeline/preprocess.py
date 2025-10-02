from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Iterable, Sequence
from xml.etree import ElementTree as ET

import spacy
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import PipelineConfig
from .models import Chunk, Document
from .ocr import build_ocr_command


NON_ALPHA = re.compile(r"[^a-z0-9]+")
MULTI_SPACE = re.compile(r"\s+")
BACK_MATTER_PATTERN = re.compile(
    r"^(references|bibliography|acknowledg(e)?ments|funding|appendix)\b",
    re.IGNORECASE,
)
CAPTION_PATTERN = re.compile(r"^(figure|fig\.|table)\s+\d", re.IGNORECASE)
DOI_PATTERN = re.compile(r"^doi[:\s]", re.IGNORECASE)
LICENSE_PATTERN = re.compile(
    r"^(creative\s+commons|©|copyright|all\s+rights\s+reserved)",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)

MINHASH_SEEDS: tuple[int, ...] = (
    0x1F3D5A1,
    0x2A1C9B3,
    0x3B7E0F5,
    0x45A2D17,
    0x59D4E2B,
    0x60FF1C3,
    0x77B3A49,
    0x88C5D6F,
    0x9AF7E81,
    0xA4C9135,
    0xB7E2F97,
    0xC8A1B2D,
    0xD9C3E4F,
    0xEAF5A61,
    0xF1B2C33,
    0x1234AB5,
)

_GROBID_AVAILABILITY: dict[str, bool] = {}

LOGGER = logging.getLogger(__name__)


def slugify(text: str) -> str:
    text = text.lower()
    text = NON_ALPHA.sub("-", text)
    text = text.strip("-")
    return text or "doc"


def load_spacy_model(name: str) -> spacy.language.Language:
    try:
        nlp = spacy.load(name)  # type: ignore[arg-type]
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    nlp.max_length = 3_000_000
    return nlp


def is_probably_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(1024)
    except OSError as exc:
        LOGGER.warning("Skipping %s: cannot read file header (%s)", path.name, exc)
        return False
    header = header.lstrip()
    return header.startswith(b"%PDF-")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pages_from_pdf(path: Path) -> list[str]:
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError) as exc:
        LOGGER.warning("Failed to read PDF %s: %s", path, exc)
        return []
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            pages.append(normalise_whitespace(text))
    return pages


def estimate_source_quality(doi: str | None, authors: Sequence[str]) -> float:
    score = 0.0
    if doi:
        score += 0.4
    if authors:
        score += 0.2
        if len(authors) >= 3:
            score += 0.1
    return min(score, 0.7)


def needs_ocr(pages: Sequence[str], min_empty_ratio: float) -> bool:
    if not pages:
        return True
    empty_or_short = sum(1 for page in pages if len(page.strip()) < 40)
    return empty_or_short / len(pages) >= min_empty_ratio


def ensure_text_layer(path: Path, config: PipelineConfig) -> Path:
    if not config.enable_ocr:
        return path
    try:
        pages = extract_pages_from_pdf(path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to probe PDF %s before OCR: %s", path, exc)
        pages = []
    if not needs_ocr(pages, config.ocr_min_empty_ratio):
        return path

    output_dir = config.work_dir / "ocr_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.ocr.pdf"
    if output_path.exists() and output_path.stat().st_mtime >= path.stat().st_mtime:
        return output_path

    command = build_ocr_command(config.ocr_command, path, output_path)
    if not command:
        return path
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if output_path.exists():
            return output_path
    except FileNotFoundError:
        LOGGER.warning(
            "OCR command %s not found; skipping OCR", command[0]
        )
    except subprocess.CalledProcessError as exc:
        LOGGER.warning("OCR command failed for %s: %s", path, exc)
    return path


def strip_headers_and_footers(path: Path, pages: list[str], config: PipelineConfig) -> list[str]:
    if not config.strip_headers_and_footers:
        return pages
    try:
        import fitz  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("PyMuPDF not installed; skipping header/footer stripping.")
        return pages

    try:
        doc = fitz.open(path)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to open %s with PyMuPDF: %s", path, exc)
        return pages

    page_blocks: list[list[tuple[str, str]]] = []
    header_counts: dict[str, int] = {}
    footer_counts: dict[str, int] = {}

    for page in doc:
        height = page.rect.height
        header_cutoff = height * 0.08
        footer_cutoff = height * 0.92
        blocks: list[tuple[str, str]] = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text, *_ = block
            cleaned = text.strip()
            if not cleaned:
                continue
            if y0 <= header_cutoff:
                header_counts[cleaned] = header_counts.get(cleaned, 0) + 1
                blocks.append(("header", cleaned))
            elif y1 >= footer_cutoff:
                footer_counts[cleaned] = footer_counts.get(cleaned, 0) + 1
                blocks.append(("footer", cleaned))
            else:
                blocks.append(("body", cleaned))
        page_blocks.append(blocks)

    doc.close()
    if not page_blocks:
        return pages

    min_repetition = max(3, int(0.3 * len(page_blocks)))
    common_headers = {text for text, count in header_counts.items() if count >= min_repetition}
    common_footers = {text for text, count in footer_counts.items() if count >= min_repetition}

    cleaned_pages: list[str] = []
    for idx, blocks in enumerate(page_blocks):
        filtered: list[str] = []
        for band, text in blocks:
            if band == "header" and text in common_headers:
                continue
            if band == "footer" and text in common_footers:
                continue
            filtered.append(text)
        if filtered:
            candidate = normalise_whitespace("\n".join(filtered))
            cleaned_pages.append(candidate)
        else:
            cleaned_pages.append(pages[idx] if idx < len(pages) else "")
    return cleaned_pages


def grobid_service_available(config: PipelineConfig) -> bool:
    if not config.auto_detect_grobid:
        return False
    base_url = config.grobid_url.rstrip("/")
    cached = _GROBID_AVAILABILITY.get(base_url)
    if cached is not None:
        return cached
    try:
        import requests  # type: ignore
    except ImportError:
        LOGGER.warning(
            "requests not installed; cannot auto-detect GROBID service availability."
        )
        _GROBID_AVAILABILITY[base_url] = False
        return False

    try:
        response = requests.get(
            base_url + "/api/isalive",
            timeout=2.0,
        )
        available = response.status_code == 200
    except Exception as exc:  # pragma: no cover - network failures
        LOGGER.debug("GROBID autodetect failed for %s: %s", base_url, exc)
        available = False

    _GROBID_AVAILABILITY[base_url] = available
    if available:
        LOGGER.info("Detected GROBID service at %s; enabling metadata extraction.", base_url)
    return available


def remove_back_matter(pages: list[str]) -> list[str]:
    cleaned: list[str] = []
    stop = False
    for page in pages:
        if stop:
            break
        lines = page.splitlines()
        trimmed: list[str] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                trimmed.append(line)
                continue
            if idx <= 6 and BACK_MATTER_PATTERN.match(stripped):
                stop = True
                break
            trimmed.append(line)
        if trimmed and not stop:
            cleaned.append(normalise_whitespace("\n".join(trimmed)))
        elif trimmed and stop:
            kept = [ln for ln in trimmed if not BACK_MATTER_PATTERN.match(ln.strip())]
            if kept:
                cleaned.append(normalise_whitespace("\n".join(kept)))
    return cleaned if cleaned else pages


def filter_page_noise(page: str) -> str:
    lines: list[str] = []
    for raw_line in page.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if CAPTION_PATTERN.match(stripped):
            continue
        if DOI_PATTERN.match(lower):
            continue
        if LICENSE_PATTERN.match(lower):
            continue
        if stripped.lower() in {"abstract", "keywords"}:
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _hash_shingle(seed: int, shingle: str) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update((seed & 0xFFFFFFFF).to_bytes(4, "little", signed=False))
    hasher.update(shingle.encode("utf-8", "ignore"))
    return int.from_bytes(hasher.digest(), "big")


def minhash_signature(text: str, shingle_size: int = 5) -> tuple[int, ...]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return ()
    if len(tokens) < shingle_size:
        shingles = [" ".join(tokens)]
    else:
        shingles = [
            " ".join(tokens[i : i + shingle_size])
            for i in range(len(tokens) - shingle_size + 1)
        ]
    if not shingles:
        return ()

    signature: list[int] = []
    for seed in MINHASH_SEEDS:
        min_value: int | None = None
        for shingle in shingles:
            value = _hash_shingle(seed, shingle)
            if min_value is None or value < min_value:
                min_value = value
        signature.append(min_value if min_value is not None else 0)
    return tuple(signature)


def is_duplicate_signature(
    signature: tuple[int, ...],
    seen: list[tuple[int, ...]],
    threshold: float = 0.85,
) -> bool:
    if not signature:
        return False
    for existing in seen:
        if not existing or len(existing) != len(signature):
            continue
        matches = sum(1 for a, b in zip(signature, existing) if a == b)
        if matches / len(signature) >= threshold:
            return True
    return False


def parse_grobid_header(xml_text: str) -> dict[str, object]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    title_el = root.find(".//tei:fileDesc/tei:titleStmt/tei:title", ns)
    title = title_el.text.strip() if title_el is not None and title_el.text else None
    authors: list[str] = []
    for pers in root.findall(".//tei:author/tei:persName", ns):
        forename = pers.findtext("tei:forename", default="", namespaces=ns) or ""
        surname = pers.findtext("tei:surname", default="", namespaces=ns) or ""
        full_name = " ".join(part for part in [forename.strip(), surname.strip()] if part)
        if full_name:
            authors.append(full_name)
    doi = root.findtext(".//tei:idno[@type='DOI']", default=None, namespaces=ns)
    doi = doi.strip() if doi else None
    return {"title": title, "authors": authors, "doi": doi}


def fetch_grobid_metadata(path: Path, config: PipelineConfig) -> dict[str, object]:
    use_grobid = config.enable_grobid or grobid_service_available(config)
    if not use_grobid:
        return {}
    try:
        import requests
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("requests not installed; skipping GROBID metadata extraction.")
        return {}

    endpoint = config.grobid_url.rstrip("/") + "/api/processHeaderDocument"
    try:
        with path.open("rb") as handle:
            response = requests.post(
                endpoint,
                params={"consolidateHeader": "1"},
                files={"input": (path.name, handle, "application/pdf")},
                timeout=config.grobid_timeout,
            )
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.warning("GROBID request failed for %s: %s", path, exc)
        return {}

    if response.status_code != 200:
        LOGGER.warning(
            "GROBID returned status %s for %s", response.status_code, path
        )
        return {}

    metadata = parse_grobid_header(response.text)
    if not metadata:
        LOGGER.warning("Failed to parse GROBID response for %s", path)
    return metadata


def normalise_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("\x0c", "\n")
    text = MULTI_SPACE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(config: PipelineConfig) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for path in sorted(config.corpus_dir.glob("*.pdf")):
        if path.name.endswith(":Zone.Identifier"):
            continue
        lower_name = path.name.lower()
        lower_stem = path.stem.lower()
        if config.skip_documents and (
            lower_name in config.skip_documents or lower_stem in config.skip_documents
        ):
            LOGGER.info("Skipping %s due to skip list entry", path.name)
            continue
        if not is_probably_pdf(path):
            LOGGER.warning("Skipping %s: file header does not look like a PDF", path.name)
            continue
        try:
            digest = file_sha256(path)
        except OSError as exc:
            LOGGER.warning("Skipping %s: failed to hash file (%s)", path.name, exc)
            continue
        if digest in seen_hashes:
            LOGGER.info("Skipping %s: duplicate content detected", path.name)
            continue
        seen_hashes.add(digest)
        text_path = ensure_text_layer(path, config)
        pages = extract_pages_from_pdf(text_path)
        if not pages:
            continue
        pages = strip_headers_and_footers(text_path, pages, config)
        pages = remove_back_matter(pages)
        processed_pages: list[str] = []
        for page in pages:
            cleaned = filter_page_noise(page)
            processed_pages.append(cleaned if cleaned else page)
        pages = [page for page in processed_pages if page.strip()]
        if not pages:
            continue
        text, page_spans = build_document_text(pages)
        metadata = fetch_grobid_metadata(path, config)
        title_candidate = metadata.get("title") if metadata else None
        title = title_candidate if isinstance(title_candidate, str) and title_candidate else path.stem.replace("-", " ")
        base_doc_id = slugify(title_candidate if isinstance(title_candidate, str) and title_candidate else path.stem)
        doc_id = base_doc_id or "doc"
        suffix = 1
        while doc_id in seen_ids:
            doc_id = f"{base_doc_id or 'doc'}-{suffix}"
            suffix += 1
        seen_ids.add(doc_id)
        authors = metadata.get("authors") if isinstance(metadata, dict) else []
        if not isinstance(authors, list):
            authors = []
        doi = metadata.get("doi") if isinstance(metadata, dict) else None
        source_quality = estimate_source_quality(doi if isinstance(doi, str) else None, authors)
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                path=path,
                text=text,
                pages=pages,
                page_spans=page_spans,
                authors=authors,
                doi=doi if isinstance(doi, str) else None,
                source_quality=source_quality,
            )
        )
    return documents


def paragraph_spans(text: str) -> Sequence[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        next_break = text.find("\n\n", start)
        if next_break == -1:
            spans.append((start, len(text)))
            break
        spans.append((start, next_break))
        start = next_break + 2
    return spans


def assign_paragraph_index(spans: Sequence[tuple[int, int]], char_start: int) -> int:
    for idx, (start, end) in enumerate(spans):
        if start <= char_start < end:
            return idx
    return len(spans) - 1 if spans else 0


def sentence_token_length(text: str) -> int:
    return max(1, len(text.split()))


def chunk_document(
    document: Document,
    nlp: spacy.language.Language,
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    spans = paragraph_spans(document.text)
    doc_obj = nlp(document.text)
    chunks: list[Chunk] = []
    sentences: list[dict[str, int | str]] = []
    chunk_index = 0
    token_budget = 0
    dedupe_signatures: list[tuple[int, ...]] = []

    def emit_chunk(buffer: list[dict[str, int | str]]) -> None:
        nonlocal chunk_index
        if not buffer:
            return
        chunk_text = " ".join(str(s["text"]) for s in buffer)
        signature = minhash_signature(chunk_text)
        if signature and is_duplicate_signature(signature, dedupe_signatures):
            return
        start = int(buffer[0]["start"])
        end = int(buffer[-1]["end"])
        paragraph_idx = assign_paragraph_index(spans, start)
        page_start, page_start_offset = locate_page(document.page_spans, start)
        page_end, page_end_offset = locate_page(document.page_spans, max(end - 1, start))
        chunk_id = f"{document.doc_id}::chunk-{chunk_index:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                title=document.title,
                text=chunk_text,
                char_start=start,
                char_end=end,
                paragraph_index=paragraph_idx,
                page_start=page_start,
                page_end=page_end,
                page_start_offset=page_start_offset,
                page_end_offset=page_end_offset,
            )
        )
        if signature:
            dedupe_signatures.append(signature)
        chunk_index += 1

    for sent in doc_obj.sents:  # type: ignore[attr-defined]
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue
        token_len = sentence_token_length(sentence_text)
        record = {
            "text": sentence_text,
            "start": sent.start_char,
            "end": sent.end_char,
            "tokens": token_len,
        }
        if token_budget + token_len > target_tokens and sentences:
            emit_chunk(sentences)
            if overlap_tokens > 0:
                tail: list[dict[str, int | str]] = []
                count = 0
                for s in reversed(sentences):
                    tail.append(s)
                    count += int(s["tokens"])
                    if count >= overlap_tokens:
                        break
                sentences = list(reversed(tail))
                token_budget = sum(int(s["tokens"]) for s in sentences)
            else:
                sentences = []
                token_budget = 0
        sentences.append(record)
        token_budget += token_len

    if sentences:
        emit_chunk(sentences)
    return chunks


def persist_documents(documents: Iterable[Document], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_payload(), ensure_ascii=True) + "\n")


def persist_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_payload(), ensure_ascii=True) + "\n")


def build_document_text(pages: list[str]) -> tuple[str, list[tuple[int, int]]]:
    if not pages:
        return "", []
    spans: list[tuple[int, int]] = []
    cursor = 0
    buffer: list[str] = []
    for idx, page in enumerate(pages):
        start = cursor
        cursor += len(page)
        spans.append((start, cursor))
        buffer.append(page)
        if idx != len(pages) - 1:
            cursor += 2  # account for "\n\n" separator
    text = "\n\n".join(buffer)
    return text, spans


def locate_page(spans: list[tuple[int, int]], char_pos: int) -> tuple[int, int]:
    if not spans:
        return 1, max(char_pos, 0)
    adjusted = max(char_pos, 0)
    for index, (start, end) in enumerate(spans):
        if adjusted < end:
            return index + 1, max(adjusted - start, 0)
    last_index = len(spans) - 1
    start, end = spans[last_index]
    return last_index + 1, min(adjusted - start, end - start)
