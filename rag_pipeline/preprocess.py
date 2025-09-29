from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import spacy
from pypdf import PdfReader

from .config import PipelineConfig
from .models import Chunk, Document


NON_ALPHA = re.compile(r"[^a-z0-9]+")
MULTI_SPACE = re.compile(r"\s+")


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


def extract_pages_from_pdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            pages.append(normalise_whitespace(text))
    return pages


def normalise_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("\x0c", "\n")
    text = MULTI_SPACE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(config: PipelineConfig) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(config.corpus_dir.glob("*.pdf")):
        if path.name.endswith(":Zone.Identifier"):
            continue
        pages = extract_pages_from_pdf(path)
        if not pages:
            continue
        text, page_spans = build_document_text(pages)
        title = path.stem.replace("-", " ")
        doc_id = slugify(path.stem)
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                path=path,
                text=text,
                pages=pages,
                page_spans=page_spans,
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
            chunk_text = " ".join(s["text"] for s in sentences)
            start = int(sentences[0]["start"])
            end = int(sentences[-1]["end"])
            paragraph_idx = assign_paragraph_index(spans, start)
            page_start, page_start_offset = locate_page(document.page_spans, start)
            page_end, page_end_offset = locate_page(
                document.page_spans, max(end - 1, start)
            )
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
            chunk_index += 1
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
        chunk_text = " ".join(s["text"] for s in sentences)
        start = int(sentences[0]["start"])
        end = int(sentences[-1]["end"])
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
