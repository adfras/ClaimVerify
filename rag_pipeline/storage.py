from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import Chunk, Document


def load_documents(path: Path) -> List[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            pages = payload.get("pages", [])
            raw_spans = payload.get("page_spans")
            if raw_spans:
                page_spans = [tuple(span) for span in raw_spans]
            elif pages:
                page_spans = []
                cursor = 0
                for idx, page in enumerate(pages):
                    start = cursor
                    cursor += len(page)
                    page_spans.append((start, cursor))
                    if idx != len(pages) - 1:
                        cursor += 2
            else:
                page_spans = []
            documents.append(
                Document(
                    doc_id=payload["doc_id"],
                    title=payload["title"],
                    path=Path(payload["path"]),
                    text=payload["text"],
                    pages=pages,
                    page_spans=page_spans,
                )
            )
    return documents


def load_chunks(path: Path) -> List[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            chunks.append(Chunk(**payload))
    return chunks


def chunk_texts(chunks: Iterable[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]
