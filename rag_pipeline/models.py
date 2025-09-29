from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Document:
    doc_id: str
    title: str
    path: Path
    text: str
    pages: list[str]
    page_spans: list[tuple[int, int]]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["page_spans"] = [list(span) for span in self.page_spans]
        return payload


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    char_start: int
    char_end: int
    paragraph_index: int
    page_start: int
    page_end: int
    page_start_offset: int
    page_end_offset: int

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def iter_chunks_by_doc(chunks: Iterable[Chunk]):
    current_doc = None
    bucket: list[Chunk] = []
    for chunk in chunks:
        if current_doc is None:
            current_doc = chunk.doc_id
        if chunk.doc_id != current_doc:
            yield current_doc, bucket
            bucket = [chunk]
            current_doc = chunk.doc_id
        else:
            bucket.append(chunk)
    if current_doc is not None and bucket:
        yield current_doc, bucket
