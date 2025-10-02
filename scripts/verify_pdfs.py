from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass
class PdfCheckResult:
    path: str
    size_bytes: int
    sha256: str
    pages: int
    characters: int
    average_chars_per_page: float
    needs_ocr: bool
    metadata_title_mismatch: bool
    warnings: list[str]


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.add("".join(current))
                current.clear()
    if current:
        tokens.add("".join(current))
    return {token for token in tokens if len(token) > 2}


def probe_pdf(path: Path) -> PdfCheckResult:
    warnings: list[str] = []
    size_bytes = path.stat().st_size
    sha256 = compute_sha256(path)
    characters = 0
    pages = 0
    average_chars = 0.0
    needs_ocr = False
    metadata_title_mismatch = False

    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        if pages == 0:
            warnings.append("no-pages")
        texts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = normalise_whitespace(text)
            texts.append(text)
            characters += len(text)
        if pages > 0:
            average_chars = characters / pages
        empty_or_short = [idx for idx, text in enumerate(texts) if len(text) < 40]
        if pages > 0 and len(empty_or_short) / pages >= 0.5:
            needs_ocr = True
        meta = reader.metadata or {}
        title = (meta.get("/Title") or "").strip()
        if title:
            title_tokens = tokenize(title)
            stem_tokens = tokenize(path.stem)
            if title_tokens:
                overlap = title_tokens & stem_tokens
                if len(overlap) / len(title_tokens) < 0.3:
                    metadata_title_mismatch = True
        if reader.is_encrypted:
            warnings.append("encrypted")
    except (PdfReadError, KeyError, ValueError, OSError) as exc:
        warnings.append(f"read-error:{exc}")

    return PdfCheckResult(
        path=str(path),
        size_bytes=size_bytes,
        sha256=sha256,
        pages=pages,
        characters=characters,
        average_chars_per_page=average_chars,
        needs_ocr=needs_ocr,
        metadata_title_mismatch=metadata_title_mismatch,
        warnings=warnings,
    )


def load_skip_list(skip_arguments: list[str], skip_file: Path | None) -> set[str]:
    skip: set[str] = set()
    for item in skip_arguments:
        cleaned = item.strip().lower()
        if cleaned:
            skip.add(cleaned)
    if skip_file and skip_file.exists():
        for line in skip_file.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip().lower()
            if cleaned and not cleaned.startswith("#"):
                skip.add(cleaned)
    return skip


def enumerate_pdfs(pdf_dir: Path, skip: set[str]) -> list[Path]:
    pdfs = []
    for entry in pdf_dir.iterdir():
        if entry.name.endswith(":Zone.Identifier"):
            continue
        if not entry.is_file() or entry.suffix.lower() != ".pdf":
            continue
        name = entry.name.lower()
        stem = entry.stem.lower()
        if name in skip or stem in skip:
            continue
        pdfs.append(entry)
    return sorted(pdfs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify PDF corpus integrity.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("full_text"),
        help="Directory containing source PDF files.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Filename or stem of a PDF to skip; may be supplied multiple times.",
    )
    parser.add_argument(
        "--skip-file",
        type=Path,
        help="Optional newline-delimited list of filenames/stems to skip.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Automatically move duplicate PDFs (after the first copy) into a _duplicates folder.",
    )
    return parser


def move_duplicates(duplicate_groups: list[list[str]], pdf_dir: Path) -> list[tuple[str, str]]:
    dedupe_dir = pdf_dir / "_duplicates"
    dedupe_dir.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[str, str]] = []
    for group in duplicate_groups:
        keep = group[0]
        for duplicate_path in group[1:]:
            source = Path(duplicate_path)
            target = dedupe_dir / source.name
            counter = 1
            while target.exists():
                target = dedupe_dir / f"{source.stem}-{counter}{source.suffix}"
                counter += 1
            shutil.move(str(source), target)
            moved.append((str(source), str(target)))
    return moved


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    pdf_dir: Path = args.pdf_dir
    if not pdf_dir.exists():
        parser.error(f"PDF directory {pdf_dir} does not exist")
    skip_file = args.skip_file
    if skip_file is None:
        candidate = Path("data/pdf_skip.txt")
        if candidate.exists():
            skip_file = candidate
    skip_entries = load_skip_list(args.skip, skip_file)
    if skip_entries:
        print(f"Skipping {len(skip_entries)} entries based on provided skip list.")
    pdfs = enumerate_pdfs(pdf_dir, skip_entries)
    if not pdfs:
        parser.error(f"No PDF files found in {pdf_dir}")

    results = [probe_pdf(pdf) for pdf in pdfs]
    duplicates: dict[str, list[str]] = defaultdict(list)
    for res in results:
        duplicates[res.sha256].append(res.path)

    total_pages = sum(res.pages for res in results)
    total_characters = sum(res.characters for res in results)
    flagged_for_ocr = [res.path for res in results if res.needs_ocr]
    unreadable = [res for res in results if any(w.startswith("read-error") for w in res.warnings)]

    print(f"Scanned {len(results)} PDFs: {total_pages} pages, {total_characters} characters extracted.")
    if flagged_for_ocr:
        print("Potential OCR candidates (>=50% short/empty pages):")
        for path in flagged_for_ocr:
            print(f"  - {path}")
    if unreadable:
        print("Unreadable PDFs:")
        for res in unreadable:
            print(f"  - {res.path}: {', '.join(res.warnings)}")
    duplicate_groups = [paths for paths in duplicates.values() if len(paths) > 1]
    if duplicate_groups:
        print("Duplicate content detected (matching SHA-256):")
        for group in duplicate_groups:
            print("  - " + ", ".join(group))
        if args.dedupe:
            moved = move_duplicates(duplicate_groups, pdf_dir)
            if moved:
                print("Moved duplicate files:")
                for src, dst in moved:
                    print(f"  - {src} -> {dst}")

    title_mismatches = [res.path for res in results if res.metadata_title_mismatch]
    if title_mismatches:
        print("Metadata title mismatch (title not found in filename):")
        for path in title_mismatches:
            print(f"  - {path}")

    if args.json:
        payload: dict[str, Any] = {
            "stats": {
                "pdf_count": len(results),
                "total_pages": total_pages,
                "total_characters": total_characters,
                "ocr_candidates": flagged_for_ocr,
                "duplicates": duplicate_groups,
                "title_mismatches": title_mismatches,
            },
            "documents": [asdict(res) for res in results],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote detailed report to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
