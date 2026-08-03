#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.io_utils import read_yaml, write_jsonl
from card_routed_rag.text_utils import normalize_space, chunk_words


def parse_pdf(reader: PdfReader) -> list[tuple[int | None, str]]:
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = normalize_space(page.extract_text() or "")
        if text:
            pages.append((i, text))
    return pages


def source_id_from_path(path: Path) -> str:
    """Build a deterministic source ID without relying on an external manifest."""
    return re.sub(r"[^A-Z0-9]+", "_", path.stem.upper()).strip("_")


def year_from_path(path: Path) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", path.stem)
    return int(match.group(0)) if match else None


def title_from_pdf(path: Path, reader: PdfReader) -> str:
    metadata_title = str(getattr(reader.metadata, "title", "") or "").strip()
    if metadata_title:
        return normalize_space(metadata_title)
    return normalize_space(path.stem.replace("_", " ").replace("-", " "))


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse local PDF files directly into chunk JSONL.")
    ap.add_argument("--config", default="configs/extraction_config.yaml")
    ap.add_argument("--source-dir", default="data/sources/pdf")
    ap.add_argument("--output", default="data/processed/document_chunks.jsonl")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    chunk_size = int(cfg.get("chunk_size_words", 350))
    overlap = int(cfg.get("chunk_overlap_words", 80))
    source_dir = Path(args.source_dir)
    pdf_paths = sorted(source_dir.glob("*.pdf"), key=lambda path: path.name.lower())
    if not pdf_paths:
        raise SystemExit(f"[error] no PDF files found in {source_dir}")

    records = []
    parsed_sources = 0
    for path in pdf_paths:
        try:
            reader = PdfReader(str(path))
            page_texts = parse_pdf(reader)
        except Exception as exc:
            print(f"[warn] failed to parse {path}: {exc}")
            continue

        source_id = source_id_from_path(path)
        source = {
            "source_id": source_id,
            "title": title_from_pdf(path, reader),
            "publisher": None,
            "year": year_from_path(path),
            "page": None,
            "section": None,
            "url": None,
            "file_name": path.name,
        }
        chunk_index = 0
        for page, text in page_texts:
            for chunk in chunk_words(text, chunk_size=chunk_size, overlap=overlap):
                chunk_index += 1
                chunk_source = dict(source)
                chunk_source["page"] = page
                records.append({
                    "chunk_id": f"{source_id}_CHUNK_{chunk_index:04d}",
                    "source": chunk_source,
                    "text": chunk,
                })
        parsed_sources += 1
        print(f"[ok] {path.name}: {len(page_texts)} text pages, {chunk_index} chunks")

    write_jsonl(records, args.output)
    print(f"[ok] wrote {len(records)} chunks from {parsed_sources} PDFs to {args.output}")


if __name__ == "__main__":
    main()
