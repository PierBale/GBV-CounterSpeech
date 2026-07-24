#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.io_utils import read_yaml, write_jsonl
from card_routed_rag.text_utils import normalize_space, chunk_words


def parse_pdf(path: Path) -> list[tuple[int | None, str]]:
    pages = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = normalize_space(page.extract_text() or "")
        if text:
            pages.append((i, text))
    return pages


def parse_html(path: Path) -> list[tuple[int | None, str]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = normalize_space(soup.get_text(" "))
    return [(None, text)] if text else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse downloaded PDF/HTML sources into chunk JSONL.")
    ap.add_argument("--sources", default="configs/sources.yaml")
    ap.add_argument("--config", default="configs/extraction_config.yaml")
    ap.add_argument("--source-dir", default="data/sources")
    ap.add_argument("--output", default="data/processed/document_chunks.jsonl")
    args = ap.parse_args()

    sources = read_yaml(args.sources)["sources"]
    cfg = read_yaml(args.config)
    chunk_size = int(cfg.get("chunk_size_words", 350))
    overlap = int(cfg.get("chunk_overlap_words", 80))

    records = []
    for src in sources:
        path = Path(args.source_dir) / src["file_name"]
        if not path.exists():
            print(f"[warn] missing source file: {path}")
            continue
        try:
            page_texts = parse_pdf(path) if src.get("file_type") == "pdf" else parse_html(path)
        except Exception as exc:
            print(f"[warn] failed to parse {path}: {exc}")
            continue
        chunk_index = 0
        for page, text in page_texts:
            for chunk in chunk_words(text, chunk_size=chunk_size, overlap=overlap):
                chunk_index += 1
                records.append({
                    "chunk_id": f"{src['source_id']}_CHUNK_{chunk_index:04d}",
                    "source": {
                        "source_id": src["source_id"],
                        "title": src["title"],
                        "publisher": src.get("publisher"),
                        "year": src.get("year"),
                        "page": page,
                        "section": None,
                        "url": src.get("url"),
                    },
                    "text": chunk,
                })
    write_jsonl(records, args.output)
    print(f"[ok] wrote {len(records)} chunks to {args.output}")


if __name__ == "__main__":
    main()
