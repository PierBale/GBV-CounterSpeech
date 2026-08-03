#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    HuggingFaceEmbedder,
    save_chunk_embedding_index,
)
from card_routed_rag.io_utils import read_jsonl, read_yaml


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Encode all parsed document chunks with a Hugging Face embedding model."
    )
    ap.add_argument("--chunks", default="data/processed/document_chunks.jsonl")
    ap.add_argument(
        "--output",
        default="data/processed/document_chunk_embeddings.npz",
    )
    ap.add_argument("--config", default="configs/retrieval_config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--device",
        default=None,
        help="Optional Sentence-Transformers device, for example cpu, cuda, or mps.",
    )
    args = ap.parse_args()

    chunks = read_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(
            f"No chunks found at {args.chunks}. Run 02_parse_sources.py first."
        )

    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    if any(not chunk_id for chunk_id in chunk_ids):
        raise SystemExit("Every chunk must have a non-empty chunk_id.")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise SystemExit("chunk_id values must be unique.")

    texts = [str(chunk.get("text", "")).strip() for chunk in chunks]
    if any(not text for text in texts):
        raise SystemExit("Every chunk must contain non-empty text.")

    cfg = read_yaml(args.config)
    model_name = args.model or cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    batch_size = args.batch_size or int(cfg.get("embedding_batch_size", 8))
    if batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    print(f"[model] loading {model_name}")
    embedder = HuggingFaceEmbedder(model_name=model_name, device=args.device)
    embeddings = embedder.encode_documents(texts, batch_size=batch_size)
    save_chunk_embedding_index(
        args.output,
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        model_name=model_name,
    )
    print(
        f"[ok] encoded {len(chunks)} chunks as {embeddings.shape[1]}-dimensional "
        f"vectors in {args.output}"
    )


if __name__ == "__main__":
    main()
