#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.chunk_retrieval import ChunkMMRRetriever
from card_routed_rag.embeddings import HuggingFaceEmbedder, load_chunk_embedding_index
from card_routed_rag.io_utils import read_jsonl, read_yaml, write_json


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Retrieve an initial dense candidate pool and rerank it with "
            "Maximal Marginal Relevance."
        )
    )
    ap.add_argument("--chunks", default="data/processed/document_chunks.jsonl")
    ap.add_argument(
        "--embeddings",
        default="data/processed/document_chunk_embeddings.npz",
    )
    ap.add_argument("--query", required=True)
    ap.add_argument("--config", default="configs/retrieval_config.yaml")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--candidate-pool-size", type=int, default=None)
    ap.add_argument("--lambda-mult", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--output",
        default="data/retrieval_outputs/retrieved_chunks.json",
    )
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    top_k = args.top_k if args.top_k is not None else int(cfg.get("top_k", 10))
    candidate_pool_size = (
        args.candidate_pool_size
        if args.candidate_pool_size is not None
        else int(cfg.get("candidate_pool_size", 100))
    )
    lambda_mult = (
        args.lambda_mult
        if args.lambda_mult is not None
        else float(cfg.get("mmr_lambda", 0.5))
    )

    chunks = read_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(
            f"No chunks found at {args.chunks}. Run 02_parse_sources.py first."
        )
    index = load_chunk_embedding_index(args.embeddings)

    print(f"[model] loading {index.model_name}")
    embedder = HuggingFaceEmbedder(model_name=index.model_name, device=args.device)
    query_embedding = embedder.encode_query(args.query)
    retriever = ChunkMMRRetriever(
        chunks=chunks,
        embeddings=index.embeddings,
        indexed_chunk_ids=index.chunk_ids,
    )
    results = retriever.retrieve(
        query_embedding,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        lambda_mult=lambda_mult,
    )

    output = {
        "input": {
            "query": args.query,
            "embedding_model": index.model_name,
            "num_chunks": len(chunks),
            "candidate_pool_size": min(candidate_pool_size, len(chunks)),
            "top_k": top_k,
            "mmr_lambda": lambda_mult,
        },
        "chunks": [
            {
                "rank": rank,
                "chunk_id": result.chunk.get("chunk_id"),
                "relevance_score": result.relevance_score,
                "mmr_score": result.mmr_score,
                "candidate_rank": result.candidate_rank,
                "text": result.chunk.get("text"),
                "source": result.chunk.get("source"),
            }
            for rank, result in enumerate(results, start=1)
        ],
    }
    write_json(output, args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2)[:3000])
    print(f"\n[ok] wrote {len(results)} retrieved chunks to {args.output}")


if __name__ == "__main__":
    main()
