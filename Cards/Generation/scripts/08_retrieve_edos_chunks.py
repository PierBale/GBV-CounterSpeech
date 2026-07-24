#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.chunk_retrieval import ChunkMMRRetriever, ChunkRetrievalResult
from card_routed_rag.embeddings import HuggingFaceEmbedder, load_chunk_embedding_index
from card_routed_rag.io_utils import read_jsonl, read_yaml, write_json


def result_to_dict(result: ChunkRetrievalResult, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": result.chunk.get("chunk_id"),
        "relevance_score": result.relevance_score,
        "mmr_score": result.mmr_score,
        "candidate_rank": result.candidate_rank,
        "text": result.chunk.get("text"),
        "source": result.chunk.get("source"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Retrieve 10 MMR-diversified chunks for every EDOS Task C label, "
            "using each label definition as the query."
        )
    )
    ap.add_argument("--chunks", default="data/processed/document_chunks.jsonl")
    ap.add_argument(
        "--embeddings",
        default="data/processed/document_chunk_embeddings.npz",
    )
    ap.add_argument("--profiles", default="configs/edos_label_profiles.yaml")
    ap.add_argument("--config", default="configs/retrieval_config.yaml")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--candidate-pool-size", type=int, default=None)
    ap.add_argument("--lambda-mult", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--output",
        default="data/retrieval_outputs/edos_label_chunks.json",
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

    profiles = read_yaml(args.profiles).get("labels", {})
    if not profiles:
        raise SystemExit(f"No EDOS label profiles found at {args.profiles}.")
    for label, profile in profiles.items():
        if not str((profile or {}).get("definition", "")).strip():
            raise SystemExit(f"Missing definition for EDOS label: {label}")

    chunks = read_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(
            f"No chunks found at {args.chunks}. Run 02_parse_sources.py first."
        )
    index = load_chunk_embedding_index(args.embeddings)
    retriever = ChunkMMRRetriever(
        chunks=chunks,
        embeddings=index.embeddings,
        indexed_chunk_ids=index.chunk_ids,
    )

    print(f"[model] loading {index.model_name}")
    embedder = HuggingFaceEmbedder(model_name=index.model_name, device=args.device)

    labels_output: dict[str, Any] = {}
    for label, profile in profiles.items():
        definition = str(profile["definition"]).strip()
        print(f"[label] {label}")
        definition = f"Counterspeech strategy for the following hate speech: {definition}"
        query_embedding = embedder.encode_query(definition)
        results = retriever.retrieve(
            query_embedding,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            lambda_mult=lambda_mult,
        )
        labels_output[label] = {
            "query": definition,
            "num_selected": len(results),
            "chunks": [
                result_to_dict(result, rank)
                for rank, result in enumerate(results, start=1)
            ],
        }
        print(f"  [ok] selected {len(results)} chunks")

    output = {
        "retrieval": {
            "task": "EDOS Task C",
            "embedding_model": index.model_name,
            "num_chunks": len(chunks),
            "num_labels": len(labels_output),
            "candidate_pool_size": min(candidate_pool_size, len(chunks)),
            "top_k_per_label": top_k,
            "mmr_lambda": lambda_mult,
            "query_field": "definition",
        },
        "labels": labels_output,
    }
    write_json(output, args.output)
    print(
        f"[done] wrote {len(labels_output)} labels and "
        f"{sum(item['num_selected'] for item in labels_output.values())} "
        f"label-chunk results to {args.output}"
    )
    print(json.dumps(output["retrieval"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
