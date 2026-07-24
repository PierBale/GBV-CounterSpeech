#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.io_utils import read_jsonl, write_json, read_yaml
from card_routed_rag.retrieval import CardRetriever, RetrievalResult


def result_to_dict(r: RetrievalResult) -> dict:
    c = r.card
    return {
        "card_id": c.get("card_id"),
        "score": r.score,
        "score_breakdown": r.breakdown,
        "claim": c.get("claim"),
        "source_quote": c.get("source_quote"),
        "primary_edos_label": c.get("primary_edos_label"),
        "secondary_edos_labels": c.get("secondary_edos_labels"),
        "edos_alignment": c.get("edos_alignment"),
        "retrieval_keywords": c.get("retrieval_keywords"),
        "source": c.get("source"),
        "validation": c.get("validation"),
    }


def run_method(retriever: CardRetriever, method: str, hate_speech: str, edos_label: str | None, top_k: int, cfg: dict) -> list[RetrievalResult]:
    if method == "basic":
        return retriever.basic(hate_speech, top_k=top_k)
    if method == "dense_mmr":
        return retriever.dense_mmr(hate_speech, top_k=top_k, lambda_mult=float(cfg.get("mmr_lambda", 0.7)))
    if method == "card_aware":
        port = cfg.get("portfolio", {}) or {}
        return retriever.card_aware(
            hate_speech,
            edos_label=edos_label,
            top_k=top_k,
            source_diversity_bonus=float(port.get("source_diversity_bonus", 0.08)),
            redundancy_penalty=float(port.get("redundancy_penalty", 0.15)),
            max_cards_per_source=int(port.get("max_cards_per_source", 2)),
        )
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve EDOS-specific quote-first evidence cards using basic, dense_mmr, or card_aware retrieval.")
    ap.add_argument("--cards", default="data/cards/validated/validated_cards.jsonl")
    ap.add_argument("--method", choices=["basic", "dense_mmr", "card_aware", "all"], default="card_aware")
    ap.add_argument("--hate-speech", required=True)
    ap.add_argument("--edos-label", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--config", default="configs/retrieval_config.yaml")
    ap.add_argument("--output", default="data/retrieval_outputs/retrieved_cards.json")
    args = ap.parse_args()

    cards = read_jsonl(args.cards)
    if not cards:
        raise SystemExit(f"No cards found at {args.cards}")
    cfg = read_yaml(args.config)
    retriever = CardRetriever(cards, weights=cfg.get("weights", {}))

    methods = ["basic", "dense_mmr", "card_aware"] if args.method == "all" else [args.method]
    output = {
        "input": {
            "hate_speech": args.hate_speech,
            "edos_label": args.edos_label,
            "top_k": args.top_k,
        },
        "methods": {},
    }
    for method in methods:
        results = run_method(retriever, method, args.hate_speech, args.edos_label, args.top_k, cfg)
        output["methods"][method] = {
            "selected_cards": [result_to_dict(r) for r in results],
            "retrieval_report": {
                "num_cards_total": len(cards),
                "num_selected": len(results),
                "unique_sources_selected": len(set(((r.card.get("source") or {}).get("source_id", "UNKNOWN")) for r in results)),
                "primary_edos_labels_selected": sorted(set(str(r.card.get("primary_edos_label")) for r in results)),
            },
        }
    write_json(output, args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2)[:3000])
    print(f"\n[ok] wrote retrieval output to {args.output}")


if __name__ == "__main__":
    main()
