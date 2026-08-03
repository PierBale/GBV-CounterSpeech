#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from collections import defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.io_utils import read_jsonl, write_jsonl
from card_routed_rag.card_validation import load_schema, schema_errors
from card_routed_rag.text_utils import normalize_space


def normalize_card(card: dict) -> dict:
    card = dict(card)
    card["status"] = card.get("status") or "candidate"
    card["chunk"] = normalize_space(card.get("chunk") or card.get("source_quote", ""))
    card.pop("source_quote", None)
    card["reasoning"] = normalize_space(card.get("reasoning", ""))
    card["argument"] = normalize_space(card.get("argument") or card.get("claim", ""))
    card.pop("claim", None)
    card["edos_alignment"] = normalize_space(card.get("edos_alignment", ""))
    card["secondary_edos_labels"] = card.get("secondary_edos_labels") or []
    card["retrieval_keywords"] = sorted(set(k.strip().lower() for k in (card.get("retrieval_keywords") or []) if str(k).strip()))
    source = card.get("source") or {}
    card["source"] = {
        "source_id": source.get("source_id") or "UNKNOWN",
        "title": source.get("title") or "",
        "publisher": source.get("publisher") or "",
        "year": source.get("year"),
        "page": source.get("page"),
        "section": source.get("section"),
        "url": source.get("url"),
        "file_name": source.get("file_name"),
    }
    validation = card.get("validation") or {}
    card["validation"] = {
        "status": validation.get("status") or "not_validated",
        "faithfulness": validation.get("faithfulness"),
        "edos_alignment": validation.get("edos_alignment"),
        "usefulness": validation.get("usefulness"),
        "notes": validation.get("notes"),
    }
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize and optionally deduplicate candidate cards.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schema", default="configs/card_schema.json")
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--coverage-output", default=None, help="Optional CSV coverage report. Defaults to <output>_coverage.csv")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    cards = [normalize_card(c) for c in read_jsonl(args.input)]
    out = []
    seen = set()
    for card in cards:
        key = (card.get("primary_edos_label"), card.get("source", {}).get("source_id"), card.get("chunk", "").lower())
        if args.dedupe and key in seen:
            continue
        seen.add(key)
        errors = schema_errors(card, schema)
        if errors:
            print(f"[skip] {card.get('card_id')} schema errors: {errors[:3]}")
            continue
        out.append(card)
    write_jsonl(out, args.output)
    print(f"[ok] wrote {len(out)} normalized cards to {args.output}")

    counts = defaultdict(int)
    for c in out:
        counts[c.get("primary_edos_label")] += 1
    print("[coverage]")
    rows = []
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")
        label_cards = [c for c in out if c.get("primary_edos_label") == label]
        rows.append({
            "primary_edos_label": label,
            "num_cards": n,
            "num_sources": len(set((c.get("source") or {}).get("source_id") for c in label_cards)),
            "card_ids": " | ".join(c.get("card_id", "") for c in label_cards),
        })
    coverage_path = args.coverage_output or str(Path(args.output).with_suffix("")) + "_coverage.csv"
    Path(coverage_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(coverage_path, index=False)
    print(f"[ok] wrote coverage report to {coverage_path}")


if __name__ == "__main__":
    main()
