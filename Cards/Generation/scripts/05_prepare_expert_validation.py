#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.io_utils import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Create expert validation CSV from candidate cards.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cards = read_jsonl(args.input)
    rows = []
    for c in cards:
        s = c.get("source", {}) or {}
        rows.append({
            "card_id": c.get("card_id"),
            "primary_edos_label": c.get("primary_edos_label"),
            "secondary_edos_labels": " | ".join(c.get("secondary_edos_labels", []) or []),
            "source_id": s.get("source_id"),
            "source_title": s.get("title"),
            "source_publisher": s.get("publisher"),
            "source_year": s.get("year"),
            "source_page": s.get("page"),
            "source_url": s.get("url"),
            "source_quote": c.get("source_quote"),
            "claim": c.get("claim"),
            "edos_alignment_text": c.get("edos_alignment"),
            "retrieval_keywords": " | ".join(c.get("retrieval_keywords", []) or []),
            "faithfulness_1_5": "",
            "edos_alignment_1_5": "",
            "usefulness_1_5": "",
            "decision_accept_reject_revision": "",
            "expert_notes": "",
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[ok] wrote validation sheet with {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
