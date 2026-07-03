#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.io_utils import read_jsonl, write_jsonl


def to_float(x):
    try:
        if pd.isna(x) or x == "":
            return None
        return float(x)
    except Exception:
        return None


def accepted(row, min_faith=4.0, min_align=4.0, min_use=3.5) -> bool:
    decision = str(row.get("decision_accept_reject_revision", "")).strip().lower()
    if decision.startswith("reject"):
        return False
    if decision.startswith("accept"):
        return True
    f = to_float(row.get("faithfulness_1_5"))
    a = to_float(row.get("edos_alignment_1_5"))
    u = to_float(row.get("usefulness_1_5"))
    if f is None or a is None or u is None:
        return False
    return f >= min_faith and a >= min_align and u >= min_use


def main() -> None:
    ap = argparse.ArgumentParser(description="Adjudicate expert validation CSV and write validated card library.")
    ap.add_argument("--cards", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default="data/validation/validation_report.csv")
    args = ap.parse_args()

    cards = {c["card_id"]: c for c in read_jsonl(args.cards)}
    df = pd.read_csv(args.validation)
    validated = []
    report_rows = []
    for _, row in df.iterrows():
        cid = row.get("card_id")
        card = cards.get(cid)
        if not card:
            continue
        f = to_float(row.get("faithfulness_1_5"))
        a = to_float(row.get("edos_alignment_1_5"))
        u = to_float(row.get("usefulness_1_5"))
        notes = row.get("expert_notes") if not pd.isna(row.get("expert_notes")) else None
        is_acc = accepted(row)
        card["validation"] = {
            "status": "accepted" if is_acc else "rejected",
            "faithfulness": f,
            "edos_alignment": a,
            "usefulness": u,
            "notes": notes,
        }
        card["status"] = "validated" if is_acc else "rejected"
        if is_acc:
            validated.append(card)
        report_rows.append({
            "card_id": cid,
            "primary_edos_label": card.get("primary_edos_label"),
            "accepted": is_acc,
            "faithfulness": f,
            "edos_alignment": a,
            "usefulness": u,
            "notes": notes,
        })
    write_jsonl(validated, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(args.report, index=False)
    print(f"[ok] wrote {len(validated)} validated cards to {args.output}")
    print(f"[ok] wrote report to {args.report}")


if __name__ == "__main__":
    main()
