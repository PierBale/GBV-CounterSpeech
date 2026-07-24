#!/usr/bin/env python3
"""
09_ensemble_dataset_predictions.py

Apply the Khan-style Mistral-7B fallback ensemble to an unlabeled dataset.

Input:
- two or more base prediction CSVs with columns:
  instance_id, text, label_pred, confidence
- one fallback prediction CSV, usually Mistral-7B, with the same rows.

Logic:
- if base models agree by majority, use the majority label;
- if base models tie, use the fallback model prediction.

This version does not require gold labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd


def majority_or_tie(labels: List[str]):
    counts = Counter(labels)
    top = counts.most_common()
    if not top:
        return None, True
    if len(top) == 1:
        return top[0][0], False
    if top[0][1] > top[1][1]:
        return top[0][0], False
    return None, True


def align_prediction_frames(base_dfs: List[pd.DataFrame], fallback_df: pd.DataFrame, align_by: str):
    if align_by == "row":
        n = len(fallback_df)
        for i, df in enumerate(base_dfs):
            if len(df) != n:
                raise ValueError(f"Base prediction file {i} has {len(df)} rows, fallback has {n}.")
        return base_dfs, fallback_df

    if align_by == "instance_id":
        if "instance_id" not in fallback_df.columns:
            raise ValueError("fallback predictions must contain instance_id for align_by=instance_id")
        aligned = []
        ids = fallback_df["instance_id"].astype(str).tolist()
        for df in base_dfs:
            if "instance_id" not in df.columns:
                raise ValueError("all base predictions must contain instance_id for align_by=instance_id")
            tmp = df.copy()
            tmp["instance_id"] = tmp["instance_id"].astype(str)
            tmp = tmp.set_index("instance_id").loc[ids].reset_index()
            aligned.append(tmp)
        return aligned, fallback_df

    raise ValueError("--align-by must be row or instance_id")


def parse_args():
    p = argparse.ArgumentParser(description="Mistral fallback ensemble for unlabeled EDOS predictions.")
    p.add_argument("--base-preds", nargs="+", required=True)
    p.add_argument("--fallback-preds", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--task", choices=["b", "c"], required=True)
    p.add_argument("--align-by", choices=["row", "instance_id"], default="instance_id")
    p.add_argument("--label-column", default="label_pred")
    p.add_argument("--confidence-column", default="confidence")
    return p.parse_args()


def main():
    args = parse_args()
    base_dfs = [pd.read_csv(p) for p in args.base_preds]
    fallback_df = pd.read_csv(args.fallback_preds)

    base_dfs, fallback_df = align_prediction_frames(base_dfs, fallback_df, args.align_by)

    rows = []
    for i in range(len(fallback_df)):
        base_votes = [str(df.loc[i, args.label_column]) for df in base_dfs]
        majority_label, tied = majority_or_tie(base_votes)
        fallback_label = str(fallback_df.loc[i, args.label_column])
        final_label = fallback_label if tied else majority_label

        base_confidences = [
            float(df.loc[i, args.confidence_column]) if args.confidence_column in df.columns else None
            for df in base_dfs
        ]
        fallback_conf = float(fallback_df.loc[i, args.confidence_column]) if args.confidence_column in fallback_df.columns else None

        rows.append({
            "instance_id": str(fallback_df.loc[i, "instance_id"]) if "instance_id" in fallback_df.columns else str(i),
            "text": fallback_df.loc[i, "text"] if "text" in fallback_df.columns else "",
            "task": f"task_{args.task}",
            "label_pred": final_label,
            "confidence": fallback_conf if tied else max([x for x in base_confidences if x is not None], default=None),
            "used_fallback": bool(tied),
            "fallback_pred": fallback_label,
            "fallback_confidence": fallback_conf,
            "base_votes": json.dumps(base_votes, ensure_ascii=False),
            "base_confidences": json.dumps(base_confidences, ensure_ascii=False),
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved ensemble predictions to {out_path}")
    print(f"Fallback used: {int(out['used_fallback'].sum())}/{len(out)}")


if __name__ == "__main__":
    main()
