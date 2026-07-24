#!/usr/bin/env python3
"""
10_write_dataset_annotations.py

Add predicted EDOS Task B and/or Task C labels to a hate-speech dataset.

Designed for WOMEN-only CONAN JSON dictionaries such as:
{
  "950": {
    "HATE_SPEECH": "...",
    "COUNTER_NARRATIVE": "...",
    "TARGET": "WOMEN",
    "VERSION": "V2"
  }
}

Output JSON keeps the original fields and adds:

"HATE_SPEECH_EDOS_PREDICTIONS": {
  "TASK_B": {
    "label": "...",
    "confidence": 0.82,
    "source": "..."
  },
  "TASK_C": {
    "label": "...",
    "confidence": 0.74,
    "parent_task_b": "...",
    "source": "..."
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

TASK_C_TO_B = {
    "1.1 threats of harm": "1. threats, plans to harm and incitement",
    "1.2 incitement and encouragement of harm": "1. threats, plans to harm and incitement",
    "2.1 descriptive attacks": "2. derogation",
    "2.2 aggressive and emotive attacks": "2. derogation",
    "2.3 dehumanising attacks & overt sexual objectification": "2. derogation",
    "3.1 casual use of gendered slurs, profanities, and insults": "3. animosity",
    "3.2 immutable gender differences and gender stereotypes": "3. animosity",
    "3.3 backhanded gendered compliments": "3. animosity",
    "3.4 condescending explanations or unwelcome advice": "3. animosity",
    "4.1 supporting mistreatment of individual women": "4. prejudiced discussions",
    "4.2 supporting systemic discrimination against women as a group": "4. prejudiced discussions",
}


def load_predictions(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    df = pd.read_csv(path)
    required = {"instance_id", "label_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV {path} missing columns: {missing}")

    result = {}
    for _, row in df.iterrows():
        instance_id = str(row["instance_id"])
        result[instance_id] = {
            "label": str(row["label_pred"]),
            "confidence": float(row["confidence"]) if "confidence" in df.columns and pd.notna(row.get("confidence")) else None,
            "source": str(path),
            "used_fallback": bool(row["used_fallback"]) if "used_fallback" in df.columns and pd.notna(row.get("used_fallback")) else None,
            "fallback_pred": str(row["fallback_pred"]) if "fallback_pred" in df.columns and pd.notna(row.get("fallback_pred")) else None,
            "base_votes": str(row["base_votes"]) if "base_votes" in df.columns and pd.notna(row.get("base_votes")) else None,
        }
    return result


def read_json_dataset(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("This annotation script expects a JSON object keyed by instance id.")
    return raw


def annotate(raw: Dict[str, Dict[str, Any]], task_b_preds, task_c_preds, overwrite: bool):
    annotated = {}
    for instance_id, item in raw.items():
        item = dict(item)
        if overwrite or "HATE_SPEECH_EDOS_PREDICTIONS" not in item:
            item["HATE_SPEECH_EDOS_PREDICTIONS"] = dict(item.get("HATE_SPEECH_EDOS_PREDICTIONS", {}))

        pred_container = item["HATE_SPEECH_EDOS_PREDICTIONS"]

        if str(instance_id) in task_b_preds:
            pred_container["TASK_B"] = {
                "label": task_b_preds[str(instance_id)]["label"],
                "confidence": task_b_preds[str(instance_id)]["confidence"],
                "source": task_b_preds[str(instance_id)]["source"],
                "used_fallback": task_b_preds[str(instance_id)].get("used_fallback"),
                "fallback_pred": task_b_preds[str(instance_id)].get("fallback_pred"),
                "base_votes": task_b_preds[str(instance_id)].get("base_votes"),
            }

        if str(instance_id) in task_c_preds:
            c_label = task_c_preds[str(instance_id)]["label"]
            pred_container["TASK_C"] = {
                "label": c_label,
                "confidence": task_c_preds[str(instance_id)]["confidence"],
                "parent_task_b": TASK_C_TO_B.get(c_label),
                "source": task_c_preds[str(instance_id)]["source"],
                "used_fallback": task_c_preds[str(instance_id)].get("used_fallback"),
                "fallback_pred": task_c_preds[str(instance_id)].get("fallback_pred"),
                "base_votes": task_c_preds[str(instance_id)].get("base_votes"),
            }

        annotated[str(instance_id)] = item
    return annotated


def write_flat_csv(annotated: Dict[str, Dict[str, Any]], out_csv: Path):
    rows = []
    for instance_id, item in annotated.items():
        preds = item.get("HATE_SPEECH_EDOS_PREDICTIONS", {})
        b = preds.get("TASK_B", {})
        c = preds.get("TASK_C", {})
        rows.append({
            "instance_id": instance_id,
            "HATE_SPEECH": item.get("HATE_SPEECH", ""),
            "COUNTER_NARRATIVE": item.get("COUNTER_NARRATIVE", ""),
            "TARGET": item.get("TARGET", ""),
            "VERSION": item.get("VERSION", ""),
            "task_b_pred": b.get("label"),
            "task_b_confidence": b.get("confidence"),
            "task_b_used_fallback": b.get("used_fallback"),
            "task_c_pred": c.get("label"),
            "task_c_confidence": c.get("confidence"),
            "task_c_parent_task_b": c.get("parent_task_b"),
            "task_c_used_fallback": c.get("used_fallback"),
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Annotate a JSON hate-speech dataset with EDOS Task B/C predictions.")
    p.add_argument("--input-json", required=True)
    p.add_argument("--task-b-preds", default=None)
    p.add_argument("--task-c-preds", default=None)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    raw = read_json_dataset(Path(args.input_json))
    task_b = load_predictions(args.task_b_preds)
    task_c = load_predictions(args.task_c_preds)

    annotated = annotate(raw, task_b, task_c, overwrite=args.overwrite)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(annotated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved annotated JSON to {out_json}")

    if args.output_csv:
        write_flat_csv(annotated, Path(args.output_csv))
        print(f"Saved flat CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
