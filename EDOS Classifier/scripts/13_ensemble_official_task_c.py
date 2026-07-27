#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from edos_khan.labels import TASK_C_LABELS
from edos_khan.official_task_c import (
    BASE_VOTERS,
    FALLBACK_MODEL,
    OFFICIAL_TASK_C_MODELS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the paper's four-model hard vote and Mistral-7B tie fallback."
        )
    )
    for model_key in (*BASE_VOTERS, FALLBACK_MODEL):
        parser.add_argument(f"--{model_key}", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metrics-json")
    parser.add_argument("--conan-json")
    parser.add_argument("--annotated-json")
    return parser.parse_args()


def load_and_validate(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    reference = frames[BASE_VOTERS[0]]
    required = {
        "row_id",
        "instance_id",
        "text",
        "pred_id",
        "label_pred",
        "confidence",
        *(f"prob_{index}" for index in range(len(TASK_C_LABELS))),
    }
    for model_key, frame in frames.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{model_key} predictions missing: {sorted(missing)}")
        if len(frame) != len(reference):
            raise ValueError(f"{model_key} row count does not match other models.")
        for column in ("row_id", "instance_id", "text"):
            left = frame[column].astype(str).tolist()
            right = reference[column].astype(str).tolist()
            if left != right:
                raise ValueError(
                    f"{model_key} is not aligned with the reference on {column}."
                )
    return frames


def ensemble_predictions(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    reference = frames[BASE_VOTERS[0]]
    rows: List[dict] = []
    for row_index in range(len(reference)):
        votes = {
            model_key: int(frames[model_key].iloc[row_index]["pred_id"])
            for model_key in BASE_VOTERS
        }
        counts = Counter(votes.values())
        highest_count = max(counts.values())
        leaders = sorted(
            label_id
            for label_id, count in counts.items()
            if count == highest_count
        )
        used_fallback = len(leaders) != 1
        if used_fallback:
            final_id = int(
                frames[FALLBACK_MODEL].iloc[row_index]["pred_id"]
            )
            decision_confidence = float(
                frames[FALLBACK_MODEL].iloc[row_index][f"prob_{final_id}"]
            )
            confidence_kind = "mistral_probability_for_fallback_label"
        else:
            final_id = leaders[0]
            decision_confidence = float(
                np.mean(
                    [
                        float(
                            frames[model_key]
                            .iloc[row_index][f"prob_{final_id}"]
                        )
                        for model_key in BASE_VOTERS
                    ]
                )
            )
            confidence_kind = "mean_base_probability_for_majority_label"

        row = {
            "row_id": int(reference.iloc[row_index]["row_id"]),
            "instance_id": str(reference.iloc[row_index]["instance_id"]),
            "text": str(reference.iloc[row_index]["text"]),
            "pred_id": final_id,
            "label_pred": TASK_C_LABELS[final_id],
            "confidence": decision_confidence,
            "confidence_kind": confidence_kind,
            "used_fallback": used_fallback,
            "top_vote_count": highest_count,
            "vote_counts": json.dumps(
                {
                    TASK_C_LABELS[label_id]: count
                    for label_id, count in sorted(counts.items())
                },
                ensure_ascii=False,
            ),
            "base_votes": json.dumps(
                {
                    key: TASK_C_LABELS[label_id]
                    for key, label_id in votes.items()
                },
                ensure_ascii=False,
            ),
            "fallback_pred": str(
                frames[FALLBACK_MODEL].iloc[row_index]["label_pred"]
            ),
        }
        if "gold_id" in reference.columns:
            row["gold_id"] = int(reference.iloc[row_index]["gold_id"])
            row["gold_label"] = str(reference.iloc[row_index]["gold_label"])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(frame: pd.DataFrame, input_paths: Dict[str, str]) -> dict:
    gold = frame["gold_id"].astype(int).tolist()
    predicted = frame["pred_id"].astype(int).tolist()
    return {
        "method": "M7-FE: hard vote over four base models; Mistral on ties",
        "num_examples": len(frame),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro")),
        "weighted_f1": float(f1_score(gold, predicted, average="weighted")),
        "fallback_used_count": int(frame["used_fallback"].sum()),
        "fallback_used_rate": float(frame["used_fallback"].mean()),
        "labels": TASK_C_LABELS,
        "confusion_matrix": confusion_matrix(
            gold,
            predicted,
            labels=list(range(len(TASK_C_LABELS))),
        ).tolist(),
        "classification_report": classification_report(
            gold,
            predicted,
            labels=list(range(len(TASK_C_LABELS))),
            target_names=TASK_C_LABELS,
            zero_division=0,
            output_dict=True,
        ),
        "prediction_files": input_paths,
        "model_repositories": {
            key: OFFICIAL_TASK_C_MODELS[key].repo_id
            for key in (*BASE_VOTERS, FALLBACK_MODEL)
        },
    }


def annotate_conan(
    input_path: Path,
    output_path: Path,
    ensemble: pd.DataFrame,
) -> None:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("CONAN input must be a JSON object keyed by instance id.")
    indexed = {
        str(row["instance_id"]): row
        for row in ensemble.to_dict(orient="records")
    }
    if set(indexed) != set(map(str, raw)):
        raise ValueError("Ensemble predictions do not cover exactly the CONAN ids.")

    for instance_id, item in raw.items():
        row = indexed[str(instance_id)]
        predictions = dict(item.get("HATE_SPEECH_EDOS_PREDICTIONS", {}))
        predictions["TASK_C"] = {
            "label": row["label_pred"],
            "confidence": row["confidence"],
            "confidence_kind": row["confidence_kind"],
            "method": "M7-FE official checkpoints",
            "used_mistral_fallback": bool(row["used_fallback"]),
            "top_vote_count": int(row["top_vote_count"]),
            "vote_counts": json.loads(row["vote_counts"]),
            "base_votes": json.loads(row["base_votes"]),
            "fallback_pred": row["fallback_pred"],
            "is_predicted_not_gold": True,
        }
        item["HATE_SPEECH_EDOS_PREDICTIONS"] = predictions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    input_paths = {
        model_key: getattr(args, model_key)
        for model_key in (*BASE_VOTERS, FALLBACK_MODEL)
    }
    frames = load_and_validate(input_paths)
    ensemble = ensemble_predictions(frames)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(output_csv, index=False)
    print(f"Saved {len(ensemble)} ensemble predictions to {output_csv}")

    has_gold = "gold_id" in ensemble.columns
    if args.metrics_json:
        if not has_gold:
            raise ValueError("--metrics-json requires gold EDOS prediction files.")
        metrics = evaluate(ensemble, input_paths)
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"EDOS Task C ensemble macro F1: {metrics['macro_f1']:.4f}")

    if args.conan_json or args.annotated_json:
        if not args.conan_json or not args.annotated_json:
            raise ValueError(
                "--conan-json and --annotated-json must be provided together."
            )
        if has_gold:
            raise ValueError("Cannot annotate CONAN with EDOS test predictions.")
        annotate_conan(
            Path(args.conan_json),
            Path(args.annotated_json),
            ensemble,
        )
        print(f"Saved annotated CONAN JSON to {args.annotated_json}")


if __name__ == "__main__":
    main()

