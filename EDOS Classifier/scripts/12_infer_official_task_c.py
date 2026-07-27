#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from edos_khan.labels import TASK_C_LABELS
from edos_khan.official_task_c import (
    OFFICIAL_TASK_C_MODELS,
    load_official_model,
    predict_probabilities,
    prediction_frame,
    read_conan_hate_speech,
    read_edos_task_c_test,
    resolve_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one official Khan et al. checkpoint on EDOS Task C test or CONAN."
        )
    )
    parser.add_argument(
        "--model",
        choices=list(OFFICIAL_TASK_C_MODELS),
        required=True,
    )
    parser.add_argument("--dataset", choices=["edos-test", "conan"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--checkpoint-root",
        default="models/official_task_c",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metrics-json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=150)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Used by DeBERTa/DTFN. Official Mistral QLoRA requires CUDA:0.",
    )
    parser.add_argument(
        "--allow-unsafe-checkpoint-load",
        action="store_true",
        help=(
            "Fallback to pickle-based torch.load only if safe weights-only loading "
            "is incompatible with the trusted authors' checkpoint."
        ),
    )
    return parser.parse_args()


def evaluation_metrics(frame: pd.DataFrame):
    gold = frame["gold_id"].astype(int).tolist()
    predicted = frame["pred_id"].astype(int).tolist()
    return {
        "dataset": "EDOS Task C official sexist test split",
        "num_examples": len(frame),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro")),
        "weighted_f1": float(f1_score(gold, predicted, average="weighted")),
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
    }


def main():
    args = parse_args()
    source = (
        read_edos_task_c_test(Path(args.input))
        if args.dataset == "edos-test"
        else read_conan_hate_speech(Path(args.input))
    )
    checkpoint = resolve_checkpoint(Path(args.checkpoint_root), args.model)
    device = torch.device(args.device)
    model, tokenizers = load_official_model(
        model_key=args.model,
        checkpoint_path=checkpoint,
        device=device,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
    )
    probabilities = predict_probabilities(
        frame=source,
        model_key=args.model,
        model=model,
        tokenizers=tokenizers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    output = prediction_frame(source, probabilities, args.model, checkpoint)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} predictions to {output_path}")

    if args.dataset == "edos-test":
        if not args.metrics_json:
            raise ValueError("--metrics-json is required with --dataset edos-test")
        metrics = evaluation_metrics(output)
        metrics["model_key"] = args.model
        metrics["model_repo"] = OFFICIAL_TASK_C_MODELS[args.model].repo_id
        metrics["checkpoint"] = str(checkpoint)
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"EDOS Task C macro F1 ({args.model}): "
            f"{metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
