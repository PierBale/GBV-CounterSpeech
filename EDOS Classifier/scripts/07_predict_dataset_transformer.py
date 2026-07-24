#!/usr/bin/env python3
"""
07_predict_dataset_transformer.py

Predict EDOS Task B or Task C labels for an unlabeled hate-speech dataset.

This script is meant for datasets such as WOMEN-only CONAN, where each item has
a HATE_SPEECH field and no gold EDOS label. It can be run once for Task B and
once for Task C, using the classifiers reproduced/trained from Khan et al. ACL 2025.

Supported model kinds:
- hf_seqcls: a standard Hugging Face sequence-classification checkpoint
  such as models/edos_task_c/deberta_dda/best_model
- mistral_lora: a Mistral-7B base model plus PEFT/LoRA adapter
  such as models/edos_task_c/mistral_dda/best_adapter

Input formats:
- JSON object keyed by instance id:
  {
    "950": {"HATE_SPEECH": "...", "COUNTER_NARRATIVE": "...", "TARGET": "WOMEN"}
  }
- JSON list of objects
- JSONL
- CSV

Examples:

Task B with DeBERTa:
    python scripts/07_predict_dataset_transformer.py \
      --task b \
      --input data/conan/WOMAN-Multitarget-CONAN.json \
      --output-csv outputs/conan/task_b_deberta_predictions.csv \
      --model-kind hf_seqcls \
      --model-dir models/edos_task_b/deberta_dda/best_model

Task C with DeBERTa:
    python scripts/07_predict_dataset_transformer.py \
      --task c \
      --input data/conan/WOMAN-Multitarget-CONAN.json \
      --output-csv outputs/conan/task_c_deberta_predictions.csv \
      --model-kind hf_seqcls \
      --model-dir models/edos_task_c/deberta_dda/best_model

Task C with Mistral LoRA:
    python scripts/07_predict_dataset_transformer.py \
      --task c \
      --input data/conan/WOMAN-Multitarget-CONAN.json \
      --output-csv outputs/conan/task_c_mistral_predictions.csv \
      --model-kind mistral_lora \
      --base-model mistralai/Mistral-7B-v0.1 \
      --adapter-dir models/edos_task_c/mistral_dda/best_adapter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from edos_khan.labels import labels_for_task


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def read_unlabeled_dataset(path: Path, text_field: str) -> Tuple[pd.DataFrame, Any, str]:
    """Return dataframe with instance_id/text plus raw object and detected format."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        if isinstance(raw, dict):
            for key, value in raw.items():
                if not isinstance(value, dict):
                    continue
                rows.append({
                    "instance_id": str(key),
                    "text": str(value.get(text_field, "")),
                    "target": value.get("TARGET", ""),
                    "version": value.get("VERSION", ""),
                    "counter_narrative": value.get("COUNTER_NARRATIVE", ""),
                })
            return pd.DataFrame(rows), raw, "json_dict"

        if isinstance(raw, list):
            for i, value in enumerate(raw):
                if not isinstance(value, dict):
                    continue
                rows.append({
                    "instance_id": str(value.get("id", i)),
                    "text": str(value.get(text_field, "")),
                    "target": value.get("TARGET", ""),
                    "version": value.get("VERSION", ""),
                    "counter_narrative": value.get("COUNTER_NARRATIVE", ""),
                })
            return pd.DataFrame(rows), raw, "json_list"

        raise ValueError("Unsupported JSON structure: expected object or list.")

    if suffix == ".jsonl":
        raw_rows = []
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                raw_rows.append(obj)
                rows.append({
                    "instance_id": str(obj.get("id", i)),
                    "text": str(obj.get(text_field, "")),
                    "target": obj.get("TARGET", ""),
                    "version": obj.get("VERSION", ""),
                    "counter_narrative": obj.get("COUNTER_NARRATIVE", ""),
                })
        return pd.DataFrame(rows), raw_rows, "jsonl"

    if suffix == ".csv":
        df = pd.read_csv(path)
        if text_field not in df.columns:
            raise ValueError(f"CSV does not contain text field {text_field!r}")
        if "instance_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "instance_id"})
        rows = pd.DataFrame({
            "instance_id": df["instance_id"].astype(str),
            "text": df[text_field].astype(str),
            "target": df["TARGET"] if "TARGET" in df.columns else "",
            "version": df["VERSION"] if "VERSION" in df.columns else "",
            "counter_narrative": df["COUNTER_NARRATIVE"] if "COUNTER_NARRATIVE" in df.columns else "",
        })
        return rows, df, "csv"

    raise ValueError(f"Unsupported input format: {suffix}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    return df


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def id2label_from_model_or_task(model, task: str) -> Dict[int, str]:
    default_labels = labels_for_task(task)
    id2label = getattr(model.config, "id2label", None) or {}
    fixed = {}
    for i in range(len(default_labels)):
        value = id2label.get(i) or id2label.get(str(i))
        if value is None or str(value).lower().startswith("label_"):
            value = default_labels[i]
        fixed[i] = str(value)
    return fixed


def load_model_and_tokenizer(args):
    if args.model_kind == "hf_seqcls":
        if not args.model_dir:
            raise ValueError("--model-dir is required for --model-kind hf_seqcls")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        return model, tokenizer

    if args.model_kind == "mistral_lora":
        if PeftModel is None:
            raise ImportError("peft is required for mistral_lora. Install: pip install peft")
        if not args.base_model or not args.adapter_dir:
            raise ValueError("--base-model and --adapter-dir are required for mistral_lora")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=len(labels_for_task(args.task)),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        return model, tokenizer

    raise ValueError(f"Unsupported model kind: {args.model_kind}")


def predict(df: pd.DataFrame, model, tokenizer, task: str, batch_size: int, max_length: int, device: torch.device):
    dataset = TextDataset(df["text"].tolist(), tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if not hasattr(model, "hf_device_map"):
        model.to(device)
    model.eval()

    all_probs = []
    all_pred_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predict Task {task.upper()}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            all_probs.append(probs.detach().cpu().numpy())
            all_pred_ids.extend(pred_ids.detach().cpu().tolist())

    probs = np.vstack(all_probs)
    id2label = id2label_from_model_or_task(model, task)
    pred_labels = [id2label[int(i)] for i in all_pred_ids]
    confidences = probs.max(axis=1)
    return all_pred_ids, pred_labels, confidences, probs, id2label


def write_prediction_csv(
    out_csv: Path,
    df: pd.DataFrame,
    task: str,
    model_name: str,
    pred_ids,
    pred_labels,
    confidences,
    probs,
    id2label: Dict[int, str],
    include_probs: bool,
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["task"] = f"task_{task}"
    out["model_name"] = model_name
    out["pred_id"] = pred_ids
    out["label_pred"] = pred_labels
    out["confidence"] = confidences

    if include_probs:
        for i in range(probs.shape[1]):
            label = id2label[i]
            safe = label.replace(" ", "_").replace(".", "_").replace("/", "_").replace("&", "and")
            out[f"prob_{i}_{safe}"] = probs[:, i]

    out.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="Predict EDOS Task B/C labels for an unlabeled hate-speech dataset.")
    p.add_argument("--task", choices=["b", "c"], required=True)
    p.add_argument("--input", required=True, help="Input JSON/JSONL/CSV dataset.")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--text-field", default="HATE_SPEECH")

    p.add_argument("--model-kind", choices=["hf_seqcls", "mistral_lora"], default="hf_seqcls")
    p.add_argument("--model-dir", default=None, help="HF seqcls checkpoint directory.")
    p.add_argument("--base-model", default=None, help="Base model for mistral_lora.")
    p.add_argument("--adapter-dir", default=None, help="PEFT adapter directory for mistral_lora.")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=150)
    p.add_argument("--device", default="auto")
    p.add_argument("--include-probs", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    df, _, _ = read_unlabeled_dataset(input_path, args.text_field)
    df = clean_dataframe(df)

    model, tokenizer = load_model_and_tokenizer(args)
    device = get_device(args.device)
    pred_ids, pred_labels, conf, probs, id2label = predict(
        df=df,
        model=model,
        tokenizer=tokenizer,
        task=args.task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    model_name = args.model_dir or args.adapter_dir or args.base_model or "unknown"
    write_prediction_csv(
        out_csv=Path(args.output_csv),
        df=df,
        task=args.task,
        model_name=model_name,
        pred_ids=pred_ids,
        pred_labels=pred_labels,
        confidences=conf,
        probs=probs,
        id2label=id2label,
        include_probs=args.include_probs,
    )


if __name__ == "__main__":
    main()
