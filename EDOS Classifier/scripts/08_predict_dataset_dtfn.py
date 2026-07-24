#!/usr/bin/env python3
"""
08_predict_dataset_dtfn.py

Predict EDOS Task B/C labels for an unlabeled hate-speech dataset using a trained
DTFN checkpoint from scripts/04_train_dtfn.py.

This is needed for the full Khan-style M7 fallback ensemble on a new dataset.

Example:

python scripts/08_predict_dataset_dtfn.py \
  --task c \
  --input data/conan/WOMAN-Multitarget-CONAN.json \
  --output-csv outputs/conan/task_c_dtfn_predictions.csv \
  --model-dir models/edos_task_c/dtfn_dda \
  --label-map data/edos/processed/task_c_khan/label_map.json \
  --deberta-model microsoft/deberta-v3-large \
  --roberta-model roberta-large \
  --batch-size 4
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


TASK_B_LABELS = [
    "1. threats, plans to harm and incitement",
    "2. derogation",
    "3. animosity",
    "4. prejudiced discussions",
]

TASK_C_LABELS = [
    "1.1 threats of harm",
    "1.2 incitement and encouragement of harm",
    "2.1 descriptive attacks",
    "2.2 aggressive and emotive attacks",
    "2.3 dehumanising attacks & overt sexual objectification",
    "3.1 casual use of gendered slurs, profanities, and insults",
    "3.2 immutable gender differences and gender stereotypes",
    "3.3 backhanded gendered compliments",
    "3.4 condescending explanations or unwelcome advice",
    "4.1 supporting mistreatment of individual women",
    "4.2 supporting systemic discrimination against women as a group",
]


class DTFN(nn.Module):
    def __init__(self, deberta_name_or_path: str, roberta_name_or_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(deberta_name_or_path, output_hidden_states=True, ignore_mismatched_sizes=True)
        self.roberta = AutoModel.from_pretrained(roberta_name_or_path, output_hidden_states=True, ignore_mismatched_sizes=True)
        hidden = self.deberta.config.hidden_size + self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, d_input_ids, d_attention_mask, r_input_ids, r_attention_mask):
        d_out = self.deberta(input_ids=d_input_ids, attention_mask=d_attention_mask)
        r_out = self.roberta(input_ids=r_input_ids, attention_mask=r_attention_mask)
        d_cls = d_out.last_hidden_state[:, 0, :]
        r_cls = r_out.last_hidden_state[:, 0, :]
        fused = torch.cat([d_cls, r_cls], dim=-1)
        return self.classifier(self.dropout(fused))


class DualTextDataset(Dataset):
    def __init__(self, texts: List[str], deberta_tok, roberta_tok, max_length: int):
        self.texts = texts
        self.deberta_tok = deberta_tok
        self.roberta_tok = roberta_tok
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        d = self.deberta_tok(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        r = self.roberta_tok(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "d_input_ids": d["input_ids"].squeeze(0),
            "d_attention_mask": d["attention_mask"].squeeze(0),
            "r_input_ids": r["input_ids"].squeeze(0),
            "r_attention_mask": r["attention_mask"].squeeze(0),
        }


def read_unlabeled_dataset(path: Path, text_field: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    rows = []

    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
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
            return pd.DataFrame(rows)
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
            return pd.DataFrame(rows)
        raise ValueError("Unsupported JSON structure.")

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append({
                    "instance_id": str(obj.get("id", i)),
                    "text": str(obj.get(text_field, "")),
                    "target": obj.get("TARGET", ""),
                    "version": obj.get("VERSION", ""),
                    "counter_narrative": obj.get("COUNTER_NARRATIVE", ""),
                })
        return pd.DataFrame(rows)

    if suffix == ".csv":
        df = pd.read_csv(path)
        if text_field not in df.columns:
            raise ValueError(f"CSV missing text field {text_field!r}")
        if "instance_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "instance_id"})
        return pd.DataFrame({
            "instance_id": df["instance_id"].astype(str),
            "text": df[text_field].astype(str),
            "target": df["TARGET"] if "TARGET" in df.columns else "",
            "version": df["VERSION"] if "VERSION" in df.columns else "",
            "counter_narrative": df["COUNTER_NARRATIVE"] if "COUNTER_NARRATIVE" in df.columns else "",
        })

    raise ValueError(f"Unsupported input extension: {suffix}")


def load_id2label(label_map_path: Path | None, task: str) -> Dict[int, str]:
    if label_map_path and label_map_path.exists():
        payload = json.loads(label_map_path.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in payload["id2label"].items()}

    labels = TASK_B_LABELS if task == "b" else TASK_C_LABELS
    return {i: label for i, label in enumerate(labels)}


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def parse_args():
    p = argparse.ArgumentParser(description="Predict an unlabeled dataset with a trained DTFN checkpoint.")
    p.add_argument("--task", choices=["b", "c"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--text-field", default="HATE_SPEECH")

    p.add_argument("--model-dir", required=True, help="Directory containing best_dtfn.pt and saved tokenizers.")
    p.add_argument("--label-map", default=None, help="Path to label_map.json from prepared EDOS data.")
    p.add_argument("--deberta-model", default="microsoft/deberta-v3-large")
    p.add_argument("--roberta-model", default="roberta-large")
    p.add_argument("--deberta-checkpoint", default=None, help="Use if DTFN was trained with local DeBERTa checkpoint.")
    p.add_argument("--roberta-checkpoint", default=None, help="Use if DTFN was trained with local RoBERTa checkpoint.")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=150)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument("--include-probs", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    checkpoint = model_dir / "best_dtfn.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing DTFN checkpoint: {checkpoint}")

    df = read_unlabeled_dataset(Path(args.input), args.text_field)
    df = df[df["text"].fillna("").astype(str).str.strip().astype(bool)].reset_index(drop=True)

    id2label = load_id2label(Path(args.label_map) if args.label_map else None, args.task)
    num_labels = len(id2label)
    device = get_device(args.device)

    deberta_tok_path = model_dir / "deberta_tokenizer"
    roberta_tok_path = model_dir / "roberta_tokenizer"
    deberta_tok = AutoTokenizer.from_pretrained(deberta_tok_path if deberta_tok_path.exists() else args.deberta_model)
    roberta_tok = AutoTokenizer.from_pretrained(roberta_tok_path if roberta_tok_path.exists() else args.roberta_model)

    deberta_source = args.deberta_checkpoint or args.deberta_model
    roberta_source = args.roberta_checkpoint or args.roberta_model

    model = DTFN(deberta_source, roberta_source, num_labels, args.dropout)
    saved = torch.load(checkpoint, map_location="cpu")
    state = saved["model_state_dict"] if isinstance(saved, dict) and "model_state_dict" in saved else saved
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset = DualTextDataset(df["text"].tolist(), deberta_tok, roberta_tok, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_probs = []
    all_pred_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predict DTFN Task {args.task.upper()}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_pred_ids.extend(pred.cpu().tolist())

    probs = np.vstack(all_probs)
    labels = [id2label[int(i)] for i in all_pred_ids]
    conf = probs.max(axis=1)

    out = df.copy()
    out["task"] = f"task_{args.task}"
    out["model_name"] = str(model_dir)
    out["pred_id"] = all_pred_ids
    out["label_pred"] = labels
    out["confidence"] = conf

    if args.include_probs:
        for i in range(probs.shape[1]):
            safe = id2label[i].replace(" ", "_").replace(".", "_").replace("/", "_").replace("&", "and")
            out[f"prob_{i}_{safe}"] = probs[:, i]

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved DTFN predictions to {out_csv}")


if __name__ == "__main__":
    main()
