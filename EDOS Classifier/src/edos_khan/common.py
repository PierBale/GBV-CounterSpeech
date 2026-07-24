from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

from .labels import TASK_B_LABELS, TASK_C_LABELS, TASK_C_TO_B, labels_for_task


def labels_for(task: str) -> List[str]:
    return labels_for_task(task)


def make_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def save_label_map(path: Path, label2id: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label2id": label2id,
        "id2label": {str(v): k for k, v in label2id.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_label_map(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    label2id = {str(k): int(v) for k, v in payload["label2id"].items()}
    id2label = {int(k): str(v) for k, v in payload["id2label"].items()}
    return label2id, id2label


def _normalize_label(x):
    return str(x).strip().replace("\n", " ")


def _task_label_column(task: str) -> str:
    if task == "b":
        return "label_category"
    if task == "c":
        return "label_vector"
    raise ValueError("task must be 'b' or 'c'")


def _load_augmentation(path: Optional[Path], task: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["text", "label"])
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Augmentation CSV not found: {path}")

    aug = pd.read_csv(path)
    if "generated_text" not in aug.columns:
        raise ValueError("Augmentation CSV must contain column 'generated_text'.")
    if "label_vector" not in aug.columns:
        raise ValueError("Augmentation CSV must contain column 'label_vector'.")

    aug = aug.copy()
    aug["label_vector"] = aug["label_vector"].map(_normalize_label)

    if task == "b":
        aug["label"] = aug["label_vector"].map(TASK_C_TO_B)
    elif task == "c":
        aug["label"] = aug["label_vector"]
    else:
        raise ValueError("task must be 'b' or 'c'")

    aug = aug[["generated_text", "label"]].rename(
        columns={"generated_text": "text"}
    )
    aug = aug.dropna(subset=["text", "label"])
    aug["text"] = aug["text"].astype(str)
    return aug.reset_index(drop=True)


def load_data(
    edos_csv: Path,
    task: str,
    augmentation_csv: Optional[Path] = None,
    include_dev: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load EDOS sexist-only splits plus optional Khan DDA augmentation.

    Task B uses label_category.
    Task C uses label_vector.
    If include_dev=True, training data = train + augmentation + dev,
    matching the paper/public-script reproduction style.
    """
    edos_csv = Path(edos_csv)
    if not edos_csv.exists():
        raise FileNotFoundError(f"EDOS CSV not found: {edos_csv}")

    df = pd.read_csv(edos_csv)
    required = {"text", "label_sexist", "label_category", "label_vector", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"EDOS CSV missing columns: {sorted(missing)}")

    df = df.copy()
    for col in ["label_sexist", "label_category", "label_vector", "split"]:
        df[col] = df[col].map(_normalize_label)

    df = df[df["label_sexist"] == "sexist"].copy()
    label_col = _task_label_column(task)
    df = df[["text", label_col, "split"]].rename(columns={label_col: "label"})
    df["text"] = df["text"].astype(str)

    train = df[df["split"] == "train"][["text", "label"]].reset_index(drop=True)
    dev = df[df["split"] == "dev"][["text", "label"]].reset_index(drop=True)
    test = df[df["split"] == "test"][["text", "label"]].reset_index(drop=True)

    aug = _load_augmentation(augmentation_csv, task)
    parts = [train]
    if len(aug):
        parts.append(aug)
    if include_dev:
        parts.append(dev)

    train_final = pd.concat(parts, ignore_index=True)
    return train_final, dev, test


def add_ids(df: pd.DataFrame, label2id: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    unknown = sorted(set(df["label"]) - set(label2id))
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")
    df["label_id"] = df["label"].map(label2id).astype(int)
    return df


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 150):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            str(self.texts[i]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item


def class_weights(y: List[int], num_labels: int, dev) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(range(num_labels))),
        y=np.array(y),
    )
    return torch.tensor(weights, dtype=torch.float, device=dev)


def metrics(y_true: List[int], y_pred: List[int], labels: List[str]) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_preds(
    out_csv: Path,
    texts: List[str],
    gold_ids: List[int],
    pred_ids: List[int],
    id2label: Dict[int, str],
    probs=None,
) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "text": texts,
        "gold_id": gold_ids,
        "gold_label": [id2label[int(i)] for i in gold_ids],
        "pred_id": pred_ids,
        "label_pred": [id2label[int(i)] for i in pred_ids],
    })

    if probs is not None:
        probs = np.asarray(probs)
        df["confidence"] = probs.max(axis=1)
        df["softmax_max_value"] = probs.max(axis=1)
        for i in range(probs.shape[1]):
            df[f"prob_{i}"] = probs[:, i]

    df.to_csv(out_csv, index=False)
