from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .labels import TASK_C_LABELS


@dataclass(frozen=True)
class OfficialModelSpec:
    key: str
    repo_id: str
    filename: str
    architecture: str


OFFICIAL_TASK_C_MODELS: Dict[str, OfficialModelSpec] = {
    "deberta9": OfficialModelSpec(
        key="deberta9",
        repo_id="sahrishkhan/edos-deberta-9-c-model",
        filename="edos-deberta-9-c-model.pth",
        architecture="deberta_seqcls",
    ),
    "dtfn3": OfficialModelSpec(
        key="dtfn3",
        repo_id="sahrishkhan/edos-roberta-deberta-3-c-model",
        filename="edos-roberta-deberta-3-c-model.pth",
        architecture="dtfn",
    ),
    "dtfn7": OfficialModelSpec(
        key="dtfn7",
        repo_id="sahrishkhan/edos-roberta-deberta-7-c-model",
        filename="edos-roberta-deberta-7-c-model.pth",
        architecture="dtfn",
    ),
    "dtfn8": OfficialModelSpec(
        key="dtfn8",
        repo_id="sahrishkhan/edos-roberta-deberta-8-c-model",
        filename="edos-roberta-deberta-8-c-model.pth",
        architecture="dtfn",
    ),
    "mistral": OfficialModelSpec(
        key="mistral",
        repo_id="sahrishkhan/edos-mistral-c-model",
        filename="edos-mistral-c-model.pth",
        architecture="mistral_qlora",
    ),
}

BASE_VOTERS: Tuple[str, ...] = ("deberta9", "dtfn3", "dtfn7", "dtfn8")
FALLBACK_MODEL = "mistral"
DEBERTA_BASE = "microsoft/deberta-v3-large"
ROBERTA_BASE = "FacebookAI/roberta-large"
MISTRAL_BASE = "mistralai/Mistral-7B-v0.1"
NUM_TASK_C_LABELS = len(TASK_C_LABELS)
DEFAULT_CHECKPOINT_ROOT = Path.home() / "edos_task_c_checkpoints"


class SingleTokenizerDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer, max_length: int):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            str(self.texts[index]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


class DualTokenizerDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        deberta_tokenizer,
        roberta_tokenizer,
        max_length: int,
    ):
        self.texts = list(texts)
        self.deberta_tokenizer = deberta_tokenizer
        self.roberta_tokenizer = roberta_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[index])
        deberta = self.deberta_tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        roberta = self.roberta_tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": deberta["input_ids"].squeeze(0),
            "attention_mask": deberta["attention_mask"].squeeze(0),
            "r_input_ids": roberta["input_ids"].squeeze(0),
            "r_attention_mask": roberta["attention_mask"].squeeze(0),
        }


class OfficialDTFN(nn.Module):
    """Exact Task C fusion head used in the authors' public scripts."""

    def __init__(self, deberta_model: nn.Module, roberta_model: nn.Module):
        super().__init__()
        self.d_model = deberta_model
        self.r_model = roberta_model
        hidden_size = (
            self.d_model.config.hidden_size + self.r_model.config.hidden_size
        )
        self.combine_head = nn.Linear(hidden_size, NUM_TASK_C_LABELS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        r_input_ids: torch.Tensor,
        r_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        deberta_hidden = self.d_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        roberta_hidden = self.r_model(
            input_ids=r_input_ids,
            attention_mask=r_attention_mask,
        ).last_hidden_state
        combined_cls = torch.cat(
            (deberta_hidden[:, 0, :], roberta_hidden[:, 0, :]),
            dim=1,
        )
        return self.combine_head(combined_cls).view(-1, NUM_TASK_C_LABELS)


def read_edos_task_c_test(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "rewire_id",
        "text",
        "label_sexist",
        "label_vector",
        "split",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"EDOS CSV missing columns: {sorted(missing)}")

    frame = frame[
        (frame["split"].astype(str).str.strip() == "test")
        & (frame["label_sexist"].astype(str).str.strip() == "sexist")
    ].copy()
    frame["label_vector"] = frame["label_vector"].astype(str).str.strip()
    unknown = sorted(set(frame["label_vector"]) - set(TASK_C_LABELS))
    if unknown:
        raise ValueError(f"Unknown EDOS Task C labels: {unknown}")
    if len(frame) != 970:
        raise ValueError(
            "Expected the official 970 sexist Task C test examples, "
            f"but found {len(frame)}. Check the EDOS file and filtering."
        )

    label2id = {label: index for index, label in enumerate(TASK_C_LABELS)}
    return pd.DataFrame(
        {
            "row_id": np.arange(len(frame), dtype=int),
            "instance_id": frame["rewire_id"].astype(str).values,
            "text": frame["text"].fillna("").astype(str).values,
            "gold_id": frame["label_vector"].map(label2id).astype(int).values,
            "gold_label": frame["label_vector"].values,
        }
    )


def read_conan_hate_speech(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("CONAN input must be a JSON object keyed by instance id.")

    rows = []
    for row_id, (instance_id, item) in enumerate(raw.items()):
        if not isinstance(item, dict):
            raise ValueError(f"CONAN item {instance_id!r} is not an object.")
        text = item.get("HATE_SPEECH")
        if text is None or not str(text).strip():
            raise ValueError(f"CONAN item {instance_id!r} has empty HATE_SPEECH.")
        rows.append(
            {
                "row_id": row_id,
                "instance_id": str(instance_id),
                "text": str(text),
            }
        )
    return pd.DataFrame(rows)


def resolve_checkpoint(checkpoint_root: Path, model_key: str) -> Path:
    spec = OFFICIAL_TASK_C_MODELS[model_key]
    checkpoint_root = checkpoint_root.expanduser().resolve()
    candidates = [
        checkpoint_root / spec.repo_id.split("/")[-1] / spec.filename,
        checkpoint_root / spec.filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Checkpoint for {model_key!r} not found. Tried:\n{rendered}\n"
        "Download it with:\n"
        "  python scripts/11_download_official_task_c.py "
        f'--output-dir "{checkpoint_root}" --models {model_key}'
    )


def load_model_state_dict(
    checkpoint_path: Path,
    allow_unsafe_checkpoint_load: bool,
) -> Mapping[str, torch.Tensor]:
    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=True,
            mmap=True,
            **load_kwargs,
        )
    except Exception as safe_error:
        if not allow_unsafe_checkpoint_load:
            raise RuntimeError(
                "Safe checkpoint loading failed. If and only if you trust the "
                "downloaded authors' checkpoint, rerun with "
                "--allow-unsafe-checkpoint-load. Original error: "
                f"{safe_error}"
            ) from safe_error
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
            mmap=True,
            **load_kwargs,
        )

    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not an authors-style training checkpoint "
            "containing model_state_dict."
        )
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("model_state_dict is not a mapping.")
    return state_dict


def _load_mistral_model():
    try:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as error:
        raise ImportError(
            "Mistral inference requires peft and bitsandbytes; install "
            "requirements-official.txt on a CUDA server."
        ) from error

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MISTRAL_BASE,
        num_labels=NUM_TASK_C_LABELS,
        quantization_config=quantization,
        device_map={"": 0},
    )
    model.config.pad_token_id = model.config.eos_token_id
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, lora_config)


def build_model_and_tokenizers(model_key: str):
    spec = OFFICIAL_TASK_C_MODELS[model_key]
    if spec.architecture == "deberta_seqcls":
        tokenizer = AutoTokenizer.from_pretrained(DEBERTA_BASE)
        model = AutoModelForSequenceClassification.from_pretrained(
            DEBERTA_BASE,
            num_labels=NUM_TASK_C_LABELS,
            ignore_mismatched_sizes=True,
        )
        return model, (tokenizer,)

    if spec.architecture == "dtfn":
        deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_BASE)
        roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE)
        deberta = AutoModel.from_pretrained(DEBERTA_BASE)
        roberta = AutoModel.from_pretrained(ROBERTA_BASE)
        return OfficialDTFN(deberta, roberta), (
            deberta_tokenizer,
            roberta_tokenizer,
        )

    if spec.architecture == "mistral_qlora":
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_BASE)
        tokenizer.pad_token = tokenizer.eos_token
        return _load_mistral_model(), (tokenizer,)

    raise ValueError(f"Unsupported architecture: {spec.architecture}")


def load_official_model(
    model_key: str,
    checkpoint_path: Path,
    device: torch.device,
    allow_unsafe_checkpoint_load: bool,
):
    model, tokenizers = build_model_and_tokenizers(model_key)
    state_dict = load_model_state_dict(
        checkpoint_path,
        allow_unsafe_checkpoint_load=allow_unsafe_checkpoint_load,
    )
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint did not match the official architecture. "
            f"Missing: {incompatible.missing_keys}; "
            f"unexpected: {incompatible.unexpected_keys}"
        )
    if model_key != FALLBACK_MODEL:
        model.to(device)
    model.eval()
    return model, tokenizers


def _model_input_device(model: nn.Module, requested: torch.device) -> torch.device:
    if hasattr(model, "hf_device_map"):
        return torch.device("cuda:0")
    return requested


def predict_probabilities(
    frame: pd.DataFrame,
    model_key: str,
    model: nn.Module,
    tokenizers: Tuple,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    if OFFICIAL_TASK_C_MODELS[model_key].architecture == "dtfn":
        dataset = DualTokenizerDataset(
            frame["text"].tolist(),
            tokenizers[0],
            tokenizers[1],
            max_length=max_length,
        )
    else:
        dataset = SingleTokenizerDataset(
            frame["text"].tolist(),
            tokenizers[0],
            max_length=max_length,
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    input_device = _model_input_device(model, device)
    probability_batches: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(input_device) for key, value in batch.items()}
            if OFFICIAL_TASK_C_MODELS[model_key].architecture == "dtfn":
                logits = model(**batch)
            else:
                logits = model(**batch).logits
            probabilities = torch.softmax(logits.float(), dim=-1)
            probability_batches.append(probabilities.cpu().numpy())

    if not probability_batches:
        return np.empty((0, NUM_TASK_C_LABELS), dtype=np.float32)
    return np.concatenate(probability_batches, axis=0)


def prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    model_key: str,
    checkpoint_path: Path,
) -> pd.DataFrame:
    if probabilities.shape != (len(source), NUM_TASK_C_LABELS):
        raise ValueError(
            "Unexpected probability matrix shape: "
            f"{probabilities.shape}; expected {(len(source), NUM_TASK_C_LABELS)}."
        )
    pred_ids = probabilities.argmax(axis=1)
    output = source.copy()
    output["model_key"] = model_key
    output["model_repo"] = OFFICIAL_TASK_C_MODELS[model_key].repo_id
    output["checkpoint"] = str(checkpoint_path)
    output["pred_id"] = pred_ids
    output["label_pred"] = [TASK_C_LABELS[index] for index in pred_ids]
    output["confidence"] = probabilities.max(axis=1)
    for label_id in range(NUM_TASK_C_LABELS):
        output[f"prob_{label_id}"] = probabilities[:, label_id]
    return output



