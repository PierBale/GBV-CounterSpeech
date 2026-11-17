"""Message subset and EDOS Task C labels predicted by the official classifier."""

import json
from pathlib import Path

import pandas as pd

from unito_amazon.card_creation.io import write_json
from unito_amazon.dataset_manager.loader import DatasetLoader
from unito_amazon.dataset_manager.sampling import sample_messages

from .config import ExperimentConfig


def label_code(label: str) -> str:
    """'2.3 dehumanising attacks & ...' -> '2.3' (EDOS and card label names differ slightly)."""
    code = str(label).strip().split(" ", 1)[0]
    if not code[:1].isdigit():
        raise ValueError(f"EDOS label without a numeric code: {label!r}")
    return code


def _optional(value):
    return None if value is None or pd.isna(value) else value


def build_subset(config: ExperimentConfig) -> dict:
    """Sample unique hateful messages per dataset (EDOS: sexist test messages only).

    sample_size=None takes every unique message, shuffled with the same seed: the
    shuffle is a permutation whose first n rows equal the sample of size n, so a
    smaller earlier subset is a prefix of the full one.
    """
    items = []
    for dataset in config.datasets:
        if dataset == "edos":
            frame = DatasetLoader.load_dataset("edos", split=config.edos_split, targets=["WOMEN"])
            frame = frame[frame["label_sexist"] == "sexist"]
        elif dataset == "multiconan":
            frame = DatasetLoader.load_dataset("multiconan", targets=["WOMEN"])
        else:
            raise ValueError(f"Unknown dataset: {dataset!r}")
        available = frame["hate_speech"].astype(str).str.strip().nunique()
        size = available if config.sample_size is None else config.sample_size
        frame = sample_messages(frame, size, config.sample_seed)
        if len(frame) < size:
            raise ValueError(f"Only {len(frame)} unique messages in {dataset}.")
        for _, row in frame.iterrows():
            items.append({
                "id": row["id"], "source": row["source"], "split": _optional(row["split"]),
                "target": _optional(row["target"]), "hateful_message": row["hate_speech"],
                "gold_edos_label": _optional(row["edos_label"]),
                "reference_counter_speech": _optional(row["counter_speech"]),
            })
        print(f"Sampled {len(frame)} unique messages from {dataset} (seed {config.sample_seed}).")
    return {"sample_size_per_dataset": "all" if config.sample_size is None else config.sample_size,
            "seed": config.sample_seed, "edos_split": config.edos_split, "items": items}


def load_or_create_subset(config: ExperimentConfig) -> list[dict]:
    """The subset is written once and then reused, so every stage uses the same messages."""
    subset = build_subset(config)
    path = config.subset_path
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved == subset:
            return subset["items"]
        if not extends(saved, subset):
            raise ValueError(f"{path} differs from the configured sample; use another output_dir.")
        # Same seed, larger sample: the old messages come first, so their generations and
        # judgments are reused. The smaller subset is kept for reference.
        archive = path.with_name(f"subset_{saved['sample_size_per_dataset']}_per_dataset.json")
        write_json(archive, saved)
        write_json(path, subset)
        print(f"Subset extended from {saved['sample_size_per_dataset']} to {subset['sample_size_per_dataset']} "
              f"messages per dataset (previous subset kept in {archive.name}).")
    else:
        write_json(path, subset)
        print(f"Subset saved to {path}")
    return subset["items"]


def extends(smaller: dict, larger: dict) -> bool:
    """True if `larger` only adds messages after those of `smaller`, per dataset."""
    if not all(smaller.get(key) == larger.get(key) for key in ("seed", "edos_split")):
        return False
    for source in {item["source"] for item in larger["items"]} | {item["source"] for item in smaller["items"]}:
        old = [item for item in smaller["items"] if item["source"] == source]
        new = [item for item in larger["items"] if item["source"] == source]
        if len(new) < len(old) or new[:len(old)] != old:
            return False
    return True


def classifier_file(config: ExperimentConfig, source: str) -> Path:
    return config.classifier_dir / ("edos" if source == "edos" else "conan") / "m7fe.csv"


def attach_predicted_labels(items: list[dict], config: ExperimentConfig) -> list[dict]:
    """Add the M7-FE prediction to every message, checking that the texts match."""
    predictions = {}
    for source in sorted({item["source"] for item in items}):
        path = classifier_file(config, source)
        if not path.exists():
            raise FileNotFoundError(
                f"Classifier output not found: {path}. Run EDOS Classifier/run_official_task_c.sh "
                f"with OUTPUT_ROOT={config.classifier_dir} first.")
        frame = pd.read_csv(path, dtype={"instance_id": str}, keep_default_na=False)
        predictions[source] = {row["instance_id"]: row for _, row in frame.iterrows()}
    labelled = []
    for item in items:
        instance_id = item["id"].removeprefix("multiconan-") if item["source"] == "multiconan" else item["id"]
        row = predictions[item["source"]].get(instance_id)
        if row is None:
            raise ValueError(f"No classifier prediction for {item['source']} {item['id']}.")
        if str(row["text"]).strip() != item["hateful_message"].strip():
            raise ValueError(f"Classifier text differs from the dataset for {item['id']}.")
        labelled.append({**item, "predicted_edos_label": row["label_pred"], "classifier": {
            "method": "M7-FE official checkpoints (Khan et al., ACL 2025)",
            "file": str(path_relative(classifier_file(config, item["source"]))),
            "confidence": float(row["confidence"]), "confidence_kind": row["confidence_kind"],
            "used_fallback": str(row["used_fallback"]) == "True",
        }})
    return labelled


def path_relative(path: Path) -> Path:
    from .config import PROJECT_ROOT
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path
