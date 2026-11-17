"""Automatic metrics (EDOS and MultiCONAN), computed as in the COLING 2025 MCG shared task.

EDOS has no reference counter-speech: it gets gen_len only, the reference-based fields are None.

Same settings as evaluation/scripts/traditional_metrics.py of
https://github.com/hitz-zentroa/eval-MCG-COLING-2025 (novelty excluded), one score per system:
  rougeL     Hugging Face evaluate "rouge", use_stemmer=True, mean over the responses
  bleu       evaluate "bleu" with default settings: corpus BLEU-4, no smoothing
  bertscore  mean F1 of evaluate "bertscore" with bert-base-multilingual-cased, no rescaling
  gen_len    mean number of whitespace-separated words of the responses
Scores are rounded to 4 decimals as in the shared task. As in the shared task, each response has
a single reference: when MultiCONAN has several references for a hateful message, one is drawn at
random, seeded with the experiment seed and the message, so every system gets the same one.
"""

import random
from collections import defaultdict
from datetime import datetime, timezone
from statistics import fmean

import numpy as np

from unito_amazon.card_creation.io import read_jsonl, write_json
from unito_amazon.dataset_manager.loader import DatasetLoader

from .config import ExperimentConfig, strategies_for

BERTSCORE_MODEL = "bert-base-multilingual-cased"


def multiconan_references() -> dict[str, list[str]]:
    """Hateful message (stripped) -> all distinct MultiCONAN references, in dataset order."""
    frame = DatasetLoader.load_dataset("multiconan", targets=["WOMEN"]).sort_values("id", kind="stable")
    references = defaultdict(list)
    for _, row in frame.iterrows():
        text = " ".join(str(row["counter_speech"] or "").split())
        key = str(row["hate_speech"]).strip()
        if text and text not in references[key]:
            references[key].append(text)
    return dict(references)


def pick_reference(references: list[str], message: str, seed: int) -> str:
    """One reference per message, independent of order and identical across systems."""
    return random.Random(f"{seed}|{message}").choice(references)


def load_metrics() -> dict:
    import evaluate

    return {name: evaluate.load(name) for name in ("rouge", "bleu", "bertscore")}


def mean(values) -> float:
    return round(float(np.mean(values)), 4)


def gen_len(predictions: list[str]) -> float:
    return mean([len(prediction.split()) for prediction in predictions])


def compute_metrics(labels: list[str], predictions: list[str], loaded: dict, device: str | None) -> dict:
    """compute_metrics of the shared-task script.

    ROUGE-L is the exact mean of the per-response scores (use_aggregator=False) instead of the
    bootstrap median of evaluate's aggregator, which estimates the same mean but changes slightly
    from run to run.
    """
    rouge = loaded["rouge"].compute(predictions=predictions, references=labels, use_stemmer=True,
                                    use_aggregator=False)["rougeL"]
    bertscore = loaded["bertscore"].compute(predictions=predictions, references=labels,
                                            model_type=BERTSCORE_MODEL, device=device)["f1"]
    return {
        "rougeL": mean(rouge),
        "bleu": round(loaded["bleu"].compute(predictions=predictions, references=labels)["bleu"], 4),
        "bertscore": mean(bertscore),
        "gen_len": gen_len(predictions),
    }


def run_metrics(config: ExperimentConfig) -> None:
    """Every dataset gets gen_len; only MultiCONAN has references for ROUGE-L, BLEU and BERTScore."""
    generations = defaultdict(lambda: defaultdict(list))
    for record in read_jsonl(config.generations_path):
        generations[record["source"]][record["generation_strategy"]].append(record)
    references = multiconan_references() if "multiconan" in config.datasets else {}
    loaded = None
    datasets = {}
    for dataset in config.datasets:
        strategies = strategies_for(dataset, config.strategies)
        scores, reference_counts = {}, []
        for strategy in strategies:
            records = generations[dataset][strategy]
            if not records:
                raise ValueError(f"No {dataset} generations for {strategy} in {config.generations_path}.")
            predictions = [record["generated_counter_speech"] or "" for record in records]
            if dataset == "multiconan":
                labels, reference_counts = [], []
                for record in records:
                    message = record["hateful_message"].strip()
                    if not (refs := references.get(message)):
                        raise ValueError(f"No MultiCONAN reference for {record['id']}.")
                    labels.append(pick_reference(refs, message, config.sample_seed))
                    reference_counts.append(len(refs))
                loaded = loaded or load_metrics()
                scores[strategy] = compute_metrics(labels, predictions, loaded, config.device)
            else:
                scores[strategy] = {"rougeL": None, "bleu": None, "bertscore": None, "gen_len": gen_len(predictions)}
            scores[strategy]["n"] = len(records)
            print(f"{dataset} {strategy}: {scores[strategy]}")
        datasets[dataset] = {"messages": len(generations[dataset][strategies[0]]), "strategies": scores}
        if reference_counts:
            datasets[dataset].update({
                "messages_with_several_references": sum(count > 1 for count in reference_counts),
                "mean_available_reference_count": fmean(reference_counts),
            })

    path = config.output_dir / "metrics_summary.json"
    write_json(path, {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "COLING 2025 MCG shared task (hitz-zentroa/eval-MCG-COLING-2025, traditional_metrics.py), "
                    "without novelty",
        "references": "MultiCONAN only: one reference per hateful message, drawn at random when there are several",
        "reference_seed": config.sample_seed,
        "bertscore_model": BERTSCORE_MODEL,
        "datasets": datasets,
    })
    print(f"Metrics written to {path}")
