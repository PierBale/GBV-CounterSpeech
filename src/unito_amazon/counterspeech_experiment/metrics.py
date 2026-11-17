"""Reference-based metrics on MultiCONAN (BLEU-4, METEOR, ROUGE-L; no BERTScore).

Each generated response is scored against every MultiCONAN reference written for the
same hateful message (multi-reference, as supported by all three metrics). Every score
comes from a library, loaded with Hugging Face evaluate through
unito_amazon.evaluation.metrics._reference_metrics: "bleu" (sentence BLEU-4, smoothed),
"meteor" (NLTK) and "rouge" (rouge-score, ROUGE-L with stemming). The paired tests use scipy.
"""

from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from statistics import fmean, stdev

from scipy.stats import wilcoxon

from unito_amazon.card_creation.io import read_jsonl, write_json, write_jsonl
from unito_amazon.dataset_manager.loader import DatasetLoader
from unito_amazon.evaluation import metrics as metric_library

from .config import ExperimentConfig, strategies_for

REFERENCE_FIELDS = ("bleu_4", "meteor", "rouge_l")


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


def metric_key(record: dict) -> tuple:
    return record["id"], record["generation_strategy"], record["generated_counter_speech"]


def run_metrics(config: ExperimentConfig) -> None:
    generations = [record for record in read_jsonl(config.generations_path) if record["source"] == "multiconan"]
    if not generations:
        raise ValueError(f"No MultiCONAN generations in {config.generations_path}.")
    references = multiconan_references()
    path = config.output_dir / "metrics_multiconan.jsonl"
    existing = {metric_key(record): record for record in read_jsonl(path)} if path.exists() else {}
    missing = [record for record in generations if metric_key(record) not in existing]
    print(f"{len(generations)} MultiCONAN generations, {len(generations) - len(missing)} already scored.")

    if missing:
        candidates, reference_lists = [], []
        for record in missing:
            refs = references.get(record["hateful_message"].strip())
            if not refs:
                raise ValueError(f"No MultiCONAN reference for {record['id']}.")
            candidates.append(" ".join(record["generated_counter_speech"].split()))
            reference_lists.append(refs)
        # Lists of references: BLEU, ROUGE and METEOR score against each reference.
        scores = metric_library._reference_metrics(
            candidates, reference_lists, bertscore_model_type=None,
            bertscore_device=config.device, bertscore_batch_size=16, include_bertscore=False,
        )
        for record, refs, score in zip(missing, reference_lists, scores, strict=True):
            existing[metric_key(record)] = {
                "id": record["id"], "source": record["source"], "generation_strategy": record["generation_strategy"],
                "generation_model": record["generation_model"], "hateful_message": record["hateful_message"],
                "generated_counter_speech": record["generated_counter_speech"],
                "response_format_valid": record.get("response_format_valid"),
                "references": refs, "reference_count": len(refs),
                "metrics": score,
            }
    ordered = [existing[metric_key(record)] for record in generations]
    write_jsonl(path, ordered)
    write_json(config.output_dir / "metrics_multiconan_summary.json", summarize_metrics(ordered, config))
    print(f"Metrics written to {path}")


def summarize_metrics(records: list[dict], config: ExperimentConfig) -> dict:
    strategies = strategies_for("multiconan", config.strategies)
    by_strategy = defaultdict(dict)
    for record in records:
        by_strategy[record["generation_strategy"]][record["id"]] = record["metrics"]
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": "multiconan",
               "references": "all MultiCONAN references of the same hateful message",
               "messages": len(by_strategy[strategies[0]]),
               "mean_reference_count": fmean(record["reference_count"] for record in records),
               "strategies": {}, "paired_wilcoxon": {}}
    for strategy in strategies:
        rows = list(by_strategy[strategy].values())
        summary["strategies"][strategy] = {
            field: {"mean": fmean(values), "sd": stdev(values) if len(values) > 1 else None, "n": len(values)}
            for field in REFERENCE_FIELDS
            if (values := [row[field] for row in rows if row.get(field) is not None])
        }
    for system_a, system_b in combinations(strategies, 2):
        ids = sorted(set(by_strategy[system_a]) & set(by_strategy[system_b]))
        for field in REFERENCE_FIELDS:
            differences = [by_strategy[system_b][i][field] - by_strategy[system_a][i][field] for i in ids]
            nonzero = [d for d in differences if d != 0]
            summary["paired_wilcoxon"][f"{system_a}|{system_b}|{field}"] = {
                "mean_difference_b_minus_a": fmean(differences), "messages": len(ids),
                "p": wilcoxon(nonzero).pvalue if len(nonzero) >= 10 else None,
            }
    return summary
