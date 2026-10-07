"""Compute automatic counter-speech metrics without loading an LLM judge."""

from __future__ import annotations

import json
from pathlib import Path

from config import (
    AUTOMATIC_METRICS_INPUT_PATHS,
    AUTOMATIC_METRICS_OUTPUT_DIR,
    BERTSCORE_BATCH_SIZE,
    BERTSCORE_DEVICE,
    BERTSCORE_MODEL_TYPE,
    DATASET,
    GENERATION_MODELS,
    GENERATION_OUTPUT_DIR,
)
from unito_amazon.evaluation.metrics import (
    REFERENCE_FREE_METRIC_FIELDS,
    REFERENCE_METRIC_FIELDS,
    compute_automatic_metrics,
    summarize_automatic_metrics,
)


METRIC_RECORD_FIELDS = (
    "id",
    "source",
    "split",
    "target",
    "edos_label",
    "generation_model",
    "generation_strategy",
    "generated_counter_speech",
    "reference_counter_speech",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric input file not found: {path}")

    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}."
                )
            records.append(record)

    if not records:
        raise ValueError(f"No records found in metric input: {path}")
    return records


def record_key(record: dict) -> tuple:
    return (
        record.get("id"),
        record.get("source"),
        record.get("generation_model"),
        record.get("generation_strategy"),
        record.get("generated_counter_speech"),
    )


def has_automatic_metrics(record: dict) -> bool:
    metrics = record.get("automatic_metrics")
    if not isinstance(metrics, dict):
        return False
    if not all(field in metrics for field in REFERENCE_FREE_METRIC_FIELDS):
        return False

    is_multiconan = (
        str(record.get("source") or "").strip().casefold() == "multiconan"
    )
    reference = record.get("reference_counter_speech")
    if isinstance(reference, (list, tuple)):
        has_reference = any(str(item or "").strip() for item in reference)
    else:
        has_reference = bool(str(reference or "").strip())
    if is_multiconan and has_reference:
        return all(metrics.get(field) is not None for field in REFERENCE_METRIC_FIELDS)
    return True


def metric_output_path(input_path: Path) -> Path:
    return AUTOMATIC_METRICS_OUTPUT_DIR / f"{input_path.stem}.metrics.jsonl"


def _metric_record(record: dict) -> dict:
    output = {field: record.get(field) for field in METRIC_RECORD_FIELDS}
    output["automatic_metrics"] = record["automatic_metrics"]
    return output


def evaluate_file(input_path: Path) -> None:
    input_records = read_jsonl(input_path)
    output_path = metric_output_path(input_path)
    existing_records = read_jsonl(output_path) if output_path.is_file() else []
    existing = {
        record_key(record): record
        for record in existing_records
        if has_automatic_metrics(record)
    }
    missing = [
        record for record in input_records if record_key(record) not in existing
    ]

    computed = {}
    if missing:
        print(
            f"Calculating automatic metrics for {len(missing)}/"
            f"{len(input_records)} outputs from {input_path.name}..."
        )
        results = compute_automatic_metrics(
            missing,
            bertscore_model_type=BERTSCORE_MODEL_TYPE,
            bertscore_device=BERTSCORE_DEVICE,
            bertscore_batch_size=BERTSCORE_BATCH_SIZE,
        )
        computed = {record_key(record): record for record in results}
    else:
        print(
            f"All automatic metrics already exist for {input_path.name}; "
            "BERTScore will not be loaded."
        )

    ordered = []
    for input_record in input_records:
        key = record_key(input_record)
        if key in computed:
            ordered.append(_metric_record(computed[key]))
        else:
            ordered.append(existing[key])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in ordered:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_automatic_metrics(ordered)
    summary["input_file"] = str(input_path)
    summary["reused_metrics"] = len(input_records) - len(missing)
    summary["new_metrics"] = len(missing)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Metrics written to {output_path.resolve()}")
    print(f"Summary written to {summary_path.resolve()}")


def get_input_paths() -> tuple[Path, ...]:
    if AUTOMATIC_METRICS_INPUT_PATHS:
        return tuple(Path(path) for path in AUTOMATIC_METRICS_INPUT_PATHS)
    return tuple(
        GENERATION_OUTPUT_DIR / f"{model}_{DATASET}_counterspeech.jsonl"
        for model in GENERATION_MODELS
    )


def main() -> None:
    for input_path in get_input_paths():
        evaluate_file(input_path)


if __name__ == "__main__":
    main()
