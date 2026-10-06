import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from config import (
    DATASET,
    GENERATION_MODELS,
    GENERATION_OUTPUT_DIR,
    JUDGE_BATCH_SIZE,
    JUDGE_DEVICE,
    JUDGE_DTYPE,
    JUDGE_INPUT_PATHS,
    JUDGE_LIMIT,
    JUDGE_MAX_RETRIES,
    JUDGE_MODELS,
    JUDGE_OUTPUT_DIR,
    JUDGE_TRUST_REMOTE_CODE,
)
from unito_amazon.evaluation.judge import LLMJudge
from unito_amazon.model_registry import resolve_decoder_model


SCORE_FIELDS = (
    "relevance",
    "respectfulness",
    "persuasiveness",
    "self_contained",
    "conciseness",
    "evidence_grounding",
    "overall",
)


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Judge input file not found: {path}")

    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if limit > 0 and len(records) >= limit:
                break

    if not records:
        raise ValueError(f"No records found in judge input: {path}")
    return records


def batched(records: list[dict], batch_size: int) -> Iterable[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def build_summary(judged_records: list[dict]) -> dict:
    valid = [
        record
        for record in judged_records
        if record["llm_judge"]["assessment"] is not None
    ]
    passed = [record for record in valid if record["llm_judge"]["passed"]]
    summary = {
        "total": len(judged_records),
        "valid_judgments": len(valid),
        "parse_failures": len(judged_records) - len(valid),
        "passed": len(passed),
        "pass_rate": len(passed) / len(valid) if valid else None,
        "mean_scores": _mean_scores(valid),
        "by_generation_model_and_strategy": {},
    }

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in valid:
        key = (
            record.get("generation_model", "unknown"),
            record.get("generation_strategy", "unknown"),
        )
        groups[key].append(record)

    for (model, strategy), group in sorted(groups.items()):
        group_passed = [item for item in group if item["llm_judge"]["passed"]]
        summary["by_generation_model_and_strategy"][f"{model}|{strategy}"] = {
            "count": len(group),
            "pass_rate": len(group_passed) / len(group),
            "mean_scores": _mean_scores(group),
        }
    return summary


def _mean_scores(records: list[dict]) -> dict:
    means = {}
    for field in SCORE_FIELDS:
        values = [
            record["llm_judge"]["assessment"][field]
            for record in records
            if record["llm_judge"]["assessment"][field] is not None
        ]
        means[field] = sum(values) / len(values) if values else None
    return means


def judge_file(
    judge: LLMJudge,
    judge_name: str,
    judge_alias: str,
    input_path: Path,
) -> None:
    records = read_jsonl(input_path, limit=JUDGE_LIMIT)
    output_path = JUDGE_OUTPUT_DIR / (
        f"{input_path.stem}.judged.{judge_alias}.jsonl"
    )
    summary_path = output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    judged_records = []
    completed = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for batch in batched(records, JUDGE_BATCH_SIZE):
            results = judge.judge_batch(batch)
            for record, result in zip(batch, results, strict=True):
                judged = dict(record)
                judged["llm_judge"] = result.model_dump()
                judged["judge_model"] = judge_name
                judged_records.append(judged)
                output_file.write(json.dumps(judged, ensure_ascii=False) + "\n")

            completed += len(batch)
            print(
                f"[{judge_name}] Judged {completed}/{len(records)} "
                f"outputs from {input_path.name}."
            )

    summary = build_summary(judged_records)
    summary["judge_model"] = judge_name
    summary["input_file"] = str(input_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Judgments written to {output_path.resolve()}")
    print(f"Summary written to {summary_path.resolve()}")


def run_judge_model(
    model_alias: str,
    input_paths: tuple[Path, ...],
) -> None:
    from unito_amazon.llm.factory import LLMFactory

    judge_model = resolve_decoder_model(model_alias)
    llm = LLMFactory.create(
        {
            "provider": "huggingface",
            "model_name": judge_model.huggingface_id,
            "device": JUDGE_DEVICE,
            "dtype": JUDGE_DTYPE,
            "trust_remote_code": JUDGE_TRUST_REMOTE_CODE,
        }
    )
    judge = LLMJudge(llm=llm, max_retries=JUDGE_MAX_RETRIES)

    try:
        for input_path in input_paths:
            judge_file(
                judge=judge,
                judge_name=judge_model.paper_name,
                judge_alias=judge_model.alias,
                input_path=input_path,
            )
    finally:
        llm.destroy()
        gc.collect()


def get_judge_input_paths() -> tuple[Path, ...]:
    if JUDGE_INPUT_PATHS:
        return tuple(Path(path) for path in JUDGE_INPUT_PATHS)
    return tuple(
        GENERATION_OUTPUT_DIR / f"{model}_{DATASET}_counterspeech.jsonl"
        for model in GENERATION_MODELS
    )


def main() -> None:
    input_paths = get_judge_input_paths()
    for model_alias in JUDGE_MODELS:
        run_judge_model(model_alias, input_paths)


if __name__ == "__main__":
    main()
