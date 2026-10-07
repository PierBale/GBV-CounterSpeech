"""Run or resume the paper's pairwise JudgeLM evaluation."""

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


BASELINE_STRATEGY = "without_rag"
VALID_WINNERS = {"response_1", "response_2", "tie"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Judge input file not found: {path}")

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
        raise ValueError(f"No records found in judge input: {path}")
    return records


def batched(records: list[dict], batch_size: int) -> Iterable[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def build_pairwise_comparisons(records: list[dict]) -> list[dict]:
    """Pair each RAG output with No-RAG for the same item and generator."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    group_order = []
    for record in records:
        key = (
            record.get("id"),
            record.get("source"),
            record.get("generation_model"),
        )
        if key not in groups:
            group_order.append(key)
        groups[key].append(record)

    comparisons = []
    for key in group_order:
        group = groups[key]
        baselines = [
            record
            for record in group
            if record.get("generation_strategy") == BASELINE_STRATEGY
        ]
        if len(baselines) != 1:
            raise ValueError(
                f"Expected exactly one {BASELINE_STRATEGY!r} output for {key}, "
                f"found {len(baselines)}."
            )
        baseline = baselines[0]
        challengers = [
            record
            for record in group
            if record.get("generation_strategy") != BASELINE_STRATEGY
        ]
        comparison_baselines = [
            (BASELINE_STRATEGY, baseline.get("generated_counter_speech"))
        ]
        reference = baseline.get("reference_counter_speech")
        if isinstance(reference, (list, tuple)):
            reference = reference[0] if reference else None
        if (
            str(baseline.get("source") or "").strip().casefold() == "multiconan"
            and str(reference or "").strip()
        ):
            comparison_baselines.append(("multiconan_reference", reference))

        for challenger in challengers:
            if challenger.get("hateful_message") != baseline.get("hateful_message"):
                raise ValueError(f"Mismatched hateful messages for comparison {key}.")
            for baseline_strategy, baseline_response in comparison_baselines:
                comparisons.append(
                    {
                        "id": baseline.get("id"),
                        "source": baseline.get("source"),
                        "split": baseline.get("split"),
                        "target": baseline.get("target"),
                        "edos_label": baseline.get("edos_label"),
                        "generation_model": baseline.get("generation_model"),
                        "baseline_strategy": baseline_strategy,
                        "challenger_strategy": challenger.get("generation_strategy"),
                        "hateful_message": baseline.get("hateful_message"),
                        "response_1_strategy": baseline_strategy,
                        "response_1": baseline_response,
                        "response_2_strategy": challenger.get("generation_strategy"),
                        "response_2": challenger.get("generated_counter_speech"),
                    }
                )

    if not comparisons:
        raise ValueError("No paper-style pairwise comparisons could be built.")
    return comparisons


def comparison_key(record: dict) -> tuple:
    return (
        record.get("id"),
        record.get("source"),
        record.get("generation_model"),
        record.get("baseline_strategy"),
        record.get("challenger_strategy"),
        record.get("response_1"),
        record.get("response_2"),
    )


def has_pairwise_judgment(record: dict) -> bool:
    result = record.get("pairwise_judge")
    return (
        isinstance(result, dict)
        and result.get("winner") in VALID_WINNERS
        and not result.get("parse_error")
    )


def output_path_for(input_path: Path, judge_alias: str) -> Path:
    filename = f"{input_path.stem}.pairwise-judged.{judge_alias}.jsonl"
    return Path(JUDGE_OUTPUT_DIR) / filename


def read_existing_judgments(path: Path) -> list[dict]:
    return read_jsonl(path) if path.is_file() else []


def _reparse_existing_judgment(record: dict) -> dict:
    """Recover a clear winner from a previously saved raw JudgeLM response."""
    if has_pairwise_judgment(record):
        return record

    result = record.get("pairwise_judge")
    if not isinstance(result, dict):
        return record
    raw_response = result.get("raw_response")
    if not isinstance(raw_response, str) or not raw_response.strip():
        return record

    parsed = LLMJudge.parse_judgment(raw_response)
    if parsed is None:
        return record
    winner, score_1, score_2 = parsed

    repaired = dict(record)
    repaired_result = dict(result)
    repaired_result.update(
        {
            "winner": winner,
            "response_1_score": score_1,
            "response_2_score": score_2,
            "parse_error": None,
            "winner_strategy": {
                "response_1": record.get("response_1_strategy")
                or record.get("baseline_strategy"),
                "response_2": record.get("response_2_strategy")
                or record.get("challenger_strategy"),
                "tie": "tie",
            }[winner],
        }
    )
    repaired["pairwise_judge"] = repaired_result
    return repaired


def _valid_existing_map(records: list[dict]) -> dict[tuple, dict]:
    valid = {}
    for record in records:
        reparsed = _reparse_existing_judgment(record)
        if has_pairwise_judgment(reparsed):
            valid[comparison_key(reparsed)] = reparsed
    return valid


def _judged_record(comparison: dict, result, judge_name: str) -> dict:
    judged = dict(comparison)
    result_data = result.model_dump()
    winner = result_data.get("winner")
    result_data["winner_strategy"] = {
        "response_1": comparison["response_1_strategy"],
        "response_2": comparison["response_2_strategy"],
        "tie": "tie",
    }.get(winner)
    judged["pairwise_judge"] = result_data
    judged["judge_model"] = judge_name
    return judged


def build_summary(records: list[dict]) -> dict:
    valid = [record for record in records if has_pairwise_judgment(record)]
    summary = {
        "total_comparisons": len(records),
        "valid_comparisons": len(valid),
        "parse_failures": len(records) - len(valid),
        "by_generation_model_baseline_and_challenger": {},
    }
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in valid:
        groups[
            (
                str(record.get("generation_model") or "unknown"),
                str(record.get("baseline_strategy") or "unknown"),
                str(record.get("challenger_strategy") or "unknown"),
            )
        ].append(record)

    for (model, baseline, challenger), group in sorted(groups.items()):
        winners = [item["pairwise_judge"]["winner"] for item in group]
        challenger_wins = winners.count("response_2")
        summary["by_generation_model_baseline_and_challenger"][
            f"{model}|{baseline}|{challenger}"
        ] = {
            "comparisons": len(group),
            "challenger_wins": challenger_wins,
            "baseline_wins": winners.count("response_1"),
            "ties": winners.count("tie"),
            "challenger_win_rate": challenger_wins / len(group),
        }
    return summary


def write_judgments(
    *,
    judge: LLMJudge | None,
    judge_name: str,
    input_path: Path,
    output_path: Path,
    comparisons: list[dict],
    existing_records: list[dict],
) -> int:
    existing = _valid_existing_map(existing_records)
    missing = [
        comparison
        for comparison in comparisons
        if comparison_key(comparison) not in existing
    ]
    if missing and judge is None:
        raise RuntimeError("Missing comparisons require an initialized JudgeLM.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    newly_judged = {}
    completed = 0
    for batch in batched(missing, JUDGE_BATCH_SIZE):
        results = judge.judge_batch(batch)
        checkpoint_records = []
        for comparison, result in zip(batch, results, strict=True):
            judged = _judged_record(comparison, result, judge_name)
            newly_judged[comparison_key(comparison)] = judged
            checkpoint_records.append(judged)

        with output_path.open("a", encoding="utf-8") as output_file:
            for record in checkpoint_records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed += len(batch)
        print(
            f"[{judge_name}] Judged {completed}/{len(missing)} missing "
            f"pairwise comparisons from {input_path.name}."
        )

    ordered = []
    for comparison in comparisons:
        key = comparison_key(comparison)
        if key in newly_judged:
            ordered.append(newly_judged[key])
        elif key in existing:
            previous = existing[key]
            reused = dict(comparison)
            reused["pairwise_judge"] = previous["pairwise_judge"]
            reused["judge_model"] = previous.get("judge_model", judge_name)
            ordered.append(reused)
        else:
            raise RuntimeError(f"No judgment produced for comparison {key!r}.")

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in ordered:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = build_summary(ordered)
    summary.update(
        {
            "judge_model": judge_name,
            "input_file": str(input_path),
            "reused_comparisons": len(comparisons) - len(missing),
            "new_comparisons": len(missing),
            "protocol": "JudgeLM pairwise RAG vs No-RAG and MultiCONAN reference",
        }
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Pairwise judgments written to {output_path.resolve()}")
    print(f"Summary written to {summary_path.resolve()}")
    return len(missing)


def run_judge_model(
    model_alias: str,
    inputs: dict[Path, list[dict]],
) -> None:
    judge_model = resolve_decoder_model(model_alias)
    plans = []
    total_missing = 0
    for input_path, comparisons in inputs.items():
        output_path = output_path_for(input_path, judge_model.alias)
        existing_records = read_existing_judgments(output_path)
        existing = _valid_existing_map(existing_records)
        missing = sum(
            comparison_key(record) not in existing for record in comparisons
        )
        total_missing += missing
        plans.append(
            (input_path, output_path, comparisons, existing_records, missing)
        )

    if total_missing == 0:
        print(
            f"[{judge_model.paper_name}] All pairwise judgments already exist; "
            "the judge model will not be loaded."
        )
        for input_path, output_path, comparisons, existing_records, _ in plans:
            write_judgments(
                judge=None,
                judge_name=judge_model.paper_name,
                input_path=input_path,
                output_path=output_path,
                comparisons=comparisons,
                existing_records=existing_records,
            )
        return

    from unito_amazon.llm.factory import LLMFactory

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
        for input_path, output_path, comparisons, existing_records, _ in plans:
            write_judgments(
                judge=judge,
                judge_name=judge_model.paper_name,
                input_path=input_path,
                output_path=output_path,
                comparisons=comparisons,
                existing_records=existing_records,
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
    inputs = {}
    for input_path in get_judge_input_paths():
        comparisons = build_pairwise_comparisons(read_jsonl(input_path))
        if JUDGE_LIMIT > 0:
            comparisons = comparisons[:JUDGE_LIMIT]
        inputs[input_path] = comparisons
        print(
            f"Built {len(comparisons)} paper-style pairwise comparisons "
            f"from {input_path.name}."
        )
    for model_alias in JUDGE_MODELS:
        run_judge_model(model_alias, inputs)


if __name__ == "__main__":
    main()
