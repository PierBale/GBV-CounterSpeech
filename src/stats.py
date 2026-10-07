"""Aggregate pairwise JudgeLM outcomes and automatic metric files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev, variance
from typing import Iterable, Sequence

from config import (
    AUTOMATIC_METRICS_OUTPUT_DIR,
    GENERATION_OUTPUT_DIR,
    JUDGE_OUTPUT_DIR,
    PROJECT_DIR,
)
from unito_amazon.evaluation.judge import LLMJudge
from unito_amazon.evaluation.metrics import (
    AUTOMATIC_METRIC_FIELDS,
    summarize_automatic_metrics,
)


VALID_WINNERS = {"response_1", "response_2", "tie"}


def discover_paths(
    requested: Sequence[Path] | None,
    default_directories: Sequence[Path],
    pattern: str,
) -> list[Path]:
    locations = list(requested) if requested is not None else list(default_directories)
    candidates = []
    for location in locations:
        location = location.expanduser()
        if location.is_file():
            candidates.append(location)
        elif location.is_dir():
            candidates.extend(sorted(location.rglob(pattern)))
        elif requested is not None:
            raise FileNotFoundError(f"Statistics input path not found: {location}")

    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def read_jsonl_files(paths: Sequence[Path]) -> list[dict]:
    records = []
    for path in paths:
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
    return records


def _pairwise_result(record: dict) -> dict:
    result = record.get("pairwise_judge")
    if not isinstance(result, dict):
        return {}
    if result.get("winner") in VALID_WINNERS and not result.get("parse_error"):
        return result

    raw_response = result.get("raw_response")
    if not isinstance(raw_response, str) or not raw_response.strip():
        return result
    parsed = LLMJudge.parse_judgment(raw_response)
    if parsed is None:
        return result

    winner, score_1, score_2 = parsed
    return {
        **result,
        "winner": winner,
        "response_1_score": score_1,
        "response_2_score": score_2,
        "parse_error": None,
        "reparsed_from_raw_response": True,
    }


def has_valid_pairwise_judgment(record: dict) -> bool:
    result = _pairwise_result(record)
    return result.get("winner") in VALID_WINNERS and not result.get("parse_error")


def summarize_pairwise(records: Sequence[dict]) -> dict:
    results = [_pairwise_result(record) for record in records]
    valid = [
        result
        for result in results
        if result.get("winner") in VALID_WINNERS and not result.get("parse_error")
    ]
    winners = [result["winner"] for result in valid]
    valid_count = len(valid)
    challenger_wins = winners.count("response_2")
    baseline_wins = winners.count("response_1")
    ties = winners.count("tie")
    return {
        "total_comparisons": len(records),
        "valid_comparisons": valid_count,
        "parse_failures": len(records) - valid_count,
        "reparsed_from_raw_response": sum(
            bool(result.get("reparsed_from_raw_response")) for result in valid
        ),
        "challenger_wins": challenger_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "rag_win_percentage": _percentage(challenger_wins, valid_count),
        "baseline_win_percentage": _percentage(baseline_wins, valid_count),
        "tie_percentage": _percentage(ties, valid_count),
    }


def summarize_metrics(records: Sequence[dict]) -> dict:
    generations = _unique_generations(records)
    summary = summarize_automatic_metrics(generations, compute_corpus_bleu=False)
    summary["unique_generations"] = len(generations)
    summary["metric_statistics"] = {
        field: _descriptive_statistics(
            record.get("automatic_metrics", {}).get(field)
            for record in generations
            if isinstance(record.get("automatic_metrics"), dict)
            and record["automatic_metrics"].get(field) is not None
        )
        for field in AUTOMATIC_METRIC_FIELDS
    }
    return summary


def _generation_key(record: dict) -> tuple:
    return (
        record.get("id"),
        record.get("source"),
        record.get("generation_model"),
        record.get("generation_strategy"),
        record.get("generated_counter_speech"),
    )


def _unique_generations(records: Sequence[dict]) -> list[dict]:
    unique = {}
    for record in records:
        key = _generation_key(record)
        current = unique.get(key)
        if current is None or (
            not current.get("automatic_metrics")
            and record.get("automatic_metrics")
        ):
            unique[key] = record
    return list(unique.values())


def _descriptive_statistics(values: Iterable[int | float]) -> dict:
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": _round(fmean(numeric)) if numeric else None,
        "variance": _round(variance(numeric)) if len(numeric) > 1 else None,
        "std": _round(stdev(numeric)) if len(numeric) > 1 else None,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
    }


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _percentage(count: int, total: int) -> float | None:
    return round(100 * count / total, 2) if total else None


def _group_records(
    records: Sequence[dict],
    fields: Sequence[str],
) -> dict[tuple[str, ...], list[dict]]:
    groups = defaultdict(list)
    for record in records:
        key = tuple(str(record.get(field) or "unknown") for field in fields)
        groups[key].append(record)
    return groups


def grouped_pairwise(
    records: Sequence[dict],
    fields: Sequence[str],
) -> list[dict]:
    output = []
    for key, group in sorted(_group_records(records, fields).items()):
        item = dict(zip(fields, key, strict=True))
        item.update(summarize_pairwise(group))
        output.append(item)
    return output


def grouped_metrics(
    records: Sequence[dict],
    fields: Sequence[str],
) -> list[dict]:
    output = []
    for key, group in sorted(_group_records(records, fields).items()):
        item = dict(zip(fields, key, strict=True))
        item.update(summarize_metrics(group))
        output.append(item)
    return output


def build_report(
    judge_records: Sequence[dict],
    metric_records: Sequence[dict],
    judge_paths: Sequence[Path],
    metric_paths: Sequence[Path],
) -> dict:
    if judge_records and not any(
        isinstance(record.get("pairwise_judge"), dict) for record in judge_records
    ):
        raise ValueError(
            "The selected judge file uses the old pointwise protocol. Run the "
            "new pairwise judge or select a *.pairwise-judged.*.jsonl file."
        )

    pairwise_detailed = grouped_pairwise(
        judge_records,
        (
            "generation_model",
            "baseline_strategy",
            "challenger_strategy",
            "judge_model",
        ),
    )
    metric_systems = grouped_metrics(
        metric_records,
        ("generation_model", "generation_strategy"),
    )
    report = {
        "protocol": "JudgeLM pairwise RAG vs No-RAG and MultiCONAN reference",
        "input_files": {
            "pairwise_judge": [str(path) for path in judge_paths],
            "automatic_metrics": [str(path) for path in metric_paths],
        },
        "overall": {
            "pairwise_judge": summarize_pairwise(judge_records),
            "automatic_metrics": summarize_metrics(metric_records),
        },
        "pairwise_judge": {
            "detailed": pairwise_detailed,
            "by_generation_model_and_challenger": grouped_pairwise(
                judge_records,
                (
                    "generation_model",
                    "baseline_strategy",
                    "challenger_strategy",
                ),
            ),
        },
        "automatic_metrics": {
            "by_system": metric_systems,
        },
    }
    report["systems"] = _combine_systems(report)
    report["comparison_tables"] = {
        "automatic_metrics": _automatic_comparison_rows(metric_systems),
        "judgelm_vs_no_rag": _pairwise_matrix(
            report["pairwise_judge"]["by_generation_model_and_challenger"],
            "without_rag",
        ),
        "judgelm_vs_multiconan_reference": _pairwise_matrix(
            report["pairwise_judge"]["by_generation_model_and_challenger"],
            "multiconan_reference",
        ),
    }
    return report


def _automatic_comparison_rows(metric_systems: Sequence[dict]) -> list[dict]:
    metrics = (
        ("BLEU-4", "bleu_4"),
        ("METEOR", "meteor"),
        ("ROUGE-L", "rouge_l"),
        ("BERTScore-F1", "bertscore_f1"),
        ("BERTScore-F1-rescaled", "bertscore_rescaled_f1"),
        ("Distinct-1", "distinct_1"),
        ("Distinct-2", "distinct_2"),
        ("Repetition Rate", "repetition_rate"),
    )
    rows = []
    for system in metric_systems:
        metric_stats = system["metric_statistics"]
        row = {
            "Model": system["generation_model"],
            "System": _strategy_label(system["generation_strategy"]),
        }
        for label, field in metrics:
            row[f"{label} mean"] = metric_stats[field]["mean"]
            row[f"{label} variance"] = metric_stats[field]["variance"]
        rows.append(row)
    return rows


def _pairwise_matrix(rows: Sequence[dict], baseline: str) -> dict:
    selected = [row for row in rows if row.get("baseline_strategy") == baseline]
    strategies = sorted(
        {row["challenger_strategy"] for row in selected},
        key=_strategy_sort_key,
    )
    models = sorted({row["generation_model"] for row in selected})
    lookup = {
        (row["generation_model"], row["challenger_strategy"]): row
        for row in selected
    }
    matrix_rows = []
    for model in models:
        cells = {}
        for strategy in strategies:
            result = lookup.get((model, strategy))
            cells[strategy] = _pairwise_cell(result) if result else None
        matrix_rows.append({"Model": model, "cells": cells})
    return {
        "baseline_strategy": baseline,
        "columns": [
            {"strategy": strategy, "label": _strategy_label(strategy)}
            for strategy in strategies
        ],
        "rows": matrix_rows,
    }


def _pairwise_cell(summary: dict) -> dict:
    challenger_wins = summary["challenger_wins"]
    baseline_wins = summary["baseline_wins"]
    valid = summary["valid_comparisons"]
    if not valid:
        result = "n/a"
    elif challenger_wins < baseline_wins:
        result = f"{challenger_wins} (lost)"
    elif challenger_wins == baseline_wins:
        result = f"{challenger_wins} (equal wins)"
    else:
        result = str(challenger_wins)

    percentages = (
        f"RAG { _format_percentage(summary['rag_win_percentage'])}; "
        f"baseline { _format_percentage(summary['baseline_win_percentage'])}; "
        f"tie { _format_percentage(summary['tie_percentage'])}"
    )
    return {
        "display": f"{result} - {percentages}" if valid else result,
        "rag_wins": challenger_wins,
        "baseline_wins": baseline_wins,
        "ties": summary["ties"],
        "rag_win_percentage": summary["rag_win_percentage"],
        "baseline_win_percentage": summary["baseline_win_percentage"],
        "tie_percentage": summary["tie_percentage"],
        "valid_comparisons": valid,
        "reparsed_from_raw_response": summary["reparsed_from_raw_response"],
        "total_comparisons": summary["total_comparisons"],
    }


def _format_percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _strategy_label(strategy: str) -> str:
    labels = {
        "without_rag": "No-RAG",
        "multiconan_reference": "MultiCONAN",
        "rag_bm25": "BM25",
        "rag_qwen3_emb_0.6b": "Qwen3-Emb-0.6B",
        "rag_sbert": "SentBERT",
        "rag_bge_m3": "BGE-M3",
    }
    return labels.get(strategy, strategy.removeprefix("rag_").replace("_", " "))


def _strategy_sort_key(strategy: str) -> tuple[int, str]:
    order = {
        "rag_bm25": 0,
        "rag_sbert": 1,
        "rag_qwen3_emb_0.6b": 2,
        "rag_bge_m3": 3,
    }
    return order.get(strategy, 100), strategy


def _combine_systems(report: dict) -> list[dict]:
    pairwise: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in report["pairwise_judge"]["by_generation_model_and_challenger"]:
        pairwise[(row["generation_model"], row["challenger_strategy"])].append(row)

    systems = []
    for metric_row in report["automatic_metrics"]["by_system"]:
        key = (metric_row["generation_model"], metric_row["generation_strategy"])
        systems.append(
            {
                "generation_model": key[0],
                "generation_strategy": key[1],
                "pairwise_comparisons": pairwise.get(key, []),
                "automatic_metrics": metric_row,
            }
        )
    return systems


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_statistics.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path = output_dir / "evaluation_statistics.md"
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")

    pairwise_rows = [
        _pairwise_csv_row(row) for row in report["pairwise_judge"]["detailed"]
    ]
    metric_rows = [
        _metric_csv_row(row) for row in report["automatic_metrics"]["by_system"]
    ]
    system_rows = [_system_csv_row(row) for row in report["systems"]]
    comparison_tables = report["comparison_tables"]
    tables = {
        "judgelm_pairwise.csv": pairwise_rows,
        "llm_judge_detailed.csv": pairwise_rows,
        "automatic_metrics_detailed.csv": metric_rows,
        "system_statistics.csv": system_rows,
        "combined_detailed.csv": system_rows,
        "automatic_metrics_comparison.csv": comparison_tables[
            "automatic_metrics"
        ],
        "judgelm_vs_no_rag.csv": _pairwise_matrix_csv_rows(
            comparison_tables["judgelm_vs_no_rag"]
        ),
        "judgelm_vs_multiconan_reference.csv": _pairwise_matrix_csv_rows(
            comparison_tables["judgelm_vs_multiconan_reference"]
        ),
    }
    outputs = []
    for filename, rows in tables.items():
        path = output_dir / filename
        _write_csv(path, rows)
        outputs.append(path)
    return json_path, markdown_path, outputs


def _pairwise_csv_row(summary: dict) -> dict:
    return {
        field: summary.get(field)
        for field in (
            "generation_model",
            "baseline_strategy",
            "challenger_strategy",
            "judge_model",
            "total_comparisons",
            "valid_comparisons",
            "parse_failures",
            "reparsed_from_raw_response",
            "challenger_wins",
            "rag_win_percentage",
            "baseline_wins",
            "baseline_win_percentage",
            "ties",
            "tie_percentage",
        )
    }


def _metric_csv_row(summary: dict) -> dict:
    row = {
        "generation_model": summary.get("generation_model"),
        "generation_strategy": summary.get("generation_strategy"),
        "unique_generations": summary["unique_generations"],
    }
    row.update(summary["reference_based"])
    row.update(summary["reference_free"])
    for field, values in summary["metric_statistics"].items():
        row[f"metric_{field}_count"] = values["count"]
        row[f"metric_{field}_mean"] = values["mean"]
        row[f"metric_{field}_variance"] = values["variance"]
        row[f"metric_{field}_std"] = values["std"]
    return row


def _system_csv_row(system: dict) -> dict:
    row = _metric_csv_row(system["automatic_metrics"])
    comparisons = {
        item["baseline_strategy"]: item
        for item in system.get("pairwise_comparisons", [])
    }
    for baseline in ("without_rag", "multiconan_reference"):
        comparison = comparisons.get(baseline)
        for field in (
            "total_comparisons",
            "valid_comparisons",
            "parse_failures",
            "reparsed_from_raw_response",
            "challenger_wins",
            "rag_win_percentage",
            "baseline_wins",
            "baseline_win_percentage",
            "ties",
            "tie_percentage",
        ):
            row[f"judgelm_vs_{baseline}_{field}"] = (
                comparison.get(field) if comparison else None
            )
    return row


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pairwise_matrix_csv_rows(matrix: dict) -> list[dict]:
    columns = matrix["columns"]
    return [
        {
            "Model": row["Model"],
            **{
                column["label"]: (
                    row["cells"][column["strategy"]]["display"]
                    if row["cells"][column["strategy"]]
                    else None
                )
                for column in columns
            },
        }
        for row in matrix["rows"]
    ]


def _markdown_pairwise_matrix(matrix: dict) -> str:
    columns = matrix["columns"]
    return _markdown_table(
        ("Model", *(column["label"] for column in columns)),
        [
            (
                row["Model"],
                *(
                    row["cells"][column["strategy"]]["display"]
                    if row["cells"][column["strategy"]]
                    else None
                    for column in columns
                ),
            )
            for row in matrix["rows"]
        ],
    )


def _pairwise_matrix_note(matrix: dict) -> str:
    counts = {
        cell["valid_comparisons"]
        for row in matrix["rows"]
        for cell in row["cells"].values()
        if cell is not None
    }
    if len(counts) == 1:
        comparison_scope = f"out of {counts.pop()} valid comparisons per cell"
    else:
        comparison_scope = "over the valid comparisons in each cell"
    return (
        f"Cells report RAG wins {comparison_scope}, followed by RAG, baseline "
        "and tie percentages over valid judgments. `(lost)` means the baseline "
        "won more comparisons; `(equal wins)` means equal RAG and baseline wins."
    )


def _markdown_report(report: dict) -> str:
    pairwise = report["overall"]["pairwise_judge"]
    metrics = report["overall"]["automatic_metrics"]
    comparisons = report["comparison_tables"]
    automatic_rows = comparisons["automatic_metrics"]
    automatic_headers = tuple(automatic_rows[0]) if automatic_rows else ("Model",)
    lines = [
        "# Evaluation statistics",
        "",
        "JudgeLM protocol: pairwise RAG vs No-RAG and MultiCONAN reference, following the paper prompt.",
        "Response 1 is the No-RAG or MultiCONAN baseline; Response 2 is the RAG challenger.",
        "Automatic-metric variance is the sample variance (denominator `n - 1`).",
        "",
        "## Overview",
        "",
        f"- Pairwise judge files: {len(report['input_files']['pairwise_judge'])}",
        f"- Automatic-metric files: {len(report['input_files']['automatic_metrics'])}",
        f"- Valid pairwise comparisons: {pairwise['valid_comparisons']} / {pairwise['total_comparisons']}",
        f"- Pairwise parse failures: {pairwise['parse_failures']}",
        (
            "- Recovered from saved raw responses: "
            f"{pairwise['reparsed_from_raw_response']}"
        ),
        f"- Unique generations with metrics: {metrics['unique_generations']}",
        "",
        "## Automatic metrics comparison",
        "",
        (
            "Reference-based metrics are populated only for records sourced "
            "from MultiCONAN; reference-free metrics are reported for every "
            "system. Each metric has separate mean and sample-variance columns."
        ),
        "",
        _markdown_table(
            automatic_headers,
            [
                tuple(row.get(header) for header in automatic_headers)
                for row in automatic_rows
            ],
        ),
        "",
        "## JudgeLM wins: RAG vs No-RAG",
        "",
        _pairwise_matrix_note(comparisons["judgelm_vs_no_rag"]),
        "",
        _markdown_pairwise_matrix(comparisons["judgelm_vs_no_rag"]),
        "",
        "## JudgeLM wins: RAG vs MultiCONAN reference",
        "",
        _pairwise_matrix_note(
            comparisons["judgelm_vs_multiconan_reference"]
        ),
        "",
        _markdown_pairwise_matrix(
            comparisons["judgelm_vs_multiconan_reference"]
        ),
        "",
        "## JudgeLM pairwise results",
        "",
        _markdown_table(
            (
                "Generator",
                "Baseline",
                "RAG challenger",
                "Judge",
                "Valid / total",
                "Reparsed",
                "RAG wins",
                "RAG win %",
                "Baseline wins",
                "Baseline win %",
                "Ties",
                "Tie %",
            ),
            [
                (
                    row["generation_model"],
                    row["baseline_strategy"],
                    row["challenger_strategy"],
                    row["judge_model"],
                    f"{row['valid_comparisons']} / {row['total_comparisons']}",
                    row["reparsed_from_raw_response"],
                    row["challenger_wins"],
                    _format_percentage(row["rag_win_percentage"]),
                    row["baseline_wins"],
                    _format_percentage(row["baseline_win_percentage"]),
                    row["ties"],
                    _format_percentage(row["tie_percentage"]),
                )
                for row in report["pairwise_judge"]["detailed"]
            ],
        ),
        "",
        "## Automatic metrics by system",
    ]

    for system in report["systems"]:
        model = system["generation_model"]
        strategy = system["generation_strategy"]
        metric_summary = system["automatic_metrics"]
        lines.extend(
            [
                "",
                f"### {_markdown_value(model)} - {_markdown_value(strategy)}",
                "",
                _markdown_table(
                    ("Metric", "n", "Mean", "Variance", "Std. dev."),
                    [
                        (
                            field,
                            values["count"],
                            values["mean"],
                            values["variance"],
                            values["std"],
                        )
                        for field, values in metric_summary[
                            "metric_statistics"
                        ].items()
                        if values["count"]
                    ],
                ),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> str:
    if not rows:
        return "_No data available._"
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_markdown_value(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body))


def _markdown_value(value: object) -> str:
    if value is None:
        return "n/a"
    text = f"{value:.6f}" if isinstance(value, float) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def print_report(report: dict) -> None:
    pairwise = report["overall"]["pairwise_judge"]
    metrics = report["overall"]["automatic_metrics"]
    print(
        f"Input files: {len(report['input_files']['pairwise_judge'])} pairwise "
        f"judge, {len(report['input_files']['automatic_metrics'])} metrics"
    )
    print(
        f"Valid pairwise comparisons: {pairwise['valid_comparisons']}/"
        f"{pairwise['total_comparisons']} | "
        f"parse failures: {pairwise['parse_failures']} | "
        f"reparsed: {pairwise['reparsed_from_raw_response']}"
    )
    print(f"Unique metric generations: {metrics['unique_generations']}")

    print("\nJudgeLM pairwise results")
    pairwise_rows = [
        (
            row["generation_model"],
            row["baseline_strategy"],
            row["challenger_strategy"],
            row["judge_model"],
            f"{row['valid_comparisons']}/{row['total_comparisons']}",
            str(row["challenger_wins"]),
            _format_percentage(row["rag_win_percentage"]),
            str(row["baseline_wins"]),
            _format_percentage(row["baseline_win_percentage"]),
            str(row["ties"]),
            _format_percentage(row["tie_percentage"]),
        )
        for row in report["pairwise_judge"]["detailed"]
    ]
    _print_table(
        (
            "generation_model",
            "baseline",
            "RAG_challenger",
            "judge_model",
            "valid/total",
            "RAG_wins",
            "RAG_win_%",
            "baseline_wins",
            "baseline_win_%",
            "ties",
            "tie_%",
        ),
        pairwise_rows,
    )

    print("\nAutomatic metrics by system (sample variance)")
    metric_rows = []
    for system in report["systems"]:
        for field, values in system["automatic_metrics"][
            "metric_statistics"
        ].items():
            if values["count"]:
                metric_rows.append(
                    (
                        system["generation_model"],
                        system["generation_strategy"],
                        field,
                        str(values["count"]),
                        _number(values["mean"]),
                        _number(values["variance"]),
                    )
                )
    _print_table(
        (
            "generation_model",
            "generation_strategy",
            "metric",
            "n",
            "mean",
            "variance",
        ),
        metric_rows,
    )


def _number(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(row)
        )

    print(format_row(headers))
    print(format_row(tuple("-" * width for width in widths)))
    for row in rows:
        print(format_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine pairwise JudgeLM and automatic-metric JSONL files."
    )
    parser.add_argument(
        "--judge-paths",
        nargs="*",
        type=Path,
        default=None,
        help="Pairwise judge JSONL files/directories (default: data/judged).",
    )
    parser.add_argument(
        "--metric-paths",
        nargs="*",
        type=Path,
        default=None,
        help="Metric JSONL files/directories (default: data/metrics).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(PROJECT_DIR) / "data" / "stats",
        help="Directory for JSON, CSV and Markdown reports (default: data/stats).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the report without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_paths = discover_paths(
        args.judge_paths,
        [Path(JUDGE_OUTPUT_DIR), Path(GENERATION_OUTPUT_DIR) / "judged"],
        "*.pairwise-judged.*.jsonl",
    )
    metric_paths = discover_paths(
        args.metric_paths,
        [Path(AUTOMATIC_METRICS_OUTPUT_DIR)],
        "*.metrics.jsonl",
    )
    if not judge_paths and not metric_paths:
        raise FileNotFoundError("No pairwise-judge or automatic-metric files found.")

    report = build_report(
        read_jsonl_files(judge_paths),
        read_jsonl_files(metric_paths),
        judge_paths,
        metric_paths,
    )
    print_report(report)
    if not args.no_save:
        json_path, markdown_path, csv_paths = write_report(report, args.output_dir)
        print(f"\nJSON report: {json_path.resolve()}")
        print(f"Markdown report: {markdown_path.resolve()}")
        print(f"CSV reports: {len(csv_paths)} files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
