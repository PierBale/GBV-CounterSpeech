"""LaTeX pairwise outcome tables, using only already aggregated statistics."""

import re
from collections import defaultdict
from fractions import Fraction
from typing import Sequence


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "\n": " ", "\r": " ",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _strategy_label(strategy: str) -> str:
    return {
        "without_rag": "No RAG", "rag_bm25": "RAG + BM25",
        "rag_qwen3_emb_0.6b": "Chunk-RAG (Qwen3-Emb-0.6B)",
        "rag_qwen3_emb_4b": "Chunk-RAG (Qwen3-Emb-4B)",
        "rag_cards": "Card-RAG (Qwen3-Emb-4B)",
        "multiconan_reference": "MultiCONAN Reference",
    }.get(strategy, strategy)


def _sort_key(row: dict) -> tuple:
    order = {"without_rag": 0, "multiconan_reference": 1, "rag_bm25": 2,
             "rag_qwen3_emb_0.6b": 3, "rag_qwen3_emb_4b": 4, "rag_cards": 5}
    baseline, system = row["baseline_strategy"], row["challenger_strategy"]
    return order.get(baseline, 100), baseline, order.get(system, 100), system


def _outcome(count: int, valid: int) -> str:
    return rf"{count} ({100 * count / valid:.2f}\%)" if valid else f"{count} (n/a)"


def _section(title: str, rows: Sequence[dict]) -> list[str]:
    lines = [rf"\multicolumn{{6}}{{l}}{{\textit{{{latex_escape(title)}}}}} \\"]
    by_baseline = defaultdict(list)
    for row in sorted(rows, key=_sort_key):
        by_baseline[row["baseline_strategy"]].append(row)
    for position, group in enumerate(by_baseline.values()):
        if position:
            lines.append(r"\midrule")
        valid_rows = [row for row in group if row["valid_comparisons"]]
        best = max((Fraction(row["challenger_wins"], row["valid_comparisons"]) for row in valid_rows), default=None)
        for row in group:
            valid = row["valid_comparisons"]
            system_wins = _outcome(row["challenger_wins"], valid)
            if len(valid_rows) > 1 and valid and Fraction(row["challenger_wins"], valid) == best:
                system_wins = rf"\textbf{{{system_wins}}}"
            cells = [
                latex_escape(_strategy_label(row["baseline_strategy"])),
                latex_escape(_strategy_label(row["challenger_strategy"])),
                f"{valid} / {row['total_comparisons']}", system_wins,
                _outcome(row["baseline_wins"], valid), _outcome(row["ties"], valid),
            ]
            lines.append(" & ".join(cells) + r" \\")
    return lines


def pairwise_latex_tables(dataset_rows: Sequence[dict], aggregate_rows: Sequence[dict]) -> str:
    """One table per decoder/judge, with dataset blocks and an overall block.

    Rows must be grouped by source/model/judge/pair for dataset_rows and
    model/judge/pair for aggregate_rows. No unobserved comparisons are invented.
    """
    groups = defaultdict(list)
    for row in dataset_rows:
        groups[(row["generation_model"], row["judge_model"])].append(row)
    tables = [r"% Generated automatically. Requires \usepackage{booktabs,graphicx}."]
    for (model, judge), rows in sorted(groups.items()):
        label = re.sub(r"[^a-zA-Z0-9]+", "-", f"{model}-{judge}").strip("-").lower()
        lines = [
            r"\begin{table*}[t]", r"\centering", r"\small",
            r"\caption{" + latex_escape(f"{judge} pairwise evaluation for {model}. ")
            + "Each row compares the system against the indicated baseline on the same messages. "
            + "Outcomes are counts and percentages over valid comparisons. "
            + "Bold marks the highest system win percentage among systems sharing a baseline "
            + "within a dataset block (including the overall block); it does not indicate statistical significance.}",
            rf"\label{{tab:pairwise-{label}}}", r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llrrrr}", r"\toprule",
            r"\textbf{Baseline} & \textbf{System} & \textbf{Valid / Total} & \textbf{System Wins} & \textbf{Baseline Wins} & \textbf{Ties} \\",
            r"\midrule",
        ]
        by_dataset = defaultdict(list)
        for row in rows:
            by_dataset[row["source"]].append(row)
        for position, (dataset, items) in enumerate(sorted(by_dataset.items())):
            if position:
                lines.append(r"\midrule")
            title = {"edos": "EDOS", "multiconan": "MultiCONAN"}.get(dataset, dataset)
            lines.extend(_section(title, items))
        overall = [row for row in aggregate_rows if row["generation_model"] == model and row["judge_model"] == judge]
        if len(by_dataset) > 1 and overall:
            lines.append(r"\midrule")
            lines.extend(_section("All datasets", overall))
        lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"])
        tables.append("\n".join(lines))
    if not groups:
        tables.append("% No pairwise comparisons available; no table generated.")
    return "\n\n".join(tables) + "\n"
