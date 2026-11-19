"""LaTeX tables of the counter-speech experiment (run from src after the judge stage).

  python counterspeech_experiment_tables.py

Writes to data/counterspeech_v2/tables/:
  judgelm_pairwise.tex        wins/losses/ties per pair and dataset (both orders combined)
  examples_edos.tex           qualitative examples, EDOS (predicted and gold label)
  examples_multiconan.tex     qualitative examples, MultiCONAN (predicted label)
  examples_edos_label_mismatch.tex  EDOS examples where predicted and gold labels differ
  metrics.tex                 automatic metrics, EDOS and MultiCONAN (after the metrics stage)
  judgelm_order_agreement.tex agreement of JudgeLM with itself across presentation orders
"""

import json
from collections import defaultdict

from unito_amazon.card_creation.io import read_jsonl
from unito_amazon.counterspeech_experiment.config import JUDGED_PAIRS, STRATEGIES, ExperimentConfig

NAMES = {
    "zero_shot": "Zero-shot",
    "rag": "RAG (Qwen3-Emb-8B)",
    "rag_label": "RAG + EDOS label (pred.)",
    "rag_gold_label": "RAG + EDOS label (gold)",
    "card_rag": "Card-RAG (Qwen3-Emb-8B)",
    "card_rag_label": "Card-RAG + EDOS label (pred.)",
    "card_rag_gold_label": "Card-RAG + EDOS label (gold)",
}
SHORT_NAMES = {"zero_shot": "Zero-shot", "rag": "RAG",
               "rag_label": "RAG + pred. label", "rag_gold_label": "RAG + gold label", "card_rag": "Card-RAG",
               "card_rag_label": "Card-RAG + pred. label", "card_rag_gold_label": "Card-RAG + gold label"}
CARD_SYSTEMS = ("card_rag", "card_rag_label", "card_rag_gold_label")
DATASETS = (("edos", "EDOS"), ("multiconan", "MultiCONAN"))
LABEL_NOTE = (r"pred.: EDOS label predicted by the EDOS classifier; gold: EDOS annotation "
              r"(available only for EDOS). Both are used to filter the cards and are stated in the prompt.")

# Hand-picked messages where a card-based system beats both Zero-shot and RAG in
# both presentation orders. edos_label_mismatch: EDOS messages where the predicted label
# differs from the gold one (gold better; both win; predicted better than gold).
EXAMPLES = {
    "edos": ["sexism2022_english-10488", "sexism2022_english-15575", "sexism2022_english-17346"],
    "multiconan": ["multiconan-1256", "multiconan-1567", "multiconan-2496"],
    "edos_label_mismatch": ["sexism2022_english-11692", "sexism2022_english-18058", "sexism2022_english-3853"],
}

LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                 "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(text: str) -> str:
    text = " ".join(str(text).split())
    return "".join(LATEX_SPECIAL.get(char, char) for char in text)


def count(value: int, total: int) -> str:
    """Math-mode content (without the $ delimiters): '335\\ (34.54\\%)'."""
    return f"{value}\\ ({100 * value / total:.2f}\\%)"


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values (statsmodels) over all the tests shown in one table."""
    from statsmodels.stats.multitest import multipletests

    keys = [key for key, p in pvalues.items() if p is not None]
    if not keys:
        return {}
    adjusted = multipletests([pvalues[key] for key in keys], method="holm")[1]
    return dict(zip(keys, map(float, adjusted)))


def pairwise_table(summary: dict) -> str:
    rows = {(key, baseline, system): summary["pairs"][f"{key}|{baseline}|{system}"]
            for key, _ in DATASETS for baseline, system in JUDGED_PAIRS
            if summary["pairs"].get(f"{key}|{baseline}|{system}", {}).get("messages")}
    adjusted = holm({name: row["sign_test_p"] for name, row in rows.items()})
    sizes = {key: {row["messages"] for name, row in summary["pairs"].items() if name.startswith(key + "|")}
             for key, _ in DATASETS}
    size_text = ", ".join(f"{title}: {'/'.join(map(str, sorted(sizes[key])))} messages"
                          for key, title in DATASETS if sizes[key])
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{JudgeLM-7B pairwise evaluation of Gemma-4-E4B counter-speech, separated by dataset "
        rf"({size_text}). Every pair is judged in both presentation orders; per message the two "
        r"outcomes are combined (win 1, tie 0.5, loss 0) and a disagreement between orders counts as a tie. "
        r"The larger of System Wins and Baseline Wins is in bold. "
        rf"$^*$: sign test on wins vs.\ losses, $p<0.05$ after Holm correction over the {len(adjusted)} "
        r"comparisons in the table. " + LABEL_NOTE + "}",
        r"\label{tab:pairwise-gemma-4-e4b-judgelm-7b}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrr}", r"\toprule",
        r"\textbf{Baseline} & \textbf{System} & \textbf{System Wins} & \textbf{Baseline Wins} & \textbf{Ties} \\",
    ]
    for key, title in DATASETS:
        lines += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\"]
        for baseline, system in JUDGED_PAIRS:
            row = rows.get((key, baseline, system))
            if row is not None:
                total = row["messages"]
                significant = adjusted.get((key, baseline, system), 1.0) < 0.05
                system_wins, baseline_wins = count(row["b_wins"], total), count(row["a_wins"], total)
                # The larger number of wins in bold (neither when they are equal).
                if row["b_wins"] > row["a_wins"]:
                    system_wins = rf"\mathbf{{{system_wins}}}"
                elif row["a_wins"] > row["b_wins"]:
                    baseline_wins = rf"\mathbf{{{baseline_wins}}}"
                if significant and row["b_wins"] > row["a_wins"]:
                    system_wins += "^*"
                elif significant:
                    baseline_wins += "^*"
                lines.append(f"{NAMES[baseline]} & {NAMES[system]} & ${system_wins}$ & ${baseline_wins}$ & "
                             f"${count(row['ties'], total)}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def combined_outcomes(judgments: list[dict]) -> dict:
    """(message id, system x, system y) -> x's points over both orders (1 win, 0.5 tie, 0 loss)."""
    points = defaultdict(float)
    for record in judgments:
        winner = record["judgment"]["winner"]
        a = record["system_a"]
        if winner == "tie":
            value = 0.5
        else:
            winning = record["response_1_strategy"] if winner == "response_1" else record["response_2_strategy"]
            value = 1.0 if winning == a else 0.0
        points[(record["id"], a, record["system_b"])] += value / 2
        points[(record["id"], record["system_b"], a)] += (1 - value) / 2
    return points


def examples_table(key: str, generations: dict, points: dict) -> str:
    gold = key.startswith("edos")
    mismatch = key == "edos_label_mismatch"
    label_caption = (
        "EDOS label: the label predicted by the EDOS classifier (used by Card-RAG + pred. label) and the "
        "gold annotation (used by Card-RAG + gold label)." if gold else
        "EDOS label: predicted by the EDOS classifier and used by Card-RAG + pred. label "
        "(MultiCONAN has no EDOS annotation)."
    )
    subject = ("EDOS messages where the label predicted by the EDOS classifier differs from the gold "
               "annotation" if mismatch else f"{dict(DATASETS)[key]} messages where evidence cards helped")
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        rf"\caption{{Counter-speech generated by Gemma-4-E4B for {subject}. "
        r"In bold, the card-based systems preferred by JudgeLM-7B over both Zero-shot and RAG in both "
        r"presentation orders. " + label_caption + "}",
        rf"\label{{tab:examples-{key.replace('_', '-')}}}",
        r"\begin{tabularx}{\textwidth}{@{}p{0.2\textwidth}X@{}}", r"\toprule",
    ]
    for position, message_id in enumerate(EXAMPLES[key]):
        records = generations[message_id]
        first = records["zero_shot"]
        if mismatch and first["edos_label"].split()[0] == first["predicted_edos_label"].split()[0]:
            raise ValueError(f"{message_id}: predicted and gold labels coincide.")
        label_rows = [rf"\textbf{{EDOS label (pred.)}} & {tex(first['predicted_edos_label'])} \\"]
        if gold:
            label_rows.append(rf"\textbf{{EDOS label (gold)}} & {tex(first['edos_label'])} \\")
        winners = {system for system in CARD_SYSTEMS if system in records
                   and points[(message_id, system, "zero_shot")] == 1 and points[(message_id, system, "rag")] == 1}
        if not winners:
            raise ValueError(f"{message_id}: no card system beats both baselines in both orders.")
        if position:
            lines.append(r"\midrule[\heavyrulewidth]")
        lines += [rf"\textbf{{Hate speech}} & {tex(first['hateful_message'])} \\", *label_rows, r"\midrule",
                  r"\textbf{System} & \textbf{Generated counter-speech} \\"]
        for system in (s for s in STRATEGIES if s in records):
            text = tex(records[system]["generated_counter_speech"])
            name = SHORT_NAMES[system]
            if system in winners:
                lines.append(rf"\textbf{{{name}}} & \textbf{{{text}}} \\")
            else:
                lines.append(rf"{name} & {text} \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def order_agreement_table(summary: dict) -> str:
    """Agreement of JudgeLM with itself when the two responses are swapped."""
    def number(value):
        return "--" if value is None else f"{value:.2f}"

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Agreement of JudgeLM-7B with itself across the two presentation orders of each pair "
        r"(outcome from the point of view of the first system: win, tie, loss). Exact: share of messages with "
        r"the same outcome in both orders; $\kappa$: Cohen's kappa; $\alpha$: Krippendorff's ordinal alpha; "
        r"$\rho$: Spearman correlation of the score differences; Flip: share of messages where each order "
        r"gives a clear but different winner.}",
        r"\label{tab:judgelm-order-agreement}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"\textbf{System A} & \textbf{System B} & \textbf{n} & \textbf{Exact} & $\kappa$ & "
        r"$\alpha$ & $\rho$ & \textbf{Flip} \\",
    ]
    agreement = summary["order_agreement"]
    for key, title in DATASETS:
        lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{\textit{{{title}}}}} \\"]
        rows = [(a, b) for a, b in JUDGED_PAIRS if f"{key}|{a}|{b}" in agreement] + [("all", None)]
        for a, b in rows:
            row = agreement[f"{key}|all pairs" if b is None else f"{key}|{a}|{b}"]
            names = (r"\textit{All pairs}", "") if b is None else (NAMES[a], NAMES[b])
            if b is None:
                lines.append(r"\cmidrule(lr){1-8}")
            lines.append(f"{names[0]} & {names[1]} & {row['messages']} & {number(row['exact_agreement'])} & "
                         f"{number(row['cohen_kappa'])} & {number(row['krippendorff_alpha_ordinal'])} & "
                         f"{number(row['spearman_score_difference'])} & {number(row['opposite_winner_rate'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


# (field, header, higher is better: bold the best value)
METRIC_COLUMNS = (("rougeL", "ROUGE-L", True), ("bleu", "BLEU", True), ("bertscore", "BERTScore", True),
                  ("gen_len", "Length", False))


def metrics_table(summary: dict) -> str:
    """Automatic metrics per dataset and system, as in the COLING 2025 MCG shared task; '--' without references."""
    multiconan = summary["datasets"].get("multiconan", {})
    sizes = ", ".join(f"{name}: {summary['datasets'][key]['messages']}" for key, name in DATASETS
                      if key in summary["datasets"])
    header = " & ".join(rf"\textbf{{{name}}}" for _, name, _ in METRIC_COLUMNS)
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        rf"\caption{{Automatic metrics of Gemma-4-E4B counter-speech ({sizes} messages), computed as in the "
        r"COLING 2025 Multilingual Counterspeech Generation shared task: ROUGE-L (with stemming), corpus "
        r"BLEU-4, mean BERTScore F1 (bert-base-multilingual-cased) and mean length in words. "
        r"Reference-based metrics only for MultiCONAN (--: EDOS has no reference counter-speech), with one "
        r"reference per message"
        + (f", drawn at random for the {multiconan['messages_with_several_references']} messages with several"
           if multiconan.get("messages_with_several_references") else "")
        + r". Best value per dataset in bold. " + LABEL_NOTE + "}",
        r"\label{tab:automatic-metrics}",
        r"\begin{tabular}{l" + "c" * len(METRIC_COLUMNS) + "}", r"\toprule",
        rf"\textbf{{System}} & {header} \\",
    ]
    for key, name in DATASETS:
        if key not in summary["datasets"]:
            continue
        rows = summary["datasets"][key]["strategies"]
        strategies = [s for s in STRATEGIES if s in rows]
        best = {field: max((rows[s][field] for s in strategies if rows[s][field] is not None), default=None)
                for field, _, bold in METRIC_COLUMNS if bold}
        lines += [r"\midrule", rf"\multicolumn{{{len(METRIC_COLUMNS) + 1}}}{{l}}{{\textit{{{name}}}}} \\"]
        for strategy in strategies:
            cells = []
            for field, _, bold in METRIC_COLUMNS:
                value = rows[strategy][field]
                if value is None:
                    cells.append("--")
                    continue
                cells.append(rf"\textbf{{{value:.3f}}}" if bold and value == best[field] else f"{value:.3f}")
            lines.append(f"{NAMES[strategy]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    config = ExperimentConfig()
    summary = json.loads((config.output_dir / "judge_summary.json").read_text(encoding="utf-8"))
    generations = defaultdict(dict)
    for record in read_jsonl(config.generations_path):
        generations[record["id"]][record["generation_strategy"]] = record
    points = combined_outcomes(read_jsonl(config.judgments_path))
    out = config.output_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "judgelm_pairwise.tex").write_text(pairwise_table(summary), encoding="utf-8")
    if "order_agreement" in summary:
        (out / "judgelm_order_agreement.tex").write_text(order_agreement_table(summary), encoding="utf-8")
    for key in EXAMPLES:
        (out / f"examples_{key}.tex").write_text(examples_table(key, generations, points), encoding="utf-8")
    metrics_summary = config.output_dir / "metrics_summary.json"
    if metrics_summary.exists():
        (out / "metrics.tex").write_text(
            metrics_table(json.loads(metrics_summary.read_text(encoding="utf-8"))), encoding="utf-8")
    else:
        print("No metrics_summary.json yet: run the metrics stage for the metrics table.")
    print(f"Tables written to {out}")
    print(r"Preamble: \usepackage{booktabs,graphicx,tabularx}")


if __name__ == "__main__":
    main()
