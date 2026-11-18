"""LaTeX tables of the context-relevance annotation (from the outputs of evaluate_retrieval.py).

- results/evaluation/context_relevance.tex, from scores_per_label.csv: per configuration, mean +-
  standard deviation over the 11 EDOS labels of mean relevance@4 and nDCG@4; best mean in bold.
- results/evaluation/agreement.tex, from agreement_common.json: agreement on the common items
  before and after the discussion, for the three annotators and each pair (anonymized as A1-A3).

Usage, from the repository root:
    python Qualtrics/retrieval_internal_annotation_final/latex_table.py
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EVALUATION = HERE / "results" / "evaluation"
MODELS = {"qwen3-embedding-8b": "Q3-Emb8B", "bge-m3": "BGE",
          "qwen3-reranker-8b": "Q3-Rer8B", "bge-reranker-v2-m3": "BGE-Rer"}
METRICS = (("mean_relevance@4", r"Mean@4"), ("nDCG@4", r"nDCG@4"))


def scores() -> pd.DataFrame:
    """Mean and sample standard deviation over the labels, best configuration first."""
    per_label = pd.read_csv(EVALUATION / "scores_per_label.csv")
    table = per_label.groupby("configuration")[[metric for metric, _ in METRICS]].agg(["mean", "std", "count"])
    return table.sort_values((METRICS[0][0], "mean"), ascending=False)


def latex(table: pd.DataFrame) -> str:
    labels = int(table[(METRICS[0][0], "count")].iloc[0])
    best = {metric: table[(metric, "mean")].max() for metric, _ in METRICS}
    header = " & ".join(rf"\textbf{{{name}}} $\uparrow$" for _, name in METRICS)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Context relevance of the paragraphs retrieved by each embedder (Emb.)--reranker (Rer.) "
        r"configuration (top-4 paragraphs per EDOS label, human gold labels). Q3-Emb8B: Qwen3-Embedding-8B; "
        r"BGE: BGE-M3; Q3-Rer8B: Qwen3-Reranker-8B; BGE-Rer: BGE-Reranker-v2-M3. "
        r"Mean@4 is the mean graded relevance "
        r"of the top-4 paragraphs, from 0 (none relevant) to 2 (all relevant; 0~=~not relevant, "
        r"1~=~partially relevant, 2~=~relevant). nDCG@4 uses the same graded gains and ranges from 0 to 1 "
        r"(1: the four most relevant judged paragraphs of the label, in the best order). "
        rf"Mean $\pm$ standard deviation over the {labels} EDOS labels; $\uparrow$: higher is better; "
        r"best mean in bold.}",
        r"\label{tab:context-relevance}",
        r"\begin{tabular}{llcc}", r"\toprule",
        rf"\textbf{{Emb.}} & \textbf{{Rer.}} & {header} \\", r"\midrule",
    ]
    for configuration, row in table.iterrows():
        embedder, reranker = configuration.split("__")
        cells = []
        for metric, _ in METRICS:
            mean, std = row[(metric, "mean")], row[(metric, "std")]
            value = rf"\textbf{{{mean:.2f}}}" if mean == best[metric] else f"{mean:.2f}"
            cells.append(rf"{value} $\pm$ {std:.2f}")
        lines.append(f"{MODELS[embedder]} & {MODELS[reranker]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


ANNOTATORS = {"marem": "A1", "arianna": "A2", "marta": "A3"}  # anonymous in the paper


def number(value: float) -> str:
    return f"{value:.2f}".replace("-", "$-$")


def agreement_latex(agreement: dict) -> str:
    """Agreement on the common items before and after the discussion of the disagreements."""
    stages = (("Before discussion", agreement), ("After discussion", agreement["after_discussion"]))
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{Inter-annotator agreement on context relevance, on the {agreement['all_three']['n_items']} "
        r"items annotated by all three annotators (A1--A3). $\alpha$: ordinal Krippendorff's alpha, from "
        r"$-1$ to 1 (1: perfect agreement, 0: chance level); Exact: exact match, the share of annotator "
        r"pairs giving the same score, from 0 to 1 (for all three annotators, the mean over the three pairs). "
        rf"After the discussion, the {agreement['n_discussed_items']} discussed items take the harmonized "
        r"score for every annotator. $\uparrow$: higher is better.}",
        r"\label{tab:context-relevance-agreement}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"\textbf{Annotators} & \textbf{$\alpha$} $\uparrow$ & \textbf{Exact} $\uparrow$ \\",
    ]
    for stage, values in stages:
        lines += [r"\midrule", rf"\multicolumn{{3}}{{l}}{{\textit{{{stage}}}}} \\"]
        rows = [("All three", values["all_three"])] + [
            ("--".join(ANNOTATORS[name] for name in pair.split("-")), row) for pair, row in values["pairwise"].items()]
        for name, row in rows:
            lines.append(f"{name} & {number(row['alpha_ordinal'])} & {number(row['pairwise_exact_agreement'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    agreement = json.loads((EVALUATION / "agreement_common.json").read_text(encoding="utf-8"))
    (EVALUATION / "agreement.tex").write_text(agreement_latex(agreement), encoding="utf-8")
    table = scores()
    summary = pd.read_csv(EVALUATION / "scores_summary.csv").set_index("configuration")
    for metric, _ in METRICS:  # same means as evaluate_retrieval.py
        pd.testing.assert_series_equal(table[(metric, "mean")].sort_index(), summary[metric].sort_index(),
                                       check_names=False)
    path = EVALUATION / "context_relevance.tex"
    path.write_text(latex(table), encoding="utf-8")
    print(f"Table written to {path}")


if __name__ == "__main__":
    main()
