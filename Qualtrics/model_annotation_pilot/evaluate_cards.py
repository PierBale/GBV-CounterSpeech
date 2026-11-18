"""Agreement, gold labels and scores of the card annotation (faithfulness, answer relevance).

Same procedure as retrieval_internal_annotation_final/evaluate_retrieval.py:
- Agreement on the common survey (45 cards, 3 annotators): ordinal Krippendorff's alpha and
  exact agreement with the pilot's `agreement_summary`, before and after the discussion (the
  discussed cards take the harmonized score for every annotator).
- Gold labels (0/1/2) for the 132 cards, per dimension: the harmonized score where filled,
  otherwise the median of the three judgments; in the private surveys, the annotator's judgment.
- Per generator model and EDOS label: mean gold score of its 6 cards; mean and standard deviation
  over the 11 labels. Both models made a card from the same paragraph at each (label, reranker
  rank), so the models are compared with a paired Wilcoxon signed-rank test (scipy) on 66 pairs.
- LaTeX tables and a per-label figure, with the two dimensions side by side.

Inputs in results/annotations/: best-model-harmonized.csv (common survey: the three annotators
plus an unnamed row with the harmonized scores) and best-model-<annotator>_*.csv.

Usage, from the repository root:
    python Qualtrics/model_annotation_pilot/evaluate_cards.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrieval_internal_annotation_pilot"))
sys.path.insert(0, str(HERE.parent / "retrieval_internal_annotation_final"))
from analyze_retrieval import agreement_summary  # noqa: E402
from plot_results import HEATMAP_CMAP, SURFACE, TEXT, TEXT_2, cell_ink  # noqa: E402  (also sets the style)

ANNOTATIONS = HERE / "results" / "annotations"
OUTPUT = HERE / "results" / "evaluation"
ANNOTATORS = ("marem", "arianna", "marta")
PAPER_NAMES = {"marem": "A1", "arianna": "A2", "marta": "A3"}  # anonymous in the paper
DIMENSIONS = {"faithfulness": "Faithfulness", "answer_relevance": "Answer Relevance"}
MODELS = {"openai/gpt-6.1-sol": "GPT-6.1 Sol", "deepseek/deepseek-v4.1-flash-1017": "DeepSeek-v4.1-flash"}
BOOTSTRAP = 10000
SEED = 20261118


def read_export(path: Path) -> pd.DataFrame:
    """Qualtrics CSV export (original or re-saved by Excel with ';')."""
    return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False, encoding="utf-8-sig")


def survey_questions(survey: str) -> dict:
    qsf = json.loads((HERE / f"{survey}.qsf").read_text(encoding="utf-8"))
    return {e["Payload"]["DataExportTag"]: e["Payload"] for e in qsf["SurveyElements"] if e["Element"] == "SQ"}


def choice_value(payload: dict) -> dict:
    """Choice code and choice text -> value: exports can hold either."""
    values = {}
    for code, choice in payload["Choices"].items():
        text = choice["Display"].strip()
        match = re.match(r"\s*(\d)", text)
        values[code] = values[text] = int(match.group(1)) if match else text
    return values


def load_survey(path: Path, survey: str) -> dict[str, pd.DataFrame]:
    """Per dimension, a respondents x cards frame of 0/1/2 (NaN where not answered)."""
    mapping = pd.read_csv(HERE / f"{survey}_question_map.csv", dtype=str, keep_default_na=False)
    questions = survey_questions(survey)
    raw = read_export(path)
    import_ids, responses = raw.iloc[1], raw.iloc[2:]
    # Qualtrics truncates long column names: find each column by its ImportId (QID).
    column_of = {json.loads(value)["ImportId"]: column for column, value in import_ids.items() if value.startswith("{")}
    if survey == "comune":
        names = [choice_value(questions["Annotator"]).get(v.strip(), "Harmonized") if v.strip() else "Harmonized"
                 for v in responses["Annotator"]]
    else:
        names = [survey] * len(responses)
    if len(set(names)) != len(names):
        raise ValueError(f"{path.name}: duplicated respondents {names}")
    frames = {}
    for dimension in DIMENSIONS:
        table = {}
        for item in mapping.itertuples():
            qid, tag = getattr(item, f"{dimension}_question_id"), getattr(item, f"{dimension}_export_tag")
            if qid not in column_of:
                raise ValueError(f"{path.name}: {qid} ({tag}) missing from the export.")
            scale = choice_value(questions[tag])
            table[item.question_id] = [scale[v.strip()] if v.strip() else np.nan for v in responses[column_of[qid]]]
        frames[dimension] = pd.DataFrame(table, index=names, dtype=float)
    return frames


def load_items() -> pd.DataFrame:
    frames = []
    for survey in ("comune", *ANNOTATORS):
        mapping = pd.read_csv(HERE / f"{survey}_question_map.csv", dtype=str, keep_default_na=False)
        mapping["survey"] = survey
        frames.append(mapping)
    items = pd.concat(frames, ignore_index=True).set_index("question_id")
    if items.index.duplicated().any():
        raise ValueError("Question maps overlap: each card must appear once.")
    items["reranker_rank"] = items["reranker_rank"].astype(int)
    return items


def agreement(judges: pd.DataFrame, harmonized: pd.Series, rng) -> dict:
    """judges: annotators x cards. "after_discussion" is higher by construction."""
    discussed = harmonized.notna()
    after = judges.copy()
    after.loc[:, discussed] = np.tile(harmonized[discussed].to_numpy(), (len(after), 1))
    pairs = [(ANNOTATORS[0], ANNOTATORS[1]), (ANNOTATORS[0], ANNOTATORS[2]), (ANNOTATORS[1], ANNOTATORS[2])]

    def summary(frame):
        return {
            "all_three": agreement_summary(frame.T.to_numpy(dtype=float), rng, BOOTSTRAP),
            "pairwise": {f"{a}-{b}": agreement_summary(frame.loc[[a, b]].T.to_numpy(dtype=float), rng, BOOTSTRAP)
                         for a, b in pairs},
            "score_distribution": {name: {str(v): int((frame.loc[name] == v).sum()) for v in (0, 1, 2)}
                                   for name in ANNOTATORS},
        }

    return {**summary(judges), "n_discussed_items": int(discussed.sum()), "after_discussion": summary(after)}


def gold_labels(common: dict, private: dict, items: pd.DataFrame) -> pd.DataFrame:
    gold = items[["survey", "generator_model", "primary_edos_label", "reranker_rank", "chunk_id"]].copy()
    for dimension in DIMENSIONS:
        judges, harmonized = common[dimension].loc[list(ANNOTATORS)], common[dimension].loc["Harmonized"]
        if judges.isna().any().any():
            raise ValueError(f"Missing {dimension} judgments in the common survey.")
        values = pd.concat([harmonized.where(harmonized.notna(), judges.median(axis=0)),
                            *[private[name][dimension].iloc[0] for name in ANNOTATORS]])
        if values.isna().any() or set(values.index) != set(items.index) or (values % 1 != 0).any():
            raise ValueError(f"Every card needs exactly one integer {dimension} gold label.")
        source = pd.Series("private", index=values.index)
        source[harmonized.index] = np.where(harmonized.notna(), "harmonized", "median_of_3")
        gold[dimension] = values.astype(int)
        gold[f"{dimension}_source"] = source
    return gold


def scores(gold: pd.DataFrame):
    per_label = (gold.groupby(["generator_model", "primary_edos_label"])[list(DIMENSIONS)].mean()
                 .reset_index().rename(columns={"primary_edos_label": "label"}))
    summary = per_label.groupby("generator_model")[list(DIMENSIONS)].agg(["mean", "std"])
    summary.columns = [f"{dimension}_{stat}" for dimension, stat in summary.columns]
    paired = gold.pivot_table(index=["primary_edos_label", "reranker_rank"], columns="generator_model",
                              values=list(DIMENSIONS))
    gpt, deepseek = MODELS
    tests = {}
    for dimension in DIMENSIONS:
        difference = paired[(dimension, gpt)] - paired[(dimension, deepseek)]
        test = wilcoxon(paired[(dimension, gpt)], paired[(dimension, deepseek)], zero_method="wilcox")
        tests[dimension] = {"pairs": int(len(difference)), "mean_difference_gpt_minus_deepseek": float(difference.mean()),
                            "gpt_better": int((difference > 0).sum()), "deepseek_better": int((difference < 0).sum()),
                            "ties": int((difference == 0).sum()), "statistic": float(test.statistic),
                            "p": float(test.pvalue)}
    return per_label, summary, tests


def number(value: float) -> str:
    return f"{value:.2f}".replace("-", "$-$")


def scores_latex(summary: pd.DataFrame) -> str:
    best = {dimension: summary[f"{dimension}_mean"].max() for dimension in DIMENSIONS}
    header = " & ".join(rf"\textbf{{{name}}} $\uparrow$" for name in DIMENSIONS.values())
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Human evaluation of the cards produced by each generator. Faithfulness: support of the "
        r"argument by its paragraph; Answer Relevance: relevance of the argument to the EDOS label. Both "
        r"range from 0 (not relevant) to 2 (relevant). $\uparrow$: higher is better; best mean in bold.}",
        r"\label{tab:cards-human}",
        r"\begin{tabular}{lcc}", r"\toprule",
        rf"\textbf{{Generator}} & {header} \\", r"\midrule",
    ]
    for model, row in summary.sort_values("faithfulness_mean", ascending=False).iterrows():
        cells = []
        for dimension in DIMENSIONS:
            mean, std = row[f"{dimension}_mean"], row[f"{dimension}_std"]
            value = rf"\textbf{{{mean:.2f}}}" if mean == best[dimension] else f"{mean:.2f}"
            cells.append(rf"{value} $\pm$ {std:.2f}")
        lines.append(f"{MODELS[model]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def agreement_latex(results: dict) -> str:
    first = results[next(iter(DIMENSIONS))]
    discussed = [(results[d]["n_discussed_items"], name) for d, name in DIMENSIONS.items()]
    after = " and ".join(f"the {n} cards discussed for {name}" for n, name in discussed if n)
    none = " and ".join(name for n, name in discussed if not n)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{Inter-annotator agreement on the cards, on the {first['all_three']['n_items']} cards "
        r"annotated by all three annotators (A1--A3). $\alpha$: ordinal Krippendorff's alpha; Exact: exact "
        rf"match, the share of annotator pairs giving the same score. After the discussion, {after} take the "
        r"harmonized score for every annotator" + (f" (no card was discussed for {none})" if none else "")
        + r". $\uparrow$: higher is better.}",
        r"\label{tab:cards-agreement}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{name}}}}}" for name in DIMENSIONS.values()) + r" \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"\textbf{Annotators}" + r" & \textbf{$\alpha$} $\uparrow$ & \textbf{Exact} $\uparrow$" * len(DIMENSIONS) + r" \\",
    ]
    for stage, key in (("Before discussion", None), ("After discussion", "after_discussion")):
        lines += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\textit{{{stage}}}}} \\"]
        rows = [("All three", "all_three", None)] + [
            ("--".join(PAPER_NAMES[n] for n in pair.split("-")), "pairwise", pair) for pair in first["pairwise"]]
        for name, group, pair in rows:
            cells = []
            for dimension in DIMENSIONS:
                values = results[dimension][key] if key else results[dimension]
                row = values[group][pair] if pair else values[group]
                cells += [number(row["alpha_ordinal"]), number(row["pairwise_exact_agreement"])]
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def figure_per_label(per_label: pd.DataFrame, path: Path):
    """Mean gold score per EDOS label and generator, the two dimensions side by side.

    No EDOS label names on the y axis: the figure sits next to the context-relevance one,
    which has the same rows in the same order."""
    cmap = HEATMAP_CMAP
    order = list(MODELS)
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(5.6, 6.4), sharey=True)
    for ax, (dimension, title) in zip(axes, DIMENSIONS.items()):
        wide = per_label.pivot(index="label", columns="generator_model", values=dimension)[order]
        image = ax.imshow(wide.to_numpy(), cmap=cmap, vmin=0, vmax=2, aspect="auto")
        for (row, column), value in np.ndenumerate(wide.to_numpy()):
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=cell_ink(cmap(value / 2)))
        ax.set_xticks(range(len(order)), [MODELS[m].replace("-v4", "\nv4") for m in order], color=TEXT, fontsize=7.5)
        ax.set_yticks(range(len(wide)), [])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(wide)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, orientation="horizontal", location="top")
        bar.outline.set_visible(False)
        bar.set_ticks(np.linspace(0, 2, 5))
        bar.ax.tick_params(labelsize=7, colors=TEXT_2)
        ax.set_title(title, fontsize=9.5, pad=58)
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    items = load_items()
    common = load_survey(ANNOTATIONS / "best-model-harmonized.csv", "comune")
    private = {name: load_survey(next(ANNOTATIONS.glob(f"best-model-{name}_*.csv")), name) for name in ANNOTATORS}
    for name, frames in private.items():
        if any(len(frame) != 1 or frame.isna().any().any() for frame in frames.values()):
            raise ValueError(f"{name}: expected one complete response.")
    results = {dimension: agreement(common[dimension].loc[list(ANNOTATORS)], common[dimension].loc["Harmonized"], rng)
               for dimension in DIMENSIONS}
    gold = gold_labels(common, private, items)
    per_label, summary, tests = scores(gold)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    gold.to_csv(OUTPUT / "gold_labels.csv", index_label="question_id", encoding="utf-8")
    per_label.to_csv(OUTPUT / "scores_per_label.csv", index=False, encoding="utf-8")
    summary.to_csv(OUTPUT / "scores_summary.csv", index_label="generator_model", encoding="utf-8")
    (OUTPUT / "agreement_common.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "paired_wilcoxon.json").write_text(json.dumps(tests, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "cards_scores.tex").write_text(scores_latex(summary), encoding="utf-8")
    (OUTPUT / "cards_agreement.tex").write_text(agreement_latex(results), encoding="utf-8")
    figure_per_label(per_label, OUTPUT / "figures" / "cards_per_label")

    print(summary.round(3).to_string())
    print(json.dumps(tests, indent=2))
    for dimension in DIMENSIONS:
        print(dimension, {stage: round(values["all_three"]["alpha_ordinal"], 3) for stage, values in
                          (("before", results[dimension]), ("after", results[dimension]["after_discussion"]))},
              "discussed", results[dimension]["n_discussed_items"])


if __name__ == "__main__":
    main()
