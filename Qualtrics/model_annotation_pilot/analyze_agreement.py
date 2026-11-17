"""Inter-annotator agreement on the common card questionnaire (comune.qsf).

Reads the raw Qualtrics export (three header rows, choice codes) and its question
map, validates it, converts choice codes to the 0/1/2 scale using the choice texts
in the QSF, and reports, separately for the two dimensions (faithfulness:
argument vs paragraph; answer_relevance: argument vs EDOS label):

- Krippendorff's alpha (ordinal: primary; interval and nominal for reference),
  overall and per annotator pair, with the `krippendorff` package;
- a 95% percentile bootstrap CI for alpha, resampling cards, with
  `scipy.stats.bootstrap`;
- percentage agreement: all three identical, at least two identical, mean
  pairwise and per pair, share of 0-vs-2 cards;
- score distribution per annotator;
- per generator model: score distribution of the per-card median and mean of all
  ratings, with a Mann-Whitney U test (scipy) on the per-card medians.

Usage, from the repository root:
    python Qualtrics/model_annotation_pilot/analyze_agreement.py --results "<export>.csv"
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, mannwhitneyu


HERE = Path(__file__).resolve().parent
LEVELS = [0, 1, 2]
DIMENSIONS = ("faithfulness", "answer_relevance")
SEED = 20261117
N_BOOTSTRAP = 10000


def alpha(ratings: pd.DataFrame, level: str) -> float:
    """ratings: items x annotators. krippendorff expects annotators x items."""
    return float(krippendorff.alpha(
        reliability_data=ratings.to_numpy(dtype=float).T,
        level_of_measurement=level, value_domain=LEVELS,
    ))


def alpha_ci(ratings: pd.DataFrame, level: str) -> list[float]:
    """Percentile bootstrap over cards (units)."""
    values = ratings.to_numpy(dtype=float)
    result = bootstrap(
        (np.arange(len(values)),),
        lambda index: krippendorff.alpha(reliability_data=values[index].T,
                                         level_of_measurement=level, value_domain=LEVELS),
        n_resamples=N_BOOTSTRAP, method="percentile", vectorized=False,
        random_state=np.random.default_rng(SEED),
    )
    return [float(result.confidence_interval.low), float(result.confidence_interval.high)]


def load(results_path: Path, map_path: Path, qsf_path: Path):
    # Qualtrics CSV export: row 0 column names, row 1 question texts, row 2 ImportIds.
    raw = pd.read_csv(results_path, header=0, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    import_ids, responses = raw.iloc[1], raw.iloc[2:]
    responses = responses[(responses["Finished"] == "1") & (responses["Status"] == "0")]
    mapping = pd.read_csv(map_path, dtype=str, keep_default_na=False)
    survey = json.loads(qsf_path.read_text(encoding="utf-8"))
    questions = {e["Payload"]["DataExportTag"]: e["Payload"] for e in survey["SurveyElements"] if e["Element"] == "SQ"}

    annotator_names = {code: choice["Display"] for code, choice in questions["Annotator"]["Choices"].items()}
    annotators = responses["Annotator"].map(annotator_names).tolist()
    if len(set(annotators)) != len(annotators):
        raise ValueError(f"Duplicate annotators among completed responses: {annotators}")

    # Qualtrics truncates long column names: find each column by its ImportId (QID).
    column_of = {json.loads(value)["ImportId"]: column for column, value in import_ids.items()
                 if value.startswith("{")}
    ratings = {dimension: {} for dimension in DIMENSIONS}
    notes = {}
    for item in mapping.itertuples():
        for dimension in DIMENSIONS:
            qid = getattr(item, f"{dimension}_question_id")
            tag = getattr(item, f"{dimension}_export_tag")
            if qid not in column_of:
                raise ValueError(f"{qid} ({tag}) missing from the export.")
            # Choice code -> scale value, read from the choice text ("0 - Not relevant").
            scale = {code: int(re.match(r"\s*(\d)", choice["Display"]).group(1))
                     for code, choice in questions[tag]["Choices"].items()}
            ratings[dimension][item.question_id] = responses[column_of[qid]].map(scale).tolist()
        # Text-entry answers are exported with ImportId "<QID>_TEXT".
        notes_column = column_of.get(f"{item.notes_question_id}_TEXT") or column_of[item.notes_question_id]
        notes[item.question_id] = responses[notes_column].tolist()
    ratings = {dimension: pd.DataFrame.from_dict(values, orient="index", columns=annotators)
               for dimension, values in ratings.items()}
    notes = pd.DataFrame.from_dict(notes, orient="index", columns=annotators)
    return ratings, notes, mapping.set_index("question_id")


def percentage_agreement(ratings: pd.DataFrame) -> dict:
    complete = ratings.dropna()
    distinct = complete.nunique(axis=1)
    pairwise = {f"{a}-{b}": float((complete[a] == complete[b]).mean())
                for a, b in itertools.combinations(complete.columns, 2)}
    return {
        "n_items": int(len(complete)),
        "all_identical": float((distinct == 1).mean()),
        "at_least_two_identical": float((distinct < complete.shape[1]).mean()),
        "mean_pairwise": float(np.mean(list(pairwise.values()))),
        "pairwise": pairwise,
        "items_with_0_vs_2": float(((complete.max(axis=1) - complete.min(axis=1)) == 2).mean()),
    }


def by_model(ratings: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    model = mapping.loc[ratings.index, "generator_model"]
    median = ratings.median(axis=1)
    report = {}
    for name in sorted(model.unique()):
        selected = model == name
        report[name] = {
            "cards": int(selected.sum()),
            "mean_rating": float(ratings[selected].to_numpy(dtype=float).mean()),
            "median_distribution": {str(v): int((median[selected] == v).sum()) for v in sorted(median.unique())},
        }
    names = sorted(model.unique())
    if len(names) == 2:
        test = mannwhitneyu(median[model == names[0]], median[model == names[1]], alternative="two-sided")
        report["mann_whitney_on_card_medians"] = {"u": float(test.statistic), "p": float(test.pvalue)}
    return report


def dimension_report(ratings: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    spread = (ratings.max(axis=1) - ratings.min(axis=1)) > 0
    return {
        "percentage_agreement": percentage_agreement(ratings),
        "krippendorff_alpha": {
            level: {"alpha": alpha(ratings, level), "ci95": alpha_ci(ratings, level)}
            for level in ("ordinal", "interval", "nominal")
        },
        "krippendorff_alpha_ordinal_pairwise": {
            f"{a}-{b}": alpha(ratings[[a, b]], "ordinal")
            for a, b in itertools.combinations(ratings.columns, 2)
        },
        "score_distribution": {
            name: {str(level): int((ratings[name] == level).sum()) for level in LEVELS}
            for name in ratings.columns
        },
        "cards_with_disagreement": int(spread.sum()),
        "by_model": by_model(ratings, mapping),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=HERE / "comune_question_map.csv")
    parser.add_argument("--qsf", type=Path, default=HERE / "comune.qsf")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "agreement")
    args = parser.parse_args()

    ratings, notes, mapping = load(args.results, args.map, args.qsf)
    report = {
        "annotators": list(ratings[DIMENSIONS[0]].columns),
        **{dimension: dimension_report(ratings[dimension], mapping) for dimension in DIMENSIONS},
        "bootstrap": {"method": "scipy.stats.bootstrap, percentile, resampling cards",
                      "replicates": N_BOOTSTRAP, "seed": SEED},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    aggregated = mapping.loc[ratings[DIMENSIONS[0]].index,
                             ["card_id", "generator_model", "primary_edos_label", "reranker_rank", "chunk_id"]]
    for dimension in DIMENSIONS:
        aggregated = aggregated.join(ratings[dimension].add_prefix(f"{dimension}_"))
        aggregated[f"{dimension}_median"] = ratings[dimension].median(axis=1)
    aggregated["notes"] = notes.apply(
        lambda row: " | ".join(f"{name}: {text}" for name, text in row.items() if text.strip()), axis=1)
    aggregated.to_csv(args.output_dir / "comune_items_aggregated.csv", index_label="question_id", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
