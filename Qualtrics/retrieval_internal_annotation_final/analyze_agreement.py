"""Inter-annotator agreement on the common context-relevance questionnaire.

Reads the raw Qualtrics export of comune.qsf (three header rows, choice codes)
and its question map, validates it, converts choice codes to the 0/1/2 scale
using the choice texts in the QSF, and reports:

- Krippendorff's alpha (ordinal: primary; interval and nominal for reference),
  overall and per annotator pair, computed with the `krippendorff` package;
- a 95% percentile bootstrap CI for alpha, resampling items, with
  `scipy.stats.bootstrap`;
- percentage agreement (simple counts with pandas): all three identical, at
  least two identical, mean pairwise and per pair, share of 0-vs-2 items;
- score distribution per annotator and the median judgment per item.

Usage, from the repository root:
    python Qualtrics/retrieval_internal_annotation_final/analyze_agreement.py
"""

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import bootstrap


HERE = Path(__file__).resolve().parent
LEVELS = [0, 1, 2]
SEED = 20261109
N_BOOTSTRAP = 10000


def alpha(ratings: pd.DataFrame, level: str) -> float:
    """ratings: items x annotators. krippendorff expects annotators x items."""
    return float(krippendorff.alpha(
        reliability_data=ratings.to_numpy(dtype=float).T,
        level_of_measurement=level, value_domain=LEVELS,
    ))


def alpha_ci(ratings: pd.DataFrame, level: str) -> list[float]:
    """Percentile bootstrap over items (units)."""
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

    ratings, notes = {}, {}
    for item in mapping.itertuples():
        tag = item.data_export_tag
        if json.loads(import_ids[tag])["ImportId"] != item.qualtrics_question_id:
            raise ValueError(f"{tag}: QID in the export differs from the question map.")
        # Choice code -> scale value, read from the choice text ("0 - Not relevant").
        scale = {code: int(re.match(r"\s*(\d)", choice["Display"]).group(1))
                 for code, choice in questions[tag]["Choices"].items()}
        ratings[item.question_id] = responses[tag].map(scale).tolist()
        notes[item.question_id] = responses[item.notes_export_tag].tolist()
    ratings = pd.DataFrame.from_dict(ratings, orient="index", columns=annotators)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results" / "comune_results.csv")
    parser.add_argument("--map", type=Path, default=HERE / "comune_question_map.csv")
    parser.add_argument("--qsf", type=Path, default=HERE / "comune.qsf")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "agreement")
    args = parser.parse_args()

    ratings, notes, mapping = load(args.results, args.map, args.qsf)
    report = {
        "annotators": list(ratings.columns),
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
        "libraries": {"krippendorff": krippendorff.__version__ if hasattr(krippendorff, "__version__") else None},
        "bootstrap": {"method": "scipy.stats.bootstrap, percentile, resampling items",
                      "replicates": N_BOOTSTRAP, "seed": SEED},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    aggregated = mapping.loc[ratings.index, ["edos_label", "chunk_id"]].join(ratings)
    aggregated["median"] = ratings.median(axis=1).astype(int)
    aggregated["all_identical"] = ratings.nunique(axis=1) == 1
    aggregated["notes"] = notes.apply(
        lambda row: " | ".join(f"{name}: {text}" for name, text in row.items() if text.strip()), axis=1)
    aggregated.to_csv(args.output_dir / "comune_items_aggregated.csv", index_label="question_id", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
