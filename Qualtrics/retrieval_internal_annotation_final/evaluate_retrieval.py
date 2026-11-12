"""Agreement, gold labels and retrieval scores of the final context-relevance annotation.

- Agreement on the common survey (29 items, 3 annotators, before discussion):
  Krippendorff's alpha (ordinal) with item-bootstrap CI and exact agreement,
  with the pilot's `agreement_summary`; plus each annotator's match with the
  harmonized label on the items discussed together.
- Gold labels (0/1/2) for the 110 (label, chunk) items: the "Harmonized" row
  where filled (discussed items), otherwise the median of the three judgments;
  in the private surveys, the single annotator's judgment.
- Per configuration and EDOS label, on its top-4: mean relevance (0-2) and
  nDCG@4 with graded gains (`sklearn.metrics.ndcg_score`; the ideal ranking uses
  every judged chunk of the label). Means over the 11 labels with a 95% bootstrap
  CI over labels (pilot's `interval`).

Statistics reuse retrieval_internal_annotation_pilot/analyze_retrieval.py; its
selection/evaluation functions are fixed to the pilot's top-3 and are not reused.
The notebook retrieval_analysis_final.ipynb walks through the same steps.

Usage, from the repository root:
    python Qualtrics/retrieval_internal_annotation_final/evaluate_retrieval.py
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrieval_internal_annotation_pilot"))
from analyze_retrieval import agreement_summary, interval  # noqa: E402

BOOTSTRAP = 10000
ANNOTATORS = ("marem", "arianna", "marta")
LEVELS = [0, 1, 2]
SEED = 20261112
TOP_K = 4
METRICS = ["mean_relevance@4", "nDCG@4"]
ANNOTATIONS = HERE / "results" / "annotations"


def read_export(path: Path) -> pd.DataFrame:
    """Qualtrics CSV export (original or re-saved by Excel with ';')."""
    return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False, encoding="utf-8-sig")


def choice_scale(qsf_path: Path) -> dict[str, dict[str, int]]:
    """Per question: choice code and choice text -> 0/1/2, from the QSF choice texts."""
    survey = json.loads(qsf_path.read_text(encoding="utf-8"))
    scales = {}
    for element in survey["SurveyElements"]:
        payload = element["Payload"] if element["Element"] == "SQ" else None
        if not payload or "-chunk_" not in payload.get("DataExportTag", ""):
            continue
        scale = {}
        for code, choice in payload["Choices"].items():
            value = int(re.match(r"\s*(\d)", choice["Display"]).group(1))
            scale[code] = value
            scale[choice["Display"].strip()] = value
        scales[payload["DataExportTag"]] = scale
    return scales


def load_survey(export_path: Path, survey: str) -> pd.DataFrame:
    """Rows: respondents (by name); columns: question_id; values 0/1/2 or NaN."""
    mapping = pd.read_csv(HERE / f"{survey}_question_map.csv", dtype=str, keep_default_na=False)
    scales = choice_scale(HERE / f"{survey}.qsf")
    raw = read_export(export_path)
    import_ids, responses = raw.iloc[1], raw.iloc[2:]
    for item in mapping.itertuples():
        if json.loads(import_ids[item.data_export_tag])["ImportId"] != item.qualtrics_question_id:
            raise ValueError(f"{export_path.name}: {item.data_export_tag} has a QID different from {survey} map.")
    names = responses["Annotator"].tolist() if survey == "comune" else [survey] * len(responses)
    table = {}
    for item in mapping.itertuples():
        scale = scales[item.data_export_tag]
        table[item.question_id] = [scale[v.strip()] if v.strip() else np.nan for v in responses[item.data_export_tag]]
    frame = pd.DataFrame(table, index=names)
    if frame.index.duplicated().any():
        raise ValueError(f"{export_path.name}: duplicated respondents {frame.index.tolist()}")
    return frame


def load_annotations(annotations: Path = ANNOTATIONS):
    """Common survey (3 annotators + Harmonized row) and the three private surveys."""
    common = load_survey(annotations / "pilot_harmonized.csv", "comune")
    private = {name: load_survey(next(annotations.glob(f"*context_relevance_{name}_*.csv")), name)
               for name in ANNOTATORS}
    return common, private


def load_items() -> pd.DataFrame:
    frames = []
    for survey in ("comune", *ANNOTATORS):
        mapping = pd.read_csv(HERE / f"{survey}_question_map.csv", dtype=str, keep_default_na=False)
        mapping["survey"] = survey
        frames.append(mapping)
    items = pd.concat(frames, ignore_index=True).set_index("question_id")
    if items.index.duplicated().any() or items.duplicated(["edos_label", "chunk_id"]).any():
        raise ValueError("Question maps overlap: each (label, chunk) item must appear once.")
    return items


def after_discussion(judges: pd.DataFrame, harmonized: pd.Series) -> pd.DataFrame:
    """Judgments after the discussion: every annotator takes the harmonized value on the
    discussed items; the other items keep their original judgments."""
    after = judges.copy()
    discussed = harmonized.notna()
    for name in after.index:
        after.loc[name, discussed] = harmonized[discussed]
    return after


def common_agreement(judges: pd.DataFrame, harmonized: pd.Series, rng) -> dict:
    """judges: annotators x items. Pilot's agreement_summary takes items x annotators.

    "after_discussion" is agreement once the discussed disagreements are resolved:
    it describes the final consensus and is higher by construction, so it is not an
    independent reliability estimate (that is "all_three", before discussion)."""
    discussed = harmonized.notna()
    after = after_discussion(judges, harmonized)
    spread_after = after.max(axis=0) - after.min(axis=0)
    return {
        "all_three": agreement_summary(judges.T.to_numpy(dtype=float), rng, BOOTSTRAP),
        "pairwise": {
            f"{a}-{b}": agreement_summary(judges.loc[[a, b]].T.to_numpy(dtype=float), rng, BOOTSTRAP)
            for a, b in [(ANNOTATORS[0], ANNOTATORS[1]), (ANNOTATORS[0], ANNOTATORS[2]), (ANNOTATORS[1], ANNOTATORS[2])]
        },
        "all_three_identical": float((judges.nunique(axis=0) == 1).mean()),
        "n_discussed_items": int(discussed.sum()),
        "after_discussion": {
            "all_three": agreement_summary(after.T.to_numpy(dtype=float), rng, BOOTSTRAP),
            "pairwise": {
                f"{a}-{b}": agreement_summary(after.loc[[a, b]].T.to_numpy(dtype=float), rng, BOOTSTRAP)
                for a, b in [(ANNOTATORS[0], ANNOTATORS[1]), (ANNOTATORS[0], ANNOTATORS[2]), (ANNOTATORS[1], ANNOTATORS[2])]
            },
            "all_three_identical": float((spread_after == 0).mean()),
            "items_unanimous_from_start": int(((judges.max(axis=0) - judges.min(axis=0)) == 0).sum()),
            "items_resolved_by_discussion": int(discussed.sum()),
            "items_still_one_point_apart": int((spread_after == 1).sum()),
        },
        "vs_harmonized_on_discussed_items": {
            name: {"exact_agreement": float((judges.loc[name, discussed] == harmonized[discussed]).mean()),
                   "mean_absolute_difference": float((judges.loc[name, discussed] - harmonized[discussed]).abs().mean())}
            for name in ANNOTATORS
        },
    }


def gold_labels(common: pd.DataFrame, private: dict, items: pd.DataFrame) -> pd.DataFrame:
    judges = common.loc[list(ANNOTATORS)]
    harmonized = common.loc["Harmonized"]
    gold_common = harmonized.where(harmonized.notna(), judges.median(axis=0))
    gold = pd.concat([gold_common, *[frame.iloc[0] for frame in private.values()]])
    if gold.isna().any() or set(gold.index) != set(items.index):
        raise ValueError("Every item needs exactly one gold label.")
    source = pd.Series("private", index=gold.index)
    source[gold_common.index] = np.where(harmonized.notna(), "harmonized", "median_of_3")
    return items[["survey", "edos_label", "chunk_id"]].join(gold.astype(int).rename("gold")).join(source.rename("gold_source"))


def runs_from_items(items: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """pipeline -> label -> chunk ids ordered by rank."""
    ranked = defaultdict(lambda: defaultdict(list))
    for _, item in items.iterrows():
        for match in json.loads(item["source_matches"]):
            ranked[match["pipeline_id"]][item["edos_label"]].append((match["reranker_rank"], item["chunk_id"]))
    runs = {}
    for pipeline, labels in ranked.items():
        runs[pipeline] = {}
        for label, pairs in labels.items():
            pairs.sort()
            if [rank for rank, _ in pairs] != list(range(1, TOP_K + 1)):
                raise ValueError(f"{pipeline} / {label}: expected ranks 1..{TOP_K}, got {pairs}")
            runs[pipeline][label] = [chunk for _, chunk in pairs]
    return runs


def label_scores(ranking: list[str], qrels: dict[str, int]) -> dict[str, float]:
    pooled = list(qrels)
    gains = np.array([[qrels[chunk] for chunk in pooled]])
    # Rank scores 4..1 for the selected chunks; 0 for the other judged chunks (below the cutoff).
    scores = np.array([[TOP_K + 1 - ranking.index(chunk) if chunk in ranking else 0 for chunk in pooled]])
    return {
        "mean_relevance@4": float(np.mean([qrels[chunk] for chunk in ranking])),
        "nDCG@4": float(ndcg_score(gains, scores, k=TOP_K)),
    }


def evaluate(runs, gold: pd.DataFrame) -> pd.DataFrame:
    qrels = defaultdict(dict)
    for row in gold.itertuples():
        qrels[row.edos_label][row.chunk_id] = int(row.gold)
    return pd.DataFrame([
        {"configuration": pipeline, "label": label, **label_scores(ranking, qrels[label])}
        for pipeline, labels in runs.items() for label, ranking in labels.items()
    ])


def summarize(per_label: pd.DataFrame, rng) -> pd.DataFrame:
    """Mean over labels with a 95% bootstrap CI over labels (pilot's interval)."""
    summary = per_label.groupby("configuration")[METRICS].mean()
    n_labels = per_label.label.nunique()
    indices = rng.integers(n_labels, size=(BOOTSTRAP, n_labels))
    for metric in METRICS:
        table = per_label.pivot(index="label", columns="configuration", values=metric)
        for configuration in summary.index:
            low, high, _ = interval(table[configuration].to_numpy()[indices].mean(axis=1))
            summary.loc[configuration, f"{metric}_ci_low"] = low
            summary.loc[configuration, f"{metric}_ci_high"] = high
    return summary.sort_values("nDCG@4", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "evaluation")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    items = load_items()
    common, private = load_annotations(args.annotations)
    agreement = common_agreement(common.loc[list(ANNOTATORS)], common.loc["Harmonized"], rng)
    gold = gold_labels(common, private, items)
    per_label = evaluate(runs_from_items(items), gold)
    summary = summarize(per_label, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold.to_csv(args.output_dir / "gold_labels.csv", index_label="question_id", encoding="utf-8")
    per_label.to_csv(args.output_dir / "scores_per_label.csv", index=False, encoding="utf-8")
    summary.to_csv(args.output_dir / "scores_summary.csv", index_label="configuration", encoding="utf-8")
    (args.output_dir / "agreement_common.json").write_text(json.dumps(agreement, indent=2) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    print(json.dumps(agreement, indent=2))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
