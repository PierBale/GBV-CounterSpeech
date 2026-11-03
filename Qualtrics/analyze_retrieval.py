"""Analyze the human relevance survey; run from any working directory.

python Qualtrics/analyze_retrieval.py
Dependencies: numpy, pandas, scipy, matplotlib (see requirements-analysis.txt).
No model inference or network access is required. Original inputs are read only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = {
    "alpha": "https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf",
    "ndcg": "https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques/",
    "tau": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html",
    "soft_labels": "https://aclanthology.org/2021.naacl-main.204/",
}


def krippendorff_alpha(ratings, level="ordinal"):
    """Units x raters; coincidence weighting supports missing ratings.

    Ordinal distances use marginal-frequency midranks, not squared raw scores.
    Units with fewer than two ratings do not contribute to either marginal.
    """
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("Expected units x annotators")
    if level not in {"ordinal", "nominal"}:
        raise ValueError(level)
    ratings = ratings[np.isfinite(ratings).sum(axis=1) >= 2]
    if not len(ratings):
        return np.nan
    values = np.unique(ratings[np.isfinite(ratings)])
    counts = np.array([(ratings == value).sum(axis=1) for value in values]).T
    m = counts.sum(axis=1)
    coincidence = np.einsum("ui,uj,u->ij", counts, counts, 1 / (m - 1))
    coincidence -= np.diag((counts / (m - 1)[:, None]).sum(axis=0))
    marginal = coincidence.sum(axis=0)
    n = marginal.sum()
    if n <= 1:
        return np.nan
    if level == "ordinal":
        midranks = np.cumsum(marginal) - marginal / 2
        distance = (midranks[:, None] - midranks[None, :]) ** 2
    else:
        distance = (values[:, None] != values[None, :]).astype(float)
    expected = np.outer(marginal, marginal) - np.diag(marginal)
    denominator = np.sum(expected * distance)
    if denominator == 0:
        return np.nan  # No marginal variation: reliability is not estimable.
    return 1 - (n - 1) * np.sum(coincidence * distance) / denominator


def interval(draws):
    valid = np.asarray(draws, dtype=float)
    valid = valid[np.isfinite(valid)]
    return (*np.quantile(valid, [0.025, 0.975]), len(valid)) if len(valid) else (np.nan, np.nan, 0)


def agreement_summary(matrix, rng, bootstrap):
    matrix = np.asarray(matrix, dtype=float)
    alpha = krippendorff_alpha(matrix)
    draws = [krippendorff_alpha(matrix[rng.integers(len(matrix), size=len(matrix))])
             for _ in range(bootstrap)]
    low, high, valid = interval(draws)
    pairwise, extreme = [], []
    for row in matrix:
        values = row[np.isfinite(row)]
        for a, b in itertools.combinations(values, 2):
            pairwise.append(a == b)
            extreme.append(abs(a - b) == 2)
    complete = matrix[np.isfinite(matrix).all(axis=1)]
    return {
        "n_items": len(matrix), "n_ratings": int(np.isfinite(matrix).sum()),
        "n_items_ge2_raters": int((np.isfinite(matrix).sum(axis=1) >= 2).sum()),
        "alpha_ordinal": alpha, "alpha_ci_low": low, "alpha_ci_high": high,
        "alpha_bootstrap_valid": valid, "alpha_nominal": krippendorff_alpha(matrix, "nominal"),
        "pairwise_exact_agreement": np.mean(pairwise) if pairwise else np.nan,
        "pairwise_extreme_disagreement": np.mean(extreme) if extreme else np.nan,
        "all_raters_exact_agreement": np.mean(np.ptp(complete, axis=1) == 0) if len(complete) else np.nan,
    }


def load_survey(results_path, mapping_path, survey_path):
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    if mapping.question_id.duplicated().any() or mapping.duplicated(["query_id", "chunk_id"]).any():
        raise ValueError("Mapping must contain one question per unique query/chunk")
    # Read the actual survey choices: Qualtrics exports choice IDs 1..3,
    # although respondents see relevance grades 0..2.
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    questions = {e["PrimaryAttribute"]: e["Payload"]
                 for e in survey["SurveyElements"] if e["Element"] == "SQ"}
    grades = {}
    for row in mapping.itertuples():
        payload = questions[row.qualtrics_question_id]
        if payload["DataExportTag"] != row.data_export_tag:
            raise ValueError(f"Survey tag mismatch: {row.question_id}")
        if payload.get("RecodeValues"):
            raise ValueError("Custom Qualtrics recodes require an explicit export mapping")
        choices = {key: int(value["Display"].split(" - ")[0])
                   for key, value in payload["Choices"].items()}
        if set(choices.values()) != {0, 1, 2}:
            raise ValueError("Expected the 0/1/2 relevance scale")
        grades[row.question_id] = choices
    with results_path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    columns = set(records[0])
    if set(mapping.question_id) != {c for c in columns if "-chunk_" in c}:
        raise ValueError("Export questions and question map differ")
    annotations, audit = [], []
    seen = set()
    for row in records:
        # Qualtrics includes question text and ImportId metadata as two data rows.
        response_id = row.get("ResponseId", "")
        if not response_id.startswith("R_"):
            continue
        if response_id in seen:
            raise ValueError("Duplicate ResponseId")
        seen.add(response_id)
        accepted = row.get("Finished") in {"1", "True", "true"} and row.get("Status") == "0"
        # Stable pseudonym, without exporting response IDs, IPs, names or dates.
        annotator = "ann_" + hashlib.sha256(response_id.encode()).hexdigest()[:8]
        audit.append({"annotator": annotator, "included": accepted,
                      "finished": row.get("Finished"), "status": row.get("Status"),
                      "n_answered": sum(bool(row[q]) for q in mapping.question_id)})
        if not accepted:
            continue
        for question in mapping.question_id:
            value = row[question].strip()
            if not value:
                continue
            if value not in grades[question]:
                raise ValueError(f"Unexpected rating code {value!r} for {question}")
            annotations.append({"annotator": annotator, "question_id": question,
                                "relevance": grades[question][value]})
    if not annotations:
        raise ValueError("No finished, non-preview responses")
    annotations = pd.DataFrame(annotations)
    matrix = annotations.pivot(index="question_id", columns="annotator", values="relevance")
    matrix = matrix.reindex(mapping.question_id)
    if matrix.notna().sum().size < 2:
        raise ValueError("At least two annotators are required")
    return mapping, annotations, matrix, pd.DataFrame(audit)


def expand_selections(mapping, run_dir):
    """Give shared judgments back to every original pipeline, preserving its rank."""
    rows = []
    lookup = {}
    for path in sorted((run_dir / "reranking").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = (path.stem, row["query_id"], row["chunk_id"])
                if key in lookup:
                    raise ValueError(f"Duplicate run item: {key}")
                lookup[key] = row
    for item in mapping.itertuples():
        if item.run_id != run_dir.name:
            raise ValueError("Map and run directory refer to different runs")
        for match in json.loads(item.source_matches):
            pipeline = match["pipeline_id"]
            key = (pipeline, item.query_id, item.chunk_id)
            source = lookup[key]
            if source["reranker_rank"] != int(match["reranker_rank"]) or source["label"] != item.edos_label:
                raise ValueError(f"Rank/label mismatch: {key}")
            rows.append({"question_id": item.question_id, "query_id": item.query_id,
                         "edos_label": item.edos_label, "chunk_id": item.chunk_id,
                         "document_source": item.document_source, "pipeline_id": pipeline,
                         "rank": int(match["reranker_rank"]),
                         "reranker_score": float(source["reranker_score"])})
    selections = pd.DataFrame(rows)
    if selections.duplicated(["pipeline_id", "query_id", "chunk_id"]).any():
        raise ValueError("Duplicated selection after expanding shared items")
    for _, group in selections.groupby(["pipeline_id", "query_id"]):
        if sorted(group["rank"]) != [1, 2, 3]:
            raise ValueError("Expected exactly the top three selections")
        scores = group.sort_values("rank").reranker_score.to_numpy()
        if not np.isfinite(scores).all() or (np.diff(scores) > 1e-10).any():
            raise ValueError("Rank and reranker score are inconsistent")
    coverage = selections.groupby("pipeline_id").query_id.agg(set)
    if any(x != set(mapping.query_id) for x in coverage):
        raise ValueError("Pipelines must cover the same queries")
    return selections


def dcg(relevance, exponential=False):
    relevance = np.asarray(relevance, dtype=float)
    gains = np.exp2(relevance) - 1 if exponential else relevance
    return float(np.sum(gains / np.log2(np.arange(len(gains)) + 2)))


def ndcg(relevance, ideal_pool, k=3, exponential=False):
    denominator = dcg(sorted(ideal_pool, reverse=True)[:k], exponential)
    return dcg(relevance[:k], exponential) / denominator if denominator > 0 else np.nan


def rank_correlations(scores, relevance):
    if len(scores) < 2 or len(set(scores)) < 2 or len(set(relevance)) < 2:
        return np.nan, np.nan
    return (float(kendalltau(scores, relevance, variant="b").statistic),
            float(spearmanr(scores, relevance).statistic))


def evaluate(selections, items, relevance_column="mean_relevance"):
    if items[relevance_column].isna().any():
        raise ValueError("Every selected/pool item needs at least one rating")
    joined = selections.merge(items[["question_id", relevance_column, "fully_relevant_fraction"]],
                              on="question_id", validate="many_to_one")
    pools = {q: group[relevance_column].to_numpy() for q, group in items.groupby("query_id")}
    rows = []
    for (pipeline, query), group in joined.groupby(["pipeline_id", "query_id"]):
        group = group.sort_values("rank")
        relevance = group[relevance_column].to_numpy()
        tau, rho = rank_correlations(group.reranker_score.tolist(), relevance.tolist())
        rows.append({"pipeline_id": pipeline, "query_id": query,
                     "edos_label": group.edos_label.iloc[0], "n_pool": len(pools[query]),
                     "mean_relevance": float(relevance.mean()),
                     "fully_relevant_fraction": float(group.fully_relevant_fraction.mean()),
                     "dcg_at3": dcg(relevance),
                     "ndcg_pool_at3": ndcg(relevance, pools[query]),
                     "ndcg_pool_at3_exponential": ndcg(relevance, pools[query], exponential=True),
                     "ndcg_order_at3": ndcg(relevance, relevance),
                     "kendall_tau_b_top3": tau, "spearman_rho_top3": rho,
                     "n_distinct_human_grades": len(set(relevance))})
    return pd.DataFrame(rows), joined


def common_pool_ordering(items, pipelines, run_dir, relevance_column="mean_relevance"):
    """Compare score order on the SAME judged candidates available to all systems.

    Never assign zero relevance or artificial scores to unretrieved candidates.
    This is conditional on the intersection, not a full-corpus reranker test.
    """
    available = {}
    for pipeline in pipelines:
        records = {}
        path = run_dir / "reranking_full" / f"{pipeline}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["query_id"], row["chunk_id"])
            if key in records:
                raise ValueError(f"Duplicate full reranking item: {key}")
            if row["pipeline_id"] != pipeline or row["run_id"] != run_dir.name:
                raise ValueError(f"Full reranking metadata mismatch in {path}")
            score = float(row["reranker_score"])
            if not np.isfinite(score):
                raise ValueError("Nonfinite reranker score")
            records[key] = score
        available[pipeline] = records
    rows = []
    for query, pool in items.groupby("query_id"):
        common = [row for row in pool.itertuples()
                  if all((query, row.chunk_id) in available[p] for p in pipelines)]
        human = [getattr(row, relevance_column) for row in common]
        for pipeline in pipelines:
            scores = [available[pipeline][query, row.chunk_id] for row in common]
            tau, rho = rank_correlations(scores, human)
            rows.append({"pipeline_id": pipeline, "query_id": query,
                         "edos_label": pool.edos_label.iloc[0], "n_common": len(common),
                         "n_pool": len(pool), "kendall_tau_b_common": tau,
                         "spearman_rho_common": rho,
                         "n_distinct_human_grades_common": len(set(human))})
    return pd.DataFrame(rows)


SUMMARY_METRICS = ["mean_relevance", "fully_relevant_fraction", "ndcg_pool_at3",
                   "ndcg_pool_at3_exponential", "ndcg_order_at3", "kendall_tau_b_top3",
                   "kendall_tau_b_common"]


def summarize_models(per_query, rng, bootstrap):
    rows = []
    for pipeline, group in per_query.groupby("pipeline_id"):
        row = {"pipeline_id": pipeline, "n_queries": len(group)}
        # Resample queries (all systems share queries), not 132 dependent selections.
        indices = rng.integers(len(group), size=(bootstrap, len(group)))
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy()
            sampled = values[indices]
            count = np.isfinite(sampled).sum(axis=1)
            draws = np.divide(np.nansum(sampled, axis=1), count,
                              out=np.full(bootstrap, np.nan), where=count > 0)
            low, high, valid = interval(draws)
            row.update({metric: group[metric].mean(), f"{metric}_n": int(np.isfinite(values).sum()),
                        f"{metric}_ci_low": low, f"{metric}_ci_high": high,
                        f"{metric}_bootstrap_valid": valid})
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("ndcg_pool_at3", ascending=False)
    result.insert(0, "rank_ndcg", result.ndcg_pool_at3.rank(method="min", ascending=False).astype(int))
    result.insert(1, "rank_relevance", result.mean_relevance.rank(method="min", ascending=False).astype(int))
    return result


def paired_comparisons(per_query, rng, bootstrap, metrics=("ndcg_pool_at3", "mean_relevance")):
    rows = []
    for metric in metrics:
        table = per_query.pivot(index="query_id", columns="pipeline_id", values=metric)
        if len(table) > 20:
            raise ValueError("Exact sign-flip test limited to 20 queries")
        for a, b in itertools.combinations(table.columns, 2):
            differences = (table[a] - table[b]).dropna().to_numpy()
            n = len(differences)
            if not n:
                continue
            signs = np.array(list(itertools.product([-1, 1], repeat=n)))
            null = (signs * differences).mean(axis=1)
            observed = differences.mean()
            p = np.mean(np.abs(null) >= abs(observed) - 1e-12)
            draws = differences[rng.integers(n, size=(bootstrap, n))].mean(axis=1)
            low, high, _ = interval(draws)
            rows.append({"metric": metric, "pipeline_a": a, "pipeline_b": b, "n_queries": n,
                         "delta_a_minus_b": observed, "delta_ci_low": low, "delta_ci_high": high,
                         "p_exact_sign_flip": p, "wins_a": int((differences > 1e-12).sum()),
                         "ties": int((abs(differences) <= 1e-12).sum()),
                         "wins_b": int((differences < -1e-12).sum())})
    result = pd.DataFrame(rows)
    for _, group in result.groupby("metric"):
        ordered = group.sort_values("p_exact_sign_flip")
        adjusted = np.minimum(1, np.maximum.accumulate(
            ordered.p_exact_sign_flip.to_numpy() * np.arange(len(ordered), 0, -1)))
        result.loc[ordered.index, "p_holm"] = adjusted
    return result


def aggregate_votes(matrix):
    """Empirical soft labels and consensus; all scores/utilities remain in 0..2.

    Soft linear relevance is E[Y], exactly the arithmetic mean. Soft exponential
    gain is E[2**Y - 1], NOT 2**E[Y] - 1. Multiplying gains by 2/3 keeps utilities
    in 0..2 and leaves nDCG invariant. These utilities are not new human ratings.
    """
    if matrix.notna().sum(axis=1).eq(0).any():
        raise ValueError("At least one rating is required for every item")
    valid = matrix.stack().to_numpy()
    if not np.isin(valid, [0, 1, 2]).all():
        raise ValueError("Human relevance grades must be 0, 1 or 2")
    result = pd.DataFrame(index=matrix.index)
    result["n_ratings"] = matrix.notna().sum(axis=1)
    for grade in range(3):
        result[f"p_relevance_{grade}"] = (matrix == grade).sum(axis=1) / result.n_ratings
    result["mean_relevance"] = matrix.mean(axis=1)
    result["median_relevance"] = matrix.median(axis=1)
    soft_expectation = result.p_relevance_1 + 2 * result.p_relevance_2
    if not np.allclose(soft_expectation, result.mean_relevance):
        raise AssertionError("Soft linear expectation must equal the mean")
    # Store the same floating representation to avoid spurious ordering of ties.
    result["soft_expected_relevance"] = result.mean_relevance
    result["fully_relevant_fraction"] = result.p_relevance_2
    result["mean_gain_exponential_0_2"] = (np.exp2(result.mean_relevance) - 1) * (2 / 3)
    result["median_gain_exponential_0_2"] = (np.exp2(result.median_relevance) - 1) * (2 / 3)
    result["soft_gain_exponential_0_2"] = (result.p_relevance_1 + 3 * result.p_relevance_2) * (2 / 3)
    return result


def compare_aggregations(selections, items, run_dir, rng, bootstrap):
    selections = selections.drop(columns=["mean_relevance", "fully_relevant_fraction"], errors="ignore")
    summaries, per_label, paired = [], [], []
    resampling_state = rng.bit_generator.state
    for aggregation, score in [("mean", "mean_relevance"), ("median", "median_relevance"),
                               ("soft", "soft_expected_relevance")]:
        # Match resamples across aggregation choices, so Monte Carlo noise
        # cannot suggest a difference between the identical linear mean/soft.
        variant_rng = np.random.default_rng()
        variant_rng.bit_generator.state = resampling_state
        metrics, _ = evaluate(selections, items, score)
        gains, _ = evaluate(selections, items, f"{aggregation}_gain_exponential_0_2")
        gains = gains[["pipeline_id", "query_id", "ndcg_pool_at3", "ndcg_order_at3"]].rename(
            columns={"ndcg_pool_at3": "ndcg_pool_at3_exponential_expected",
                     "ndcg_order_at3": "ndcg_order_at3_exponential"})
        metrics = metrics.merge(gains, on=["pipeline_id", "query_id"], validate="one_to_one")
        # The exponential column always denotes the gain appropriate to the
        # aggregation: transform consensus for mean/median; expected gain for soft.
        metrics["ndcg_pool_at3_exponential"] = metrics.pop("ndcg_pool_at3_exponential_expected")
        common = common_pool_ordering(items, sorted(selections.pipeline_id.unique()), run_dir, score)
        metrics = metrics.merge(common.drop(columns=["edos_label", "n_pool"]),
                                on=["pipeline_id", "query_id"], validate="one_to_one")
        metrics["rank_ndcg_within_label"] = metrics.groupby("query_id").ndcg_pool_at3.rank(method="min", ascending=False)
        metrics["rank_ndcg_exponential_within_label"] = metrics.groupby("query_id").ndcg_pool_at3_exponential.rank(method="min", ascending=False)
        metrics["aggregation"] = aggregation
        summary = summarize_models(metrics, variant_rng, bootstrap)
        summary["rank_ndcg_exponential"] = summary.ndcg_pool_at3_exponential.rank(method="min", ascending=False).astype(int)
        summary["aggregation"] = aggregation
        summaries.append(summary)
        per_label.append(metrics)
        tests = paired_comparisons(metrics, variant_rng, bootstrap,
                                   metrics=("mean_relevance", "ndcg_pool_at3", "ndcg_pool_at3_exponential"))
        tests["aggregation"] = aggregation
        paired.append(tests)
    comparison = pd.concat(summaries, ignore_index=True)
    all_tests = pd.concat(paired, ignore_index=True)
    ordered = all_tests.sort_values("p_exact_sign_flip")
    all_tests.loc[ordered.index, "p_holm_all_configs"] = np.minimum(1, np.maximum.accumulate(
        ordered.p_exact_sign_flip.to_numpy() * np.arange(len(ordered), 0, -1)))
    details = pd.concat(per_label, ignore_index=True)
    # nDCG is undefined for all-zero ideal pools. Match the query support when
    # comparing aggregation choices, rather than comparing 11-query and 10-query means.
    ndcg_columns = ["ndcg_pool_at3", "ndcg_pool_at3_exponential"]
    finite_queries = details.groupby("query_id")[ndcg_columns].agg(lambda s: np.isfinite(s).all()).all(axis=1)
    valid_queries = finite_queries[finite_queries].index
    matched_summaries = []
    for aggregation, group in details[details.query_id.isin(valid_queries)].groupby("aggregation"):
        variant_rng = np.random.default_rng()
        variant_rng.bit_generator.state = resampling_state
        matched = summarize_models(group, variant_rng, bootstrap)
        matched["aggregation"] = aggregation
        matched["rank_ndcg_exponential"] = matched.ndcg_pool_at3_exponential.rank(method="min", ascending=False).astype(int)
        matched_summaries.append(matched)
    matched = pd.concat(matched_summaries, ignore_index=True)
    keep = ["aggregation", "pipeline_id", "n_queries", "rank_ndcg", "rank_ndcg_exponential"]
    keep += [c for c in matched if c.startswith("ndcg_pool_at3")]
    return comparison, details, all_tests, matched[keep]


def make_plots(summary, by_label, agreement, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.fonttype"] = "none"

    pipelines = summary.pipeline_id.tolist()
    short = {p: p.replace("qwen3-embedding-8b", "Qwen-E").replace("qwen3-reranker-8b", "Qwen-R")
             .replace("bge-reranker-v2-m3", "BGE-R").replace("bge-m3", "BGE-E").replace("__", " + ")
             for p in pipelines}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for ax, metric, title, bounds in zip(
            axes, ["mean_relevance", "ndcg_pool_at3"],
            ["Rilevanza media (0-2)", "nDCG@3 sul pool annotato"], [(0, 2), (0, 1)]):
        values = summary[metric].to_numpy()
        errors = np.array([values - summary[f"{metric}_ci_low"], summary[f"{metric}_ci_high"] - values])
        x = np.arange(len(pipelines))
        ax.bar(x, values, yerr=errors, color="#287a8c", capsize=3, width=.65)
        ax.set_xticks(x, [short[p] for p in pipelines], rotation=15, ha="right")
        ax.set(ylim=bounds, title=title, xlabel="Configurazione",
               ylabel="Score umano (0–2)" if metric == "mean_relevance" else "nDCG@3")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        for position, value, upper in zip(x, values, summary[f"{metric}_ci_high"]):
            # Put values above the confidence interval so labels remain readable.
            ax.annotate(f"{value:.2f}", xy=(position, max(value, upper)),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.suptitle("Confronto configurazioni — IC 95% bootstrap sulle query")
    fig.savefig(out_dir / "model_ranking.svg", metadata={"Date": None})
    plt.close(fig)

    labels = sorted(by_label.edos_label.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained")
    for ax, metric, title, maximum in zip(axes, ["mean_relevance", "ndcg_pool_at3"],
                                          ["Rilevanza media", "nDCG@3 sul pool"], [2, 1]):
        pivot = by_label.pivot(index="edos_label", columns="pipeline_id", values=metric).reindex(index=labels, columns=pipelines)
        im = ax.pcolormesh(pivot.to_numpy(), vmin=0, vmax=maximum, cmap="YlGnBu")
        ax.set_ylim(len(labels), 0)
        ax.set_xticks(np.arange(len(pipelines)) + .5, [short[p] for p in pipelines], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)) + .5, [label.split()[0] for label in labels])
        ax.set(ylabel="Etichetta EDOS", title=title)
        for i in range(len(labels)):
            for j in range(len(pipelines)):
                val = pivot.iloc[i, j]
                ax.text(j + .5, i + .5, f"{val:.2f}", ha="center", va="center", color="white" if val > maximum * .6 else "#12212a")
        colorbar = fig.colorbar(im, ax=ax, shrink=.8)
        colorbar.solids.set_rasterized(False)
    fig.suptitle("Pattern per etichetta — una query per etichetta")
    fig.savefig(out_dir / "patterns_by_label.svg", metadata={"Date": None})
    plt.close(fig)

    local = agreement[agreement.scope == "label"].sort_values("edos_label")
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    vals = local.alpha_ordinal.to_numpy()
    # Percentile intervals need not contain the point estimate: draw endpoints directly.
    y = np.arange(len(local))
    ax.hlines(y, local.alpha_ci_low, local.alpha_ci_high, color="#287a8c")
    ax.scatter(vals, y, color="#287a8c")
    ax.set_yticks(y, [x.split()[0] for x in local.edos_label])
    ax.invert_yaxis()
    ax.axvline(0, color="#9a4d32", linestyle="--", linewidth=1)
    ax.set(xlabel="Alpha ordinale (IC 95%, bootstrap degli item)", ylabel="Etichetta EDOS",
           title="Accordo tra annotatori per etichetta")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(out_dir / "agreement_by_label.svg", metadata={"Date": None})
    plt.close(fig)


def plot_aggregation_comparison(comparison, per_label, matched, out_dir):
    import matplotlib.pyplot as plt

    pipelines = comparison[comparison.aggregation == "mean"].sort_values("rank_ndcg").pipeline_id.tolist()
    names = {p: p.replace("qwen3-embedding-8b", "Qwen-E").replace("qwen3-reranker-8b", "Qwen-R")
             .replace("bge-reranker-v2-m3", "BGE-R").replace("bge-m3", "BGE-E").replace("__", " + ")
             for p in pipelines}
    schemes = [("mean", "Media", "#287a8c"), ("median", "Mediana", "#be6d33"),
               ("soft", "Soft label", "#715caa")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    n_common = int(matched.n_queries.iloc[0])
    for ax, metric, title, maximum in zip(axes,
            ["mean_relevance", "ndcg_pool_at3", "ndcg_pool_at3_exponential"],
            ["Score umano (0–2), 11 query", f"nDCG lineare, {n_common} query comuni", f"nDCG esponenziale, {n_common} query comuni"], [2, 1, 1]):
        x = np.arange(len(pipelines))
        for j, (scheme, label, color) in enumerate(schemes):
            source = comparison if metric == "mean_relevance" else matched
            group = source[source.aggregation == scheme].set_index("pipeline_id").reindex(pipelines)
            centers = x + (j - 1) * .24
            ax.bar(centers, group[metric], width=.22, color=color, label=label)
            ax.vlines(centers, group[f"{metric}_ci_low"], group[f"{metric}_ci_high"], color="#24343d", linewidth=.8)
            for position, value, upper in zip(centers, group[metric], group[f"{metric}_ci_high"]):
                ax.annotate(f"{value:.2f}", xy=(position, max(value, upper)),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x, [names[p] for p in pipelines], rotation=20, ha="right")
        ax.set(ylim=(0, maximum), title=title, xlabel="Configurazione",
               ylabel="Score umano (0–2)" if metric == "mean_relevance" else "nDCG@3")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Media, mediana e soft label — IC 95% sulle query")
    fig.savefig(out_dir / "aggregation_comparison.svg", metadata={"Date": None})
    plt.close(fig)

    labels = sorted(per_label.edos_label.unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 11), layout="constrained")
    for row, metric, gain in [(0, "ndcg_pool_at3", "lineare"), (1, "ndcg_pool_at3_exponential", "esponenziale")]:
        for column, (scheme, label, _) in enumerate(schemes):
            ax = axes[row, column]
            pivot = per_label[per_label.aggregation == scheme].pivot(
                index="edos_label", columns="pipeline_id", values=metric).reindex(index=labels, columns=pipelines)
            mesh = ax.pcolormesh(pivot.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu")
            ax.set_ylim(len(labels), 0)
            ax.set_xticks(np.arange(len(pipelines)) + .5, [names[p] for p in pipelines], rotation=35, ha="right")
            ax.set_yticks(np.arange(len(labels)) + .5, [s.split()[0] for s in labels])
            ax.set(title=f"{label} — gain {gain}", ylabel="Etichetta EDOS")
            for i in range(len(labels)):
                for j in range(len(pipelines)):
                    value = pivot.iloc[i, j]
                    ax.text(j + .5, i + .5, f"{value:.2f}" if np.isfinite(value) else "n.d.",
                            ha="center", va="center", color="white" if value > .6 else "#12212a")
    colorbar = fig.colorbar(mesh, ax=axes, shrink=.65, label="nDCG@3 sul pool annotato")
    colorbar.solids.set_rasterized(False)
    fig.suptitle("Confronto per etichetta — una query per etichetta")
    fig.savefig(out_dir / "aggregation_patterns_by_label.svg", metadata={"Date": None})
    plt.close(fig)


def markdown_table(frame, columns):
    # Avoid an optional tabulate dependency.
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}" if np.isfinite(value) else "n.d."
        return str(value)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame[columns].itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(summary, agreement, pairwise, label_patterns, sensitivity, audit, out_dir, bootstrap, seed,
                 agreement_pairs, aggregation_tests, aggregation_matched):
    global_agreement = agreement[agreement.scope == "global"].iloc[0]
    best = summary.iloc[0]
    strongest_label = label_patterns.sort_values("alpha_ordinal", ascending=False).iloc[0]
    weakest_pool = label_patterns.sort_values("pool_mean_relevance").iloc[0]
    significance = ("Nessuna differenza tra sistemi è significativa al 5% dopo Holm."
                    if not (pairwise.p_holm < .05).any() else
                    "Alcuni confronti sono significativi al 5% dopo Holm; vedere la tabella appaiata.")
    mean_order = sensitivity[sensitivity.aggregation == "mean"].sort_values("rank_ndcg").pipeline_id.tolist()
    median_order = sensitivity[sensitivity.aggregation == "median"].sort_values("rank_ndcg").pipeline_id.tolist()
    stability = ("Il ranking con consenso medio e mediano coincide."
                 if mean_order == median_order else
                 "Il ranking cambia passando dal consenso medio a quello mediano; consultare la tabella di sensibilità.")
    text = f"""# Valutazione umana del retrieval

{audit['n_annotators']} annotatori, {audit['n_items']} coppie query–passaggio uniche,
{audit['n_queries']} query su {audit['n_labels']} etichette EDOS e
{audit['n_selections']} selezioni top-3 (4 configurazioni). Run: {audit['run_id']}.

## Risultati principali

- L'accordo corretto per il caso è basso (alpha ordinale {global_agreement.alpha_ordinal:.3f}):
  il ranking va interpretato come esplorativo e dipende da giudizi con disaccordo sostanziale.
- **{best.pipeline_id}** ha la migliore media di rilevanza ({best.mean_relevance:.3f}/2)
  e il migliore nDCG@3 ({best.ndcg_pool_at3:.3f}) con il consenso medio.
  {significance}
- L'accordo più alto è per **{strongest_label.edos_label}**
  (alpha {strongest_label.alpha_ordinal:.3f}). Il pool meno rilevante è quello di
  **{weakest_pool.edos_label}** ({weakest_pool.pool_mean_relevance:.3f}/2).
- {stability} La classifica dei singoli annotatori è esportata separatamente.

## Accordo prima dell'aggregazione

Alpha di Krippendorff **ordinale**: **{global_agreement.alpha_ordinal:.3f}**,
IC 95% [{global_agreement.alpha_ci_low:.3f}, {global_agreement.alpha_ci_high:.3f}].
Accordo esatto tra coppie di annotatori: {global_agreement.pairwise_exact_agreement:.1%};
accordo unanime: {global_agreement.all_raters_exact_agreement:.1%};
disaccordi estremi 0 contro 2: {global_agreement.pairwise_extreme_disagreement:.1%}.
Non vengono duplicate le annotazioni dei passaggi condivisi.

### Accordo per coppia di annotatori

La misura principale anche per le coppie è **alpha di Krippendorff ordinale**,
calcolato sui soli item valutati da entrambi gli annotatori, mantenendo i voti umani 0/1/2.
Gli IC 95% sono bootstrap degli item con i due voti ricampionati insieme.
Il kappa quadratico resta nel CSV come diagnostica aggiuntiva.
Alpha ordinale e kappa quadratico usano distanze e modelli di accordo atteso
diversi: i loro valori non sono direttamente intercambiabili.
L'alpha globale è stimato congiuntamente sui tre annotatori, non è la media degli alpha per coppia.

{markdown_table(agreement_pairs, ['annotator_a', 'annotator_b', 'n_items', 'alpha_ordinal', 'alpha_ci_low', 'alpha_ci_high', 'exact_agreement'])}

## Ranking delle configurazioni

Ordine principale per nDCG@3 sul pool comune annotato di ogni query.
La rilevanza media è una seconda classifica, indipendente dalla posizione.
La frazione `fully_relevant_fraction` è la quota di singoli giudizi uguali a 2,
non la precisione calcolata su una soglia del consenso.

{markdown_table(summary, ['rank_ndcg', 'rank_relevance', 'pipeline_id', 'mean_relevance', 'ndcg_pool_at3', 'ndcg_order_at3', 'kendall_tau_b_top3', 'kendall_tau_b_common'])}

Gli intervalli delle metriche e il numero di query valide sono in `model_ranking.csv`.
I confronti appaiati sono in `paired_comparisons.csv`: delta A−B, IC 95%, test
esatto a inversione dei segni e correzione di Holm **separata per metrica**.
Il test presuppone scambiabilità dei due sistemi sotto l'ipotesi nulla;
con sole 11 query la potenza statistica è limitata.

{markdown_table(pairwise, ['metric', 'pipeline_a', 'pipeline_b', 'delta_a_minus_b', 'delta_ci_low', 'delta_ci_high', 'p_holm'])}

## Pattern per etichetta

`label_patterns.csv` riassume l'accordo sugli item unici, la rilevanza dei pool
e i vincitori per rilevanza/nDCG. I pari merito sono mantenuti.
`metrics_by_label.csv` contiene ogni configurazione × etichetta e la posizione
relativa del sistema; `patterns_by_label.svg` consente il confronto visivo.

{markdown_table(label_patterns, ['edos_label', 'n_items', 'alpha_ordinal', 'pool_mean_relevance', 'best_ndcg', 'best_relevance'])}

## Sensibilità al consenso

Il confronto usa le stesse selezioni e lo stesso pool annotato per ogni query.
La media e la mediana producono uno score scalare su 0–2. Le **soft label**
conservano le frequenze empiriche `p0`, `p1`, `p2`, esportate in `item_consensus.csv`.
La loro rilevanza attesa è `p1 + 2*p2`: coincide esattamente con la media dei voti.
Quindi, con gain lineare, media e soft label danno gli stessi score e nDCG.

Per studiare l'effetto della distribuzione si aggiunge il gain esponenziale:

- media: `g = 2^mean − 1`;
- mediana: `g = 2^median − 1`;
- soft label: `g = E[2^Y − 1] = p1 + 3*p2`.

È una scelta esplicita di utilità per questa analisi: `E[g(Y)]` non coincide con
`g(E[Y])`. Il gain esponenziale atteso può favorire item polarizzati a parità di media;
non è automaticamente una stima più corretta della rilevanza. I gain sono riscalati
con il fattore 2/3 su 0–2 (nDCG è invariato); restano utilità derivate, distinte
dai voti umani. Lo score umano riportato per soft label è sempre `E[Y]` su 0–2.

Per ciascuna variante nDCG divide DCG dei gain per l'ideale ottenuto ordinando
gli stessi gain nel pool. È **nDCG dei gain attesi**, non la media degli nDCG
dei singoli annotatori né un'aspettativa del rapporto su possibili etichette.
Il denominatore è comune alle quattro pipeline entro ogni query e variante.

{markdown_table(sensitivity, ['aggregation', 'pipeline_id', 'mean_relevance', 'rank_ndcg', 'ndcg_pool_at3', 'rank_ndcg_exponential', 'ndcg_pool_at3_exponential'])}

**Query valide e confronto a parità di supporto.** Con la mediana, l'etichetta 3.3
ha ideale tutto zero: il suo nDCG non è definito. Le medie disponibili sopra usano
11 query per media/soft e 10 per mediana; non sono un confronto sullo stesso supporto.
Il confronto di nDCG fra aggregazioni usa quindi le
**{int(aggregation_matched.n_queries.iloc[0])} query valide in tutte le varianti**:

{markdown_table(aggregation_matched, ['aggregation', 'pipeline_id', 'n_queries', 'rank_ndcg', 'ndcg_pool_at3', 'rank_ndcg_exponential', 'ndcg_pool_at3_exponential'])}

Questi risultati e i loro IC sono in `aggregation_common_query_ranking.csv`.
Nel grafico di confronto, il pannello score umano usa tutte le 11 query,
mentre i pannelli nDCG usano le query valide comuni. Non si sostituisce con zero
il nDCG non definito. Le tabelle per etichetta mostrano esplicitamente i mancanti.

`aggregation_sensitivity.csv` contiene anche IC 95%, query valide e Kendall tau-b.
`aggregation_metrics_by_label.csv` mostra ogni aggregazione × query × pipeline.
`aggregation_comparison.svg` e `aggregation_patterns_by_label.svg` mostrano il confronto.
`aggregation_paired_comparisons.csv` contiene i test appaiati per score umano e
i due nDCG: `p_holm` corregge le sei coppie entro aggregazione e metrica;
`p_holm_all_configs` corregge l'intera famiglia di {len(aggregation_tests)} confronti.
Con questa correzione complessiva, i confronti con p < 0,05 sono
{int((aggregation_tests.p_holm_all_configs < .05).sum())}.

`annotator_model_ranking.csv` mostra inoltre la classifica separata per annotatore,
senza attribuire a nessuno il ruolo di gold. L'agreement si calcola sui voti originali
e non cambia passando da media a mediana o soft label.

## Metodo e limiti

- Dal QSF si ricava la corrispondenza tra codici Qualtrics 1/2/3 e rilevanza 0/1/2.
  Le due righe di metadati, le anteprime e le risposte non terminate sono escluse.
  Eventuali giudizi mancanti restano mancanti; non sono convertiti in zero.
- Un item è una **coppia query–chunk**, perché lo stesso chunk può essere rilevante
  per un'etichetta e irrilevante per un'altra. Per i sistemi, `source_matches`
  restituisce ogni item condiviso a tutti i sistemi e ai rispettivi rank originali.
- Alpha ordinale usa distanze basate sulle frequenze marginali delle categorie.
  L'alpha nominale e l'accordo esatto sono diagnostiche aggiuntive.
- Consenso principale: media dei voti 0/1/2. Questo assegna utilità lineare alla scala.
  DCG@3 = somma di `relevance_i / log2(i+1)`, i=1..3.
  nDCG@3 divide DCG per l'ideale dei **migliori tre item nell'unione annotata**
  dei top-3 di tutte le configurazioni della stessa query: denominatore uguale
  per tutti i sistemi. `ndcg_pool_at3_exponential` usa `2^relevance−1` come sensibilità.
- `ndcg_order_at3` usa come ideale solo i tre item scelti dallo stesso sistema:
  misura l'ordine condizionatamente alla selezione, può essere alto anche con
  passaggi poco utili. Se l'ideale è tutto zero, nDCG è non definito ed escluso
  dalla media; il numero di valori validi è sempre esportato.
- Kendall tau-b confronta **score del reranker e consenso umano nella stessa query**,
  con score alto = rilevanza alta, gestendo i pari merito. Spearman rho è esportato
  come controllo. Le correlazioni non sono definite con meno di due item o valori
  costanti; non sono sostituite da zero e non si concatenano score di query diverse.
- `kendall_tau_b_top3` valuta i tre item propri di ogni sistema. Per un confronto
  dell'ordine sullo **stesso insieme di item**, `kendall_tau_b_common` usa i soli item
  annotati presenti in `reranking_full` di tutte e quattro le configurazioni.
  L'intersezione contiene {audit['n_common_items']} item; copertura e query utilizzabili
  sono in `ordering_common_pool.csv`. Il confronto resta condizionato a questo sottoinsieme.
- Le metriche globali sono medie macro delle query, con uguale peso per etichetta
  in questo dataset. IC dei modelli: bootstrap delle query; confronti: bootstrap
  appaiato. IC alpha: bootstrap degli item, globale e per etichetta.
  {bootstrap} repliche, seed {seed}. Inferenza condizionata ai tre annotatori osservati;
  non stima la variabilità di una popolazione di annotatori.
- C'è **una sola query per etichetta**. I pattern per etichetta sono descrittivi,
  non prove di differenze generalizzabili. L'alpha globale può anche riflettere
  differenze tra etichette: va letto insieme agli alpha per etichetta.
- Passaggi riutilizzati tra etichette e documenti comuni introducono dipendenze;
  gli IC basati su item/query sono approssimazioni. I sistemi condividono anche
  alcune selezioni: non vanno trattate come campioni indipendenti.
- Solo i top-3 riuniti sono annotati. Non si possono stimare recall sull'intero
  corpus, nDCG@10 o qualità di tutto il ranking senza ulteriori giudizi.
  Il pool è costruito dai sistemi valutati: l'ideale è relativo a quel pool.
- Si confrontano **pipeline embedding + reranker**, non modelli isolati:
  l'effetto dei due componenti non si identifica causalmente da questo campione.

## Riferimenti

- Krippendorff, *Computing Krippendorff's Alpha-Reliability*: {SOURCES['alpha']}
- Järvelin & Kekäläinen (2002), *Cumulated Gain-based Evaluation of IR Techniques*,
  ACM TOIS, DOI 10.1145/582415.582418: {SOURCES['ndcg']}
- SciPy, `kendalltau`, formula tau-b e gestione dei pari merito: {SOURCES['tau']}
- Fornaciari et al. (2021), soft label come distribuzioni dei giudizi: {SOURCES['soft_labels']}

## Riproduzione

```powershell
python Qualtrics/analyze_retrieval.py --bootstrap {bootstrap} --seed {seed}
python -m unittest discover -s Qualtrics/tests -v
```

Input e hash SHA-256 sono in `analysis_audit.json`. I dati originali non sono modificati.
"""
    (out_dir / "report.md").write_text(text, encoding="utf-8")


def run_analysis(results_path=HERE / "Best-retrieval-results.csv",
                 mapping_path=HERE / "Best_retrieval_config_question_map.csv",
                 survey_path=HERE / "Best_retrieval_config.qsf",
                 run_dir=ROOT / "data/card_creation_comparison/runs/confronto-02",
                 out_dir=HERE / "analysis", bootstrap=10000, seed=20261103):
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    paths = [Path(p) for p in (results_path, mapping_path, survey_path, run_dir, out_dir)]
    results_path, mapping_path, survey_path, run_dir, out_dir = paths
    rng = np.random.default_rng(seed)
    mapping, annotations, matrix, respondents = load_survey(results_path, mapping_path, survey_path)
    selections = expand_selections(mapping, run_dir)
    items = mapping[["question_id", "query_id", "edos_label", "chunk_id", "document_source"]].copy()
    consensus = aggregate_votes(matrix)
    for column in consensus:
        items[column] = consensus[column].to_numpy()
    agreement = [{"scope": "global", "edos_label": "ALL",
                  **agreement_summary(matrix.to_numpy(), rng, bootstrap)}]
    for label, group in items.groupby("edos_label"):
        agreement.append({"scope": "label", "edos_label": label,
                          **agreement_summary(matrix.loc[group.question_id].to_numpy(), rng, bootstrap)})
    agreement = pd.DataFrame(agreement)
    # Separate stream preserves the existing global/model bootstrap results.
    pair_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    pairs = []
    for a, b in itertools.combinations(matrix.columns, 2):
        sub = matrix[[a, b]].dropna()
        values = sub.to_numpy()
        if not len(values):
            pairs.append({"annotator_a": a, "annotator_b": b, "n_items": 0,
                          "alpha_ordinal": np.nan, "alpha_ci_low": np.nan,
                          "alpha_ci_high": np.nan, "alpha_bootstrap_valid": 0,
                          "exact_agreement": np.nan, "cohen_kappa_quadratic": np.nan})
            continue
        pair_summary = agreement_summary(values, pair_rng, bootstrap)
        pair_counts = np.array([np.bincount(values[:, i].astype(int), minlength=3) for i in range(2)])
        distance = (np.arange(3)[:, None] - np.arange(3)[None, :]) ** 2 / 4
        expected = np.outer(*pair_counts) / len(values) ** 2
        observed_disagreement = np.mean((values[:, 0] - values[:, 1]) ** 2 / 4)
        expected_disagreement = np.sum(expected * distance)
        pairs.append({"annotator_a": a, "annotator_b": b, "n_items": len(values),
                      **{key: pair_summary[key] for key in
                         ["alpha_ordinal", "alpha_ci_low", "alpha_ci_high", "alpha_bootstrap_valid"]},
                      "exact_agreement": np.mean(values[:, 0] == values[:, 1]),
                      "cohen_kappa_quadratic": 1 - observed_disagreement / expected_disagreement if expected_disagreement else np.nan})
    agreement_pairs = pd.DataFrame(pairs)
    per_query, selections = evaluate(selections, items)
    common = common_pool_ordering(items, sorted(selections.pipeline_id.unique()), run_dir)
    per_query = per_query.merge(common.drop(columns=["edos_label", "n_pool"]),
                                on=["pipeline_id", "query_id"], validate="one_to_one")
    per_query["rank_ndcg_within_label"] = per_query.groupby("query_id").ndcg_pool_at3.rank(method="min", ascending=False)
    summary = summarize_models(per_query, rng, bootstrap)
    paired = paired_comparisons(per_query, rng, bootstrap)
    aggregation_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    sensitivity, aggregation_by_label, aggregation_tests, aggregation_matched = compare_aggregations(
        selections, items, run_dir, aggregation_rng, bootstrap)
    annotator_rankings = []
    for annotator in matrix:
        individual_items = items.copy()
        individual_items["individual"] = matrix[annotator].to_numpy()
        if individual_items.individual.isna().any():
            continue  # Do not silently rank systems on different incomplete pools.
        metrics, _ = evaluate(selections.drop(columns=["mean_relevance", "fully_relevant_fraction"]), individual_items, "individual")
        grouped = metrics.groupby("pipeline_id")[["mean_relevance", "ndcg_pool_at3"]].mean().reset_index()
        grouped["annotator"] = annotator
        grouped["rank_ndcg"] = grouped.ndcg_pool_at3.rank(method="min", ascending=False).astype(int)
        annotator_rankings.append(grouped)
    label_patterns = agreement[agreement.scope == "label"].copy()
    label_patterns = label_patterns.merge(items.groupby("edos_label").mean_relevance.mean().rename("pool_mean_relevance"), on="edos_label")
    for metric, column in [("ndcg_pool_at3", "best_ndcg"), ("mean_relevance", "best_relevance")]:
        best = per_query[per_query[metric].sub(per_query.groupby("edos_label")[metric].transform("max")).abs() < 1e-12]
        winners = best.groupby("edos_label").pipeline_id.agg("; ".join).rename(column)
        label_patterns = label_patterns.merge(winners, on="edos_label", how="left")
    inputs = [results_path, mapping_path, survey_path]
    inputs += sorted((run_dir / "reranking").glob("*.jsonl"))
    inputs += sorted((run_dir / "reranking_full").glob("*.jsonl"))
    audit = {"run_id": run_dir.name, "n_annotators": len(matrix.columns), "n_items": len(items),
             "n_queries": items.query_id.nunique(), "n_labels": items.edos_label.nunique(),
             "n_selections": len(selections), "n_missing_ratings": int(matrix.isna().sum().sum()),
             "n_common_items": int(common.groupby("query_id").n_common.first().sum()),
             "bootstrap": bootstrap, "seed": seed, "sources": SOURCES,
             "pairwise_bootstrap_seed_sequence": [seed, 1],
             "aggregation_bootstrap_seed_sequence": [seed, 2],
             "soft_label_definition": "Empirical vote distribution; linear E[Y]; exponential E[2^Y-1]",
             "annotators_skipped_in_individual_ranking": [a for a in matrix if matrix[a].isna().any()],
             "inputs_sha256": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"annotations_long.csv": annotations, "annotators.csv": respondents,
               "item_consensus.csv": items, "pipeline_selections.csv": selections,
               "agreement.csv": agreement, "agreement_pairwise.csv": agreement_pairs,
               "metrics_by_label.csv": per_query, "model_ranking.csv": summary,
               "paired_comparisons.csv": paired, "ordering_common_pool.csv": common,
               "aggregation_sensitivity.csv": sensitivity, "label_patterns.csv": label_patterns,
               "aggregation_metrics_by_label.csv": aggregation_by_label,
               "aggregation_common_query_ranking.csv": aggregation_matched,
               "aggregation_paired_comparisons.csv": aggregation_tests}
    if annotator_rankings:
        outputs["annotator_model_ranking.csv"] = pd.concat(annotator_rankings, ignore_index=True)
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    (out_dir / "analysis_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    make_plots(summary, per_query, agreement, out_dir)
    plot_aggregation_comparison(sensitivity, aggregation_by_label, aggregation_matched, out_dir)
    write_report(summary, agreement, paired, label_patterns, sensitivity, audit, out_dir, bootstrap, seed,
                 agreement_pairs, aggregation_tests, aggregation_matched)
    return {"summary": summary, "agreement": agreement, "per_label": per_query,
            "label_patterns": label_patterns, "paired": paired, "audit": audit,
            "agreement_pairs": agreement_pairs, "aggregation_comparison": sensitivity,
            "aggregation_matched": aggregation_matched,
            "aggregation_by_label": aggregation_by_label, "aggregation_tests": aggregation_tests}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "Best-retrieval-results.csv")
    parser.add_argument("--mapping", type=Path, default=HERE / "Best_retrieval_config_question_map.csv")
    parser.add_argument("--survey", type=Path, default=HERE / "Best_retrieval_config.qsf")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/card_creation_comparison/runs/confronto-02")
    parser.add_argument("--out-dir", type=Path, default=HERE / "analysis")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    analysis = run_analysis(args.results, args.mapping, args.survey, args.run_dir,
                            args.out_dir, args.bootstrap, args.seed)
    print(analysis["summary"][["rank_ndcg", "pipeline_id", "mean_relevance", "ndcg_pool_at3", "kendall_tau_b_common"]].to_string(index=False))
    print(analysis["agreement"].query("scope == 'global'").to_string(index=False))
    print(f"Report: {args.out_dir / 'report.md'}")
