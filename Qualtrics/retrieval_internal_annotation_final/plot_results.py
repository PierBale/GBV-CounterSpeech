"""Figures for the context-relevance evaluation (reads results/evaluation/*).

Run evaluate_retrieval.py first, then, from the repository root:
    python Qualtrics/retrieval_internal_annotation_final/plot_results.py
PNG (300 dpi) and PDF are written to results/evaluation/figures/.
"""


import json
import textwrap
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


HERE = Path(__file__).resolve().parent
EVALUATION = HERE / "results" / "evaluation"
FIGURES = EVALUATION / "figures"

# Reference palette (dataviz skill), light mode; validated for the slots used.
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

NAMES = {
    "qwen3-embedding-8b__qwen3-reranker-8b": "Qwen3-Emb + Qwen3-Rerank",
    "bge-m3__qwen3-reranker-8b": "BGE-M3 + Qwen3-Rerank",
    "qwen3-embedding-8b__bge-reranker-v2-m3": "Qwen3-Emb + BGE-Rerank",
    "bge-m3__bge-reranker-v2-m3": "BGE-M3 + BGE-Rerank",
}
ORDER = list(NAMES)  # fixed: colour follows the configuration, not its rank

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": TEXT,
    "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def short_label(label: str) -> str:
    """Full EDOS label, wrapped on two lines instead of truncated."""
    return textwrap.fill(label, width=34)


def figure_model_ranking(summary: pd.DataFrame):
    """As the pilot's model_ranking: mean relevance (0-2) and nDCG, 95% CI over labels."""
    data = summary.loc[ORDER[::-1]]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.8), sharey=True)
    panels = [("mean_relevance@4", "Mean relevance of the top-4 (0–2)", 2.0, "{:.2f}"),
              ("nDCG@4", "nDCG@4 on the judged pool", 1.0, "{:.3f}")]
    for ax, (metric, title, limit, fmt) in zip(axes, panels):
        values = data[metric]
        ax.barh(y, values, height=0.55, color=SERIES[0], zorder=2)
        ax.errorbar(values, y, xerr=[values - data[f"{metric}_ci_low"], data[f"{metric}_ci_high"] - values],
                    fmt="none", ecolor=TEXT, elinewidth=1, capsize=3, zorder=3)
        for yi, (value, high) in enumerate(zip(values, data[f"{metric}_ci_high"])):
            ax.text(high + limit * 0.015, yi, fmt.format(value), va="center", color=TEXT, fontsize=8.5)
        ax.set_xlim(0, limit)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=9.5)
    axes[0].set_yticks(y, [NAMES[c] for c in data.index], color=TEXT)
    fig.suptitle("Configurations compared — mean over 11 EDOS labels, 95% bootstrap CI over labels",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.03)
    save(fig, "1_model_ranking")


def figure_factorial(summary: pd.DataFrame):
    rerankers = [("bge-reranker-v2-m3", "BGE-Rerank"), ("qwen3-reranker-8b", "Qwen3-Rerank")]
    embedders = [("bge-m3", "BGE-M3"), ("qwen3-embedding-8b", "Qwen3-Emb")]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for i, (embedder, name) in enumerate(embedders):
        values = [summary.loc[f"{embedder}__{reranker}", "nDCG@4"] for reranker, _ in rerankers]
        ax.plot([0, 1], values, color=SERIES[i], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(1.06, values[1], f"{name}  {values[1]:.3f}", va="center", color=TEXT, fontsize=8.5)
        ax.text(-0.06, values[0], f"{values[0]:.3f}", va="center", ha="right", color=TEXT_2, fontsize=8)
    ax.set_xticks([0, 1], [name for _, name in rerankers], color=TEXT)
    ax.set_xlim(-0.35, 1.75)
    ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("nDCG@4")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Embedder × reranker: both effects add up")
    save(fig, "3_embedder_x_reranker")


def figure_heatmap(per_label: pd.DataFrame):
    """As the pilot's patterns_by_label: mean relevance and nDCG@4 per EDOS label."""
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.4), sharey=True)
    for ax, (metric, title, maximum) in zip(axes, [("mean_relevance@4", "Mean relevance of the top-4 (0–2)", 2.0),
                                                   ("nDCG@4", "nDCG@4", 1.0)]):
        wide = per_label.pivot(index="label", columns="configuration", values=metric)[ORDER]
        image = ax.imshow(wide.to_numpy(), cmap=cmap, vmin=0, vmax=maximum, aspect="auto")
        for (row, column), value in np.ndenumerate(wide.to_numpy()):
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if value >= 0.55 * maximum else TEXT)
        ax.set_xticks(range(len(ORDER)), [NAMES[c].replace(" + ", "\n+ ") for c in ORDER], color=TEXT, fontsize=7.5)
        ax.set_yticks(range(len(wide)), [short_label(label) for label in wide.index], color=TEXT, fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(ORDER)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(wide)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, orientation="horizontal", location="top")
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7, colors=TEXT_2)
        ax.set_title(title, fontsize=9.5, pad=58)
    fig.suptitle("Patterns per EDOS label — one query per label", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", y=1.05)
    save(fig, "4_patterns_per_label")


def figure_agreement(agreement: dict):
    # Values from evaluate_retrieval.py (pilot's agreement_summary): alpha with
    # item-bootstrap CI; exact agreement = identical judgments (pairs: pairwise
    # exact agreement; all three: share of items with three identical judgments).
    rows = [(pair.replace("-", " – "), stats["alpha_ordinal"], stats["alpha_ci_low"], stats["alpha_ci_high"],
             stats["pairwise_exact_agreement"]) for pair, stats in agreement["pairwise"].items()]
    overall = agreement["all_three"]
    rows.append(("all three", overall["alpha_ordinal"], overall["alpha_ci_low"], overall["alpha_ci_high"],
                 agreement["all_three_identical"]))
    rows = rows[::-1]
    names = [r[0] for r in rows]
    alphas, lows, highs, percent = (np.array([r[i] for r in rows]) for i in range(1, 5))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
    colors = [SERIES[0] if name != "all three" else "#104281" for name in names]
    axes[0].hlines(range(len(names)), lows, highs, color=TEXT_2, linewidth=1.2, zorder=2)
    axes[0].scatter(alphas, range(len(names)), color=colors, s=60, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, value in enumerate(alphas):
        # Above the point, so the label never collides with the threshold line.
        axes[0].text(value, yi + 0.14, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
    axes[0].set_xlim(-0.4, 1)
    axes[0].axvline(0, color=GRID, linewidth=1, zorder=1)
    axes[1].barh(range(len(names)), percent, height=0.55, color=colors, zorder=2)
    for yi, value in enumerate(percent):
        axes[1].text(value + 0.02, yi, f"{value:.0%}", va="center", fontsize=8.5, color=TEXT)
    axes[1].set_xlim(0, 1)
    for ax, title in ((axes[0], "Krippendorff's α (ordinal), 95% CI"), (axes[1], "Exact agreement")):
        ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=9.5)
    axes[0].axvline(0.667, color=TEXT_2, linewidth=1, linestyle=(0, (3, 2)), zorder=3)
    axes[0].text(0.675, len(names) - 0.55, "0.667 (Krippendorff's\ntentative threshold)", fontsize=7, color=TEXT_2, va="top")
    axes[0].set_yticks(range(len(names)), names, color=TEXT)
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle("Agreement on the common survey (29 items, before discussion)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", y=1.04)
    save(fig, "6_agreement_common")


def main():
    summary = pd.read_csv(EVALUATION / "scores_summary.csv", index_col="configuration")
    per_label = pd.read_csv(EVALUATION / "scores_per_label.csv")
    agreement = json.loads((EVALUATION / "agreement_common.json").read_text(encoding="utf-8"))
    figure_model_ranking(summary)
    figure_factorial(summary)
    figure_heatmap(per_label)
    figure_agreement(agreement)
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()
