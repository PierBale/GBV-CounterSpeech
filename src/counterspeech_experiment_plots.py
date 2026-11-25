"""Figures of the JudgeLM pairwise results (run from src after the judge stage).

  python counterspeech_experiment_plots.py [--output-dir ../data/counterspeech_v4]

Writes to <output dir>/figures/ (.pdf and .png), one panel per dataset, both presentation orders
combined as in judgelm_pairwise.tex; *: sign test p < 0.05 after Holm correction over all the pairs shown.
  judgelm_pairwise_bars    one bar per pair: system wins | ties | orders disagree | baseline wins
  judgelm_pairwise_matrix  net win rate (wins - losses) / messages of the row system against the column
                           system; lower triangle only (the matrix is antisymmetric); blank: pair not judged
"""

import json

from counterspeech_experiment_tables import DATASETS, SHORT_NAMES, holm, summary_pairs
from unito_amazon.counterspeech_experiment.config import STRATEGIES, ExperimentConfig

# Bars: red (baseline wins) / blue (system wins).
LOSS, NEUTRAL, WIN = "#e34948", "#f0efec", "#256abf"
INK, MUTED, SURFACE = "#1f1f1d", "#6b6b66", "#ffffff"
# Matrix: diverging orange (row loses) <- neutral gray -> azure (row wins), darker at the ends.
MATRIX_SCALE = ("#b9520f", "#f29a52", "#f0efec", "#6fb6e6", "#1f74b5")


def net_matrix(summary: dict, key: str, adjusted: dict):
    """Systems present in the dataset, {(row, col): (net, significant, wins, losses, ties)}."""
    cells = {}
    for a, b in summary_pairs(summary):
        row = summary["pairs"].get(f"{key}|{a}|{b}")
        if not row or not row.get("messages"):
            continue
        n, significant = row["messages"], adjusted.get((key, a, b), 1.0) < 0.05
        cells[(a, b)] = ((row["a_wins"] - row["b_wins"]) / n, significant, row["a_wins"], row["b_wins"], row["ties"])
        cells[(b, a)] = ((row["b_wins"] - row["a_wins"]) / n, significant, row["b_wins"], row["a_wins"], row["ties"])
    systems = [s for s in STRATEGIES if any(s in pair for pair in cells)]
    return systems, cells


def plot(summary: dict, out_stem) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    adjusted = holm({(key, a, b): summary["pairs"][f"{key}|{a}|{b}"]["sign_test_p"]
                     for key, _ in DATASETS for a, b in summary_pairs(summary)
                     if summary["pairs"].get(f"{key}|{a}|{b}", {}).get("messages")})
    panels = [(title, *net_matrix(summary, key, adjusted)) for key, title in DATASETS]
    panels = [panel for panel in panels if panel[1]]
    limit = max(abs(cell[0]) for _, _, cells in panels for cell in cells.values())
    limit = max(0.1, -(-limit * 10 // 1) / 10)  # round up to a tenth
    cmap = LinearSegmentedColormap.from_list("orange_azure", MATRIX_SCALE)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    # Lower triangle only (the matrix is antisymmetric): each pair once, from the point of view of
    # the system that comes later in STRATEGIES; no diagonal, so rows start at the 2nd system and
    # columns stop at the second-to-last.
    widths = [len(systems) - 1 for _, systems, _ in panels]
    # Same systems in every panel: the row labels are shown once, on the left.
    shared_rows = all(systems == panels[0][1] for _, systems, _ in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(0.62 * sum(widths) + 2.6, 0.62 * max(widths) + 1.6),
                             gridspec_kw={"width_ratios": widths, "wspace": 0.1 if shared_rows else 0.55},
                             squeeze=False)
    for position, (ax, (title, systems, cells)) in enumerate(zip(axes[0], panels)):
        rows, cols = systems[1:], systems[:-1]
        for i, row in enumerate(rows):
            for j, col in enumerate(cols[:i + 1]):
                cell = cells.get((row, col))
                face = cmap(norm(cell[0])) if cell else SURFACE
                edge = SURFACE if cell else "#d9d8d4"
                # 2px surface gap between cells; unjudged cells as a faint outline.
                ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor=face, edgecolor=edge,
                                       linewidth=0.6 if not cell else 0))
                if cell:
                    net, significant = cell[0], cell[1]
                    dark = abs(norm(net) - 0.5) > 0.36  # white text only on the darker ends
                    points = round(100 * net)
                    ax.text(j + 0.5, i + 0.5, f"{points:+d}{'*' if significant else ''}" if points else "0",
                            ha="center", va="center", fontsize=7.5,
                            color=SURFACE if dark else INK, fontweight="bold" if significant else "normal")
        ax.set_xlim(0, len(cols))
        ax.set_ylim(len(rows), 0)
        ax.set_aspect("equal")
        ax.set_xticks([k + 0.5 for k in range(len(cols))], [SHORT_NAMES[s] for s in cols],
                      rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks([k + 0.5 for k in range(len(rows))],
                      [] if shared_rows and position else [SHORT_NAMES[s] for s in rows])
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.set_xlabel("Opponent (column)", color=MUTED)
    axes[0][0].set_ylabel("System (row)", color=MUTED)

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0].tolist(),
                            fraction=0.025, pad=0.02, shrink=0.8)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=7)
    colorbar.set_ticks([-limit, 0, limit], labels=[f"{-100 * limit:.0f}\nrow loses", "0", f"+{100 * limit:.0f}\nrow wins"])
    colorbar.set_label("Net win rate (pp): wins − losses, both orders", color=MUTED, fontsize=7)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


TIE, DISAGREE, HATCH = "#bdbbb4", "#e9e7e2", "#a9a7a0"


def plot_bars(summary: dict, out_stem) -> None:
    """One 100% bar per judged pair: system wins | ties | orders disagree | baseline wins (both
    orders combined, as in judgelm_pairwise.tex, whose Ties are the sum of the two middle parts:
    a tie in both orders, or a clear but different winner in each order); one panel per dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pairs = summary_pairs(summary)
    adjusted = holm({(key, a, b): summary["pairs"][f"{key}|{a}|{b}"]["sign_test_p"]
                     for key, _ in DATASETS for a, b in pairs
                     if summary["pairs"].get(f"{key}|{a}|{b}", {}).get("messages")})
    panels = []
    for key, title in DATASETS:
        rows = [(a, b, summary["pairs"][f"{key}|{a}|{b}"]) for a, b in pairs
                if summary["pairs"].get(f"{key}|{a}|{b}", {}).get("messages")]
        if rows:
            panels.append((key, title, rows))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    heights = [len(rows) for _, _, rows in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 0.27 * sum(heights) + 1.3 * len(panels)),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.35}, squeeze=False)
    for ax, (key, title, rows) in zip(axes[:, 0], panels):
        labels = []
        for y, (baseline, system, row) in enumerate(rows):
            n = row["messages"]
            agreement = summary["order_agreement"][f"{key}|{baseline}|{system}"]
            if agreement["messages"] != n:
                raise ValueError(f"{key}|{baseline}|{system}: order agreement on a different number of messages.")
            # Combined ties = tie in both orders + opposite clear winners (opposite_winner_rate).
            disagree = agreement["opposite_winner_rate"]
            # Table convention: A = baseline, B = system.
            shares = (row["b_wins"] / n, row["ties"] / n - disagree, disagree, row["a_wins"] / n)
            left = 0.0
            for share, color, ink, hatch in zip(shares, (WIN, TIE, DISAGREE, LOSS), (SURFACE, INK, INK, SURFACE),
                                                (None, None, "////", None)):
                ax.barh(y, share, left=left, height=0.72, color=color, edgecolor=SURFACE, linewidth=1.2)
                if hatch:  # texture on the disagreement part, so it does not rely on gray tone alone
                    ax.barh(y, share, left=left, height=0.72, color="none", edgecolor=HATCH, hatch=hatch,
                            linewidth=0)
                if share >= 0.07:
                    ax.text(left + share / 2, y, f"{100 * share:.0f}", ha="center", va="center",
                            fontsize=7, color=ink)
                left += share
            significant = adjusted.get((key, baseline, system), 1.0) < 0.05
            labels.append(f"{SHORT_NAMES[system]}  vs  {SHORT_NAMES[baseline]}" + ("  *" if significant else ""))
        ax.set_yticks(range(len(rows)), labels)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.axvline(0.5, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
        ax.tick_params(length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{title} ({rows[0][2]['messages']} messages)", fontsize=9, fontweight="bold", loc="left")
    axes[0, 0].legend(handles=[Patch(color=WIN, label="System wins"), Patch(color=TIE, label="Tie"),
                               Patch(facecolor=DISAGREE, edgecolor=HATCH, hatch="////", linewidth=0,
                                     label="Orders disagree"),
                               Patch(color=LOSS, label="Baseline wins")],
                      loc="lower left", bbox_to_anchor=(0, 1.06 + 0.6 / heights[0]), ncol=4, frameon=False,
                      fontsize=7.5, handlelength=1.2, borderaxespad=0)
    axes[-1, 0].set_xlabel("Share of messages, both presentation orders combined. Orders disagree: a different clear "
                           "winner in each order.  *: sign test p < 0.05, Holm",
                           color=MUTED, fontsize=7)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    import argparse
    import dataclasses
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, help="experiment folder (default: data/counterspeech_v4)")
    args = parser.parse_args()
    config = ExperimentConfig()
    if args.output_dir:
        config = dataclasses.replace(config, output_dir=args.output_dir.resolve())
    summary = json.loads((config.output_dir / "judge_summary.json").read_text(encoding="utf-8"))
    out = config.output_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot(summary, out / "judgelm_pairwise_matrix")
    plot_bars(summary, out / "judgelm_pairwise_bars")
    print(f"Figures written to {out}: judgelm_pairwise_matrix and judgelm_pairwise_bars (.pdf/.png)")


if __name__ == "__main__":
    main()
