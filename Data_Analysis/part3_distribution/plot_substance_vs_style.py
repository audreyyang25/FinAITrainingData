"""plot_substance_vs_style.py -- how big is the family gap, in each feature space?

Within-family vs across-family mean pairwise JSD, at two FREQUENCY-based
granularities (importance weights and vocabulary coarsening both dropped):

    open vocabulary (556 features, selection frequency) -- emergent extraction
    20 rubric criteria (binary presence)                -- fixed codebook

In BOTH spaces, across-family pairs diverge more than within-family pairs -- the
gap the JSD clustering rests on. The two spaces live on very different absolute
scales (556 vs 20 features), so each granularity gets its OWN panel/axis; read
within-vs-across inside a panel, and the across/within ratio in the panel title.

Inputs:
    outputs/js_divergence_freq.csv       (Part 2: nearest_neighbor --suffix _freq)
    outputs/rubric_matrix_<judge>.csv    (Part 3: rubric_analysis)

    python -m part3_distribution.plot_substance_vs_style
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shared.config import part_output, ANTHROPIC, OPENAI, GEMINI, QWEN
from shared.jsd_stats import jsd

FIGDIR = part_output("part3_distribution", "figures")
FAM = {m["model"]: m["family"]
       for fam in (ANTHROPIC, OPENAI, GEMINI, QWEN) for m in fam}

BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#86857f", "#e8e7e2"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _within_across(D, models):
    w = [D[(a, b)] for a, b in itertools.combinations(models, 2) if FAM[a] == FAM[b]]
    x = [D[(a, b)] for a, b in itertools.combinations(models, 2) if FAM[a] != FAM[b]]
    return float(np.mean(w)), float(np.mean(x))


def _open_vocab():
    M = pd.read_csv(part_output("part2_coverage", "js_divergence_freq.csv"), index_col=0)
    models = [m for m in M.index if m in FAM]           # drop gold
    D = {(a, b): M.loc[a, b] for a, b in itertools.combinations(models, 2)}
    return _within_across(D, models)


def _rubric(judge):
    R = pd.read_csv(part_output("part3_distribution", f"rubric_matrix_{judge}.csv"), index_col=0)
    models = [m for m in R.index if m in FAM]
    D = {(a, b): jsd(R.loc[a].values, R.loc[b].values)
         for a, b in itertools.combinations(models, 2)}
    return _within_across(D, models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="llama-3_3-70b-instruct")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)

    panels = [
        ("Open vocabulary  (556 features, selection frequency)", _open_vocab()),
        ("20 rubric criteria  (binary presence)", _rubric(args.judge)),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.4))
    for ax, (title, (w, x)) in zip(axes, panels):
        ax.barh(1, w, height=0.5, color=ORANGE, zorder=2)      # within
        ax.barh(0, x, height=0.5, color=BLUE, zorder=2)        # across
        ax.text(w, 1, f"  within  {w:.4f}", va="center", fontsize=9.5, color="#333")
        ax.text(x, 0, f"  across  {x:.4f}", va="center", fontsize=9.5, color="#333")
        ax.set_yticks([])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(0, x * 1.4)
        ax.set_title(f"{title}      across / within = {x / w:.2f}",
                     fontsize=10.5, loc="left", fontweight="600", pad=6)
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(left=False)

    axes[-1].set_xlabel("mean pairwise JS divergence (bits)")
    fig.suptitle("Across-family pairs diverge more than within-family "
                 "-- in both feature spaces",
                 fontsize=12.5, fontweight="600", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(FIGDIR, "within_across_bars.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
