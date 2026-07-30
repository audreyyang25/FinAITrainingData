"""plot_model_tilts.py -- each MODEL's compositional tilt across the 20 rubric
criteria, as diverging bars, grouped by family.

The per-model companion to plot_family_tilts.py: instead of averaging the three
tiers within a family, every one of the 12 models gets its own bar, so you can see
whether a family's tiers tilt TOGETHER (a coherent family signature) or spread
apart. Each bar is the model's deviation from the 12-model mean engagement share
(percentage points) -- centered on 0, so over/under-emphasis is the picture.
Because the families are balanced (3 models each), that 12-model mean equals the
4-family mean used in plot_family_tilts, so the two figures share a baseline.

Layout: 4 family panels; within each, 3 grouped bars per criterion (frontier / mid
/ small), COLORED BY TIER (identity), with polarity shown by direction from 0.
Criteria ordered by cross-model spread: the most model-discriminating on top.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import output_path
from rubric_analysis import load_scores, presence_matrix
from coarsen_jsd import TIER
from rubric import LABELS

# Tier colors = categorical slots 1-3 (blue/orange/aqua), validated all-pairs.
# Color now carries TIER IDENTITY (not sign); sign is read from bar direction.
TIER_ORDER = ["frontier", "mid", "small"]
TIER_COLOR = {"frontier": "#2a78d6", "mid": "#eb6834", "small": "#1baf7a"}
TIER_LABEL = {"frontier": "Frontier", "mid": "Mid", "small": "Small"}
EDGE, MUTED, GRID = "#3a3a38", "#86857f", "#e8e7e2"
FAMS = ["Anthropic", "OpenAI", "Gemini", "Qwen"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

short = lambda i: f"{i}. " + (LABELS[i] if len(LABELS[i]) <= 32 else LABELS[i][:31] + "…")


def main():
    recs, _ = load_scores("llama-3_3-70b-instruct")
    rate, _ = presence_matrix(recs)
    rate = rate.loc[[m for m in rate.index if m in TIER]]
    norm = rate.div(rate.sum(1), axis=0)          # each model -> distribution over criteria
    permodel = norm.T                              # criteria x 12 models (NO family averaging)
    dev = permodel.sub(permodel.mean(1), axis=0) * 100   # pp deviation from the 12-model mean

    # Criteria ordered by how much the models spread on them (most discriminating on top).
    order = (permodel.max(1) - permodel.min(1)).sort_values(ascending=False).index.tolist()
    dev = dev.loc[order]

    y = np.arange(len(order))[::-1]
    offset = {"frontier": 0.25, "mid": 0.0, "small": -0.25}   # frontier top within each group
    h = 0.23
    xmax = np.abs(dev.values).max() * 1.15
    tick = max(1, int(xmax))
    ticks = list(range(-tick, tick + 1))

    fig, axes = plt.subplots(1, 4, figsize=(15, 8.4), sharey=True)
    for ax, f in zip(axes, FAMS):
        models = sorted((m for m in dev.columns if TIER[m][0] == f),
                        key=lambda m: TIER_ORDER.index(TIER[m][1]))
        for m in models:
            key = TIER[m][1]
            ax.barh(y + offset[key], dev[m].values, height=h,
                    color=TIER_COLOR[key], edgecolor=EDGE, linewidth=0.3, zorder=3)
        ax.axvline(0, color="#b9b8b3", lw=1, zorder=2)
        ax.set_xlim(-xmax, xmax)
        ax.set_title(f, fontsize=12.5, fontweight="600", pad=8)
        ax.tick_params(left=False)
        ax.grid(True, axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:+d}".replace("+0", "0").replace("-", "−") for t in ticks],
                           fontsize=9, color=MUTED)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([short(i) for i in order], fontsize=9)
    axes[0].set_ylim(-0.7, len(order) - 0.3)

    handles = [Patch(facecolor=TIER_COLOR[t], edgecolor=EDGE, linewidth=0.3,
                     label=TIER_LABEL[t]) for t in TIER_ORDER]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.985),
               frameon=False, fontsize=10, ncol=3, columnspacing=1.4, handlelength=1.1)

    fig.suptitle("Each model's compositional tilt across the 20 criteria",
                 x=0.5, y=0.995, fontsize=14, fontweight="700")
    fig.text(0.5, 0.94,
             "Deviation from the 12-model mean engagement share (pp), one bar per model. "
             "Bars right = emphasizes more, left = less. Tiers that tilt together = a coherent family signature.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.02, "deviation from mean (percentage points of engagement share)",
             ha="center", fontsize=10, color="#444")

    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    p = os.path.join(output_path("figures"), "model_tilts.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
