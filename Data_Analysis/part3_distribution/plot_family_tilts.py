"""plot_family_tilts.py -- each family's compositional tilt across the 20 rubric
criteria, as diverging bars.

Raw normalized shares differ by only ~1-2pp, so plotting them directly hides the
signal. Instead each bar is the family's deviation from the 4-family mean share
(percentage points of engagement) -- centered on 0, so over/under-emphasis IS the
picture. This is exactly what drives the JSD clustering (composition, not level).
Criteria ordered by cross-family spread: the discriminating ones on top.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shared.config import part_output
from part3_distribution.rubric_analysis import load_scores, presence_matrix
from shared.jsd_stats import TIER
from part3_distribution.rubric import LABELS

BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#86857f", "#e8e7e2"
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
    norm = rate.div(rate.sum(1), axis=0)                       # each model -> distribution
    fam = pd.DataFrame({f: norm.loc[[m for m in norm.index if TIER[m][0] == f]].mean()
                        for f in FAMS})
    dev = fam.sub(fam.mean(1), axis=0) * 100                    # pp deviation from grand mean
    order = (fam.max(1) - fam.min(1)).sort_values(ascending=False).index.tolist()
    dev = dev.loc[order]

    y = np.arange(len(order))[::-1]
    xmax = np.abs(dev.values).max() * 1.18

    fig, axes = plt.subplots(1, 4, figsize=(14, 7.6), sharey=True)
    for ax, f in zip(axes, FAMS):
        vals = dev[f].values
        colors = [BLUE if v >= 0 else ORANGE for v in vals]
        ax.barh(y, vals, height=0.66, color=colors, zorder=3)
        ax.axvline(0, color="#b9b8b3", lw=1, zorder=2)
        ax.set_xlim(-xmax, xmax)
        ax.set_title(f, fontsize=12.5, fontweight="600", pad=8)
        ax.tick_params(left=False)
        ax.grid(True, axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["−1", "0", "+1"], fontsize=9, color=MUTED)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([short(i) for i in order], fontsize=9)
    axes[0].set_ylim(-0.6, len(order) - 0.4)

    fig.suptitle("Each family's compositional tilt across the 20 criteria",
                 x=0.5, y=0.995, fontsize=14, fontweight="700")
    fig.text(0.5, 0.94,
             "Deviation from the 4-family mean engagement share (pp). "
             "Blue = emphasizes more, orange = less. This composition is what drives the clustering.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.02, "deviation from mean (percentage points of engagement share)",
             ha="center", fontsize=10, color="#444")

    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    figdir = part_output("part3_distribution", "figures")
    os.makedirs(figdir, exist_ok=True)
    p = os.path.join(figdir, "family_tilts.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
