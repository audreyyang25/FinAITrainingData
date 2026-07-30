"""plot_three_ratios.py -- the headline: cleaner axes sharpen family separation.

Across-family / within-family JSD ratio at three measurement granularities, from
noisiest to cleanest:
    547 raw features (freq)     -- emergent extraction, phrasing noise included
     21 semantic concepts       -- post-hoc grouping of the 547
     20 rubric criteria (binary)-- fixed codebook, one judge, kappa-validated

If clustering were a vocabulary artifact, cleaning the axes would shrink the
ratio. It grows -- the noise was masking the family signal.
"""

import os
import matplotlib.pyplot as plt

from config import output_path

FIGDIR = output_path("figures")
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#86857f", "#e8e7e2"

# (label, ratio, permutation p) -- ratios from coarsen_jsd (freq) and rubric_analysis
DATA = [
    ("547 raw features\n(emergent, freq)", 1.25, "<.001"),
    ("21 semantic\nconcepts", 1.37, ".076"),
    ("20 rubric criteria\n(binary, validated)", 1.66, ".012"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = range(len(DATA))
    ratios = [d[1] for d in DATA]
    # color the cleanest bar orange, the rest blue
    colors = [BLUE, BLUE, ORANGE]

    ax.axhline(1.0, color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.text(1.5, 1.02, "1.0 = no family effect", color=MUTED, fontsize=9,
            va="bottom", ha="center")

    ax.bar(x, ratios, width=0.6, color=colors, zorder=3)
    for i, (label, r, p) in enumerate(DATA):
        ax.text(i, r + 0.015, f"{r:.2f}", ha="center", va="bottom",
                fontsize=13, fontweight="700", color="#222")
        ax.text(i, r + 0.075, f"p {p}", ha="center", va="bottom",
                fontsize=9.5, color=MUTED)

    ax.set_xticks(list(x))
    ax.set_xticklabels([d[0] for d in DATA], fontsize=10)
    ax.set_ylim(1.0, 1.85)
    ax.set_ylabel("across / within family JSD")
    ax.set_title("Cleaner axes sharpen family separation",
                 fontsize=13.5, fontweight="600", loc="left", pad=26)
    ax.annotate("", xy=(2, 1.60), xytext=(0, 1.30),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                connectionstyle="arc3,rad=-0.15"), zorder=2)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "three_ratio_comparison.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
