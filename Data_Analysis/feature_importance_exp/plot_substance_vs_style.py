"""plot_substance_vs_style.py -- figures for the question "why do same-family
models cluster in JSD space: substance, intensity, or vocabulary?"

Reads what the analysis scripts already wrote (all from the Llama extraction):
    outputs/jsd_matrix_p.csv, jsd_matrix_freq.csv   (importance_distribution-style)
    outputs/coarsen_jsd_tfidf_freq.csv, coarsen_jsd_llama_freq.csv  (coarsen_jsd.py)

Writes three PNGs to outputs/figures/:
    intensity_scatter.png   -- Test 1: dropping importance weights barely moves any pair
    coarsening_ratio.png    -- Test 2: family separation survives collapsing the vocabulary
    within_across_bars.png  -- the gap itself, both weightings

    python plot_substance_vs_style.py
"""

import itertools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import output_path, ANTHROPIC, OPENAI, GEMINI, QWEN

FIGDIR = output_path("figures")
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


def _pairs(P, F):
    models = list(P.index)
    rows = []
    for a, b in itertools.combinations(models, 2):
        rows.append((P.loc[a, b], F.loc[a, b], FAM[a] == FAM[b]))
    return np.array([(p, f) for p, f, _ in rows]), np.array([s for *_, s in rows])


def fig_scatter():
    P = pd.read_csv(output_path("jsd_matrix_p.csv"), index_col=0)
    F = pd.read_csv(output_path("jsd_matrix_freq.csv"), index_col=0)
    xy, same = _pairs(P, F)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    lo, hi = 0.03, 0.125
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.text(0.118, 0.118, "y = x", color=MUTED, style="italic",
            ha="right", va="bottom", fontsize=10)
    ax.scatter(xy[~same, 0], xy[~same, 1], s=46, c=BLUE, alpha=0.55,
               edgecolors="white", linewidths=0.9, label="across-family", zorder=2)
    ax.scatter(xy[same, 0], xy[same, 1], s=60, c=ORANGE, alpha=0.95,
               edgecolors="white", linewidths=1.1, label="within-family", zorder=3)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("JSD with importance weights")
    ax.set_ylabel("JSD, selection frequency only")
    ax.set_title("Dropping the importance weights barely moves any pair",
                 fontsize=12.5, fontweight="600", loc="left", pad=30)
    ax.text(0, 1.035, "Each dot is one of 66 model pairs · nearest-neighbour family purity 9/12 either way",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "intensity_scatter.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def fig_ratio():
    t = pd.read_csv(output_path("coarsen_jsd_tfidf_freq.csv"))
    l = pd.read_csv(output_path("coarsen_jsd_llama_freq.csv"))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axhline(1.0, color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.text(6, 1.006, "no family effect", color=MUTED, fontsize=9.5, va="bottom")
    for df, color, name in ((t, BLUE, "TF-IDF (lexical)"),
                            (l, ORANGE, "Llama (semantic)")):
        ax.plot(df.k, df.ratio, "-o", color=color, lw=2.4, ms=7,
                mec="white", mew=1.4, label=name, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(560, 4.5)                         # inverted: coarser to the right
    ax.set_ylim(0.95, 1.45)
    ax.set_xticks([547, 200, 100, 50, 25, 12, 6])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("feature groups  (547 to 5, log scale — coarser toward the right)")
    ax.set_ylabel("across / within family JSD")
    ax.set_title("Family separation survives collapsing the vocabulary",
                 fontsize=12.5, fontweight="600", loc="left", pad=30)
    ax.text(0, 1.035, "Ratio > 1 = families differ on substance, not phrasing. It holds down to ~5 concepts.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "coarsening_ratio.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def fig_bars():
    data = []
    for label, name in (("Importance-weighted", "p"), ("Frequency only", "freq")):
        M = pd.read_csv(output_path(f"jsd_matrix_{name}.csv"), index_col=0)
        models = list(M.index)
        w = [M.loc[a, b] for a, b in itertools.combinations(models, 2) if FAM[a] == FAM[b]]
        x = [M.loc[a, b] for a, b in itertools.combinations(models, 2) if FAM[a] != FAM[b]]
        data.append((label, np.mean(w), np.mean(x)))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(data))[::-1]
    h = 0.34
    for i, (label, w, x) in enumerate(data):
        yy = y[i]
        ax.barh(yy + h / 2 + 0.02, w, height=h, color=ORANGE, zorder=2)
        ax.barh(yy - h / 2 - 0.02, x, height=h, color=BLUE, zorder=2)
        ax.text(w + 0.001, yy + h / 2 + 0.02, f"within  {w:.4f}", va="center", fontsize=9.5, color="#333")
        ax.text(x + 0.001, yy - h / 2 - 0.02, f"across  {x:.4f}", va="center", fontsize=9.5, color="#333")
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data], fontsize=11)
    ax.set_xlim(0, 0.105)
    ax.set_xlabel("mean pairwise JSD (547-feature resolution)")
    ax.set_title("Across-family pairs diverge ~25% more, with or without intensity",
                 fontsize=12.5, fontweight="600", loc="left", pad=14)
    ax.grid(True, axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(left=False)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "within_across_bars.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    for f in (fig_scatter, fig_ratio, fig_bars):
        print("wrote", f())


if __name__ == "__main__":
    main()
