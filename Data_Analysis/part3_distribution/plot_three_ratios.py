"""plot_three_ratios.py -- does family separation hold from the noisy open
vocabulary down to the clean fixed codebook?

across / within family JSD ratio at two FREQUENCY-based granularities (no
importance weights, no vocabulary coarsening):

    open vocabulary (556 features, selection frequency) -- emergent extraction,
                                                            phrasing noise included
    20 rubric criteria (binary presence)                -- fixed codebook, one judge

Ratio > 1 means across-family pairs diverge more than within-family pairs. If the
family clustering were a phrasing artifact, scoring on the fixed 20-criterion
codebook would wash it out. It doesn't -- the ratio holds and sharpens -- so the
separation is substantive, not vocabulary.

Ratios and permutation p are computed here from the analysis artifacts:
    outputs/js_divergence_freq.csv       (Part 2: nearest_neighbor --suffix _freq)
    outputs/rubric_matrix_<judge>.csv    (Part 3: rubric_analysis)

    python -m part3_distribution.plot_three_ratios
    python -m part3_distribution.plot_three_ratios --judge llama-3_3-70b-instruct
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
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#86857f", "#e8e7e2"
FAM = {m["model"]: m["family"]
       for fam in (ANTHROPIC, OPENAI, GEMINI, QWEN) for m in fam}
N_PERM = 20000

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _ratio(D, models, perm=N_PERM, seed=0):
    """across/within mean JSD ratio + permutation p, from a pairwise-JSD dict
    D[(a, b)] over `models` (all of which must be in FAM). The null makes the
    family labels exchangeable across models."""
    fams = [FAM[m] for m in models]
    idx = list(itertools.combinations(range(len(models)), 2))

    def sep(labels):
        w = [D[(models[i], models[j])] for i, j in idx if labels[i] == labels[j]]
        x = [D[(models[i], models[j])] for i, j in idx if labels[i] != labels[j]]
        return np.mean(w), np.mean(x)

    within, across = sep(fams)
    ratio = across / within
    rng = np.random.default_rng(seed)
    null = np.empty(perm)
    for b in range(perm):
        w, x = sep(list(rng.permutation(fams)))
        null[b] = x / w
    p = float((np.sum(null >= ratio) + 1) / (perm + 1))
    return ratio, p


def _open_vocab_D():
    """pairwise JSD over the open vocabulary (selection frequency), gold dropped."""
    M = pd.read_csv(part_output("part2_coverage", "js_divergence_freq.csv"), index_col=0)
    models = [m for m in M.index if m in FAM]
    D = {(a, b): M.loc[a, b] for a, b in itertools.combinations(models, 2)}
    return D, models


def _rubric_D(judge):
    """pairwise JSD over the 20-criterion presence-rate distributions."""
    R = pd.read_csv(part_output("part3_distribution", f"rubric_matrix_{judge}.csv"), index_col=0)
    models = [m for m in R.index if m in FAM]
    D = {(a, b): jsd(R.loc[a].values, R.loc[b].values)
         for a, b in itertools.combinations(models, 2)}
    return D, models


def _fmt_p(p):
    return "<.001" if p < 0.001 else ("%.3f" % p).lstrip("0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="llama-3_3-70b-instruct")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)

    Dov, mov = _open_vocab_D()
    Dru, mru = _rubric_D(args.judge)
    r_ov, p_ov = _ratio(Dov, mov)
    r_ru, p_ru = _ratio(Dru, mru)

    DATA = [
        ("open vocabulary\n(556 features, freq)", r_ov, _fmt_p(p_ov), BLUE),
        ("20 rubric criteria\n(binary, fixed)",   r_ru, _fmt_p(p_ru), ORANGE),
    ]
    print(f"open-vocab ratio {r_ov:.2f} (p {_fmt_p(p_ov)})   "
          f"rubric ratio {r_ru:.2f} (p {_fmt_p(p_ru)})")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    x = range(len(DATA))
    top = max(d[1] for d in DATA)

    ax.axhline(1.0, color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.text(0.5, 1.008, "1.0 = no family effect", color=MUTED,
            fontsize=9, va="bottom", ha="center")

    ax.bar(x, [d[1] for d in DATA], width=0.55,
           color=[d[3] for d in DATA], zorder=3)
    for i, (label, r, p, _c) in enumerate(DATA):
        ax.text(i, r + (top - 1) * 0.02, f"{r:.2f}", ha="center", va="bottom",
                fontsize=13, fontweight="700", color="#222")
        ax.text(i, r + (top - 1) * 0.10, f"p {p}", ha="center", va="bottom",
                fontsize=9.5, color=MUTED)

    ax.annotate("", xy=(1, r_ru * 0.97), xytext=(0, r_ov * 1.03),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                connectionstyle="arc3,rad=-0.15"), zorder=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels([d[0] for d in DATA], fontsize=10)
    ax.set_ylim(1.0, 1.0 + (top - 1) * 1.35)
    ax.set_ylabel("across / within family JSD")
    ax.set_title("Cleaner axes sharpen family separation\n"
                 "the fixed codebook does not wash the clustering out",
                 fontsize=13, fontweight="600", loc="left", pad=18)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "separation_ratio.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
