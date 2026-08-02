"""plot_three_ratios.py -- does family separation hold from the noisy open
vocabulary down to the clean fixed codebook?

Renders the across/within family JSD ratio at two granularities -- open vocabulary
(556 features, selection frequency) vs the fixed 20-criterion rubric. Ratio > 1
means across-family pairs diverge more than within-family pairs; if the clustering
were a phrasing artifact the fixed codebook would wash it out, but it sharpens.

This is a PURE READER: the ratios and permutation p come from
    outputs/part3_distribution/stats_<judge>.json   (written by rubric_analysis)
so this figure and plot_substance_vs_style always agree. Regenerate the stats
with `python -m part3_distribution.rubric_analysis` (which itself needs Part 2's
js_divergence_freq.csv).

    python -m part3_distribution.plot_three_ratios
    python -m part3_distribution.plot_three_ratios --judge llama-3_3-70b-instruct
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

from shared.config import part_output

FIGDIR = part_output("part3_distribution", "figures")
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#86857f", "#e8e7e2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _fmt_p(p):
    return "<.001" if p < 0.001 else ("%.3f" % p).lstrip("0")


def load_stats(judge):
    path = part_output("part3_distribution", f"stats_{judge}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run "
                         f"`python -m part3_distribution.rubric_analysis` first.")
    S = json.load(open(path))
    if S.get("open_vocab") is None:
        raise SystemExit("stats file has no open_vocab -- run "
                         "`python -m part2_coverage.nearest_neighbor --suffix _freq` "
                         "then rubric_analysis.")
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="llama-3_3-70b-instruct")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)

    S = load_stats(args.judge)
    ov, ru = S["open_vocab"], S["rubric"]
    r_ov, r_ru = ov["ratio"], ru["ratio"]
    DATA = [
        ("open vocabulary\n(556 features, freq)", r_ov, _fmt_p(ov["p"]), BLUE),
        ("20 rubric criteria\n(binary, fixed)",   r_ru, _fmt_p(ru["p"]), ORANGE),
    ]
    print(f"open-vocab ratio {r_ov:.2f} (p {_fmt_p(ov['p'])})   "
          f"rubric ratio {r_ru:.2f} (p {_fmt_p(ru['p'])})")

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
