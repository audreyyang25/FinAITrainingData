#!/usr/bin/env python3
"""C5: did abstention hide memorization?

  python Memorization/viz/make_forced_figure.py [--dark]

Every model was asked the same court-opinion questions twice: once with UNKNOWN
available, then -- on exactly the rows it declined -- again with UNKNOWN removed.
If the abstentions were concealing recall, the forced answers should score near
the voluntary ones. If they were honest, the forced answers should fall toward
the null floor.

Gemini is absent by design rather than omission: it abstained on ~11% of items,
so there was no meaningful subset to re-ask and its voluntary number is already
close to unbiased.

Color stays keyed to model, matching C1/C3 -- the grouping variable here is the
prompt condition, which is carried by position instead.
"""
from __future__ import annotations
import argparse, csv, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import LIGHT, DARK, MODEL_LABEL, NULL_FLOOR, FIGS, style
from make_comparison import save

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
DS = os.path.join(BASE, "datasets")

MODELS = [("anthropic__claude-opus-4", "s1"), ("openai__gpt-5", "s3")]
POPS = [("scores_v2", "Answered voluntarily\n(UNKNOWN was available)"),
        ("scores_forced", "Forced to answer\n(the abstained rows)"),
        ("scores_merged", "All items combined\n(selection-free)")]


def read(d, slug):
    p = os.path.join(DS, d, f"scores_court_opinions_qa_v2__{slug}.csv")
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, newline="")))


def stat(rows):
    # Same admissibility rule as the other figures: tier D, minus the leaked Q03.
    rows = [r for r in rows if r.get("tier") == "D" and r.get("qid") != "Q03"]
    a = [r for r in rows if r.get("nonanswer") == "attempt"]
    return (statistics.mean(float(r["longest_run"]) for r in a), len(a)) if a else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()
    C = DARK if args.dark else LIGHT
    plt.rcParams.update({"font.family": "sans-serif", "axes.titleweight": "regular"})

    fg, ax = plt.subplots(figsize=(9.4, 4.6))
    fg.patch.set_facecolor(C["surface"])
    h = 0.74 / len(MODELS)
    for i, (slug, slot) in enumerate(MODELS):
        ys, vs, ns = [], [], []
        for j, (d, _) in enumerate(POPS):
            m, n = stat(read(d, slug))
            if m is None:
                continue
            ys.append(j + (i - (len(MODELS) - 1) / 2) * h)
            vs.append(m); ns.append(n)
        ax.barh(ys, vs, height=h * 0.88, color=C[slot], label=MODEL_LABEL[slug])
        for y, v, n in zip(ys, vs, ns):
            ax.text(v + 0.06, y, f"{v:.2f}   n={n:,}", va="center",
                    color=C["ink2"], fontsize=8)

    ax.axvline(NULL_FLOOR, color=C["ink2"], linewidth=1.2, linestyle=(0, (4, 3)))
    # Below the last bar, not above the first: the y-axis is inverted, so a
    # negative y puts this in the title's lap.
    ax.text(NULL_FLOOR + 0.05, len(POPS) - 0.45, f"null floor {NULL_FLOOR}",
            color=C["ink2"], fontsize=8, style="italic")
    ax.set_yticks(range(len(POPS)))
    ax.set_yticklabels([lbl for _, lbl in POPS], color=C["ink"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.6)
    ax.xaxis.grid(True, color=C["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    style(ax, C, xlabel="mean longest verbatim run (tokens)")
    ax.set_title("Forced answers score worse than voluntary ones —\n"
                 "the abstentions were honest, not concealment",
                 color=C["ink"], fontsize=12, pad=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["ink2"], loc="lower right")
    save(fg, "c5_forced_vs_voluntary", args.dark)


if __name__ == "__main__":
    main()
