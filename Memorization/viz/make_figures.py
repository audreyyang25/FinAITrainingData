#!/usr/bin/env python3
"""Figures comparing control texts against court opinions.

Reads the scored CSVs in datasets/scores/ and writes PNGs to datasets/figures/.
Court figures are skipped with a note if those runs have not happened yet, so
this is safe to run at any point.

  python Memorization/viz/make_figures.py
  python Memorization/viz/make_figures.py --dark

Design notes (the parts that are decisions, not taste):
  * Form before color. Ch.1 is magnitude across nominal sources -> sorted
    horizontal bars, one hue per *identity* (control vs court), never a
    darker-where-bigger ramp: that would double-encode bar length as hue.
  * Palette is the validated default. Slots 1-3 clear every all-pairs gate in
    both modes (worst CVD dE 9.2 light / 9.4 dark; normal-vision 24.0 / 20.9).
    Light-mode aqua sits at 2.74:1 on the surface, below 3:1, so the relief rule
    applies -- every mark carries a visible direct label.
  * No dual axes anywhere. Counts and token scores never share a plot.
"""
from __future__ import annotations
import argparse, csv, glob, os, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
SCORES = os.path.join(BASE, "datasets", "scores")
FIGS = os.path.join(BASE, "datasets", "figures")
NULL_FLOOR = 1.0        # longest_run, from null_floor.json (v2 question set;
                        # the earlier 0.9 was measured on the v1 set, which still
                        # had Q05/Q06/Q15/Q18 in it)

LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e3e2de",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a")
DARK = dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a37",
            s1="#3987e5", s2="#d95926", s3="#199e70")

WORK_LABEL = {"constitution": "US Constitution", "gatsby": "The Great Gatsby",
              "pride": "Pride and Prejudice", "mobydick": "Moby-Dick",
              "gatsby_shuf": "Gatsby, scrambled"}
MODEL_LABEL = {"anthropic__claude-opus-4": "Claude Opus 4",
               "google__gemini-2.5-pro": "Gemini 2.5 Pro",
               "openai__gpt-5": "GPT-5"}


def load(tag):
    """tag -> {model_slug: [rows]} for scores_<tag>__<model>.csv"""
    out = {}
    for f in sorted(glob.glob(os.path.join(SCORES, f"scores_{tag}__*.csv"))):
        slug = os.path.basename(f)[len(f"scores_{tag}__"):-4]
        out[slug] = list(csv.DictReader(open(f, newline="")))
    return out


def attempts(rows):
    """Genuine attempts only. A prose refusal scored as an attempt contributes
    ~1 token and drags the mean to the floor -- it is a non-answer, not a miss."""
    return [r for r in rows if r.get("unknown") != "1" and r.get("refusal") != "1"
            and r.get("errored") != "1"]


def mean(rows, col="longest_run"):
    v = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    return statistics.mean(v) if v else 0.0


def style(ax, C, xlabel=None, ylabel=None):
    ax.set_facecolor(C["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"]); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=C["ink2"], labelsize=9, length=0)
    if xlabel: ax.set_xlabel(xlabel, color=C["ink2"], fontsize=9, labelpad=8)
    if ylabel: ax.set_ylabel(ylabel, color=C["ink2"], fontsize=9, labelpad=8)


def fig(C, w, h):
    f, ax = plt.subplots(figsize=(w, h), dpi=170)
    f.patch.set_facecolor(C["surface"])
    return f, ax


def save(f, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    f.savefig(p, bbox_inches="tight", facecolor=f.get_facecolor())
    plt.close(f)
    print(f"   wrote {p}")


# ---------------------------------------------------------------- figure 1
def fig_recall_by_source(C, ctrl, court):
    """Magnitude across nominal sources. The headline."""
    bars = []
    for work, label in WORK_LABEL.items():
        rs = [r for m in ctrl.values() for r in m if r["case_id"].rsplit("_ch", 1)[0] == work]
        a = attempts(rs)
        if a:
            bars.append((label, mean(a), "control"))
    for slug, rows in court.items():
        a = attempts([r for r in rows if r["tier"] == "D"])
        if a:
            bars.append((f"Court opinions — {MODEL_LABEL.get(slug, slug)}", mean(a), "court"))
    if not bars:
        return
    bars.sort(key=lambda b: b[1])

    f, ax = fig(C, 8.2, 0.42 * len(bars) + 1.9)
    style(ax, C, xlabel="Mean longest verbatim run (tokens)")
    ys = range(len(bars))
    cols = [C["s1"] if b[2] == "control" else C["s2"] for b in bars]
    ax.barh(list(ys), [b[1] for b in bars], height=0.55, color=cols, zorder=3)
    ax.set_yticks(list(ys)); ax.set_yticklabels([b[0] for b in bars], color=C["ink"], fontsize=9.5)
    # Direct label on every mark — also the relief for light-mode contrast.
    for y, b in zip(ys, bars):
        ax.text(b[1] + max(x[1] for x in bars) * 0.015, y, f"{b[1]:.1f}",
                va="center", fontsize=9, color=C["ink2"])
    ax.axvline(NULL_FLOOR, color=C["ink2"], lw=1, zorder=2)
    ax.text(NULL_FLOOR, len(bars) - 0.35, f"  chance floor {NULL_FLOOR}",
            fontsize=8.5, color=C["ink2"], va="top")
    ax.xaxis.grid(True, color=C["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Memorized text reproduces verbatim; court opinions do not",
                 color=C["ink"], fontsize=12, pad=14, loc="left")
    # Legend only the categories that have marks — a legend entry for absent data
    # tells the reader a series exists when it does not.
    present = [(k, lbl, col) for k, lbl, col in
               (("control", "Control texts", C["s1"]), ("court", "Court opinions", C["s2"]))
               if any(b[2] == k for b in bars)]
    if len(present) >= 2:
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in present],
                  [l for _, l, _ in present], frameon=False, fontsize=9,
                  labelcolor=C["ink2"], loc="lower right")
    save(f, "fig1_recall_by_source.png")


# ---------------------------------------------------------------- figure 2
def fig_pre_post(C, court):
    """Two groups per model — the study's actual design."""
    models, pre, post = [], [], []
    for slug, rows in court.items():
        a = attempts([r for r in rows if r["tier"] == "D"])
        p = [r for r in a if r["arm"] == "pre_cutoff"]
        q = [r for r in a if r["arm"] == "post_cutoff"]
        if p or q:
            models.append(MODEL_LABEL.get(slug, slug)); pre.append(mean(p)); post.append(mean(q))
    if not models:
        return
    f, ax = fig(C, 7.4, 0.85 * len(models) + 2.1)
    style(ax, C, xlabel="Mean longest verbatim run (tokens)")
    ys = range(len(models)); h = 0.3
    ax.barh([y + h / 2 + 0.02 for y in ys], pre, height=h, color=C["s1"], label="Pre-cutoff", zorder=3)
    ax.barh([y - h / 2 - 0.02 for y in ys], post, height=h, color=C["s2"], label="Post-cutoff", zorder=3)
    for y, v in zip(ys, pre):
        ax.text(v + 0.05, y + h / 2 + 0.02, f"{v:.1f}", va="center", fontsize=9, color=C["ink2"])
    for y, v in zip(ys, post):
        ax.text(v + 0.05, y - h / 2 - 0.02, f"{v:.1f}", va="center", fontsize=9, color=C["ink2"])
    ax.axvline(NULL_FLOOR, color=C["ink2"], lw=1, zorder=2)
    ax.set_yticks(list(ys)); ax.set_yticklabels(models, color=C["ink"], fontsize=10)
    ax.xaxis.grid(True, color=C["grid"], lw=0.7, zorder=0); ax.set_axisbelow(True)
    ax.set_title("Court opinions: before vs after each model's training cutoff",
                 color=C["ink"], fontsize=12, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=C["ink2"], loc="lower right")
    save(f, "fig2_pre_post_cutoff.png")


# ---------------------------------------------------------------- figure 3
def fig_prefix_length(C, ctrl, court):
    """Ordered x (prefix words) -> line. Same break, same answer, longer prompt."""
    qmap = {"Q21": 6, "Q22": 12, "Q23": 24, "Q24": 48}
    series = []
    ca = [r for m in ctrl.values() for r in attempts(m) if r["qid"] in qmap
          and r["case_id"].rsplit("_ch", 1)[0] != "gatsby_shuf"]
    if ca:
        series.append(("Control texts", C["s1"],
                       [(n, mean([r for r in ca if r["qid"] == q])) for q, n in qmap.items()]))
    co = [r for m in court.values() for r in attempts(m) if r["qid"] in qmap]
    if co:
        series.append(("Court opinions", C["s2"],
                       [(n, mean([r for r in co if r["qid"] == q])) for q, n in qmap.items()]))
    if not series:
        return
    f, ax = fig(C, 7.0, 4.4)
    style(ax, C, xlabel="Prompt length (words before a fixed break)",
          ylabel="Mean longest verbatim run (tokens)")
    for label, col, pts in series:
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2,
                marker="o", markersize=8, label=label, zorder=3)
        for x, y in pts:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color=C["ink2"])
    ax.axhline(NULL_FLOOR, color=C["ink2"], lw=1, zorder=2)
    ax.text(6, NULL_FLOOR, " chance floor", fontsize=8.5, color=C["ink2"], va="bottom")
    ax.set_xticks(list(qmap.values()))
    ax.yaxis.grid(True, color=C["grid"], lw=0.7, zorder=0); ax.set_axisbelow(True)
    ax.set_title("Does a longer prompt improve verbatim recall?",
                 color=C["ink"], fontsize=12, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=C["ink2"])
    save(f, "fig3_prefix_length.png")


# ---------------------------------------------------------------- figure 4
def fig_disposition(C, ctrl, court):
    """How each model responds — the axis that accuracy alone hides."""
    rows_out = []
    for work, label in WORK_LABEL.items():
        for slug, rows in ctrl.items():
            rs = [r for r in rows if r["case_id"].rsplit("_ch", 1)[0] == work]
            if rs:
                rows_out.append((f"{label} · {MODEL_LABEL.get(slug, slug)}", rs))
    for slug, rows in court.items():
        rs = [r for r in rows if r["tier"] == "D"]
        if rs:
            rows_out.append((f"Court opinions · {MODEL_LABEL.get(slug, slug)}", rs))
    if not rows_out:
        return
    f, ax = fig(C, 8.6, 0.32 * len(rows_out) + 1.6)
    style(ax, C, xlabel="Share of questions (%)")
    for i, (label, rs) in enumerate(rows_out):
        n = len(rs)
        a = 100 * len(attempts(rs)) / n
        u = 100 * sum(1 for r in rs if r.get("unknown") == "1") / n
        d = 100 * sum(1 for r in rs if r.get("refusal") == "1") / n
        left = 0.0
        for val, col in ((a, C["s1"]), (u, C["s3"]), (d, C["s2"])):
            if val > 0:
                # 2px surface gap between stacked segments
                ax.barh(i, val, left=left, height=0.55, color=col,
                        edgecolor=C["surface"], linewidth=1.4, zorder=3)
                if val >= 7:
                    ax.text(left + val / 2, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=8, color=C["surface"], zorder=4)
            left += val
    ax.set_yticks(range(len(rows_out)))
    ax.set_yticklabels([r[0] for r in rows_out], color=C["ink"], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_title("Response disposition: attempted, declined, or refused",
                 color=C["ink"], fontsize=12, pad=14, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ("s1", "s3", "s2")]
    ax.legend(handles, ["Attempted", "UNKNOWN", "Refused (policy)"], frameon=False,
              fontsize=9, labelcolor=C["ink2"], ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.06))
    save(f, "fig4_response_disposition.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true", help="render on the dark surface")
    args = ap.parse_args()
    C = DARK if args.dark else LIGHT

    ctrl = load("controls")
    court = load("court_opinions_v2")
    print(f"control score files : {len(ctrl)}  {sorted(ctrl)}")
    print(f"court score files   : {len(court)} {sorted(court) or '(none yet)'}\n")
    if not ctrl and not court:
        print("nothing to plot — run score_answers.py first"); return

    fig_recall_by_source(C, ctrl, court)
    fig_disposition(C, ctrl, court)
    if court:
        fig_pre_post(C, court)
    fig_prefix_length(C, ctrl, court)
    if not court:
        print("\n   (fig2 pre/post skipped — no court scores yet; rerun after those finish)")


if __name__ == "__main__":
    main()
