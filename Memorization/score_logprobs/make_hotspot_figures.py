#!/usr/bin/env python3
"""Figures for the hot-spot sweep and the span experiment. No GPU.

  python make_hotspot_figures.py

L5  WHAT THE HOT SPOTS ARE. Every one of the 400 most confidently-predicted
    windows, decomposed by why it is predictable, split by arm. The arm split is
    the argument: post-cutoff opinions did not exist when the model was trained,
    so if they contribute as many hot spots with the same composition, the
    phenomenon is a property of judicial writing rather than memorisation.

L6  THE SPAN EXPERIMENT. The 197 windows the profile could not adjudicate --
    those restating something earlier in the same opinion, where a 2k-token
    context window supplies the answer -- retested with a 50-word prefix and
    nothing else. Three groups:
      TGT pre-cutoff spans (could be memorised), CTL random spans from the same
      opinions (controls for selection), FLR post-cutoff spans (cannot be
      memorised; the floor).
    Left panel pairs TGT against CTL within each opinion, because spans from one
    opinion are not independent. Right panel is the comparison that decides it.
"""
from __future__ import annotations
import argparse, collections, csv, glob, os, statistics, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")

try:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "viz"))
    from make_figures import LIGHT as C, style              # noqa: E402
except Exception:
    C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e3e2de",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a")

    def style(ax, C, xlabel=None, ylabel=None):
        ax.set_facecolor(C["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C["grid"])
        ax.tick_params(colors=C["ink2"], labelsize=9, length=0)
        if xlabel:
            ax.set_xlabel(xlabel, color=C["ink2"], fontsize=9, labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, color=C["ink2"], fontsize=9, labelpad=8)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# L5 is a FUNNEL, not two views of the same data. Stage 1 (left) tests every hot
# spot for recurrence; whatever survives -- and only that -- reaches stage 2
# (right). The two panels are disjoint by construction, so the same span is never
# counted twice. An earlier version put the regex-decided verdicts in the left
# panel as well, which double-counted them against the right panel's categories.
STAGE1 = [
    ("in_document_repetition", "restated earlier in the same opinion", "#b9b8b3"),
    ("recurring_furniture",    "wording recurs in post-cutoff opinions", "#8f8e89"),
    ("__survives",             "neither — carried to panel B", "#f0efec"),
]
# Two patterns carry nearly all of stage 2. scotus_boilerplate, pleading_template
# and court_furniture between them account for 4 spans, all post-cutoff, so they
# are pooled rather than given three near-invisible segments -- but they must be
# pooled, not dropped, or the post bar stops short of 100%.
STAGE2 = [
    ("statutory_citation", "statutory citation\n15 U.S.C. § 78j(b), 17 C.F.R. § 240.10b-5", "#7b8f9e"),
    ("quoted_precedent",   "quoted precedent\nreporter cite, (2d Cir. 2003), block quote", "#5d6f7d"),
    ("__other",            "syllabus / pleading template", "#a8b4bd"),
    ("__residual",         "residual — no test fired", "#eb6834"),
]
MINOR = {"scotus_boilerplate", "pleading_template", "court_furniture"}


def l5(rows, out):
    """Two-stage funnel. Panel A tests all 399 hot spots for recurrence; only the
    spans that survive both tests appear in panel B. Disjoint by construction."""
    THR = 0.25
    def survives(r):
        return (float(r["self_repetition"]) < THR
                and float(r["recur_in_post"]) < THR)

    arms = [("pre", "Pre-cutoff\nmodel may have seen these"),
            ("post", "Post-cutoff\nmodel never saw these")]
    fg, axes = plt.subplots(1, 2, figsize=(13.6, 4.4),
                            gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.32})
    fg.patch.set_facecolor(C["surface"])

    # ---- Panel A: recurrence tests, both arms ------------------------------
    ax = axes[0]
    for j, (key, lab) in enumerate(arms):
        sub = [r for r in rows if key in r["arm_manifest"]]
        left = 0.0
        for v, vlab, col in STAGE1:
            n = (sum(1 for r in sub if survives(r)) if v == "__survives"
                 else sum(1 for r in sub if r["verdict"] == v and not survives(r)))
            if not n:
                continue
            pct = 100 * n / len(sub)
            ax.barh(j, pct, left=left, height=0.5, color=col,
                    edgecolor=C["surface"], linewidth=1.4,
                    label=vlab if j == 0 else None)
            if pct >= 8:
                ax.text(left + pct / 2, j, f"{n}", ha="center", va="center",
                        fontsize=8.5,
                        color=C["ink2"] if v == "__survives" else C["surface"])
            left += pct
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(
        [f"{l}\nn={sum(1 for r in rows if k in r['arm_manifest'])}"
         for k, l in arms], color=C["ink"], fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(1.55, -0.55)
    ax.xaxis.grid(True, color=C["grid"], lw=0.7); ax.set_axisbelow(True)
    style(ax, C, xlabel="% of that arm's hot spots")
    ax.set_title("A · Does the wording recur?", color=C["ink"],
                 fontsize=10.5, pad=8, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=C["ink2"], ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.24), columnspacing=1.2)

    # ---- Panel B: only the survivors --------------------------------------
    ax = axes[1]

    # One bucket per span. A span can match several patterns -- a block quote
    # usually carries a reporter cite AND a statutory cite -- so counting
    # `category == v or category_near == v` independently made the panel sum to
    # 102 of 97. Resolve the same way classify_hotspots does: the span's own
    # match wins, the neighbourhood match is the fallback. Both arms are bucketed
    # identically, which is the point -- if the post arm looks the same, the
    # composition is a property of the genre and not of training exposure.
    def bucket(r):
        c = r["category"] if r["category"] != "-" else r["category_near"]
        if c == "-":
            return "__residual"
        return "__other" if c in MINOR else c

    # Register each legend label on its FIRST appearance in any arm. Keying it to
    # j == 0 dropped "syllabus / pleading template", which occurs only in the post
    # arm, leaving an unexplained segment in the chart.
    seen = set()
    for j, (key, lab) in enumerate(arms):
        pool = [r for r in rows if survives(r) and key in r["arm_manifest"]]
        if not pool:
            continue
        counts = collections.Counter(bucket(r) for r in pool)
        left = 0.0
        for v, vlab, col in STAGE2:
            n = counts.get(v, 0)
            if not n:
                continue
            pct = 100 * n / len(pool)
            ax.barh(j, pct, left=left, height=0.5, color=col,
                    edgecolor=C["surface"], linewidth=1.4,
                    label=None if v in seen else vlab)
            seen.add(v)
            if pct >= 6:
                ax.text(left + pct / 2, j, f"{n}", ha="center", va="center",
                        fontsize=8.5, color=C["surface"])
            left += pct
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(
        [f"{sum(1 for r in rows if survives(r) and k in r['arm_manifest'])} survivors"
         for k, _ in arms], color=C["ink"], fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(1.55, -0.55)
    ax.xaxis.grid(True, color=C["grid"], lw=0.7); ax.set_axisbelow(True)
    style(ax, C, xlabel="% of survivors")
    ax.set_title("B · If not, what is it?", color=C["ink"],
                 fontsize=10.5, pad=8, loc="left")
    ax.legend(frameon=False, fontsize=7.6, labelcolor=C["ink2"],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.1)

    fg.suptitle("The most memorised-looking text in 115 opinions is not memorised\n"
                "both arms decompose the same way — and the model never saw the "
                "post-cutoff opinions",
                color=C["ink"], fontsize=12, x=0.006, ha="left", y=1.06)
    fg.savefig(out, bbox_inches="tight", dpi=180, facecolor=C["surface"])
    plt.close(fg)
    print(f"  wrote {out}")


def l6(rows, out):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["qid"][:3]][r["case_id"]].append(float(r["logp_per_token"]))
    cm = {g: {c: statistics.mean(v) for c, v in d.items()} for g, d in by.items()}
    if not all(g in cm for g in ("TGT", "CTL", "FLR")):
        print("  L6 skipped — need TGT/CTL/FLR rows"); return

    fg, axes = plt.subplots(1, 2, figsize=(12.4, 5.4),
                            gridspec_kw={"width_ratios": [1.15, 1]}, sharey=True)
    fg.patch.set_facecolor(C["surface"])

    # Left: paired within opinion. Spans from one opinion are not independent, so
    # the honest unit is the case, and the paired direction is the result.
    ax = axes[0]
    shared = sorted(set(cm["TGT"]) & set(cm["CTL"]))
    up = 0
    for c in shared:
        a, b = cm["CTL"][c], cm["TGT"][c]
        up += b > a
        ax.plot([0, 1], [a, b], color=C["s2"] if b > a else C["ink2"],
                lw=1.3, alpha=0.55, marker="o", markersize=4)
    ax.plot([0, 1], [statistics.mean(cm["CTL"][c] for c in shared),
                     statistics.mean(cm["TGT"][c] for c in shared)],
            color=C["ink"], lw=2.6, marker="o", markersize=7, zorder=5, label="mean")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["random spans\n(CTL)", "hot spots\n(TGT)"],
                       color=C["ink"], fontsize=9)
    ax.set_xlim(-0.25, 1.25)
    ax.yaxis.grid(True, color=C["grid"], lw=0.7); ax.set_axisbelow(True)
    style(ax, C, ylabel="mean logprob per token (50-word prefix)")
    ax.set_title(f"Hot spots beat random spans in {up}/{len(shared)} opinions",
                 color=C["ink"], fontsize=10.5, pad=8, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["ink2"], loc="lower right")

    # Right: the comparison that decides it. One dot per opinion.
    ax = axes[1]
    groups = [("TGT", "Pre-cutoff\nhot spots", C["s2"]),
              ("FLR", "Post-cutoff\nhot spots", C["s3"]),
              ("CTL", "Random\nspans", C["ink2"])]
    import random as _r
    _r.seed(0)
    for i, (g, lab, col) in enumerate(groups):
        v = list(cm[g].values())
        ax.scatter([i + _r.uniform(-0.11, 0.11) for _ in v], v, s=42, color=col,
                   alpha=0.7, edgecolor="none")
        mu = statistics.mean(v)
        ax.plot([i - 0.26, i + 0.26], [mu, mu], color=C["ink"], lw=2.4, zorder=5)
        ax.text(i + 0.30, mu, f"{mu:+.2f}\nn={len(v)}", va="center", fontsize=8,
                color=C["ink2"])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([l for _, l, _ in groups], color=C["ink"], fontsize=9)
    ax.set_xlim(-0.5, len(groups) - 0.2)
    ax.yaxis.grid(True, color=C["grid"], lw=0.7); ax.set_axisbelow(True)
    style(ax, C)
    ax.set_title("…but not the opinions the model never saw  (p = 0.57)",
                 color=C["ink"], fontsize=10.5, pad=8, loc="left")

    fg.suptitle("Predictable, but not memorised\n"
                "one dot per opinion; post-cutoff opinions post-date the training "
                "data, so their level is the floor",
                color=C["ink"], fontsize=12, x=0.006, ha="left", y=1.0)
    fg.tight_layout()
    fg.savefig(out, bbox_inches="tight", dpi=180, facecolor=C["surface"])
    plt.close(fg)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", default="")
    ap.add_argument("--spans", default="")
    ap.add_argument("--out-dir", default=os.path.join(DS, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cp = args.classified or sorted(glob.glob(os.path.join(DS, "logprobs", "classified__*.csv")))[0]
    rows = list(csv.DictReader(open(cp, newline="")))
    print(f"{len(rows)} classified windows")
    l5(rows, os.path.join(args.out_dir, "l5_hotspot_composition.png"))

    sp = args.spans or sorted(glob.glob(os.path.join(DS, "logprobs", "*stage2*.csv")))[0]
    srows = [r for r in csv.DictReader(open(sp, newline="")) if r["trunc_n"] == "25"]
    print(f"{len(srows)} span rows at trunc_n=25")
    l6(srows, os.path.join(args.out_dir, "l6_span_experiment.png"))


if __name__ == "__main__":
    main()
