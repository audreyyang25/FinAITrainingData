#!/usr/bin/env python3
"""Two figures from the logprob pass. No GPU -- run these back on the laptop.

  python make_curve.py --logprobs datasets/logprobs/logprobs__controls__*.csv
  python make_curve.py --logprobs ... --scores datasets/scores/scores_controls__meta-llama__llama-3.1-70b-instruct.csv

L1  EXTRACTABILITY CURVE. For a fixed suffix length n, the fraction of passages
    with p >= threshold, swept over a log grid. One line per corpus. This is the
    shape Cooper et al. report, and the reason score_logprobs.py emits a row per
    truncation length: logp_sum is not comparable across different n.

L2  CALIBRATION SCATTER. logp_per_token against the black-box longest_run for
    the same (case_id, qid). This is the figure the whole exercise is for. If
    Gatsby sits high on logprob and low on longest_run, then the 3.6 tokens the
    sampling probe measured is the PROXY failing, not the model lacking the
    text. If both are low, Llama genuinely does not carry it and the tension
    with Cooper et al. is about the serving stack or the instruct tuning.
"""
from __future__ import annotations
import argparse, collections, csv, glob, math, os, statistics, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")

# House palette if the viz layer is importable; standalone fallback so this file
# still runs on a bare box with only matplotlib.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "viz"))
    from make_figures import LIGHT as C, style          # noqa: E402
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

WORK_LABEL = {"constitution": "US Constitution", "gatsby": "The Great Gatsby",
              "gatsby_shuf": "Gatsby, scrambled"}
SLOTS = ["s1", "s2", "s3"]


def load(paths):
    rows = []
    for pat in paths:
        for f in sorted(glob.glob(pat)):
            rows += list(csv.DictReader(open(f, newline="")))
    for r in rows:
        r["_lp"] = float(r["logp_sum"])
        r["_lppt"] = float(r["logp_per_token"])
    return rows


def group_of(r):
    """Corpus grouping: control works by name, court opinions by arm."""
    w = r.get("work", "")
    if w in WORK_LABEL:
        return WORK_LABEL[w]
    return f"Court opinions ({r.get('arm', '?').replace('_cutoff', '')}-cutoff)"


def l1_curve(rows, n, out):
    sub = [r for r in rows if str(r["trunc_n"]) == str(n)]
    if not sub:
        print(f"  L1 skipped — no rows with trunc_n={n}")
        return
    groups = collections.OrderedDict()
    for r in sub:
        groups.setdefault(group_of(r), []).append(r["_lp"])
    grid = [10 ** -e for e in range(1, 13)]

    fg, ax = plt.subplots(figsize=(8.4, 5.2))
    fg.patch.set_facecolor(C["surface"])
    for i, (name, lps) in enumerate(groups.items()):
        frac = [100 * sum(1 for v in lps if v >= math.log(t)) / len(lps) for t in grid]
        ax.plot(grid, frac, marker="o", markersize=5, lw=2,
                color=C[SLOTS[i % len(SLOTS)]],
                alpha=1.0 if "scrambled" not in name else 0.5,
                label=f"{name}  (n={len(lps)})")
    ax.set_xscale("log")
    ax.invert_xaxis()          # easier thresholds to the right, as in the paper
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=C["grid"], lw=0.7)
    ax.set_axisbelow(True)
    style(ax, C, xlabel="extraction threshold p  (log scale)",
          ylabel=f"% of {n}-token passages with P(suffix | prefix) >= p")
    ax.set_title(f"(n,p)-discoverable extraction, n={n}\n"
                 "teacher-forced probability of the true continuation",
                 color=C["ink"], fontsize=12, pad=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["ink2"])
    fg.savefig(out, bbox_inches="tight", dpi=180, facecolor=C["surface"])
    plt.close(fg)
    print(f"  wrote {out}")


def l2_calibration(rows, scores_csv, n, out):
    """logp_per_token vs the black-box longest_run, joined on (case_id, qid)."""
    if not scores_csv or not os.path.exists(scores_csv):
        print("  L2 skipped — pass --scores <black-box scores csv>")
        return
    bb = {}
    for r in csv.DictReader(open(scores_csv, newline="")):
        if r.get("nonanswer") == "attempt":
            bb[(r["case_id"], r["qid"])] = float(r["longest_run"])
    sub = [r for r in rows if str(r["trunc_n"]) == str(n)
           and (r["case_id"], r["qid"]) in bb]
    if not sub:
        print("  L2 skipped — no overlap between logprob rows and scores")
        return
    groups = collections.OrderedDict()
    for r in sub:
        groups.setdefault(group_of(r), []).append(
            (r["_lppt"], bb[(r["case_id"], r["qid"])]))

    fg, ax = plt.subplots(figsize=(8.0, 5.6))
    fg.patch.set_facecolor(C["surface"])
    for i, (name, pts) in enumerate(groups.items()):
        ax.scatter([p for p, _ in pts], [q for _, q in pts], s=26,
                   color=C[SLOTS[i % len(SLOTS)]],
                   alpha=0.75 if "scrambled" not in name else 0.4,
                   edgecolor="none", label=f"{name}  (n={len(pts)})")
    allp = [p for v in groups.values() for p, _ in v]
    allq = [q for v in groups.values() for _, q in v]
    if len(allp) > 2 and len(set(allp)) > 1:
        mx, my = statistics.mean(allp), statistics.mean(allq)
        num = sum((a - mx) * (b - my) for a, b in zip(allp, allq))
        den = (sum((a - mx) ** 2 for a in allp) * sum((b - my) ** 2 for b in allq)) ** .5
        if den:
            ax.text(0.02, 0.97, f"r = {num/den:+.3f}", transform=ax.transAxes,
                    va="top", color=C["ink2"], fontsize=9, style="italic")
    ax.xaxis.grid(True, color=C["grid"], lw=0.7)
    ax.yaxis.grid(True, color=C["grid"], lw=0.7)
    ax.set_axisbelow(True)
    style(ax, C, xlabel="mean logprob per token (teacher-forced)",
          ylabel="longest verbatim run, tokens (black-box sampling)")
    ax.set_title("Does the black-box proxy track the real metric?\n"
                 "high logprob + low run = the proxy is missing memorisation",
                 color=C["ink"], fontsize=12, pad=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["ink2"], loc="lower right")
    fg.savefig(out, bbox_inches="tight", dpi=180, facecolor=C["surface"])
    plt.close(fg)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprobs", nargs="+",
                    default=[os.path.join(DS, "logprobs", "logprobs__controls__*.csv")])
    ap.add_argument("--scores", help="black-box scores csv, for the L2 join")
    ap.add_argument("--n", default="25", help="trunc_n to plot")
    ap.add_argument("--out-dir", default=os.path.join(DS, "figures"))
    args = ap.parse_args()

    rows = load(args.logprobs)
    if not rows:
        sys.exit(f"no rows matched {args.logprobs}")
    print(f"{len(rows)} logprob rows; trunc_n present: "
          f"{sorted({str(r['trunc_n']) for r in rows})}")
    nwarn = sum(1 for r in rows if r.get("align_warning"))
    if nwarn:
        print(f"  NOTE: {nwarn} rows carry align_warning — inspect before publishing")
    os.makedirs(args.out_dir, exist_ok=True)
    l1_curve(rows, args.n, os.path.join(args.out_dir, f"l1_extractability_n{args.n}.png"))
    l2_calibration(rows, args.scores, args.n,
                   os.path.join(args.out_dir, "l2_logprob_vs_blackbox.png"))


if __name__ == "__main__":
    main()
