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


def slug(m):
    """Normalise a model id for comparison across HF and OpenRouter casing:
    meta-llama/Llama-3.1-70B-Instruct  vs  meta-llama/llama-3.1-70b-instruct."""
    return m.strip().lower().replace("/", "__")


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
    """logp_per_token vs the black-box longest_run, joined on (case_id, qid).

    `rows` MUST already be filtered to one model. Joining several models' logprobs
    against one black-box file compares, say, 8B-base's probabilities to
    70B-Instruct's sampling behaviour -- a meaningless pairing that still produces
    a plausible-looking scatter and an r value. main() enforces the filter.
    """
    if not scores_csv or not os.path.exists(scores_csv):
        print("  L2 skipped — pass --scores <black-box scores csv>")
        return
    models = {r["model"] for r in rows}
    if len(models) > 1:
        print(f"  L2 skipped — {len(models)} models in the row set; filter with --model")
        return
    bb_model = os.path.basename(scores_csv).split("__", 1)[-1][:-4]
    if slug(bb_model) != slug(next(iter(models))):
        print(f"  L2 WARNING: logprobs are {next(iter(models))} but the scores file "
              f"is {bb_model}. These are different models; the join is not a "
              f"calibration. Pass matching --model/--scores.")
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


def l3_base_vs_instruct(rows, n, out):
    """Does instruction tuning suppress memorisation, and is it work-specific?"""
    fam = collections.OrderedDict()
    for r in rows:
        if str(r["trunc_n"]) != str(n):
            continue
        m = r["model"]
        base = m.replace("-Instruct", "")
        fam.setdefault(base, {}).setdefault("inst" if m.endswith("-Instruct") else "base",
                                            []).append(r)
    works = ["constitution", "gatsby", "gatsby_shuf"]
    sizes = [b for b in fam if "base" in fam[b] and "inst" in fam[b]]
    if not sizes:
        print("  L3 skipped — need both a base and an -Instruct file")
        return
    fg, axes = plt.subplots(1, len(sizes), figsize=(5.6 * len(sizes), 4.6), sharex=True)
    axes = axes if len(sizes) > 1 else [axes]
    fg.patch.set_facecolor(C["surface"])
    h = 0.36
    for ax, size in zip(axes, sizes):
        for j, w in enumerate(works):
            for k, cond in enumerate(("base", "inst")):
                v = [float(r["logp_per_token"]) for r in fam[size][cond] if r["work"] == w]
                if not v:
                    continue
                y = j + (k - 0.5) * h
                mu = statistics.mean(v)
                ax.barh(y, mu, height=h * 0.85, color=C[SLOTS[j % 3]],
                        alpha=1.0 if cond == "base" else 0.45,
                        label=cond if j == 0 else None)
                ax.text(mu - 0.12, y, f"{mu:.2f}", va="center", ha="right",
                        fontsize=7.5, color=C["ink2"])
        ax.set_yticks(range(len(works)))
        ax.set_yticklabels([WORK_LABEL[w] for w in works], color=C["ink"], fontsize=9)
        ax.invert_yaxis()
        ax.xaxis.grid(True, color=C["grid"], lw=0.7)
        ax.set_axisbelow(True)
        style(ax, C, xlabel="mean logprob per token")
        ax.set_title(size.split("/")[-1], color=C["ink"], fontsize=10, pad=8, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=C["ink2"], loc="lower left")
    fg.suptitle("Instruction tuning suppresses Gatsby, not the Constitution\n"
                "solid = base, faded = -Instruct; closer to 0 is more memorised",
                color=C["ink"], fontsize=12, x=0.006, ha="left", y=0.99)
    fg.tight_layout()
    fg.savefig(out, bbox_inches="tight", dpi=180, facecolor=C["surface"])
    plt.close(fg)
    print(f"  wrote {out}")


def series_key(r):
    """Series a row belongs to for the prefix sweep.

    Control works keep their own name; court rows collapse to one series per
    arm. NOTE the arm here is the QA file's baked-in global 2023-12-31 split,
    copied through by pairs.py -- it is NOT recomputed per model the way
    score_answers.py does it. That is correct for Llama 3.1 (cutoff 2023-12)
    and would be WRONG for any other model.
    """
    w = r.get("work", "")
    if w in WORK_LABEL:
        return w
    return "court_pre" if "pre" in r.get("arm", "") else "court_post"


SERIES = [("constitution", "US Constitution", "s1"),
          ("gatsby",       "The Great Gatsby", "s2"),
          ("court_pre",    "Court opinions (pre-cutoff)", "s3"),
          ("court_post",   "Court opinions (post-cutoff)", "s3")]


def l4_prefix_length(rows, n, out):
    """Extraction vs prefix length -- Q21-Q24 share a break point and vary only
    the amount of context. This is the axis that reconciles a low pooled number
    with Cooper et al., who use ~50-token prefixes throughout."""
    QW = {"Q21": 6, "Q22": 12, "Q23": 24, "Q24": 48}
    sub = [r for r in rows if str(r["trunc_n"]) == str(n) and r["qid"] in QW]
    if not sub:
        print("  L4 skipped — no Q21-Q24 rows")
        return
    for r in sub:
        r["_key"] = series_key(r)
    # FACET BY MODEL SIZE. Colour is spent on the corpus and linestyle on
    # base/instruct; two sizes on one axis produced duplicated legend labels
    # with no way to tell 8B from 70B.
    fams = sorted({r["model"].replace("-Instruct", "") for r in sub})
    fg, axes = plt.subplots(1, len(fams), figsize=(6.2 * len(fams), 5.2), sharey=True)
    axes = axes if len(fams) > 1 else [axes]
    fg.patch.set_facecolor(C["surface"])
    for ax, fam in zip(axes, fams):
        for key, label, slot in SERIES:
            for inst in (False, True):
                m = fam + ("-Instruct" if inst else "")
                pts = []
                for q, wd in sorted(QW.items(), key=lambda t: t[1]):
                    v = [float(r["logp_per_token"]) for r in sub
                         if r["model"] == m and r["_key"] == key and r["qid"] == q]
                    if v:
                        pts.append((wd, statistics.mean(v)))
                if len(pts) < 2:
                    continue
                # post-cutoff court is the floor reference, drawn thinner so it
                # reads as a companion to its pre-cutoff line rather than a peer.
                ax.plot([p for p, _ in pts], [q for _, q in pts],
                        marker="o" if key != "court_post" else "s",
                        markersize=6 if key != "court_post" else 4,
                        lw=2 if key != "court_post" else 1.2,
                        color=C[slot], linestyle="--" if inst else "-",
                        alpha=(0.5 if inst else 1.0) * (0.6 if key == "court_post" else 1.0),
                        label=f"{label} — {'instruct' if inst else 'base'}")
        ax.axhline(-0.5, color=C["ink2"], lw=1, linestyle=(0, (2, 3)))
        ax.set_xticks(list(QW.values()))
        ax.xaxis.grid(True, color=C["grid"], lw=0.7)
        ax.yaxis.grid(True, color=C["grid"], lw=0.7)
        ax.set_axisbelow(True)
        style(ax, C, xlabel="prefix length (words before a fixed break)")
        ax.set_title(fam.split("/")[-1], color=C["ink"], fontsize=10, pad=8, loc="left")
    axes[0].set_ylabel("mean logprob per token", color=C["ink2"], fontsize=9, labelpad=8)
    axes[0].text(6, -0.5, " memorisation threshold", fontsize=7.5,
                 color=C["ink2"], va="bottom")
    axes[-1].legend(frameon=False, fontsize=7.5, labelcolor=C["ink2"],
                    loc="lower right", ncol=1)
    fg.suptitle("More context rescues Gatsby, not the court opinions\n"
                "Gatsby crosses the threshold by 48 words; court opinions stay flat "
                "and pre/post-cutoff never separate",
                color=C["ink"], fontsize=12, x=0.006, ha="left", y=0.99)
    fg.tight_layout()
    fg.savefig(out, bbox_inches="tight", dpi=180, facecolor=C["surface"])
    plt.close(fg)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logprobs", nargs="+",
                    default=[os.path.join(DS, "logprobs", "logprobs__controls__*.csv")])
    ap.add_argument("--scores", help="black-box scores csv, for the L2 join")
    ap.add_argument("--n", default="25", help="trunc_n to plot")
    ap.add_argument("--model", help="restrict to one model id. Omit and every "
                                    "model found gets its own pair of figures.")
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
    # One figure set PER MODEL. Pooling models onto a single axis was the bug
    # that made an 8B-vs-70B scatter look like a calibration curve.
    present = sorted({r["model"] for r in rows})
    wanted = [m for m in present if not args.model or slug(m) == slug(args.model)]
    if not wanted:
        sys.exit(f"--model {args.model} not found; have: {present}")
    for m in wanted:
        sub = [r for r in rows if r["model"] == m]
        tag = slug(m).replace("meta-llama__", "")
        print(f"\n=== {m}  ({len(sub)} rows) ===")
        l1_curve(sub, args.n, os.path.join(
            args.out_dir, f"l1_extractability_n{args.n}__{tag}.png"))
        l2_calibration(sub, args.scores, args.n, os.path.join(
            args.out_dir, f"l2_logprob_vs_blackbox__{tag}.png"))
    # Cross-model figures: these compare models, so they sit outside the loop.
    print("\n=== cross-model ===")
    l3_base_vs_instruct(rows, args.n,
                        os.path.join(args.out_dir, "l3_base_vs_instruct.png"))
    l4_prefix_length(rows, args.n,
                     os.path.join(args.out_dir, "l4_prefix_length.png"))


if __name__ == "__main__":
    main()
