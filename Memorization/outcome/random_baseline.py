#!/usr/bin/env python3
"""Do the models beat a guesser that knows nothing but the verdict distribution?

  python Memorization/outcome/random_baseline.py
  python Memorization/outcome/random_baseline.py --exclude-other
  python Memorization/outcome/random_baseline.py --arm post_cutoff \
      --judged .../judged__anthropic__claude-opus-5__predict.csv

TWO baselines, because they differ by 14 points and only reporting the low one
would flatter the models:

  PRIOR-MATCHED   draw a disposition per case from the empirical distribution of
                  true verdicts. Expected accuracy is sum(p^2). This is the
                  "weighted random" design -- correct, but on imbalanced classes
                  it is a WEAK baseline.
  ALWAYS-MODE     always answer the single most common verdict. Expected
                  accuracy is max(p). Equally knowledge-free and strictly
                  harder whenever the distribution is skewed, which this one is
                  (`mixed` is 46% of pre-cutoff attempts).

A model that beats prior-matched but not always-mode has learned nothing a
one-line heuristic does not already know.

The prior is computed PER MODEL over that model's own attempted rows, not over
the whole corpus. Each model faces a different case mix -- arms are resolved
against its own cutoff, and declines remove a self-selected subset -- so a
shared prior would be scored against the wrong denominator.

The p-value is one-sided: P(random >= observed) under the prior-matched draw.
"""
from __future__ import annotations
import argparse, collections, csv, os, random, statistics, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize import graded
from models import TARGETS

REPO = os.path.dirname(os.path.dirname(HERE))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")
DEFAULT_JUDGED = os.path.join(DS, "outcome_scores", "judged__anthropic__claude-opus-5.csv")
DEFAULT_OUT = os.path.join(DS, "figures", "random_baseline.csv")
# Carries all FOUR baselines o6 draws, so the CSV and the figure cannot drift.
#   prior_matched_pct        mean of the simulated null -- drives the p-value
#   prior_matched_exact_pct  closed form sum(p^2) -- what o6 ticks. The two agree
#                            to ~0.1pt; they differ only by simulation noise.
COLS = ["condition", "model", "label", "n", "observed_pct",
        "uniform_pct", "prior_matched_exact_pct", "always_affirmed_pct",
        "always_mode_pct", "mode_label",
        "prior_matched_pct", "prior_ci_lo", "prior_ci_hi", "p_value",
        "strongest_baseline", "strongest_baseline_pct",
        "beats_prior", "beats_mode", "beats_all"]


def simulate(actual, prior_labels, prior_weights, trials, seed):
    """Empirical null: redraw every case's answer from the prior, count matches."""
    rng = random.Random(seed)
    n = len(actual)
    out = []
    for _ in range(trials):
        draw = rng.choices(prior_labels, weights=prior_weights, k=n)
        out.append(100 * sum(a == b for a, b in zip(actual, draw)) / n)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default=DEFAULT_JUDGED)
    ap.add_argument("--arm", default="pre_cutoff",
                    choices=["pre_cutoff", "post_cutoff", "any"])
    ap.add_argument("--exclude-other", action="store_true",
                    help="drop `other` as a truth class, as o3 does -- it is a "
                         "judge-assigned catch-all on both sides")
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.judged, newline="")) if graded(r)]
    if args.arm != "any":
        rows = [r for r in rows if r["arm"] == args.arm]
    rows = [r for r in rows if not int(r.get("declined") or 0)]
    if args.exclude_other:
        before = len(rows)
        rows = [r for r in rows if r["disposition_actual"].strip().lower() != "other"]
        if before == len(rows):
            # `other` was retired in favour of granted/denied/judgment, so this
            # flag now removes nothing. Left in place for older judged files, but
            # it must say so -- silently writing a "_no_other" file identical to
            # the base one is how a stale artifact gets quoted months later.
            print("  NOTE: --exclude-other removed 0 rows; `other` is not in this "
                  "judged file's vocabulary. Output is identical to the default.\n")

    cond = f"{args.arm}{'_no_other' if args.exclude_other else ''}"
    out_rows = []
    print(f"{os.path.basename(args.judged)}  arm={args.arm}  "
          f"{'excluding' if args.exclude_other else 'including'} `other`  "
          f"trials={args.trials}\n")
    print(f"{'model':17s} {'n':>4s} {'observed':>9s} {'uniform':>8s} "
          f"{'prior Σp²':>9s} {'alw.affirm':>10s} {'alw.mode':>10s}  p")

    for mid, lbl, _ in list(TARGETS) + [(None, "POOLED", None)]:
        s = rows if mid is None else [r for r in rows if r["model"] == mid]
        if not s:
            continue
        actual = [r["disposition_actual"].strip().lower() for r in s]
        obs = 100 * sum(1 for r in s
                        if r["disposition_claimed"].strip().lower()
                        == r["disposition_actual"].strip().lower()) / len(s)
        cnt = collections.Counter(actual)
        labels = list(cnt)
        weights = [cnt[d] for d in labels]
        null = simulate(actual, labels, weights, args.trials, args.seed)
        lo, hi = null[int(0.025 * args.trials)], null[int(0.975 * args.trials)]
        p = sum(1 for v in null if v >= obs) / args.trials
        mode, mc = cnt.most_common(1)[0]
        mode_pct = 100 * mc / len(s)
        # The same four o6 ticks, computed per model on its own attempted set.
        uni = 100 / len(cnt)
        exact = 100 * sum((c / len(s)) ** 2 for c in cnt.values())
        aff = 100 * cnt.get("affirmed", 0) / len(s)
        strongest, strongest_pct = max(
            [("uniform random", uni), ("prior-matched", exact),
             ('always "affirmed"', aff), (f'always "{mode}"', mode_pct)],
            key=lambda t: t[1])
        out_rows.append(dict(
            condition=cond, model=mid or "ALL", label=lbl, n=len(s),
            observed_pct=round(obs, 1),
            uniform_pct=round(uni, 1), prior_matched_exact_pct=round(exact, 1),
            always_affirmed_pct=round(aff, 1),
            always_mode_pct=round(mode_pct, 1), mode_label=mode,
            prior_matched_pct=round(statistics.mean(null), 1),
            prior_ci_lo=round(lo, 1), prior_ci_hi=round(hi, 1), p_value=round(p, 4),
            strongest_baseline=strongest, strongest_baseline_pct=round(strongest_pct, 1),
            beats_prior=int(obs > hi), beats_mode=int(obs > mode_pct),
            beats_all=int(obs > strongest_pct)))
        print(f"{lbl:17s} {len(s):4d} {obs:8.1f}% {uni:7.1f}% {exact:8.1f}% "
              f"{aff:9.1f}% {mode_pct:9.1f}%  {p:.4f}"
              + ("" if obs > strongest_pct else f"   <- below {strongest}"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {args.out}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
