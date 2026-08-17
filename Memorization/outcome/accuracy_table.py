#!/usr/bin/env python3
"""Per-model accuracy for both outcome experiments, as one tidy CSV.

  python Memorization/outcome/accuracy_table.py
  python Memorization/outcome/accuracy_table.py --out somewhere/else.csv

Long format, one row per (condition, model), so the two experiments can be
filtered apart or joined without reshaping:

  pre_cutoff_recall   the original run -- cases inside training data, and the
                      default SYSTEM prompt offered UNKNOWN, so n_attempted is a
                      subset each model selected for itself. n_declined says how
                      big that selection was; it ranges from 7 to 84.
  post_cutoff_predict --predict --arm post_cutoff -- cases outside training data
                      and declining was forbidden, so n_attempted is everything.

The two therefore differ in BOTH arm and prompt. Compare them, but do not read
the difference as a pure arm effect.

`dispo_hit_pct` is exact-match on the disposition label. Read it against
`baseline_pct` in the same row -- the always-guess-the-modal-disposition rate
for that condition, which is the floor a model beats by knowing nothing.
"""
from __future__ import annotations
import argparse, collections, csv, os, statistics, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize import graded
from models import TARGETS

REPO = os.path.dirname(os.path.dirname(HERE))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")
SCORES = os.path.join(DS, "outcome_scores")
DEFAULT_OUT = os.path.join(DS, "figures", "outcome_accuracy.csv")

CONDITIONS = [
    ("pre_cutoff_recall", "judged__anthropic__claude-opus-5.csv", "pre_cutoff"),
    ("post_cutoff_predict", "judged__anthropic__claude-opus-5__predict.csv", "post_cutoff"),
]
COLS = ["condition", "model", "label", "n_graded", "n_attempted", "n_declined",
        "outcome", "reasoning", "combined", "dispo_hit_pct", "baseline_pct",
        "modal_disposition"]


def summarise(rows):
    """One record from a set of judged rows, or None if nothing was attempted."""
    att = [r for r in rows if not int(r.get("declined") or 0)]
    if not att:
        return None
    hit = sum(1 for r in att
              if r["disposition_claimed"].strip().lower()
              == r["disposition_actual"].strip().lower())
    return dict(n_graded=len(rows), n_attempted=len(att),
                n_declined=len(rows) - len(att),
                outcome=round(statistics.mean(float(r["outcome_score"]) for r in att), 3),
                reasoning=round(statistics.mean(float(r["reasoning_score"]) for r in att), 3),
                combined=round(statistics.mean(float(r["combined"]) for r in att), 3),
                dispo_hit_pct=round(100 * hit / len(att), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default=SCORES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    out_rows = []
    for cond, fname, armv in CONDITIONS:
        path = os.path.join(args.scores_dir, fname)
        if not os.path.exists(path):
            print(f"  [ ] {cond}: no {fname} — skipped")
            continue
        rows = [r for r in csv.DictReader(open(path, newline=""))
                if graded(r) and r["arm"] == armv]
        att = [r for r in rows if not int(r.get("declined") or 0)]
        # Baseline is per condition, not global: the modal disposition differs
        # between the arms, so a shared number would be wrong for one of them.
        act = collections.Counter(r["disposition_actual"].strip().lower() for r in att)
        modal, n_modal = act.most_common(1)[0] if act else ("", 0)
        base = round(100 * n_modal / max(1, sum(act.values())), 1)
        for mid, lbl, _ in list(TARGETS) + [(None, "POOLED", None)]:
            s = summarise(rows if mid is None else [r for r in rows if r["model"] == mid])
            if not s:
                continue
            out_rows.append(dict(condition=cond, model=mid or "ALL", label=lbl,
                                 baseline_pct=base, modal_disposition=modal, **s))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader(); w.writerows(out_rows)
    print(f"wrote {args.out}  ({len(out_rows)} rows)")
    for r in out_rows:
        print(f"  {r['condition']:20s} {r['label']:16s} n={r['n_attempted']:4d} "
              f"outcome={r['outcome']:.2f} reasoning={r['reasoning']:.2f} "
              f"dispo={r['dispo_hit_pct']:.0f}%")


if __name__ == "__main__":
    main()
