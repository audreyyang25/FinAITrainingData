#!/usr/bin/env python3
"""Aggregate judged outcome scores by model and arm.

  python Memorization/outcome/summarize.py
  python Memorization/outcome/summarize.py --judged <file.csv> --out <dir>

Only rows the judge actually graded are counted. Rows carrying an API error or
an unparseable verdict are excluded and reported separately -- averaging over
them as zeros would read a billing failure as a model getting the case wrong.

Writes summary_by_model.csv and summary_by_model_arm.csv, so the numbers are
citable without re-deriving them, and re-running after the remaining rows are
graded just overwrites both.
"""
from __future__ import annotations
import argparse, csv, glob, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LABEL, CUTOFF

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")
JUDGED = os.path.join(DS, "outcome_scores")


def graded(r) -> bool:
    """A usable verdict: no transport error, no parse failure, numeric scores."""
    if (r.get("error") or "").strip() or (r.get("parse_error") or "").strip():
        return False
    try:
        float(r["outcome_score"]); float(r["reasoning_score"])
    except (TypeError, ValueError, KeyError):
        return False
    return True


def agg(rows) -> dict:
    f = lambda k: statistics.mean(float(r[k]) for r in rows)
    # Among ATTEMPTS only. A decline carries disposition_claimed="none", and
    # counting those as wrong dispositions conflates "did not answer" with
    # "answered wrong" -- 410 of 814 rows are declines, which drags the figure
    # from 59% to 29% and makes it a measure of willingness, not accuracy.
    dispo = [r for r in rows if r.get("disposition_actual") and not int(r.get("declined") or 0)]
    hit = sum(1 for r in dispo
              if r.get("disposition_claimed", "").strip().lower()
              == r["disposition_actual"].strip().lower())
    return {
        "n": len(rows),
        "outcome": round(f("outcome_score"), 3),
        "reasoning": round(f("reasoning_score"), 3),
        "combined": round(f("combined"), 3),
        # The gap is diagnostic: knowing the disposition while inventing the
        # rationale is the signature of knowledge from a summary, not the opinion.
        "outcome_minus_reasoning": round(f("outcome_score") - f("reasoning_score"), 3),
        "declined_pct": round(100 * sum(int(r.get("declined") or 0) for r in rows) / len(rows), 1),
        "dispo_exact_pct": round(100 * hit / len(dispo), 1) if dispo else "",
        "n_attempted": len(dispo),
        "pct_scoring_1": round(100 * sum(1 for r in rows if float(r["combined"]) >= 0.99) / len(rows), 1),
        "pct_scoring_0": round(100 * sum(1 for r in rows if float(r["combined"]) <= 0.01) / len(rows), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", action="append", help="judged CSV(s); default: primary judge")
    ap.add_argument("--out", default=JUDGED)
    args = ap.parse_args()

    files = args.judged or [f for f in sorted(glob.glob(os.path.join(JUDGED, "*.csv")))
                            if not any(s in os.path.basename(f)
                                       for s in ("_altjudge", "_web", "_noweb", "_smoke",
                                                 "summary_"))]
    if not files:
        sys.exit(f"no judged files in {JUDGED}")

    allrows, dropped = [], 0
    for f in files:
        for r in csv.DictReader(open(f, newline="")):
            (allrows.append(r) if graded(r) else None)
            dropped += 0 if graded(r) else 1

    if not allrows:
        sys.exit("no graded rows yet")
    print(f"{len(allrows)} graded rows   ({dropped} excluded: API errors or unparsed)\n")

    models = sorted({r["model"] for r in allrows},
                    key=lambda m: -agg([r for r in allrows if r["model"] == m])["combined"])

    w = 10
    print(f"{'model':<17}{'cutoff':<12}{'n':>5}{'outcome':>9}{'reason':>8}{'comb':>7}"
          f"{'o-r':>7}{'dispo=':>8}{'(natt)':>8}{'100%':>7}{'0%':>6}")
    print("-" * 94)
    by_model = []
    for m in models:
        a = agg([r for r in allrows if r["model"] == m])
        by_model.append({"model": m, "label": LABEL.get(m, m), "cutoff": CUTOFF.get(m, ""), **a})
        print(f"{LABEL.get(m,m)[:16]:<17}{CUTOFF.get(m,''):<12}{a['n']:>5}{a['outcome']:>9.2f}"
              f"{a['reasoning']:>8.2f}{a['combined']:>7.2f}{a['outcome_minus_reasoning']:>7.2f}"
              f"{str(a['dispo_exact_pct'])+'%':>8}{a['n_attempted']:>8}"
              f"{str(a['pct_scoring_1'])+'%':>7}{str(a['pct_scoring_0'])+'%':>6}")

    print(f"\n{'model':<17}{'arm':<13}{'n':>5}{'outcome':>9}{'reason':>8}{'comb':>7}{'declined':>10}")
    print("-" * 69)
    by_arm = []
    for m in models:
        for arm in ("pre_cutoff", "post_cutoff"):
            s = [r for r in allrows if r["model"] == m and r["arm"] == arm]
            if not s:
                continue
            a = agg(s)
            by_arm.append({"model": m, "label": LABEL.get(m, m), "arm": arm, **a})
            print(f"{LABEL.get(m,m)[:16]:<17}{arm:<13}{a['n']:>5}{a['outcome']:>9.2f}"
                  f"{a['reasoning']:>8.2f}{a['combined']:>7.2f}{str(a['declined_pct'])+'%':>10}")

    os.makedirs(args.out, exist_ok=True)
    for name, rows_ in [("summary_by_model.csv", by_model),
                        ("summary_by_model_arm.csv", by_arm)]:
        p = os.path.join(args.out, name)
        with open(p, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows_[0].keys()))
            wr.writeheader(); wr.writerows(rows_)
        print(f"\nwrote {p}")

    # post_cutoff is the empirical floor: a model cannot know an outcome decided
    # after its training data ends, so whatever it scores there is what guessing
    # from case name, court, and area of law is worth.
    post = [r for r in allrows if r["arm"] == "post_cutoff"]
    if post:
        print(f"\nempirical floor (all models, post-cutoff, n={len(post)}): "
              f"combined {statistics.mean(float(r['combined']) for r in post):.2f}")


if __name__ == "__main__":
    main()
