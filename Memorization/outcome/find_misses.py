#!/usr/bin/env python3
"""Pre-cutoff cases the models got wrong -- where ignorance is not the excuse.

  python Memorization/outcome/find_misses.py
  python Memorization/outcome/find_misses.py --max-score 0.5 --by-model
  python Memorization/outcome/find_misses.py --judged .../judged__..._predict.csv --arm post_cutoff

A post-cutoff miss is expected: the case is outside training data. A PRE-cutoff
miss is not, so these are the rows worth reading by hand -- they are either a
real knowledge gap, a case too obscure to have been written about, or a judge
error. Sorted by how many models missed the same case, because a case every
model fails is a different problem from one a single model fails.

Declines are excluded by default. A decline scores 0 by rubric but means "did
not try", which is not the same claim as "tried and was wrong"; --include-declined
folds them back in.

Captions are not in the judged CSV, so they are joined from datasets/outcomes/.
"""
from __future__ import annotations
import argparse, collections, csv, glob, os, statistics, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize import graded                       # the one definition of "judged"
from models import LABEL

REPO = os.path.dirname(os.path.dirname(HERE))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")
JUDGED = os.path.join(DS, "outcome_scores", "judged__anthropic__claude-opus-5.csv")


def captions(pred_dirs):
    """case_id -> caption, from whichever prediction CSVs exist."""
    out = {}
    for d in pred_dirs:
        for f in glob.glob(os.path.join(d, "*.csv")):
            for r in csv.DictReader(open(f, newline="")):
                if r.get("caption"):
                    out.setdefault(r["case_id"], r["caption"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default=JUDGED)
    ap.add_argument("--arm", default="pre_cutoff",
                    choices=["pre_cutoff", "post_cutoff", "any"])
    ap.add_argument("--max-score", type=float, default=0.25,
                    help="outcome_score at or below this counts as a miss "
                         "(0.25 = the rubric's 'substantially wrong' band)")
    ap.add_argument("--include-declined", action="store_true")
    ap.add_argument("--by-model", action="store_true",
                    help="one line per (case, model) instead of one per case")
    ap.add_argument("--csv", help="write the rows here instead of printing a table")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.judged, newline="")) if graded(r)]
    if args.arm != "any":
        rows = [r for r in rows if r["arm"] == args.arm]
    if not args.include_declined:
        rows = [r for r in rows if not int(r.get("declined") or 0)]
    miss = [r for r in rows if float(r["outcome_score"]) <= args.max_score]
    cap = captions([os.path.join(DS, "outcomes"), os.path.join(DS, "outcomes_predict")])

    print(f"{os.path.basename(args.judged)}   arm={args.arm}   "
          f"outcome_score <= {args.max_score}   declines "
          f"{'included' if args.include_declined else 'excluded'}")
    print(f"{len(miss)} misses out of {len(rows)} graded attempts "
          f"({100*len(miss)/max(1,len(rows)):.1f}%)\n")

    if args.by_model:
        recs = [{"case_id": r["case_id"], "caption": cap.get(r["case_id"], r["case_id"]),
                 "model": LABEL.get(r["model"], r["model"]), "date_filed": r["date_filed"],
                 "court_level": r["court_level"], "outcome": r["outcome_score"],
                 "reasoning": r["reasoning_score"], "actual": r["disposition_actual"],
                 "claimed": r["disposition_claimed"], "justification": r["justification"]}
                for r in sorted(miss, key=lambda x: (x["case_id"], x["model"]))]
        for x in recs:
            print(f"{x['outcome']:>4}  {x['model']:<16} {x['caption'][:58]:<58} "
                  f"{x['actual']} -> {x['claimed']}")
    else:
        by = collections.defaultdict(list)
        for r in miss:
            by[r["case_id"]].append(r)
        allby = collections.defaultdict(list)
        for r in rows:
            allby[r["case_id"]].append(float(r["outcome_score"]))
        recs = []
        for cid, rs in by.items():
            recs.append({"case_id": cid, "caption": cap.get(cid, cid),
                         "date_filed": rs[0]["date_filed"], "court_level": rs[0]["court_level"],
                         "n_models_missed": len(rs), "n_models_graded": len(allby[cid]),
                         "mean_outcome_all_models": round(statistics.mean(allby[cid]), 3),
                         "actual": rs[0]["disposition_actual"],
                         "models": "; ".join(sorted(LABEL.get(r["model"], r["model"]) for r in rs))})
        recs.sort(key=lambda x: (-x["n_models_missed"], x["mean_outcome_all_models"]))
        print(f"{'miss':>5} {'mean':>5}  {'filed':<11} {'caption':<56} actual")
        for x in recs:
            print(f"{x['n_models_missed']}/{x['n_models_graded']:<3} "
                  f"{x['mean_outcome_all_models']:>5}  {x['date_filed'][:10]:<11} "
                  f"{x['caption'][:56]:<56} {x['actual']}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader(); w.writerows(recs)
        print(f"\nwrote {args.csv}  ({len(recs)} rows)")


if __name__ == "__main__":
    main()
