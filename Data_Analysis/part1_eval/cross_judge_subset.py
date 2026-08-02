"""cross_judge_subset.py -- CROSS-MODEL JUDGING SUBSET.

Scores a small, deliberately score-diverse subset of the generations with THREE
independent judges instead of one, adding the cumulative `overall_quality`
dimension (anchored 1.0 / 0.7 / 0.5 / 0.3 / 0.0 rubric + explanation).

Why a subset: judging all ~4.5k generations three times over is ~13.5k calls at
several thousand input tokens each. Inter-rater reliability does not need the full
corpus -- 50 items spread across the score range gives an agreement estimate and a
reviewable packet for a fraction of the cost.

Why these judges: all three families sit OUTSIDE the 12-model generator pool
(anthropic / google / openai / qwen), so no judge ever grades its own family's
output. The archived pilot measured self-preference at +0.21 (claude) and +0.17
(gpt) when judges and generators overlapped; an independent panel removes that term
rather than having to correct for it.

Selection: stratified by the EXISTING single-judge (llama) content_similarity, so
the subset spans clear failures through clear successes rather than piling up in the
0.8-1.0 mode where most of the corpus sits. Sampling is random within each band and
seeded, so the subset is reproducible. Answers that fail setup.generation's validity
check (stub / pointer answers) are excluded -- they are a known generation bug and
would waste judge calls.

  # select + judge with all three judges
  python -m part1_eval.cross_judge_subset

  # preview the subset without spending anything
  python -m part1_eval.cross_judge_subset --dry-run

  # one judge at a time
  python -m part1_eval.cross_judge_subset --judges deepseek/deepseek-v3.2

Outputs, per judge, to outputs/part1_eval/cross_model_subset/<judge-slug>/, plus
subset_manifest.csv naming the 50 selected (dataset, case_id, model) items.
"""

import argparse
import json
import os
import random
from pathlib import Path

from part1_eval.general_judge import (run_eval, DEFAULT_WORKERS, judge_workers,
                                      ALL_DIMENSIONS)
from setup.generation import invalid_reason

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_GENS = ROOT / "generations.json"
DEFAULT_BASELINE = ROOT / "outputs" / "part1_eval" / "llama" / "judgments_detail.jsonl"
DEFAULT_OUT = ROOT / "outputs" / "part1_eval" / "cross_model_subset"

PANEL = [
    "meta-llama/llama-3.3-70b-instruct",   # incumbent; bf16-pinned
    "deepseek/deepseek-v3.2",              # already exercised in part3_distribution
    "mistralai/mistral-large-2512",        
]

# Bands over the baseline judge's content_similarity. Deliberately uneven at the
# bottom: sub-0.5 is rare (~2% of the corpus) but it is where judges are most likely
# to disagree, so it gets the same allocation as the crowded 0.9-1.0 band.
BANDS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]


def load_baseline_scores(path=DEFAULT_BASELINE):
    """(dataset, case_id, model) -> baseline judge's content_similarity."""
    scores = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cs = (r.get("content_similarity") or {}).get("content_similarity")
        if cs is not None:
            scores[(r["dataset"], r["case_id"], r["model"])] = cs
    return scores


def select_subset(generations, scores, n=50, seed=0):
    """Stratified random sample across BANDS. Bands that cannot fill their quota pass
    the shortfall to the next band, so the total is always n when the corpus allows."""
    rng = random.Random(seed)
    by_band = {b: [] for b in BANDS}
    for g in generations:
        if not all(k in g for k in ("dataset", "case_id", "model")):
            continue                       # unkeyable orphan row; see setup/generation.py
        if g["model"] == "gold":
            continue
        if invalid_reason(g.get("answer")):
            continue                       # known-bad generation; don't spend judges on it
        cs = scores.get((g["dataset"], g["case_id"], g["model"]))
        if cs is None:
            continue                       # not judged by the baseline -> no band
        for lo, hi in BANDS:
            if lo <= cs < hi:
                by_band[(lo, hi)].append({**g, "baseline_cs": cs})
                break

    per_band, picked, carry = n // len(BANDS), [], 0
    for band in BANDS:
        pool = by_band[band]
        want = per_band + carry
        take = min(want, len(pool))
        carry = want - take
        picked.extend(rng.sample(pool, take))
    # Any residual shortfall (a band ran dry at the end) is topped up from what's left.
    if len(picked) < n:
        chosen = {(p["dataset"], p["case_id"], p["model"]) for p in picked}
        rest = [g for pool in by_band.values() for g in pool
                if (g["dataset"], g["case_id"], g["model"]) not in chosen]
        picked.extend(rng.sample(rest, min(n - len(picked), len(rest))))
    return picked


def to_judge_rows(picked):
    """generations.json shape -> the shape run_eval expects (answer -> response)."""
    return [{"dataset": p["dataset"], "case_id": p["case_id"], "model": p["model"],
             "response": p["answer"], "run": p.get("run", 0),
             "model_family": p.get("model_family"), "model_key": p.get("model_key")}
            for p in picked]


def write_manifest(picked, out_dir):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = Path(out_dir) / "subset_manifest.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "case_id", "model", "model_key", "baseline_cs", "answer_chars"])
        for p in sorted(picked, key=lambda r: r["baseline_cs"]):
            w.writerow([p["dataset"], p["case_id"], p["model"], p.get("model_key"),
                        round(p["baseline_cs"], 3), len(p.get("answer") or "")])
    return path


def _slug(model):
    return model.split("/")[-1].replace(".", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", default=str(DEFAULT_GENS))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                    help="single-judge judgments_detail.jsonl used to stratify")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judges", nargs="*", default=PANEL)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--dimensions", nargs="*", default=["overall_quality"],
                    choices=list(ALL_DIMENSIONS),
                    help="dimensions to score (default: overall_quality only -- one judge "
                         "call per response instead of four or five). 'outcome' also "
                         "triggers gold-verdict pre-classification, which cannot represent "
                         "references that give different verdicts for different advisors.")
    ap.add_argument("--dry-run", action="store_true",
                    help="select and report the subset, make no judge calls")
    args = ap.parse_args()

    generations = json.loads(Path(args.generations).read_text())
    scores = load_baseline_scores(args.baseline)
    picked = select_subset(generations, scores, n=args.n, seed=args.seed)

    manifest = write_manifest(picked, args.out)
    print(f"selected {len(picked)} of {len(generations)} generations -> {manifest}")
    for lo, hi in BANDS:
        k = sum(1 for p in picked if lo <= p["baseline_cs"] < hi)
        print(f"  baseline_cs [{lo}, {hi}): {k}")
    n_models = len({p["model"] for p in picked})
    n_ds = len({p["dataset"] for p in picked})
    print(f"  spanning {n_models} models, {n_ds} datasets")

    if args.dry_run:
        print("\n--dry-run: no judge calls made")
        return

    rows = to_judge_rows(picked)
    for judge in args.judges:
        out_dir = Path(args.out) / _slug(judge)
        # Provider-pinned judges declare a lower worker count in JUDGES; an explicit
        # --workers on the command line still wins.
        w = args.workers if args.workers != DEFAULT_WORKERS else judge_workers(judge, DEFAULT_WORKERS)
        print(f"\n=== {judge} -> {out_dir} ({w} workers, {len(args.dimensions)} dim) ===")
        run_eval(rows, out_dir=str(out_dir), judge=judge,
                 workers=w, dimensions=tuple(args.dimensions))


if __name__ == "__main__":
    main()
