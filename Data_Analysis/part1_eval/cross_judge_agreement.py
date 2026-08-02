"""cross_judge_agreement.py -- reliability analysis for the cross-model judging subset.

Reads every judge's judgments_detail.jsonl under outputs/part1_eval/cross_model_subset/
and reports how much the panel actually agrees, on three levels:

  NUMERIC   pairwise Pearson/Spearman and mean absolute difference on the 0-1 scores,
            plus Krippendorff's alpha (interval) across the whole panel.
  ANCHOR    exact and adjacent agreement on the five-level rubric anchor. This is the
            metric that survives scale drift -- two judges at 0.65 and 0.75 disagree
            numerically but both said "sound".
  ITEMS     per-item spread, so the widest disagreements can be pulled for review.

Optionally merges human labels (--human labels.csv, columns
dataset,case_id,model,anchor[,overall_quality]) and treats the human as one more
coder, which turns the same report into a VALIDITY check rather than a
reliability one -- LLM judges can agree with each other and still all be wrong.

  python -m part1_eval.cross_judge_agreement
  python -m part1_eval.cross_judge_agreement --human outputs/part1_eval/human_labels.csv
  python -m part1_eval.cross_judge_agreement --metric content_similarity
"""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE.parent / "outputs" / "part1_eval" / "cross_model_subset"
KEY = ["dataset", "case_id", "model"]


def load_panel(root, metric="overall_quality"):
    """-> long DataFrame [dataset, case_id, model, judge, score, anchor]."""
    rows = []
    for detail in sorted(Path(root).glob("*/judgments_detail.jsonl")):
        for line in detail.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            block = r.get(metric)
            if not isinstance(block, dict):
                continue
            score = block.get(metric)
            if score is None:
                continue
            rows.append({
                "dataset": r["dataset"], "case_id": r["case_id"], "model": r["model"],
                "judge": detail.parent.name, "score": float(score),
                "anchor": block.get("anchor"),
            })
    return pd.DataFrame(rows)


def merge_human(df, path):
    """Add human labels as an extra 'judge' named human."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            score = r.get("overall_quality") or r.get("anchor")
            if not score:
                continue
            rows.append({"dataset": r["dataset"], "case_id": int(r["case_id"]),
                         "model": r["model"], "judge": "human",
                         "score": float(score),
                         "anchor": float(r["anchor"]) if r.get("anchor") else None})
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def krippendorff_alpha_interval(wide):
    """Krippendorff's alpha, interval metric, over a units x coders matrix with NaNs.

    alpha = 1 - Do/De, where Do is the mean squared difference between ratings of the
    SAME unit and De the mean squared difference between all pairs of ratings anywhere.
    Unequal coverage is handled natively, which matters here because a judge that fails
    schema validation on an item simply has no value for it.
    """
    units = [row.dropna().values for _, row in wide.iterrows()]
    units = [u for u in units if len(u) > 1]
    if not units:
        return float("nan")
    n = sum(len(u) for u in units)
    if n < 2:
        return float("nan")

    do = 0.0
    for u in units:
        pair_sum = sum((a - b) ** 2 for a, b in combinations(u, 2)) * 2  # ordered pairs
        do += pair_sum / (len(u) - 1)
    do /= n

    allv = [v for u in units for v in u]
    de = sum((a - b) ** 2 for a, b in combinations(allv, 2)) * 2 / (n * (n - 1))
    return float("nan") if de == 0 else 1.0 - do / de


def report(df, metric):
    judges = sorted(df.judge.unique())
    print(f"metric: {metric}   judges: {judges}")
    print(f"items: {df.groupby(KEY).ngroups}   ratings: {len(df)}\n")

    wide = df.pivot_table(index=KEY, columns="judge", values="score")

    print("=== per-judge scale usage ===")
    usage = wide.agg(["count", "mean", "std", "median"]).T
    usage["at_ceiling_%"] = [(wide[j] == wide[j].max()).mean() * 100 for j in wide.columns]
    print(usage.round(3).to_string(), "\n")

    print("=== pairwise agreement ===")
    rows = []
    for a, b in combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        if len(both) < 3:
            continue
        rows.append({
            "pair": f"{a} ~ {b}", "n": len(both),
            "pearson": round(both[a].corr(both[b]), 3),
            "spearman": round(both[a].corr(both[b], method="spearman"), 3),
            "mean_diff": round((both[a] - both[b]).mean(), 3),
            "mean_abs_diff": round((both[a] - both[b]).abs().mean(), 3),
        })
    if rows:
        print(pd.DataFrame(rows).sort_values("spearman").to_string(index=False), "\n")

    alpha = krippendorff_alpha_interval(wide)
    print(f"=== Krippendorff's alpha (interval, whole panel): {alpha:.3f} ===")
    print("   >0.80 reliable · 0.67-0.80 tentative · <0.67 not reliable\n")

    if df.anchor.notna().any():
        aw = df.pivot_table(index=KEY, columns="judge", values="anchor")
        levels = [0.0, 0.3, 0.5, 0.7, 1.0]
        idx = {v: i for i, v in enumerate(levels)}
        exact = adj = tot = 0
        for _, row in aw.iterrows():
            vals = row.dropna().values
            for a, b in combinations(vals, 2):
                tot += 1
                if a == b:
                    exact += 1
                if abs(idx.get(a, -9) - idx.get(b, 9)) <= 1:
                    adj += 1
        if tot:
            print("=== anchor agreement (5-level rubric) ===")
            print(f"exact:    {exact}/{tot} ({exact/tot*100:.1f}%)")
            print(f"adjacent: {adj}/{tot} ({adj/tot*100:.1f}%)")
            print(f"alpha (anchors): {krippendorff_alpha_interval(aw):.3f}\n")

    spread = (wide.max(axis=1) - wide.min(axis=1)).rename("spread")
    print("=== per-item spread ===")
    print(spread.describe().round(3).to_string())
    print(f"\nwidest disagreements:")
    top = pd.concat([wide, spread], axis=1).nlargest(10, "spread")
    print(top.round(2).to_string())
    return wide, spread


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--metric", default="overall_quality",
                    choices=["overall_quality", "content_similarity"])
    ap.add_argument("--human", default=None,
                    help="CSV of human labels: dataset,case_id,model,anchor[,overall_quality]")
    ap.add_argument("--out", default=None, help="write per-item spread CSV here")
    args = ap.parse_args()

    df = load_panel(args.dir, args.metric)
    if df.empty:
        print(f"no judgments with '{args.metric}' under {args.dir} -- "
              "run part1_eval.cross_judge_subset first")
        return
    if args.human:
        df = merge_human(df, args.human)

    wide, spread = report(df, args.metric)
    if args.out:
        pd.concat([wide, spread], axis=1).round(3).to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
