"""Step 4: aggregation. Turn one run's judgments.jsonl into the analyses we care
about, AND return a flat dict of scalar metrics so run.py can compare runs.

  - validity report (how many judgments each judge failed to format)
  - generator x judge score matrix, per condition  -> self-bias diagonal
  - self vs other (does a judge favor its own generations?)
  - question effect (with_q - no_q), overall and per dataset
  - per-judge leniency / per-generator quality
"""

import os
import pandas as pd

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judgments.jsonl")
pd.set_option("display.width", 120)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def run_aggregation(judgments_path=PATH, out_dir=None, verbose=True):
    """Analyze one run. Returns {metric_name: scalar} for cross-run comparison."""
    df = pd.read_json(judgments_path, lines=True)
    df["valid"] = df["valid"].fillna(False)
    invalid_rate = float((~df["valid"]).mean())

    if verbose:
        section("VALIDITY (invalid = judge response we couldn't parse a score from)")
        print(f"total judgments: {len(df)}   invalid: {(~df['valid']).sum()}")
        print(df[~df["valid"]].groupby("judge").size().rename("invalid_count").to_frame())

    df = df[df["valid"]].copy()
    df["is_self"] = df["generator"] == df["judge"]

    if verbose:
        for cond in ["no_q", "with_q"]:
            section(f"GENERATOR (rows) x JUDGE (cols) mean score  [{cond}]")
            sub = df[df["condition"] == cond]
            print(sub.pivot_table(index="generator", columns="judge",
                                  values="score", aggfunc="mean").round(3))
        section("SELF vs OTHER (judge scoring its own generation vs everyone else's)")
        print(df.pivot_table(index="is_self", columns="condition",
                             values="score", aggfunc="mean").round(3))
        gap = (df[df["condition"] == "no_q"]
               .pivot_table(index="judge", columns="is_self", values="score", aggfunc="mean"))
        gap["self_minus_other"] = gap.get(True) - gap.get(False)
        print("\nper-judge self-preference [no_q]:")
        print(gap.round(3))
        section("QUESTION EFFECT (with_q - no_q mean score)")
        by_cond = df.pivot_table(index="dataset", columns="condition",
                                 values="score", aggfunc="mean")
        by_cond["delta"] = by_cond["with_q"] - by_cond["no_q"]
        print(by_cond.round(3))
        section("MARGINALS")
        print("per-judge mean score (leniency):")
        print(df.groupby("judge").score.mean().round(3).sort_values())
        print("\nper-generator mean score (quality):")
        print(df.groupby("generator").score.mean().round(3).sort_values())
        nq = df[df["condition"] == "no_q"]
        have = nq["guessed_question"].notna().mean() if len(nq) else 0
        section("QUESTION-GUESSING (no_q): share of judgments with a guess present")
        print(f"{have:.1%} of no_q judgments include a guessed_question")

    # ---- Flat scalar metrics (the variance-test inputs) ----
    metrics = {"invalid_rate": round(invalid_rate, 4),
               "overall_mean": round(float(df.score.mean()), 4)}
    for j, v in df.groupby("judge").score.mean().items():
        metrics[f"judge_mean[{j}]"] = round(float(v), 4)
    for g, v in df.groupby("generator").score.mean().items():
        metrics[f"gen_mean[{g}]"] = round(float(v), 4)

    nq = df[df["condition"] == "no_q"]
    metrics["self_minus_other_no_q"] = round(
        float(nq[nq.is_self].score.mean() - nq[~nq.is_self].score.mean()), 4)
    metrics["question_delta_overall"] = round(
        float(df[df.condition == "with_q"].score.mean()
              - df[df.condition == "no_q"].score.mean()), 4)
    piv = df.pivot_table(index="dataset", columns="condition", values="score", aggfunc="mean")
    for ds in piv.index:
        metrics[f"qdelta[{ds}]"] = round(float(piv.loc[ds, "with_q"] - piv.loc[ds, "no_q"]), 4)

    # Save the headline matrix next to the judgments (or in out_dir).
    dest = out_dir or os.path.dirname(judgments_path)
    df[df.condition == "no_q"].pivot_table(
        index="generator", columns="judge", values="score", aggfunc="mean"
    ).round(3).to_csv(os.path.join(dest, "matrix_no_q.csv"))

    return metrics


if __name__ == "__main__":
    run_aggregation()
