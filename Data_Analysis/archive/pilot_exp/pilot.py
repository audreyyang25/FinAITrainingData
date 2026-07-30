"""Pilot pipeline: full-dataset generation -> judging (no_q + with_q) ->
self-bias / question-effect aggregation. Outputs under results/pilot_run/.

Rehomed from the old complete_run.py. The cross-run variance scaffold is kept:
add a second entry to RUNS (same pipeline, different data_dir) to compare runs.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Data_Analysis/ on sys.path
from config import DATA_DIR
from generations import run_generation
from judge import run_judging
from aggregate import run_aggregation

HERE = os.path.dirname(os.path.abspath(__file__))   # Data_Analysis/pilot_exp
DA_ROOT = os.path.dirname(HERE)                      # Data_Analysis
RESULTS = os.path.join(DA_ROOT, "results")

# Add the second dataset here when it's ready -- same pipeline, new data_dir.
RUNS = [
    {"name": "pilot_run", "data_dir": DATA_DIR, "limit": 15},
    # {"name": "run2", "data_dir": "/abs/path/to/second/dataset", "limit": 15},
]


def run_one(run):
    out_dir = os.path.join(RESULTS, run["name"])
    os.makedirs(out_dir, exist_ok=True)
    gen_path = os.path.join(out_dir, "generations.jsonl")
    jud_path = os.path.join(out_dir, "judgments.jsonl")

    print(f"\n{'#' * 70}\n# RUN: {run['name']}  (data_dir={run['data_dir']})\n{'#' * 70}")
    run_generation(out_paths=[gen_path], data_dir=run["data_dir"], limit=run["limit"])
    run_judging(gen_path=gen_path, out_path=jud_path, data_dir=run["data_dir"])
    return run_aggregation(jud_path, out_dir=out_dir)


if __name__ == "__main__":
    results = {r["name"]: run_one(r) for r in RUNS}

    # ---- Variance test scaffold: one column per run, spread across runs. ----
    if len(results) > 1:
        cols = list(results.keys())
        m = pd.DataFrame(results)              # index = metric, columns = runs
        m["mean"] = m[cols].mean(axis=1)
        m["std"] = m[cols].std(axis=1, ddof=1)
        m["var"] = m[cols].var(axis=1, ddof=1)
        print("\n" + "=" * 70 + "\nCROSS-RUN VARIANCE\n" + "=" * 70)
        print(m.round(4))
        m.round(6).to_csv(os.path.join(RESULTS, "variance.csv"))
        print(f"\nsaved {os.path.join(RESULTS, 'variance.csv')}")
    else:
        print("\n(only one run -- add a second to RUNS to get the variance comparison)")
