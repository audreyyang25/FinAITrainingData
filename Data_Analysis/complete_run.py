import os
import pandas as pd
from config import DATA_DIR
from generations import run_generation
from judge import run_judging
from aggregate import run_aggregation

BASE = os.path.dirname(os.path.abspath(__file__))

# Add the second dataset here when it's ready — same pipeline, new data_dir.
RUNS = [
    {"name": "pilot_run", "data_dir": DATA_DIR, "limit": 15},
    # {"name": "v2", "data_dir": "/abs/path/to/second/dataset", "limit": 15},
]


def run_one(run):
    out_dir = os.path.join(BASE, "results", run["name"])
    os.makedirs(out_dir, exist_ok=True)
    gen_path = os.path.join(out_dir, "generations.jsonl")
    jud_path = os.path.join(out_dir, "judgments.jsonl")

    print(f"\n{'#' * 70}\n# RUN: {run['name']}  (data_dir={run['data_dir']})\n{'#' * 70}")
    run_generation(out_path=gen_path, data_dir=run["data_dir"], limit=run["limit"])
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
        m.round(6).to_csv(os.path.join(BASE, "results", "variance.csv"))
        print(f"\nsaved {os.path.join(BASE, 'runs', 'variance.csv')}")
    else:
        print("\n(only one run — add a second to RUNS to get the variance comparison)")
