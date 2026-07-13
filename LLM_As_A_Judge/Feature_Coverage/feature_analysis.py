import math
import pandas as pd

from utils import (
    load_jsonl,
)

from config import output_path


INPUT = output_path("generations.jsonl")



def entropy(values):

    # Normalize by the actual total rather than assuming 100 -- importance sums
    # are only approximately 100 (see IMPORTANCE_SUM_TOLERANCE in schemas.py).
    total = sum(values)

    if total <= 0:
        return 0.0

    probs = [
        v / total
        for v in values
    ]

    return -sum(
        p * math.log(p)
        for p in probs
        if p > 0
    )



def compute_entropy(suffix=""):


    records = load_jsonl(
        INPUT
    )


    rows = []


    for r in records:


        if r["model"] == "gold":
            continue


        values = [
            f["importance"]
            for f in r["features"]
        ]


        rows.append(
            {
                "model":
                    r["model"],

                "case_id":
                    r["case_id"],

                "entropy":
                    entropy(values),
            }
        )


    df = pd.DataFrame(rows)


    df.to_csv(
        output_path(f"entropy{suffix}.csv"),
        index=False
    )



if __name__ == "__main__":
    compute_entropy()