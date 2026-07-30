import pandas as pd

from shared.utils import load_json
from shared.config import part_output

# Independent-Llama extraction (Part 2), not the legacy self-report case_features.
CASE_FEATURES = part_output("part2_coverage", "llama/case_feat.json")

def compute_coverage(suffix=""):
    cases = load_json(
        CASE_FEATURES,
        default={},
    )
    rows = []
    for case in cases.values():
        superset = case["superset"]
        n = len(superset)
        # Coverage is scored by # of features covered by specific model generation / # of features in the superset of features for that question
        per_model = {}

        for e in case["features"].values():

            slot = per_model.setdefault(
                e["model"],
                {
                    "model_family": e["model_family"],
                    "indices": set(),
                },
            )

            if e["superset_index"] is not None:
                slot["indices"].add(e["superset_index"])

        for model, slot in per_model.items():

            found = len(slot["indices"])

            rows.append(
                {
                    "dataset": case["dataset"],
                    "case_id": case["case_id"],
                    "model_family": slot["model_family"],
                    "model": model,
                    "features_found": found,
                    "total_features": n,
                    "coverage": found / n if n else 0,
                }
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        part_output("part2_coverage", f"coverage{suffix}.csv"),
        index=False,
    )

if __name__ == "__main__":
    compute_coverage()
