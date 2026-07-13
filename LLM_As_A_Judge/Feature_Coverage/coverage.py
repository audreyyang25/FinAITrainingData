import pandas as pd

from utils import load_json
from config import output_path


CASE_FEATURES = output_path("case_features.json")



def compute_coverage(suffix=""):

    cases = load_json(
        CASE_FEATURES,
        default={},
    )


    rows = []


    for case in cases.values():

        superset = case["superset"]

        n = len(superset)


        # Per source, the distinct superset features it covered. Driven
        # entirely from the id registry -- each feature already carries its
        # canonical superset index, so there is no text matching to get wrong.
        # Gold (the reference answer) is scored too: it is one of the sources
        # that built the superset, so its coverage is the baseline for how much
        # of the pooled reasoning the reference itself contains.
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
        output_path(f"coverage{suffix}.csv"),
        index=False,
    )



if __name__ == "__main__":
    compute_coverage()
