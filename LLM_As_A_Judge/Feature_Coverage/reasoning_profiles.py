import pandas as pd

from utils import load_json
from config import output_path


GLOBAL_FEATURES = output_path("global_features.json")

CASE_FEATURES = output_path("case_features.json")

OUTPUT = output_path("reasoning_profiles.csv")



def build_profiles():

    global_space = load_json(
        GLOBAL_FEATURES,
        default={},
    )

    global_mapping = global_space.get("mapping")

    if not global_mapping:
        print(
            "global_features.json missing or empty -- run "
            "global_canonicalization first. Skipping profiles."
        )
        return


    cases = load_json(
        CASE_FEATURES,
        default={},
    )


    rows = []


    for case in cases.values():

        superset = case["superset"]


        for e in case["features"].values():

            if e["model_family"] == "gold":
                continue

            idx = e["superset_index"]

            if idx is None:
                continue

            # chain: raw feature -> superset index -> canonical text -> global
            case_feature = superset[idx]

            global_feature = global_mapping.get(case_feature)

            if not global_feature:
                continue

            rows.append(
                {
                    "model": e["model"],
                    "model_family": e["model_family"],
                    "global_feature": global_feature,
                    "importance": e["importance"],
                    "case_id": case["case_id"],
                }
            )


    df = pd.DataFrame(rows)


    profile = (
        df
        .groupby(
            [
                "model",
                "model_family",
                "global_feature",
            ]
        )
        .agg(
            appearances=(
                "case_id",
                "count",
            ),
            mean_importance=(
                "importance",
                "mean",
            ),
        )
        .reset_index()
    )


    totals = (
        df
        .groupby("model")
        .case_id
        .nunique()
    )


    profile["frequency"] = (
        profile.apply(
            lambda r:
            r.appearances
            /
            totals[r.model],
            axis=1,
        )
    )


    profile.to_csv(
        OUTPUT,
        index=False,
    )



if __name__ == "__main__":
    build_profiles()
