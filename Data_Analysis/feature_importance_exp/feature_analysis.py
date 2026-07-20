"""feature_analysis.py -- how concentrated is each model's reasoning?

Entropy is computed over the CANONICALIZED features (superset slots), not the
model's raw self-reported list.  A model that names the same underlying factor
twice under different phrasings splits its importance across two raw entries
(0.3 + 0.2) where it should hold one (0.5) -- which inflates measured
diffuseness.  That duplication is model-specific (Sonnet ~0.9 dupes/case, Opus
~0.1), so scoring on raw features biases entropy along exactly the axis we use
it to compare models.  Summing importance into superset slots first removes it,
and puts this stage on the same alphabet as importance_distribution.py.

Emits both the raw entropy and the scale-free EVENNESS:

    H        = -sum p ln p                    in [0, ln k]
    evenness = H / ln k                       in [0, 1]

H alone is not comparable across cases: its ceiling is ln k, and k (features
named) varies 3-17 here, so H tracks k at r ~ 0.96 and is mostly a verbosity
measure.  Evenness divides the ceiling out -- 1.0 = importance spread perfectly
flat, low = the model committed to a few decisive features.  Use evenness to
compare, and as a covariate; H alone is near-collinear with k.
"""

import math
from collections import defaultdict

import pandas as pd

from utils import load_json
from config import output_path

INPUT = output_path("case_features.json")


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

    return -sum(p * math.log(p) for p in probs if p > 0)


def compute_entropy(suffix=""):

    cases = load_json(INPUT, default={})

    rows = []

    for case in cases.values():

        # model -> superset slot -> summed importance.  A feature that failed to
        # canonicalize (superset_index None) keeps its own slot rather than
        # being dropped: discarding it would silently lose importance mass and
        # penalize models whose features happened not to map.
        slots = defaultdict(lambda: defaultdict(float))
        families = {}

        for e in case["features"].values():

            if e["model_family"] == "gold":
                continue

            idx = e["superset_index"]
            key = idx if idx is not None else ("unmapped", e["feature"])

            slots[e["model"]][key] += e["importance"]
            families[e["model"]] = e["model_family"]

        for model, allocation in slots.items():

            values = list(allocation.values())
            k = len(values)

            h = entropy(values)
            ceiling = math.log(k) if k > 1 else 0.0

            rows.append(
                {
                    # dataset is part of the key: case_id restarts at 1 in every
                    # dataset, so (model, case_id) alone is NOT unique and any
                    # join on it silently fans out.
                    "dataset":
                        case["dataset"],

                    "case_id":
                        case["case_id"],

                    "model_family":
                        families[model],

                    "model":
                        model,

                    "features_named":
                        k,

                    "entropy":
                        h,

                    "max_entropy":
                        ceiling,

                    # Undefined for a single feature (ceiling is 0) -- leave
                    # blank rather than inventing a 0 or a 1.  Clamped because a
                    # perfectly uniform allocation gives h == ceiling only to
                    # within float error, which would otherwise print 1.0000...2
                    # and make "evenness > 1" useless as a bug signal.
                    "evenness":
                        min(1.0, h / ceiling) if ceiling > 0 else None,
                }
            )

    df = pd.DataFrame(rows).sort_values(["dataset", "case_id", "model"])

    df.to_csv(
        output_path(f"entropy{suffix}.csv"),
        index=False
    )

    print(f"Wrote entropy{suffix}.csv ({len(df)} rows)")
    print(f"  entropy  mean {df.entropy.mean():.3f}  "
          f"range {df.entropy.min():.3f}-{df.entropy.max():.3f}")
    print(f"  evenness mean {df.evenness.mean():.3f}  "
          f"range {df.evenness.min():.3f}-{df.evenness.max():.3f}")


if __name__ == "__main__":
    compute_entropy()
