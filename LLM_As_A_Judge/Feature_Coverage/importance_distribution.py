"""importance_distribution.py -- per-model reasoning-attention distributions.

For each model, P_m(f) = share of that model's total self-reported importance
that went to global feature f, aggregated over all cases. Every model lives on
the SAME axis (the global vocabulary), so the distributions are directly
comparable: we also emit a pairwise Jensen-Shannon divergence between models.

Pure local computation off case_features.json + global_features.json -- no API
calls. Gold is excluded (it is the reference, not a contestant).

Outputs:
  importance_matrix.csv  -- rows = model, cols = global feature, values = P_m(f)
  js_divergence.csv      -- symmetric model x model JS divergence (log2, [0,1])
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from utils import load_json
from config import output_path


CASE_FEATURES = output_path("case_features.json")
GLOBAL_FEATURES = output_path("global_features.json")


def _accumulate_mass():
    """mass[model][global_feature] = total importance summed over all cases."""

    cases = load_json(CASE_FEATURES, default={})
    global_space = load_json(GLOBAL_FEATURES, default={"mapping": {}})
    global_mapping = global_space["mapping"]

    mass = defaultdict(lambda: defaultdict(float))

    for case in cases.values():

        superset = case["superset"]

        for e in case["features"].values():

            if e["model_family"] == "gold":
                continue

            idx = e["superset_index"]
            if idx is None:
                continue

            global_feature = global_mapping.get(superset[idx])
            if not global_feature:
                continue

            mass[e["model"]][global_feature] += e["importance"]

    return mass


def _kl(p, m):
    """KL(p || m) in bits, summed over p's support (m > 0 there by construction)."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / m[mask])))


def _js_divergence(p, q):
    """Jensen-Shannon divergence (log2): 0 = identical, 1 = disjoint support."""
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def build_distribution(suffix=""):

    mass = _accumulate_mass()

    # Normalize each model's mass to a probability distribution.
    rows = []
    for model, feats in mass.items():
        total = sum(feats.values())
        if total <= 0:
            continue
        for global_feature, m in feats.items():
            rows.append({
                "model": model,
                "global_feature": global_feature,
                "probability": m / total,
            })

    if not rows:
        print("No importance mass found -- run global_canonicalization first.")
        return

    df = pd.DataFrame(rows)

    # Model x global-feature matrix of P_m(f); missing cells are 0.
    matrix = (
        df
        .pivot(index="model", columns="global_feature", values="probability")
        .fillna(0.0)
        .sort_index()
    )
    matrix.to_csv(output_path(f"importance_matrix{suffix}.csv"))

    # Pairwise JS divergence over the shared feature axis.
    models = list(matrix.index)
    P = matrix.to_numpy()
    n = len(models)

    js = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _js_divergence(P[i], P[j])
            js[i, j] = d
            js[j, i] = d

    js_out = output_path(f"js_divergence{suffix}.csv")
    pd.DataFrame(js, index=models, columns=models).to_csv(js_out)

    print(f"Wrote importance_matrix{suffix}.csv "
          f"({matrix.shape[0]} models x {matrix.shape[1]} global features)")
    print(f"Wrote {js_out} ({n} x {n} model JS-divergence matrix)")


if __name__ == "__main__":
    build_distribution()
