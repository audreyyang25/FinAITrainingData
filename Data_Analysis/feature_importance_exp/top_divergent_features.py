"""top_divergent_features.py -- for each model pair, the features that drive
their JS divergence, read from importance_matrix{suffix}.csv.

JSD is a sum over features:

    JSD(p, q) = sum_f contrib_f,
    contrib_f = 0.5 * p(f) * log2(p(f)/m(f)) + 0.5 * q(f) * log2(q(f)/m(f)),
    m = (p + q) / 2,   contrib_f >= 0.

So ranking features by contrib_f gives EXACTLY the features responsible for a
pair's divergence -- weighted the way the metric weights them, unlike |p - q|.

Requires importance_matrix{suffix}.csv, which nearest_neighbor.py writes:
    python nearest_neighbor.py --dir outputs/llama --suffix _llama

Output: top_divergent_features{suffix}.csv, long format, top-N features per pair.
"""

import argparse
import itertools

import numpy as np
import pandas as pd

from config import output_path


def _term(x, m):
    """0.5 * x * log2(x / m), with the 0*log0 := 0 convention (no warnings)."""
    out = np.zeros_like(x)
    nz = x > 0
    out[nz] = 0.5 * x[nz] * np.log2(x[nz] / m[nz])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_llama",
                    help="reads importance_matrix{suffix}.csv (default _llama)")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    M = pd.read_csv(output_path(f"importance_matrix{args.suffix}.csv"), index_col=0)
    models = list(M.index)
    feats = list(M.columns)
    P = {m: M.loc[m].to_numpy(dtype=float) for m in models}

    rows = []
    for a, b in itertools.combinations(models, 2):
        p, q = P[a], P[b]
        m = 0.5 * (p + q)
        contrib = _term(p, m) + _term(q, m)     # per-feature; sums to JSD(a, b)
        total = float(contrib.sum())

        for rank, j in enumerate(np.argsort(contrib)[::-1][:args.top], 1):
            rows.append({
                "model_a": a,
                "model_b": b,
                "pair_jsd": total,
                "rank": rank,
                "feature": feats[j],
                "jsd_contribution": float(contrib[j]),
                "share_of_pair_jsd": float(contrib[j] / total) if total > 0 else 0.0,
                "p_a": float(p[j]),
                "p_b": float(q[j]),
                "over_weighted_by": a if p[j] > q[j] else b,
            })

    out = pd.DataFrame(rows)
    path = output_path(f"top_divergent_features{args.suffix}.csv")
    out.to_csv(path, index=False)

    n_pairs = len(list(itertools.combinations(models, 2)))
    print(f"{n_pairs} model pairs x top {args.top} = {len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
