"""rubric_analysis.py -- turn binary rubric scores into the same JSD / clustering
analysis used for the feature vocabulary, so the two are directly comparable.

Per model, the distribution is the 20-criterion PRESENCE-rate vector (fraction of
the model's cases where each criterion is engaged), normalized to sum to 1. Then
the exact JSD / nearest-neighbour / permutation machinery from jsd_stats runs
on it -- if families still separate on these clean, uniformly-scored axes, the
family clustering is substance, not vocabulary or phrasing.

  python rubric_analysis.py                       # diagnostics + clustering (llama)
  python rubric_analysis.py --judge gpt-5_4-mini  # a specific judge's scores

Diagnostics (base rates, always shown) validate the codebook: a criterion that
fires ~never or ~always is uninformative; a big base-rate gap between two judges
flags a fuzzy definition.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from shared.config import part_output
from shared.jsd_stats import jsd, summarize, TIER      # reuse the exact stats
from part3_distribution.rubric import IDS, LABELS
from shared.utils import load_jsonl


def load_scores(judge_slug):
    path = os.path.join(part_output("part3_distribution", "rubric"), f"{judge_slug}.jsonl")
    rows = load_jsonl(path)
    # present[(model, case)] = {id: bool}
    recs = {}
    for r in rows:
        recs[(r["model"], f"{r['dataset']}:{r['case_id']}")] = {
            int(k): bool(v) for k, v in r["present"].items()}
    return recs, path


def presence_matrix(recs):
    """models x 20 criteria, value = fraction of the model's cases criterion present."""
    models = sorted({m for m, _ in recs})
    counts = {m: np.zeros(len(IDS)) for m in models}
    ncase = {m: 0 for m in models}
    for (m, _), d in recs.items():
        ncase[m] += 1
        for j, i in enumerate(IDS):
            if d.get(i):
                counts[m][j] += 1
    rate = pd.DataFrame({m: counts[m] / ncase[m] for m in models},
                        index=IDS).T          # models x criteria
    return rate, ncase


def _family_ratio(D, models, n_perm=2000, seed=0):
    """within/across-family mean JSD, their ratio, and a permutation p, from a
    pairwise-JSD dict D[(a, b)] over `models` (family via TIER; the null makes
    family labels exchangeable across models)."""
    fams = [TIER[m][0] for m in models]
    pairs = list(itertools.combinations(range(len(models)), 2))

    def wx(labels):
        w = [D[(models[i], models[j])] for i, j in pairs if labels[i] == labels[j]]
        x = [D[(models[i], models[j])] for i, j in pairs if labels[i] != labels[j]]
        return float(np.mean(w)), float(np.mean(x))

    within, across = wx(fams)
    ratio = across / within
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        w, x = wx(list(rng.permutation(fams)))
        null[k] = x / w
    return {"within": within, "across": across, "ratio": ratio,
            "p": float((null >= ratio).mean()), "null_mean": float(null.mean()),
            "n_perm": n_perm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="llama-3_3-70b-instruct")
    args = ap.parse_args()

    recs, path = load_scores(args.judge)
    rate, ncase = presence_matrix(recs)
    # gold (the reference answer) is scored too but isn't a model family -- drop it
    # from the clustering; it has no TIER entry.
    rate = rate.loc[[m for m in rate.index if m in TIER]]
    models = list(rate.index)
    print(f"{len(models)} models, {sum(ncase[m] for m in models)} scored answers  ({path})\n")

    # ---- diagnostics: per-criterion base rate (pooled over models) ----
    pooled = rate.mean(axis=0)
    print("per-criterion presence rate (pooled), flag <5% or >95%:")
    for i in IDS:
        r = pooled[i]
        flag = "  <-- uninformative" if (r < 0.05 or r > 0.95) else ""
        print(f"  [{i:>2}] {r:5.0%}  {LABELS[i][:50]}{flag}")

    # ---- clustering (needs >=2 families with >=2 models each) ----
    fams = [TIER[m][0] for m in models if m in TIER]
    from collections import Counter
    fc = Counter(fams)
    if len(models) < 6 or sum(v >= 2 for v in fc.values()) < 2:
        print(f"\n[skip clustering] only {len(models)} models "
              f"({dict(fc)}) -- run the full 12-model set for the family test.")
        return

    M = rate.div(rate.sum(axis=1), axis=0)     # normalize each model to a distribution
    purity, within, across, nn = summarize(M, models)

    # --- rubric granularity: within/across-family JSD + permutation p ---
    Dru = {(a, b): jsd(M.loc[a].values, M.loc[b].values)
           for a, b in itertools.combinations(models, 2)}
    ru = _family_ratio(Dru, models)
    ru["nn_family_purity"] = purity

    print(f"\nRUBRIC clustering (20 criteria, binary presence):")
    print(f"  nn family purity : {purity:.0%}")
    print(f"  within JSD       : {ru['within']:.4f}")
    print(f"  across JSD       : {ru['across']:.4f}")
    print(f"  ratio            : {ru['ratio']:.2f}")
    print(f"  permutation p    : {ru['p']:.3f}   (null mean {ru['null_mean']:.2f})")
    print("  ratio>1 & p<.05 -> families differ on substantive criteria")

    pd.DataFrame(Dru.items(), columns=["pair", "jsd"]).to_csv(
        part_output("part3_distribution", f"rubric_jsd_pairs_{args.judge}.csv"), index=False)
    M.to_csv(part_output("part3_distribution", f"rubric_matrix_{args.judge}.csv"))

    # --- open-vocab granularity: read Part 2's frequency JSD matrix (gold dropped) ---
    ov = None
    ov_path = part_output("part2_coverage", "js_divergence_freq.csv")
    if os.path.exists(ov_path):
        JS = pd.read_csv(ov_path, index_col=0)
        ov_models = [m for m in JS.index if m in TIER]
        Dov = {(a, b): float(JS.loc[a, b])
               for a, b in itertools.combinations(ov_models, 2)}
        ov = _family_ratio(Dov, ov_models)
        print(f"\nOPEN-VOCAB (556 features, freq): ratio {ov['ratio']:.2f}  p {ov['p']:.3f}")
    else:
        print(f"\n[open-vocab skipped] {ov_path} missing -- run "
              f"part2_coverage.nearest_neighbor --suffix _freq first.")

    # One stats file, READ by both ratio plots so they cannot drift apart.
    stats = {
        "judge": args.judge,
        "n_models": len(models),
        "n_answers": int(sum(ncase[m] for m in models)),
        "open_vocab": ov,
        "rubric": ru,
        "criterion_base_rate": {int(i): float(pooled[i]) for i in IDS},
    }
    stats_path = part_output("part3_distribution", f"stats_{args.judge}.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  wrote {stats_path}")


if __name__ == "__main__":
    main()
