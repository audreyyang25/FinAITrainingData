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
    ratio = across / within

    rng = np.random.default_rng(0)
    D = {(a, b): jsd(M.loc[a].values, M.loc[b].values)
         for a, b in itertools.combinations(models, 2)}

    def sep(labels):
        lm = dict(zip(models, labels))
        w = [d for (a, b), d in D.items() if lm[a] == lm[b]]
        x = [d for (a, b), d in D.items() if lm[a] != lm[b]]
        return np.mean(x) / np.mean(w)

    null = np.array([sep(list(rng.permutation(fams))) for _ in range(2000)])
    p = float((null >= ratio).mean())

    print(f"\nRUBRIC clustering (20 criteria, binary presence):")
    print(f"  nn family purity : {purity:.0%}")
    print(f"  within JSD       : {within:.4f}")
    print(f"  across JSD       : {across:.4f}")
    print(f"  ratio            : {ratio:.2f}")
    print(f"  permutation p    : {p:.3f}   (null mean {null.mean():.2f})")
    print("  ratio>1 & p<.05 -> families differ on substantive criteria")

    pd.DataFrame(D.items(), columns=["pair", "jsd"]).to_csv(
        part_output("part3_distribution", f"rubric_jsd_pairs_{args.judge}.csv"), index=False)
    M.to_csv(part_output("part3_distribution", f"rubric_matrix_{args.judge}.csv"))


if __name__ == "__main__":
    main()
