"""jsd_stats.py -- the shared JS-divergence machinery.

Extracted from the (removed) coarsening experiment so Part 2 (nearest_neighbor)
and Part 3 (rubric_analysis, tilt plots) share ONE implementation of the metric
and the family bookkeeping:

    TIER       model slug -> (family, tier)
    jsd        Jensen-Shannon divergence in bits (0 = identical)
    summarize  nearest-neighbour family purity + within/across-family mean JSD
    load       read an independent-extractor dir -> per-model feature mass/counts

jsd/summarize are metric-only: they operate on whatever per-model distribution
the caller passes (selection frequency, rubric presence rates, ...). `load`
returns both an importance-weighted `mass` and a selection `named` count; callers
pick whichever they want.
"""

import itertools
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from shared.config import ANTHROPIC, GEMINI, OPENAI, QWEN

TIER = {m["model"]: (m["family"], m["key"])
        for fam in (ANTHROPIC, OPENAI, GEMINI, QWEN) for m in fam}


def load(extract_dir):
    """mass[m][f], named[m][f], per-model importance totals, and per-model case
    counts -- from an independent-extractor run directory (case_feat.json +
    global_features.json)."""
    with open(os.path.join(extract_dir, "case_feat.json")) as fh:
        cases = json.load(fh)
    with open(os.path.join(extract_dir, "global_features.json")) as fh:
        mapping = json.load(fh)["mapping"]

    mass = defaultdict(lambda: defaultdict(float))
    named = defaultdict(lambda: defaultdict(int))
    total = defaultdict(float)
    seen = defaultdict(set)

    for key, case in cases.items():
        superset = case["superset"]
        hit = defaultdict(lambda: defaultdict(float))
        for e in case["features"].values():
            m = e["model"]
            seen[m].add(key)
            total[m] += e["importance"]
            i = e["superset_index"]
            if i is None:
                continue
            g = mapping.get(superset[i])
            if g:
                hit[m][g] += e["importance"]
        for m, d in hit.items():
            for g, v in d.items():
                if v > 0:
                    mass[m][g] += v
                    named[m][g] += 1

    return mass, named, total, {m: len(s) for m, s in seen.items()}


def _kl(p, m):
    k = p > 0
    return float(np.sum(p[k] * np.log2(p[k] / m[k])))


def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def summarize(M, models):
    """Nearest-neighbour purity and within/across-family mean JSD over the
    per-model distribution matrix M (rows = models in `models`)."""
    D = pd.DataFrame({a: {b: jsd(M.loc[a].values, M.loc[b].values)
                          for b in models} for a in models})
    np.fill_diagonal(D.values, np.inf)

    nn = {m: D[m].idxmin() for m in models}
    purity = sum(TIER[m][0] == TIER[nn[m]][0] for m in models) / len(models)

    within, across = [], []
    for a, b in itertools.combinations(models, 2):
        (within if TIER[a][0] == TIER[b][0] else across).append(D.loc[a, b])

    return purity, float(np.mean(within)), float(np.mean(across)), nn
