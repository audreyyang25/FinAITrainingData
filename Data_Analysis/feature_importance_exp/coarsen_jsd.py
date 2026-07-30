"""coarsen_jsd.py -- does family clustering in JS-divergence space survive
coarsening the feature vocabulary?

The question: models of the same family are nearest neighbours in JSD space. Is
that SUBSTANCE (families attend to different legal concepts) or VOCABULARY
(families phrase the same concept differently, and the canonicalizer splits
those phrasings into separate global features)?

Test: merge the ~556 global features into K groups for a sweep of K, recompute
the JSD matrix at each granularity, and track how family structure holds up.

    clustering PERSISTS as K shrinks -> substance
    clustering WASHES OUT as K shrinks -> vocabulary granularity artifact

Grouping is TF-IDF + agglomerative (cosine, average linkage) over the feature
strings -- deterministic and model-free, so it adds no family bias of its own.
Note the limit: it merges LEXICALLY similar strings. Cross-family paraphrases of
one concept that share few words may survive as separate groups, which biases
this test TOWARD finding persistence. Read a null result as strong, a positive
result as suggestive.

All inputs come from the independent-extractor run (outputs/llama), never the
models' self-reported importances.

    python coarsen_jsd.py
    python coarsen_jsd.py --ks 556 200 100 50 25 10 --weight freq
"""

import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ANTHROPIC, GEMINI, OPENAI, QWEN, output_path

TIER = {m["model"]: (m["family"], m["key"])
        for fam in (ANTHROPIC, OPENAI, GEMINI, QWEN) for m in fam}


def load(extract_dir):
    """mass[m][f], named[m][f], per-model totals -- from the llama extraction."""
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
    """Nearest-neighbour purity and within/across-family mean JSD."""
    D = pd.DataFrame({a: {b: jsd(M.loc[a].values, M.loc[b].values)
                          for b in models} for a in models})
    np.fill_diagonal(D.values, np.inf)

    nn = {m: D[m].idxmin() for m in models}
    purity = sum(TIER[m][0] == TIER[nn[m]][0] for m in models) / len(models)

    within, across = [], []
    for a, b in itertools.combinations(models, 2):
        (within if TIER[a][0] == TIER[b][0] else across).append(D.loc[a, b])

    return purity, float(np.mean(within)), float(np.mean(across)), nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/llama",
                    help="independent-extractor output dir (default outputs/llama)")
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[400, 200, 100, 50, 25, 12, 6],
                    help="group counts to sweep (full vocabulary is added automatically)")
    ap.add_argument("--weight", choices=["p", "freq"], default="p",
                    help="p = importance share (freq x intensity); freq = selection rate only")
    ap.add_argument("--grouping", choices=["tfidf", "llama"], default="tfidf",
                    help="tfidf = lexical sweep over --ks; llama = semantic "
                         "concept/domain levels from group_features_llama.py")
    ap.add_argument("--perm", type=int, default=0,
                    help="permutation reps for the family-separation null (0 = skip)")
    args = ap.parse_args()

    mass, named, total, ncase = load(args.dir)
    models = sorted(m for m in mass if m != "gold")
    feats = sorted({g for m in models for g in mass[m]})

    if args.weight == "p":
        raw = pd.DataFrame({m: [mass[m].get(g, 0) / total[m] for g in feats]
                            for m in models}, index=feats).T
    else:
        raw = pd.DataFrame({m: [named[m].get(g, 0) / ncase[m] for g in feats]
                            for m in models}, index=feats).T

    print(f"{len(models)} models x {len(feats)} global features "
          f"(weight={args.weight}, source={args.dir})\n")

    # levels: (K, label-array or None for the full vocabulary)
    levels = [(len(feats), None)]
    if args.grouping == "tfidf":
        X = np.asarray(TfidfVectorizer(stop_words="english",
                                       sublinear_tf=True).fit_transform(feats).todense())
        for k in sorted(args.ks, reverse=True):
            if k < len(feats):
                levels.append((k, AgglomerativeClustering(
                    n_clusters=k, metric="cosine", linkage="average").fit_predict(X)))
    else:
        with open(os.path.join(args.dir, "feature_groups.json")) as fh:
            groups = json.load(fh)
        for field in ("feature_to_concept", "feature_to_domain"):
            lab = np.array([groups[field].get(f, -1) for f in feats])
            levels.append((len(set(lab)), lab))
        print(f"semantic grouping by {groups['grouper']}\n")

    rng = np.random.default_rng(0)
    fams = [TIER[m][0] for m in models]

    head = f"{'K':>6} {'nn purity':>10} {'within':>9} {'across':>9} {'ratio':>7}"
    print(head + (f" {'p':>7}" if args.perm else ""))
    rows = []
    for k, lab in levels:
        if lab is None:
            M = raw.div(raw.sum(1), axis=0)
        else:
            M = raw.T.groupby(lab).sum().T
            M = M.div(M.sum(1), axis=0)

        purity, within, across, nn = summarize(M, models)
        ratio = across / within if within else float("nan")
        line = f"{k:>6} {purity:>9.0%} {within:>9.4f} {across:>9.4f} {ratio:>7.2f}"

        p = None
        if args.perm:
            # Null: family labels are exchangeable across the 12 models.
            D = {(a, b): jsd(M.loc[a].values, M.loc[b].values)
                 for a, b in itertools.combinations(models, 2)}

            def sep(labels):
                lm = dict(zip(models, labels))
                w = [d for (a, b), d in D.items() if lm[a] == lm[b]]
                x = [d for (a, b), d in D.items() if lm[a] != lm[b]]
                return np.mean(x) / np.mean(w)

            null = np.array([sep(list(rng.permutation(fams)))
                             for _ in range(args.perm)])
            p = float((null >= ratio).mean())
            line += f" {p:>7.3f}"

        print(line)
        rows.append(dict(k=k, nn_purity=purity, within=within,
                         across=across, ratio=ratio, p=p))

    path = output_path(f"coarsen_jsd_{args.grouping}_{args.weight}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nratio = across-family JSD / within-family JSD.")
    print("  stays > 1 as K falls -> families differ on substance")
    print("  decays toward 1      -> family structure was vocabulary granularity")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
