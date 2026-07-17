"""family_tier_analysis.py -- how families and tiers differ in what they weight,
from importance_matrix{suffix}.csv (default: the Llama complete-cases run).

  Q1 cross-family  : do families diverge on the SAME features or different ones?
  Q2 within-family : which features do tiers (frontier vs small) weight apart?
  Q3 generalization: does the tier gradient hold ACROSS families?

Gold is excluded (not a family/tier). Also writes
family_tier_features{suffix}.csv -- per-feature detail (family mean weights,
cross-family variance, each family's frontier-small gradient, the universal
average gradient, and whether the gradient's sign agrees across all 4 families),
sorted by |avg gradient|, so you can examine specific features beyond the top-N.
"""

import argparse
import itertools
from collections import Counter

import numpy as np
import pandas as pd

from config import output_path, ANTHROPIC, OPENAI, GEMINI, QWEN

TIERS = {m["model"]: (m["family"], m["key"])
         for fam in (ANTHROPIC, OPENAI, GEMINI, QWEN) for m in fam}
FAMS = ["Anthropic", "OpenAI", "Gemini", "Qwen"]

short = lambda s: s if len(s) <= 56 else s[:55] + "…"


def _term(x, m):
    out = np.zeros_like(x)
    nz = x > 0
    out[nz] = 0.5 * x[nz] * np.log2(x[nz] / m[nz])
    return out


def jsd_contrib(p, q):
    m = 0.5 * (p + q)
    return _term(p, m) + _term(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_llama_complete",
                    help="reads importance_matrix{suffix}.csv")
    args = ap.parse_args()

    M = pd.read_csv(output_path(f"importance_matrix{args.suffix}.csv"), index_col=0)
    feats = np.array(M.columns)
    models = [m for m in M.index if m != "gold"]

    famP = {}
    for f in FAMS:
        v = np.mean([M.loc[m].to_numpy() for m in models if TIERS[m][0] == f], axis=0)
        famP[f] = v / v.sum()

    # Q1 -------------------------------------------------------------
    print("Q1. CROSS-FAMILY: same features or different?")
    FP = np.vstack([famP[f] for f in FAMS])
    var = FP.var(axis=0)
    print("  top features separating the families (cross-family variance):")
    for j in np.argsort(var)[::-1][:8]:
        vals = " ".join(f"{f[:3]}={famP[f][j]:.3f}" for f in FAMS)
        print(f"    {var[j]:.5f}  {vals}  {short(feats[j])}")

    pc = {p: jsd_contrib(famP[p[0]], famP[p[1]])
          for p in itertools.combinations(FAMS, 2)}
    C = np.corrcoef(np.vstack(list(pc.values())))
    print(f"  mean corr between 6 family-pair divergence patterns: "
          f"{C[np.triu_indices(6, 1)].mean():.2f}  (high=same features)")
    rec = Counter()
    for v in pc.values():
        for j in np.argsort(v)[::-1][:10]:
            rec[j] += 1
    print("  features recurring across family pairs (top-10 of N/6):")
    for j, ct in rec.most_common(5):
        print(f"    {ct}/6  {short(feats[j])}")

    # Q2 + gradients -------------------------------------------------
    grad = {}
    for f in FAMS:
        fr = next(m for m in models if TIERS[m] == (f, "frontier"))
        sm = next(m for m in models if TIERS[m] == (f, "small"))
        grad[f] = M.loc[fr].to_numpy() - M.loc[sm].to_numpy()

    print("\nQ2. WITHIN-FAMILY tier gradient (frontier - small):")
    for f in FAMS:
        g = grad[f]
        fr = ", ".join(short(feats[j])[:32] for j in np.argsort(g)[::-1][:2])
        sm = ", ".join(short(feats[j])[:32] for j in np.argsort(g)[:2])
        print(f"  {f:10s} frontier+: {fr}")
        print(f"  {'':10s} small+:    {sm}")

    # Q3 -------------------------------------------------------------
    print("\nQ3. Does the tier gradient GENERALIZE across families?")
    G = np.vstack([grad[f] for f in FAMS])
    Cg = np.corrcoef(G)
    print("        " + " ".join(f"{f[:4]:>6s}" for f in FAMS))
    for i, f in enumerate(FAMS):
        print(f"  {f[:4]:6s} " + " ".join(f"{Cg[i, k]:6.2f}" for k in range(4)))
    print(f"  mean off-diagonal corr: {Cg[np.triu_indices(4, 1)].mean():.2f}  "
          f"(high=universal tier effect; ~0=family-specific)")

    avg = G.mean(axis=0)
    consistent = (np.sign(G) == np.sign(avg)).all(axis=0)
    print("  frontier weights MORE across all 4 families:")
    for j in [j for j in np.argsort(avg)[::-1] if consistent[j]][:5]:
        print(f"    +{avg[j]:.3f}  {short(feats[j])}")
    print("  small weights MORE across all 4 families:")
    for j in [j for j in np.argsort(avg) if consistent[j]][:5]:
        print(f"    {avg[j]:.3f}  {short(feats[j])}")

    # per-feature CSV for examination --------------------------------
    df = pd.DataFrame({"feature": feats})
    for f in FAMS:
        df[f] = famP[f]
    df["cross_family_var"] = var
    for f in FAMS:
        df[f"grad_{f}"] = grad[f]
    df["avg_grad"] = avg
    df["sign_consistent"] = consistent
    df = df.sort_values("avg_grad", key=lambda s: s.abs(), ascending=False)

    path = output_path(f"family_tier_features{args.suffix}.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path} ({len(df)} features)")


if __name__ == "__main__":
    main()
