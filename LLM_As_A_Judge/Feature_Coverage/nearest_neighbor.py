"""nearest_neighbor.py -- who does gold reason like, and is the geometry real?

Builds the model x global-feature importance matrix WITH gold included (unlike
importance_distribution.py, which drops gold as the reference), then runs four
tests.  Absolute JSD is badly upward-biased at this sample size (400 cases, ~500
features): two draws from the SAME distribution score ~0.06, which is where most
of the observed pairwise values sit.  So nothing here reads magnitudes.  Every
claim is a RANKING on a shared case set, where the common sampling noise cancels,
and every ranking is checked against a null.

  1. REPLICATION.  Build the whole distance matrix twice on disjoint halves of
     the cases and correlate them.  If the geometry is sampling noise the two
     halves disagree.  (Observed r ~ 0.92 against a label-shuffled null of ~0.00.)
     NOTE: this detects noise, not systematic bias -- see MISSING DATA below.

  2. WHAT DRIVES DISTANCE.  Regress the 66 pairwise JSDs on |verbosity gap| and
     same-family.  This replaces an earlier --control flag that residualized the
     profiles against verbosity covariates.  That control was wrong twice over:
     it silently switched metric (residuals aren't distributions, so JSD is
     undefined and it fell back to correlation distance), and it controlled for a
     MEDIATOR -- verbosity is itself a family trait (Anthropic ~8.8 features/case,
     Google ~5.0), so regressing it out deletes the very signal being measured.
     The pairwise regression answers the question the control was reaching for
     without either flaw, and it finds verbosity explains R^2 = 0.000 of the
     divergence while same-family explains 0.174 (p < 1e-4).  There is nothing to
     control away.

  3. FAMILY RECOVERY.  Leave-one-out nearest-neighbour family accuracy against a
     permutation null, plus the BOOTSTRAP STABILITY of each NN edge -- because a
     "hit" on a pair separated by 0.001 is luck, not evidence.

  4. GOLD.  Ranked JS divergence from gold to every model, with the bootstrap
     probability that each is gold's nearest neighbour.

MISSING DATA.  Generation failed for some cases (claude-haiku-4.5 is short 82 of
400, qwen3.5-9b short 61; everyone else <5).  A model estimated from fewer cases
has a sparser profile and therefore an inflated JSD to EVERYONE -- a systematic
bias that replicates across splits and so survives test 1.  --complete-cases
restricts every model to the cases where all 13 generators succeeded, which is
the only way to make the comparison genuinely apples-to-apples.

Outputs (figures/):
  nn_embedding{suffix}.png   -- classical MDS of JS distance, gold starred,
                                arrows = each model's nearest neighbour
  gold_ranking{suffix}.png   -- gold's distance to every model
"""

import argparse
import itertools
import json
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import output_path

# Same Okabe-Ito assignment visualize_distributions.py uses -- keep families
# reading identically across every figure in the project.
FAMILY_COLORS = {
    "anthropic": "#E69F00",
    "openai":    "#0072B2",
    "google":    "#009E73",
    "qwen":      "#D55E00",
}
GOLD_COLOR = "#333333"

N_BOOT = 2000
N_PERM = 20000
N_SPLIT = 200


def family(model):
    return model.split("/")[0]


# ============================================================
# Data
# ============================================================

def load_mass(complete_cases=False):
    """(cases x models x features) importance tensor, gold included.

    Keeping the case axis un-summed is what makes the bootstrap possible: we
    resample cases and re-sum, rather than resampling the aggregate.

    complete_cases: keep only cases where every generator produced features, so
    each model's profile is estimated from an identical case set.
    """
    with open(output_path("case_features.json")) as fh:
        cases = json.load(fh)
    with open(output_path("global_features.json")) as fh:
        mapping = json.load(fh)["mapping"]

    keys = sorted(cases)
    feats = sorted({v for v in mapping.values() if v})
    models = sorted({e["model"]
                     for c in cases.values()
                     for e in c["features"].values()})

    if complete_cases:
        keys = [k for k in keys
                if {e["model"] for e in cases[k]["features"].values()} == set(models)]

    fidx = {f: i for i, f in enumerate(feats)}
    midx = {m: i for i, m in enumerate(models)}

    mass = np.zeros((len(keys), len(models), len(feats)))
    named = defaultdict(list)          # model -> features named, per case
    evens = defaultdict(list)          # model -> evenness, per case

    for ci, k in enumerate(keys):
        superset = cases[k]["superset"]
        per_case = defaultdict(list)

        for e in cases[k]["features"].values():
            idx = e["superset_index"]
            if idx is None:
                continue
            global_feature = mapping.get(superset[idx])
            if not global_feature:
                continue
            mass[ci, midx[e["model"]], fidx[global_feature]] += e["importance"]
            per_case[e["model"]].append((idx, e["importance"]))

        for m, vals in per_case.items():
            total = sum(v for _, v in vals)
            if total <= 0:
                continue
            p = np.array([v for _, v in vals]) / total
            p = p[p > 0]          # 0*log0 := 0; without the mask numpy gives nan
            # Deduplicate by superset slot to match coverage.py's features_found:
            # a model that names the same slot twice has not attended to two
            # features.
            k_named = len({i for i, _ in vals})
            h = float(-np.sum(p * np.log(p)))
            named[m].append(k_named)
            # Evenness, not raw entropy: raw H tracks k at r ~ 0.95 and is mostly
            # a verbosity restatement (see feature_analysis.py).
            evens[m].append(h / np.log(k_named) if k_named > 1 else np.nan)

    covars = {m: {"verbosity": float(np.mean(named[m])),
                  "evenness": float(np.nanmean(evens[m]))} for m in models}
    return mass, models, feats, covars, len(keys)


# ============================================================
# Divergence
# ============================================================

def _normalize(v):
    total = v.sum()
    return v / total if total > 0 else v


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits; matches importance_distribution.py."""
    m = 0.5 * (p + q)

    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / m[mask])))

    return 0.5 * kl(p) + 0.5 * kl(q)


def profiles(mass, models, rows=None):
    """model -> probability distribution over features, summed across `rows`."""
    if rows is None:
        rows = np.arange(mass.shape[0])
    return {m: _normalize(mass[rows, i].sum(0)) for i, m in enumerate(models)}


def divergence_matrix(P, models):
    n = len(models)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = D[j, i] = js_divergence(P[models[i]], P[models[j]])
    return D


# ============================================================
# 1. Is the geometry real, or sampling noise?
# ============================================================

def replication(mass, models):
    """Correlate the distance matrix built on two DISJOINT halves of the cases.

    Signal reproduces across independent data; noise does not.  The null shuffles
    one half's model labels, destroying the correspondence while preserving the
    distribution of distances.
    """
    n = mass.shape[0]
    rng = np.random.default_rng(0)
    iu = np.triu_indices(len(models), 1)

    obs, null = [], []
    for _ in range(N_SPLIT):
        p = rng.permutation(n)
        A = divergence_matrix(profiles(mass, models, p[:n // 2]), models)
        B = divergence_matrix(profiles(mass, models, p[n // 2:]), models)
        obs.append(np.corrcoef(A[iu], B[iu])[0, 1])

        perm = rng.permutation(len(models))
        Bs = B[np.ix_(perm, perm)]
        null.append(np.corrcoef(A[iu], Bs[iu])[0, 1])

    obs, null = np.array(obs), np.array(null)
    cut = np.percentile(null, 95)
    print(f"  split-half r  = {obs.mean():.3f}  (5th pct {np.percentile(obs, 5):.3f})")
    print(f"  shuffled null = {null.mean():.3f}  (95th pct {cut:.3f})")
    print(f"  -> observed beats the null in {np.mean(obs > cut):.0%} of {N_SPLIT} splits")
    print("  NOTE: this rules out sampling noise, NOT systematic bias "
          "(e.g. missing data).")


# ============================================================
# 2. What actually drives distance: verbosity, or family?
# ============================================================

def distance_drivers(P, models, covars):
    """Regress pairwise JSD on |verbosity gap| and same-family.

    The question the old --control flag was reaching for, asked properly: does
    family predict reasoning similarity BEYOND what a verbosity difference would
    predict?  Verbosity is not a nuisance to be residualized out -- it is itself
    part of house style -- so we test whether it explains anything, rather than
    assuming it does and removing it.
    """
    contest = [m for m in models if m != "gold"]
    pairs = list(itertools.combinations(contest, 2))

    y = np.array([js_divergence(P[a], P[b]) for a, b in pairs])
    gap = np.array([abs(covars[a]["verbosity"] - covars[b]["verbosity"])
                    for a, b in pairs])
    same = np.array([1.0 if family(a) == family(b) else 0.0 for a, b in pairs])

    def fit(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta, 1 - (y - X @ beta).var() / y.var()

    ones = np.ones(len(pairs))
    _, r2_verb = fit(np.column_stack([ones, gap]))
    beta, r2_full = fit(np.column_stack([ones, gap, same]))

    rng = np.random.default_rng(0)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        perm = rng.permutation(contest)
        lab = {m: family(p) for m, p in zip(contest, perm)}
        sp = np.array([1.0 if lab[a] == lab[b] else 0.0 for a, b in pairs])
        bp, *_ = np.linalg.lstsq(np.column_stack([ones, gap, sp]), y, rcond=None)
        null[b] = bp[2]
    p = (np.sum(null <= beta[2]) + 1) / (N_PERM + 1)

    print(f"  {len(pairs)} model pairs.  JSD ~ 1 + |verbosity gap| + same_family")
    print(f"    |verbosity gap|   {beta[1]:+.5f} per feature   "
          f"[R^2 alone = {r2_verb:.3f}]")
    print(f"    same family       {beta[2]:+.4f}               "
          f"[R^2 added = {r2_full - r2_verb:+.3f}]   perm p = {p:.4f}")
    print(f"  -> verbosity explains ~nothing; family is what the metric encodes.")


# ============================================================
# 3. Family recovery + how stable each nearest-neighbour edge is
# ============================================================

def family_recovery(D, models):
    """Leave-one-out NN family accuracy + permutation null (gold held out)."""
    keep = [i for i, m in enumerate(models) if m != "gold"]
    sub = D[np.ix_(keep, keep)]
    names = [models[i] for i in keep]

    def nn(i):
        return next(j for j in np.argsort(sub[i]) if j != i)

    hits = [family(names[nn(i)]) == family(names[i]) for i in range(len(names))]
    acc = float(np.mean(hits))

    counts = Counter(family(m) for m in names)
    chance = float(np.mean([(counts[family(m)] - 1) / (len(names) - 1)
                            for m in names]))

    rng = np.random.default_rng(1)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        lab = [family(names[p]) for p in rng.permutation(len(names))]
        null[b] = np.mean([lab[nn(i)] == lab[i] for i in range(len(names))])
    p = (np.sum(null >= acc) + 1) / (N_PERM + 1)

    print(f"  NN family accuracy {acc:.0%} ({sum(hits)}/{len(names)})   "
          f"chance {chance:.0%}   permutation p = {p:.4f}")
    return names, hits


def edge_stability(mass, models, names, hits):
    """Bootstrap P(each model's nearest neighbour is X).

    A family "hit" on a pair separated by 0.001 bits is luck.  This says which
    edges are load-bearing and which are coin flips.
    """
    contest = [m for m in models if m != "gold"]
    rng = np.random.default_rng(2)
    n = mass.shape[0]
    win = Counter()

    for _ in range(500):
        rows = rng.integers(0, n, n)
        P = profiles(mass, models, rows)
        for m in contest:
            nn = min((x for x in contest if x != m),
                     key=lambda x: js_divergence(P[m], P[x]))
            win[(m, nn)] += 1

    hit_of = dict(zip(names, hits))
    for m in contest:
        top, share = max(((x, c) for (mm, x), c in win.items() if mm == m),
                         key=lambda t: t[1])
        share /= 500
        tag = "same-family" if family(top) == family(m) else "CROSS-family"
        flag = "  <- coin flip" if share < 0.65 else ""
        mark = "HIT " if hit_of.get(m) else "miss"
        print(f"  {mark} {m:30s} -> {top:30s} {share:4.0%}  {tag}{flag}")


# ============================================================
# 4. Gold
# ============================================================

def gold_ranking(mass, models):
    """Ranked distance from gold to each model, with a case bootstrap."""
    contest = [m for m in models if m != "gold"]

    def rank(rows):
        P = profiles(mass, models, rows)
        return {m: js_divergence(P["gold"], P[m]) for m in contest}

    point = rank(None)

    rng = np.random.default_rng(0)
    n = mass.shape[0]
    draws = defaultdict(list)
    wins = Counter()

    for _ in range(N_BOOT):
        r = rank(rng.integers(0, n, n))
        for m, d in r.items():
            draws[m].append(d)
        wins[min(r, key=r.get)] += 1

    # Error bars = bootstrap SPREAD about the point estimate, not a percentile
    # interval.  Resampling with replacement leaves only ~63% distinct cases, so
    # every bootstrap profile is sparser than the real one and its plug-in JSD is
    # biased UP: the percentile interval lands entirely ABOVE the point estimate
    # and a pivotal correction overshoots entirely below it.  Neither is honest.
    # The bias is near-common across models so it cancels in the comparison we
    # actually make; we show variance only and let the win counts, which are
    # bias-invariant, carry the inference.
    ci = {}
    for m in contest:
        half = 1.96 * float(np.std(draws[m]))
        ci[m] = (max(0.0, point[m] - half), point[m] + half)

    return point, ci, wins


# ============================================================
# Figures
# ============================================================

def _mds(D):
    """Classical MDS (PCoA) on the JS *distance* sqrt(JSD).

    sqrt(JSD) is a true metric (JSD itself is not), so it is the right thing to
    embed.  Double-center the squared-distance matrix and take the top two
    eigenvectors: B = -0.5 * J D^2 J,  coords = V sqrt(L).
    """
    Dm = np.sqrt(np.clip(D, 0, None))
    n = len(Dm)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:2]
    coords = V[:, order] * np.sqrt(np.clip(w[order], 0, None))
    return coords, float(np.sum(np.clip(w[order], 0, None))
                         / np.clip(w, 0, None).sum())


def plot_embedding(D, models, suffix):
    coords, explained = _mds(D)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_facecolor("white")

    # Nearest-neighbour arrows are computed in the FULL space, not from the 2D
    # coordinates -- only ~38% of the distance structure survives the projection,
    # so an arrow may well point to a node that is not the closest on screen.
    for i, m in enumerate(models):
        j = next(k for k in np.argsort(D[i]) if k != i)
        ax.annotate("", xy=coords[j], xytext=coords[i],
                    arrowprops=dict(arrowstyle="-|>", color="#BBBBBB", lw=1.2,
                                    shrinkA=9, shrinkB=11, alpha=0.9),
                    zorder=1)

    for i, m in enumerate(models):
        is_gold = m == "gold"
        ax.scatter(*coords[i], s=420 if is_gold else 150,
                   c=GOLD_COLOR if is_gold else FAMILY_COLORS[family(m)],
                   marker="*" if is_gold else "o",
                   edgecolors="white", linewidths=2, zorder=3)
        ax.annotate("GOLD" if is_gold else m.split("/")[1], coords[i],
                    xytext=(0, 13 if is_gold else 11), textcoords="offset points",
                    ha="center", fontsize=9.5 if is_gold else 8.5,
                    fontweight="bold" if is_gold else "normal", color="#222222")

    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="white", ms=9,
                          label=f) for f, c in FAMILY_COLORS.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", mfc=GOLD_COLOR,
                              mec="white", ms=15, label="gold"))
    ax.legend(handles=handles, frameon=False, loc="best", fontsize=9)

    ax.set_title("Reasoning-attention space: models near each other weight the "
                 "same features\nclassical MDS on JS distance  |  arrows point "
                 "to each model's nearest neighbour (computed in full space)",
                 fontsize=11.5, pad=14, loc="left")
    ax.set_xlabel(f"MDS 1   ({explained:.0%} of distance structure survives in 2D)",
                  fontsize=9, color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    path = output_path(os.path.join("figures", f"nn_embedding{suffix}.png"))
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_gold_ranking(point, ci, wins, suffix):
    order = sorted(point, key=point.get)
    y = np.arange(len(order))[::-1]
    vals = [point[m] for m in order]
    lo = [point[m] - ci[m][0] for m in order]
    hi = [ci[m][1] - point[m] for m in order]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(y, vals, height=0.62,
            color=[FAMILY_COLORS[family(m)] for m in order],
            edgecolor="white", linewidth=2, zorder=2)
    ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor="#555555",
                elinewidth=1.3, capsize=3, zorder=3)

    for yi, m in zip(y, order):
        share = wins[m] / N_BOOT
        note = f"  {point[m]:.3f}"
        if share > 0.005:
            note += f"   ({share:.0%} of bootstraps = nearest)"
        ax.text(ci[m][1] + max(vals) * 0.015, yi, note, va="center",
                fontsize=8.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([m.split("/")[1] for m in order], fontsize=9)
    ax.set_xlabel("distance from GOLD  --  JS divergence (bits)", fontsize=9.5)
    ax.set_title("Which model reasons most like gold?\n"
                 f"lower = closer  |  bars are +/-1.96 sd over {N_BOOT} case "
                 "bootstraps (spread, not a bias-corrected CI)",
                 fontsize=11.5, pad=14, loc="left")
    ax.set_xlim(0, max(ci[m][1] for m in order) * 1.42)
    ax.grid(axis="x", color="#E8E8E8", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [plt.Line2D([], [], marker="s", ls="", mfc=c, mec="none", ms=9,
                          label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)

    path = output_path(os.path.join("figures", f"gold_ranking{suffix}.png"))
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--complete-cases", action="store_true",
                    help="only use cases where all 13 generators succeeded, so "
                         "every model is estimated from an identical case set")
    args = ap.parse_args()

    os.makedirs(output_path("figures"), exist_ok=True)

    mass, models, feats, covars, n_cases = load_mass(args.complete_cases)
    mode = "COMPLETE CASES ONLY" if args.complete_cases else "all cases"
    print(f"{n_cases} cases x {len(models)} models x {len(feats)} features "
          f"[{mode}]\n")

    P = profiles(mass, models)
    D = divergence_matrix(P, models)

    print("1. REPLICATION -- is the geometry real?")
    replication(mass, models)

    print("\n2. WHAT DRIVES DISTANCE -- verbosity or family?")
    distance_drivers(P, models, covars)

    print("\n3. FAMILY RECOVERY (gold held out)")
    names, hits = family_recovery(D, models)
    print()
    edge_stability(mass, models, names, hits)

    print("\n4. GOLD")
    point, ci, wins = gold_ranking(mass, models)
    for m in sorted(point, key=point.get):
        print(f"  {m:30s} {point[m]:.4f}  [{ci[m][0]:.4f}, {ci[m][1]:.4f}]  "
              f"nearest in {wins[m] / N_BOOT:6.1%}")

    by_family = Counter()
    for m, c in wins.items():
        by_family[family(m)] += c
    print("\n  P(gold's nearest neighbour is from family X):")
    for f, c in by_family.most_common():
        print(f"    {f:12s} {c / N_BOOT:6.1%}")

    print("\nFIGURES")
    plot_embedding(D, models, args.suffix)
    plot_gold_ranking(point, ci, wins, args.suffix)


if __name__ == "__main__":
    main()
