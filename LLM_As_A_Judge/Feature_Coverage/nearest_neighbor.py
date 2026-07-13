"""nearest_neighbor.py -- who does gold reason like?

Builds the model x global-feature importance matrix WITH gold included (unlike
importance_distribution.py, which drops gold as the reference), then asks two
questions:

  1. Does JS divergence encode provenance at all?  Leave-one-out nearest-
     neighbour family recovery over the 12 contestants, against a permutation
     null.  If shuffling the family labels reproduces the accuracy, the geometry
     is noise.

  2. Which model is gold closest to?  Ranked JS divergence with a case-level
     bootstrap CI, plus P(model is gold's nearest neighbour) over resamples.

Because absolute JSD is badly upward-biased at this sample size (400 cases,
~500 features), NOTHING here reads the magnitudes.  Every claim is a *ranking*
on a shared case set, where the common sampling noise cancels.

  --control  additionally residualizes each model's distribution against
             per-model covariates (mean importance entropy, mean features named)
             before ranking, to test whether the answer is family signal or just
             a capability/verbosity gradient.  See docstring of _residualize.

Outputs (figures/):
  nn_embedding{suffix}.png   -- classical MDS of JS distance, gold starred,
                                arrows = each model's nearest neighbour
  gold_ranking{suffix}.png   -- gold's distance to every model, bootstrap CIs
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


def family(model):
    return model.split("/")[0]


# ============================================================
# Data
# ============================================================

def load_mass():
    """(cases x models x features) importance tensor, gold included.

    Keeping the case axis un-summed is what makes the bootstrap possible: we
    resample cases and re-sum, rather than resampling the aggregate.
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

    fidx = {f: i for i, f in enumerate(feats)}
    midx = {m: i for i, m in enumerate(models)}

    mass = np.zeros((len(keys), len(models), len(feats)))
    named = defaultdict(list)          # model -> features named, per case
    ents = defaultdict(list)           # model -> importance entropy, per case

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
            # Deduplicate by superset slot so this matches coverage.py's
            # features_found exactly -- a model that names the same slot twice
            # has not attended to two features.
            named[m].append(len({i for i, _ in vals}))
            ents[m].append(float(-np.sum(p * np.log(p))))

    covars = np.array([[np.mean(ents[m]), np.mean(named[m])] for m in models])
    return mass, models, feats, covars


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
# Covariate control
# ============================================================

def _residualize(P, models, covars):
    """Strip the part of each model's profile predicted by its covariates.

    Stack the profiles into A (n_models x n_features).  Build design matrix
    X = [1, entropy, n_features_named] (n_models x 3) and project A onto the
    orthogonal complement of X's column space:

        H = X (X'X)^-1 X'        (hat matrix, n_models x n_models)
        R = (I - H) A            (residual profiles)

    Column f of A is regressed on X across MODELS -- so we are removing the
    component of "how much attention does this model give feature f" that is
    linearly explainable by how diffuse / how verbose that model is.  What is
    left is the idiosyncratic part.

    R's rows are no longer probability distributions (they contain negatives and
    sum to ~0), so JSD is undefined on them.  We therefore rank on CORRELATION
    distance, 1 - corr(r_i, r_j), which is scale-free and defined on signed
    vectors.  Only the ranking is comparable to the JSD ranking, not the values.
    """
    A = np.array([P[m] for m in models])
    X = np.column_stack([np.ones(len(models)), covars])
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    R = (np.eye(len(models)) - H) @ A

    C = np.corrcoef(R)
    return 1.0 - C


# ============================================================
# Tests
# ============================================================

def family_recovery(D, models, verbose=True):
    """Leave-one-out NN family accuracy + permutation p-value (gold excluded)."""
    keep = [i for i, m in enumerate(models) if m != "gold"]
    sub = D[np.ix_(keep, keep)]
    names = [models[i] for i in keep]

    def nn(i):
        order = np.argsort(sub[i])
        return next(j for j in order if j != i)

    hits = [family(names[nn(i)]) == family(names[i]) for i in range(len(names))]
    acc = float(np.mean(hits))

    if verbose:
        for i, m in enumerate(names):
            tag = "HIT " if hits[i] else "miss"
            print(f"  {m:32s} -> {names[nn(i)]:32s} {tag}")

    counts = Counter(family(m) for m in names)
    chance = float(np.mean([(counts[family(m)] - 1) / (len(names) - 1)
                            for m in names]))

    rng = np.random.default_rng(1)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        perm = rng.permutation(len(names))
        lab = [family(names[p]) for p in perm]
        null[b] = np.mean([lab[nn(i)] == lab[i] for i in range(len(names))])

    p = (np.sum(null >= acc) + 1) / (N_PERM + 1)
    print(f"\n  NN family accuracy {acc:.0%} ({sum(hits)}/{len(names)})  "
          f"chance {chance:.0%}  permutation p = {p:.4f}")
    return acc, p


def gold_ranking(mass, models, covars, control=False):
    """Ranked distance from gold to each model, with a case bootstrap."""
    contest = [m for m in models if m != "gold"]

    def rank(rows):
        P = profiles(mass, models, rows)
        if control:
            D = _residualize(P, models, covars)
            g = models.index("gold")
            return {m: D[g, models.index(m)] for m in contest}
        return {m: js_divergence(P["gold"], P[m]) for m in contest}

    point = rank(None)

    rng = np.random.default_rng(0)
    n_cases = mass.shape[0]
    draws = defaultdict(list)
    wins = Counter()

    for _ in range(N_BOOT):
        rows = rng.integers(0, n_cases, n_cases)
        r = rank(rows)
        for m, d in r.items():
            draws[m].append(d)
        wins[min(r, key=r.get)] += 1

    # Error bars = bootstrap SPREAD about the point estimate, not a percentile
    # interval.  Resampling with replacement leaves only ~63% distinct cases, so
    # every bootstrap profile is sparser than the real one and its plug-in JSD is
    # biased UP; the percentile interval lands entirely ABOVE theta and a
    # pivotal correction overshoots entirely below it.  Neither is honest.  The
    # bias is near-common across models, so it cancels in the comparison we
    # actually make -- we therefore show variance only (theta +/- 1.96 sd) and
    # let the win counts, which are bias-invariant, carry the inference.
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
    var = np.clip(w, 0, None)
    return coords, float(np.sum(np.clip(w[order], 0, None)) / var.sum())


def plot_embedding(D, models, suffix):
    coords, explained = _mds(D)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_facecolor("white")

    # nearest-neighbour arrows: the structure the accuracy stat is counting
    for i, m in enumerate(models):
        j = next(k for k in np.argsort(D[i]) if k != i)
        ax.annotate(
            "", xy=coords[j], xytext=coords[i],
            arrowprops=dict(arrowstyle="-|>", color="#BBBBBB", lw=1.2,
                            shrinkA=9, shrinkB=11, alpha=0.9),
            zorder=1,
        )

    for i, m in enumerate(models):
        is_gold = m == "gold"
        ax.scatter(
            *coords[i],
            s=420 if is_gold else 150,
            c=GOLD_COLOR if is_gold else FAMILY_COLORS[family(m)],
            marker="*" if is_gold else "o",
            edgecolors="white", linewidths=2, zorder=3,
        )
        label = "GOLD" if is_gold else m.split("/")[1]
        ax.annotate(label, coords[i], xytext=(0, 13 if is_gold else 11),
                    textcoords="offset points", ha="center",
                    fontsize=9.5 if is_gold else 8.5,
                    fontweight="bold" if is_gold else "normal",
                    color="#222222")

    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="white", ms=9,
                          label=f) for f, c in FAMILY_COLORS.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", mfc=GOLD_COLOR,
                              mec="white", ms=15, label="gold"))
    ax.legend(handles=handles, frameon=False, loc="best", fontsize=9)

    ax.set_title("Reasoning-attention space: models near each other weight the "
                 "same features\nclassical MDS on JS distance  |  arrows point "
                 "to each model's nearest neighbour",
                 fontsize=11.5, pad=14, loc="left")
    ax.set_xlabel(f"MDS 1   ({explained:.0%} of distance structure in 2D)",
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


def plot_gold_ranking(point, ci, wins, suffix, control):
    order = sorted(point, key=point.get)
    y = np.arange(len(order))[::-1]
    vals = [point[m] for m in order]
    lo = [point[m] - ci[m][0] for m in order]
    hi = [ci[m][1] - point[m] for m in order]
    colors = [FAMILY_COLORS[family(m)] for m in order]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(y, vals, height=0.62, color=colors, edgecolor="white", linewidth=2,
            zorder=2)
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
    metric = ("correlation distance, covariates removed" if control
              else "JS divergence (bits)")
    ax.set_xlabel(f"distance from GOLD  --  {metric}", fontsize=9.5)
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

    tag = "_controlled" if control else ""
    path = output_path(os.path.join("figures", f"gold_ranking{tag}{suffix}.png"))
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--control", action="store_true",
                    help="residualize on entropy + n_features before ranking")
    args = ap.parse_args()

    os.makedirs(output_path("figures"), exist_ok=True)

    mass, models, feats, covars = load_mass()
    print(f"{mass.shape[0]} cases x {len(models)} models x {len(feats)} features\n")

    P = profiles(mass, models)
    D = divergence_matrix(P, models)

    print("LEAVE-ONE-OUT NEAREST NEIGHBOUR (gold held out of this test)")
    family_recovery(D, models)

    print("\nGOLD RANKING" + ("  [covariates controlled]" if args.control else ""))
    point, ci, wins = gold_ranking(mass, models, covars, control=args.control)
    for m in sorted(point, key=point.get):
        print(f"  {m:32s} {point[m]:.4f}   "
              f"[{ci[m][0]:.4f}, {ci[m][1]:.4f}]   nearest in {wins[m]/N_BOOT:6.1%}")

    by_family = Counter()
    for m, c in wins.items():
        by_family[family(m)] += c
    print("\n  P(gold's nearest neighbour is from family X):")
    for f, c in by_family.most_common():
        print(f"    {f:12s} {c/N_BOOT:6.1%}")

    print("\nFIGURES")
    plot_embedding(D, models, args.suffix)
    plot_gold_ranking(point, ci, wins, args.suffix, args.control)


if __name__ == "__main__":
    main()
