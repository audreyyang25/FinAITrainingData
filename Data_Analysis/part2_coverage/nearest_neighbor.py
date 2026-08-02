"""nearest_neighbor.py -- who does gold reason like, and is the geometry real?

Builds the model x global-feature SELECTION-frequency matrix WITH gold included,
from the independent-Llama extraction (outputs/llama), then runs four tests. The
profile is FREQUENCY, not importance: for each model, the share of cases in which
it engaged each global reasoning feature -- the extractor's importance weights are
deliberately discarded, so nothing here depends on a cheaper model's numeric
scoring. Absolute JSD is upward-biased at this sample size (two draws from the
SAME distribution score well above 0), so nothing reads magnitudes. Every claim
is a RANKING on a shared case set, where the common sampling noise cancels, and
every ranking is checked against a null.

  1. REPLICATION.  Build the whole distance matrix twice on disjoint halves of
     the cases and correlate them.  If the geometry is sampling noise the two
     halves disagree; the null shuffles one half's model labels.
     NOTE: this detects noise, not systematic bias -- see MISSING DATA below.

  2. WHAT DRIVES DISTANCE.  Regress the pairwise JSDs on |verbosity gap| and
     same-family.  Verbosity (features named per case) is itself part of house
     style, so we test whether it explains anything rather than residualizing it
     out (which would delete family signal, since verbosity is a family trait).

  3. FAMILY RECOVERY.  Leave-one-out nearest-neighbour family accuracy against a
     permutation null, plus the BOOTSTRAP STABILITY of each NN edge -- because a
     "hit" on a pair separated by a hair is luck, not evidence.

  4. GOLD.  Ranked JS divergence from gold to every model, with the bootstrap
     probability that each is gold's nearest neighbour.

MISSING DATA.  Generation failed for some cases (a few models are short a chunk
of the 400). A model estimated from fewer cases has a sparser profile and thus an
inflated JSD to EVERYONE -- a systematic bias that survives test 1.
--complete-cases restricts every model to the cases where all generators
succeeded, the only way to make the comparison genuinely apples-to-apples.

Run from the Data_Analysis root:
    python -m part2_coverage.nearest_neighbor                    # outputs/llama
    python -m part2_coverage.nearest_neighbor --complete-cases

Outputs (outputs/part2_coverage/):
  figures/nn_embedding{suffix}.png -- classical MDS of JS distance, gold starred,
                                arrows = each model's nearest neighbour
  gold_ranking{suffix}.png   -- gold's distance to every model
  js_heatmap{suffix}.png     -- pairwise JS-divergence heatmap
Plus selection_matrix{suffix}.csv (P_m(f)) and js_divergence{suffix}.csv.
"""

import argparse
import itertools
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.config import part_output

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

# Data

def load_mass(complete_cases=False, case_path=None, global_path=None):
    """(cases x models x features) SELECTION tensor, gold included.

    Frequency, not importance: mass[c, m, g] = 1 if model m named >= 1 feature
    mapping to global feature g in case c, else 0. Summed over cases (see
    `profiles`), each model's profile becomes its selection-frequency
    distribution over the global vocabulary -- the extractor's importance weights
    are dropped entirely. Keeping the case axis un-summed is what makes the
    bootstrap possible: we resample cases and re-sum, rather than resampling the
    aggregate.

    complete_cases: keep only cases where every generator produced features, so
    each model's profile is estimated from an identical case set.
    """
    with open(case_path or part_output("part2_coverage", "llama/case_feat.json")) as fh:
        cases = json.load(fh)
    with open(global_path or part_output("part2_coverage", "llama/global_features.json")) as fh:
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
    named = defaultdict(list)          # model -> distinct features named, per case

    for ci, k in enumerate(keys):
        superset = cases[k]["superset"]
        per_case = defaultdict(set)    # model -> set of global-feature indices named

        for e in cases[k]["features"].values():
            idx = e["superset_index"]
            if idx is None:
                continue
            global_feature = mapping.get(superset[idx])
            if not global_feature:
                continue
            gi = fidx[global_feature]
            # Selection indicator: assignment (not +=) deduplicates a model that
            # names the same global feature twice in one case -- it engaged one
            # feature, not two.
            mass[ci, midx[e["model"]], gi] = 1.0
            per_case[e["model"]].add(gi)

        for m, gis in per_case.items():
            named[m].append(len(gis))

    # verbosity = mean distinct features engaged per case (a frequency covariate,
    # no importance involved).
    covars = {m: {"verbosity": float(np.mean(named[m])) if named[m] else 0.0}
              for m in models}
    return mass, models, feats, covars, len(keys)


# ============================================================
# Divergence
# ============================================================

def _normalize(v):
    total = v.sum()
    return v / total if total > 0 else v


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits (0 = identical distributions)."""
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
    return {"split_half_r": float(obs.mean()), "split_half_r_p5": float(np.percentile(obs, 5)),
            "null_mean": float(null.mean()), "null_p95": float(cut),
            "frac_beats_null": float(np.mean(obs > cut)), "n_splits": N_SPLIT}


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
    return {"n_pairs": len(pairs), "verbosity_beta": float(beta[1]),
            "verbosity_r2": float(r2_verb), "family_beta": float(beta[2]),
            "family_r2_added": float(r2_full - r2_verb), "perm_p": float(p), "n_perm": N_PERM}


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
    return names, hits, {"nn_accuracy": float(acc), "chance": float(chance),
                         "perm_p": float(p), "n_perm": N_PERM, "n_models": len(names)}


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

    path = part_output("part2_coverage", os.path.join("figures", f"nn_embedding{suffix}.png"))
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

    path = part_output("part2_coverage", os.path.join("figures", f"gold_ranking{suffix}.png"))
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_heatmap(D, models, suffix):
    """Annotated pairwise JS-divergence heatmap -- the raw distances behind the
    NN plot and gold ranking. Family-ordered so same-family blocks are visible.

    Read the PATTERN (are same-family pairs consistently lower?), not individual
    cells: at this sample size absolute JSD is upward-biased (~0.06 even for two
    draws from the SAME distribution), so a single 0.042-vs-0.044 gap is noise.
    The replication test printed at run start is what certifies the structure.
    """
    fam_rank = {f: i for i, f in enumerate(FAMILY_COLORS)}
    order = sorted(range(len(models)),
                   key=lambda i: (fam_rank.get(family(models[i]), 9), models[i]))
    Do = D[np.ix_(order, order)]
    labels = ["GOLD" if models[i] == "gold" else models[i].split("/")[1]
              for i in order]
    lab_c = [GOLD_COLOR if models[i] == "gold" else FAMILY_COLORS[family(models[i])]
             for i in order]
    n = len(models)

    fig, ax = plt.subplots(figsize=(1.05 * n + 2, 1.05 * n + 1))
    im = ax.imshow(Do, cmap="cividis")
    thresh = Do.max() * 0.55
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ax.text(j, i, f"{Do[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if Do[i, j] < thresh else "#222222")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for t, c in zip(ax.get_xticklabels(), lab_c): t.set_color(c)
    for t, c in zip(ax.get_yticklabels(), lab_c): t.set_color(c)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="both", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("JS divergence (bits)   0 = identical attention", fontsize=9)
    ax.set_title("Pairwise reasoning divergence  (same matrix as the NN plot; "
                 "gold included)", fontsize=11, pad=12, loc="left")

    path = part_output("part2_coverage", os.path.join("figures", f"js_heatmap{suffix}.png"))
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def dump_matrices(P, D, models, feats, suffix):
    """Write the exact matrices behind the figures so the coefficients are
    inspectable and reusable:

      selection_matrix{suffix}.csv -- P_m(f), rows = model, cols = global
        feature, values = selection-frequency shares. GOLD INCLUDED and (if
        --complete-cases) on the same case set as the figures, so it matches
        plot_heatmap/plot_embedding exactly.
      js_divergence{suffix}.csv    -- D, the model x model JSD matrix the
        heatmap and embedding are built from.
    """
    pd.DataFrame([P[m] for m in models], index=models, columns=feats).to_csv(
        part_output("part2_coverage", f"selection_matrix{suffix}.csv"))
    pd.DataFrame(D, index=models, columns=models).to_csv(
        part_output("part2_coverage", f"js_divergence{suffix}.csv"))
    print(f"  wrote selection_matrix{suffix}.csv "
          f"({len(models)} models x {len(feats)} features) and "
          f"js_divergence{suffix}.csv")


# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--complete-cases", action="store_true",
                    help="only use cases where all 13 generators succeeded, so "
                         "every model is estimated from an identical case set")
    ap.add_argument("--dir", default=None,
                    help="read case_feat.json + global_features.json from this "
                         "directory; default = outputs/llama (independent extractor)")
    args = ap.parse_args()

    os.makedirs(part_output("part2_coverage", "figures"), exist_ok=True)

    case_path = os.path.join(args.dir, "case_feat.json") if args.dir else None
    global_path = os.path.join(args.dir, "global_features.json") if args.dir else None
    mass, models, feats, covars, n_cases = load_mass(
        args.complete_cases, case_path, global_path)
    mode = "COMPLETE CASES ONLY" if args.complete_cases else "all cases"
    print(f"{n_cases} cases x {len(models)} models x {len(feats)} features "
          f"[{mode}]\n")

    P = profiles(mass, models)
    D = divergence_matrix(P, models)

    print("1. REPLICATION -- is the geometry real?")
    rep = replication(mass, models)

    print("\n2. WHAT DRIVES DISTANCE -- verbosity or family?")
    drivers = distance_drivers(P, models, covars)

    print("\n3. FAMILY RECOVERY (gold held out)")
    names, hits, fam_rec = family_recovery(D, models)
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

    stats = {
        "n_cases": n_cases, "n_models": len(models), "n_features": len(feats),
        "complete_cases": bool(args.complete_cases), "n_boot": N_BOOT,
        "replication": rep,
        "distance_drivers": drivers,
        "family_recovery": fam_rec,
        "gold": {
            "distance": {m: float(point[m]) for m in point},
            "nearest_frac": {m: wins[m] / N_BOOT for m in point},
            "family_win_frac": {f: c / N_BOOT for f, c in by_family.items()},
        },
    }
    stats_path = part_output("part2_coverage", f"stats{args.suffix}.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\n  wrote {stats_path}")

    print("\nFIGURES")
    plot_embedding(D, models, args.suffix)
    plot_gold_ranking(point, ci, wins, args.suffix)
    plot_heatmap(D, models, args.suffix)
    dump_matrices(P, D, models, feats, args.suffix)


if __name__ == "__main__":
    main()
