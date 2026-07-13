"""visualize_distributions.py -- three views of the model reasoning-attention data.

  1. model-similarity clustermap  (from js_divergence.csv)  -- who reasons alike
  2. attention heatmap            (from importance_matrix.csv) -- most divisive features
  3. signature heatmap            (deviation from the mean)  -- each model's fingerprint

Also prints within-family vs across-family mean JS divergence.

Reusable across dataset sizes:
  python visualize_distributions.py --suffix _250
  python visualize_distributions.py --matrix outputs/importance_matrix_500.csv \\
         --js outputs/js_divergence_500.csv --suffix _500

Color: perceptually-uniform, CVD-safe maps -- 'cividis' for magnitude,
'RdBu_r' centered at 0 for the diverging signature. Family tags use the
Okabe-Ito colorblind-safe categorical palette.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


# Okabe-Ito (colorblind-safe), fixed order by family.
FAMILY_COLORS = {
    "anthropic": "#E69F00",   # orange
    "openai":    "#0072B2",   # blue
    "google":    "#009E73",   # green
    "qwen":      "#D55E00",   # vermillion
}

TOP_N = 25          # features shown in the attention / signature heatmaps
LABEL_CHARS = 46    # feature-label truncation


def family_of(model):
    return model.split("/")[0]


def short(feature):
    f = feature.strip()
    return f if len(f) <= LABEL_CHARS else f[: LABEL_CHARS - 1] + "…"


def order_by_family(models):
    fam_rank = {f: i for i, f in enumerate(FAMILY_COLORS)}
    return sorted(models, key=lambda m: (fam_rank.get(family_of(m), 9), m))


def plot_similarity(js, outpath):
    """Fig 1: clustered heatmap + dendrogram of pairwise JS divergence."""
    dist = squareform(js.values, checks=False)
    L = linkage(dist, method="average")

    row_colors = pd.Series(
        {m: FAMILY_COLORS.get(family_of(m), "#999999") for m in js.index},
        name="family",
    )

    g = sns.clustermap(
        js,
        row_linkage=L,
        col_linkage=L,
        cmap="cividis",
        row_colors=row_colors,
        col_colors=row_colors,
        figsize=(9.5, 9.5),
        linewidths=0.4,
        cbar_kws={"label": "JS divergence  (0 = identical attention)"},
        xticklabels=[m.split("/")[-1] for m in js.columns],
        yticklabels=[m.split("/")[-1] for m in js.index],
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.fig.suptitle("Model reasoning similarity  (clustered by JS divergence)",
                   y=1.02, fontsize=13, fontweight="bold")
    g.ax_heatmap.legend(
        handles=[Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()],
        title="family", loc="upper left", bbox_to_anchor=(1.25, 1.15),
        frameon=False, fontsize=9,
    )
    g.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def _heatmap(ax, data, models, feats, cmap, norm, cbar_label):
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([short(f) for f in feats], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.split("/")[-1] for m in models], fontsize=8.5)
    # color the y tick labels by family
    for tick, m in zip(ax.get_yticklabels(), models):
        tick.set_color(FAMILY_COLORS.get(family_of(m), "#333333"))
    ax.set_xticks(np.arange(-.5, len(feats), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="both", length=0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label(cbar_label, fontsize=9)
    return im


def plot_attention(matrix, outpath):
    """Fig 2: top features by cross-model variance; raw P_m(f)."""
    var = matrix.var(axis=0).sort_values(ascending=False)
    feats = list(var.head(TOP_N).index)
    models = order_by_family(list(matrix.index))
    data = matrix.loc[models, feats].values

    fig, ax = plt.subplots(figsize=(13, 6.5))
    _heatmap(ax, data, models, feats, "cividis",
             matplotlib.colors.Normalize(vmin=0, vmax=data.max()),
             "P_m(f)  attention share")
    ax.set_title("Where models spend reasoning attention  "
                 "(top 25 most-divisive features)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_signature(matrix, outpath):
    """Fig 3: deviation from the cross-model mean (diverging)."""
    dev = matrix.subtract(matrix.mean(axis=0), axis=1)
    feats = list(dev.abs().max(axis=0).sort_values(ascending=False).head(TOP_N).index)
    models = order_by_family(list(matrix.index))
    data = dev.loc[models, feats].values
    lim = np.abs(data).max()

    fig, ax = plt.subplots(figsize=(13, 6.5))
    _heatmap(ax, data, models, feats, "RdBu_r",
             TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
             "P_m(f) − mean   (red = over-weights)")
    ax.set_title("Each model's reasoning signature  "
                 "(deviation from the average model)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def family_divergence(js):
    within, across = [], []
    models = list(js.index)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            d = js.iloc[i, j]
            (within if family_of(models[i]) == family_of(models[j]) else across).append(d)
    print("\n=== reasoning divergence ===")
    print(f"  mean WITHIN-family JS : {np.mean(within):.3f}")
    print(f"  mean ACROSS-family JS : {np.mean(across):.3f}")
    verdict = ("families share a house style"
               if np.mean(within) < np.mean(across)
               else "no family effect")
    print(f"  -> {verdict} (within < across means style is inherited from the family)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="Reads outputs/*<suffix>.csv and writes figures<suffix>.png")
    ap.add_argument("--matrix", default=None, help="override matrix path")
    ap.add_argument("--js", default=None, help="override js path")
    ap.add_argument("--outdir", default="outputs/figures")
    args = ap.parse_args()

    s = args.suffix

    # By default, --suffix picks both the input tables and the output names.
    matrix_path = args.matrix or f"outputs/importance_matrix{s}.csv"
    js_path = args.js or f"outputs/js_divergence{s}.csv"

    # resolve relative to this file so cwd doesn't matter
    here = os.path.dirname(os.path.abspath(__file__))
    resolve = lambda p: p if os.path.isabs(p) else os.path.join(here, p)
    outdir = resolve(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    matrix = pd.read_csv(resolve(matrix_path), index_col=0)
    js = pd.read_csv(resolve(js_path), index_col=0)

    plot_similarity(js, os.path.join(outdir, f"model_similarity{s}.png"))
    plot_attention(matrix, os.path.join(outdir, f"attention_heatmap{s}.png"))
    plot_signature(matrix, os.path.join(outdir, f"signature_heatmap{s}.png"))
    family_divergence(js)
    print(f"\nWrote 3 figures to {outdir}/ (suffix '{s or 'none'}')")


if __name__ == "__main__":
    main()
