"""Shared figure helpers: render result tables/matrices as PNGs (headless).

Each helper takes a full output `path`. Importing matplotlib is deferred to call
time so a machine without it still runs everything except the figures (callers
wrap these in try/except ImportError and fall back to CSVs).
"""

import os


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_table_fig(df, title, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(df.columns) + 2), 0.5 * len(df) + 1.4))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=12)
    tbl = ax.table(cellText=df.round(4).astype(str).values,
                   colLabels=list(df.columns), rowLabels=[str(i) for i in df.index],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  figure: {os.path.basename(path)}")


def save_bar(series, title, path, ylabel="mean delta"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in series]
    series.plot(kind="bar", ax=ax, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  figure: {os.path.basename(path)}")


def save_score_heatmap(pivot, title, path, cbar_label="mean score"):
    """Heatmap for raw scores (sequential, data-range colormap; red=low/worst)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.3 * len(pivot.columns) + 3, 0.6 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, cmap="RdYlGn",
                   vmin=pivot.values.min(), vmax=pivot.values.max(), aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  figure: {os.path.basename(path)}")


def save_heatmap(pivot, title, path, cbar_label="delta"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 3, 0.6 * len(pivot.index) + 2))
    vmax = max(0.001, abs(pivot.values).max())
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  figure: {os.path.basename(path)}")
