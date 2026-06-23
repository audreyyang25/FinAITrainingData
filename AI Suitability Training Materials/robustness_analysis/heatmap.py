import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from load import load_all

FIG_DIR = "/Users/audreyyang/FinAITrainingData/AI Suitability Training Materials/robustness_analysis/figures"

def plot_heatmap(ct, title, fname, fmt="d", cmap="Blues", figsize=(8, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ct.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index)
    for (i, j), val in np.ndenumerate(ct.values):
        ax.text(j, i, format(val, fmt), ha="center", va="center")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="count")
    fig.savefig(f"{FIG_DIR}/{fname}", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    df = load_all()
    std = df[df["content_type"] == "standard"]
    plot_heatmap(pd.crosstab(std["topic"], std["difficulty"]),
                "Topic x Difficulty (standard)", "topic_difficulty.png")
    plot_heatmap(pd.crosstab(std["topic"], std["compliant"]),
                "Topic x Compliant (standard)", "topic_compliant.png")
    plot_heatmap(pd.crosstab(std["difficulty"], std["compliant"]),
                "Difficulty x Compliant (standard)", "difficulty_compliant.png")
    plot_heatmap(pd.crosstab(df["topic"], df["content_type"]),
                "Topic x Content Type", "topic_content_type.png")
