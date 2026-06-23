import pandas as pd
import matplotlib.pyplot as plt
from load import load_all

FIG_DIR = "/Users/audreyyang/FinAITrainingData/AI Suitability Training Materials/robustness_analysis/figures"

def difficulty_by_type(df):
    d = df[df["difficulty"].notna()].copy()
    order = ["easy", "medium", "hard", "expert"]
    d["difficulty"] = pd.Categorical(d["difficulty"], categories=order, ordered=True)
    return pd.crosstab(d["difficulty"], d["content_type"])

def plot_difficulty(df):
    counts = difficulty_by_type(df)
    ax = counts.plot(kind="bar")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Number of Examples")
    ax.set_title("Difficulty Distribution by Content Type")
    
    for container in ax.containers:
        ax.bar_label(container, padding=3)

    fig = ax.get_figure()
    fig.savefig(FIG_DIR + "/difficulty_by_type.png", bbox_inches="tight")

    plt.show()

def completeness(df, field):
    present = df.groupby("content_type")[field].apply(lambda s: s.notna().sum())
    total = df.groupby("content_type")[field].size()
    out = pd.DataFrame({
        "labeled":      present,
        "unlabeled":    total - present,
        "total":        total,
    })
    out["frac_labeled"] = out["labeled"] / out["total"]
    return out

if __name__ == "__main__":
    df = load_all()
    plot_difficulty(df)
    print(completeness(df, "difficulty"))