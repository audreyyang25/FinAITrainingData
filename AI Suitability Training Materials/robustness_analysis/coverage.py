import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from load import load_all

FIG_DIR = "/Users/audreyyang/FinAITrainingData/AI Suitability Training Materials/robustness_analysis/figures"

def topic_counts(df):
    return df["topic"].value_counts().sort_values(ascending=True)

def plot_table_counts(df):
    counts = topic_counts(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(counts.index, counts.values)
    ax.set_xlabel("Number of Examples")
    ax.set_ylabel("Topic")
    ax.set_title("Training Examples per Topic")
    fig.savefig(FIG_DIR + "/topic_counts.png", bbox_inches="tight")
    plt.show()

def topic_by_type(df):
    return pd.crosstab(df["topic"], df["content_type"])

if __name__ == "__main__":
    df = load_all()
    print(topic_counts(df))
    print(topic_by_type(df))
    plot_table_counts(df)

