from load import load_all
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = "/Users/audreyyang/FinAITrainingData/AI Suitability Training Materials/robustness_analysis/figures"

def compliant_balance(df):
    std = df[df["content_type"] == "standard"]
    return std["compliant"].value_counts()

def compliant_by_topic(df, normalize=False):
    std = df[df["content_type"] == "standard"]
    return pd.crosstab(std["topic"], std["compliant"],
                       normalize="index" if normalize else False)

def zero_compliant_topics(df):
    counts = compliant_by_topic(df)
    if True in counts.columns:
        return counts[counts[True] == 0]
    else:
        raise ValueError("All topics are missing compliant examples.")
    
def label_distribution(df, field, subset=None):
    s = df[field] if subset is None else df.loc[subset, field]
    counts = s.value_counts(dropna=False)
    props  = s.value_counts(dropna=False, normalize=True)
    return pd.DataFrame({"count": counts, "proportion": props})

def plot_compliant_by_topic(df):
    counts = compliant_by_topic(df)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=True).index]
    ax = counts.plot(kind="barh", stacked=True)
    ax.set_xlabel("Compliance Counts")
    ax.set_ylabel("Topic")
    ax.set_title("Compliance Distribution by Topic")

    fig = ax.get_figure()
    fig.savefig(FIG_DIR + "/compliance_by_topic.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    df = load_all()
    plot_compliant_by_topic(df)
    print(label_distribution(df, "compliant", subset=(df["content_type"]=="standard")))
    print(zero_compliant_topics(df))