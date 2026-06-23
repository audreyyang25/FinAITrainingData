from load import load_all
from heatmap import plot_heatmap
import pandas as pd

def yn_label(text):
    if not isinstance(text, str):
        return "missing"
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    elif t.startswith("no"):
        return "no"
    else:
        return "other"

if __name__ == "__main__":
    df = load_all()
    adv = df[df["content_type"] == "adversarial"]
    adv["answer_type"] = adv["correct_answer"].apply(yn_label)
    print(adv["answer_type"].value_counts())

    other = adv.loc[adv["answer_type"] == "other", "correct_answer"]
    print(other.str.strip().str.split().str[0].str.lower().value_counts())

    print(adv["adversarial_type"].value_counts())
    
    plot_heatmap(pd.crosstab(adv["topic"], adv["answer_type"]),
                 "Topic x Answer Type (adversarial)", "topic_answer_type.png")
    plot_heatmap(pd.crosstab(adv["topic"], adv["adversarial_type"]),
                 "Topic x Adversarial Type (adversarial)", "topic_adversarial_type.png")
