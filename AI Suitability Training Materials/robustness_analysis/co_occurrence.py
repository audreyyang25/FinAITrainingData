import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy

from load import load_all

FIG_DIR = "/Users/audreyyang/FinAITrainingData/AI Suitability Training Materials/robustness_analysis/figures"

TEXT_FIELD = "fact_pattern"

PROFESSIONS = [
    "schoolteacher", "teacher", "professor", "nurse", "physician", "doctor",
    "dentist", "pharmacist", "engineer", "accountant", "lawyer", "attorney",
    "executive", "manager", "business owner", "entrepreneur", "contractor",
    "electrician", "plumber", "mechanic", "farmer", "realtor", "salesperson",
    "police officer", "firefighter", "veteran", "military", "pilot",
    "scientist", "consultant", "banker", "secretary", "clerk",
    "widow", "widower", "retiree", "retired",
]

def _build_name_extractor():
    """Return a function text -> list[str] of PERSON names."""
    nlp = spacy.load("en_core_web_sm")

    def extract(text):
        doc = nlp(text)
        return [e.text for e in doc.ents if e.label_ == "PERSON"]

    print("[names] using spaCy en_core_web_sm")
    return extract


def first_name_of(full_name):
    """'Margaret Olin' -> 'Margaret'."""
    return full_name.split()[0] if full_name else None


# Regex extractors for structured entities like age, dollar earnings, and professions
AGE_RE = re.compile(r"\bage[d]?\s+(\d{1,3})\b|\b(\d{1,3})-year-old\b", re.I)
DOLLAR_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")

def extract_ages(text):
    out = []
    for m in AGE_RE.finditer(text):
        val = m.group(1) or m.group(2)
        if val:
            out.append(int(val))
    return out

def extract_dollars(text):
    return DOLLAR_RE.findall(text)

def extract_professions(text):
    t = text.lower()
    return [p for p in PROFESSIONS if p in t]


# Build a per-record entity table
def build_entity_table(df):
    extract_names = _build_name_extractor()
    rows = []
    for _, r in df.iterrows():
        text = r.get(TEXT_FIELD)
        if not isinstance(text, str) or not text.strip():
            continue
        names = extract_names(text)
        rows.append({
            "topic": r.get("topic"),
            "content_type": r.get("content_type"),
            "primary_name": first_name_of(names[0]) if names else None,
            "all_names": names,
            "professions": extract_professions(text),
            "ages": extract_ages(text),
            "dollars": extract_dollars(text),
        })
    return pd.DataFrame(rows)

# Diversity metrics
def diversity_report(ent):
    print("\n=== NAME DIVERSITY (primary client name) ===")
    names = ent["primary_name"].dropna()
    print(f"records with a name: {len(names)}")
    print(f"unique names: {names.nunique()}  "
          f"(type-token ratio = {names.nunique() / max(len(names), 1):.2f})")
    print("top 20 names:")
    print(names.value_counts().head(20).to_string())

    print("\n=== AGE DISTRIBUTION ===")
    ages = ent["ages"].explode().dropna()
    if len(ages):
        print(ages.describe().to_string())
        print("most common ages:")
        print(ages.value_counts().head(10).to_string())

    print("\n=== MOST REPEATED DOLLAR AMOUNTS ===")
    dollars = ent["dollars"].explode().dropna()
    if len(dollars):
        print(dollars.value_counts().head(15).to_string())

    print("\n=== PROFESSION FREQUENCY ===")
    profs = ent["professions"].explode().dropna()
    if len(profs):
        print(profs.value_counts().to_string())


# Profession x Name co-occurrence
def cooccurrence(ent, top_names=25):
    """Explode (profession, name) pairs and cross-tab them."""
    pairs = ent[["professions", "primary_name"]].copy()
    pairs = pairs[pairs["primary_name"].notna()]
    pairs = pairs.explode("professions").dropna(subset=["professions"])
    ct = pd.crosstab(pairs["professions"], pairs["primary_name"])

    # keep only the most frequent names so the heatmap is legible
    keep = ct.sum(axis=0).sort_values(ascending=False).head(top_names).index
    ct = ct[keep]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]
    return ct


def concentration_report(ent):
    """For each profession, how dominant is its single most common name?"""
    pairs = ent[["professions", "primary_name"]].copy()
    pairs = pairs[pairs["primary_name"].notna()]
    pairs = pairs.explode("professions").dropna(subset=["professions"])

    rows = []
    for prof, grp in pairs.groupby("professions"):
        vc = grp["primary_name"].value_counts()
        rows.append({
            "profession": prof,
            "n_examples": len(grp),
            "unique_names": grp["primary_name"].nunique(),
            "top_name": vc.index[0],
            "top_name_count": int(vc.iloc[0]),
            "top_name_share": vc.iloc[0] / len(grp),
        })
    return pd.DataFrame(rows).sort_values("top_name_share", ascending=False)


def plot_cooccurrence(ct, fname="profession_name_cooccurrence.png"):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(ct.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index)
    for (i, j), val in np.ndenumerate(ct.values):
        if val:
            ax.text(j, i, val, ha="center", va="center", fontsize=7)
    ax.set_title("Profession × client first-name (counts)")
    fig.colorbar(im, ax=ax, label="count")
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = load_all()
    ent = build_entity_table(df)

    diversity_report(ent)

    print("\n=== PROFESSION NAME CONCENTRATION "
          "(high top_name_share = repetitive) ===")
    conc = concentration_report(ent)
    print(conc.to_string(index=False))

    ct = cooccurrence(ent)
    plot_cooccurrence(ct)
    print(f"\nSaved heatmap -> {os.path.join(FIG_DIR, 'profession_name_cooccurrence.png')}")

    # Persist the raw entity table for any follow-up analysis.
    out_csv = os.path.join(os.path.dirname(__file__), "entities.csv")
    ent.to_csv(out_csv, index=False)
    print(f"Saved entity table -> {out_csv}")
