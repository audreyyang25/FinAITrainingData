"""Gender-effect experiment: does the CLIENT's gender change how models' answers
score, holding the case (and demographic) constant?

    delta = score(male) - score(female)     (+ => male variant scored higher)

Unique to this experiment: pairing the name-swapped male/female variants. Generation,
judging (with-question only), and the paired-delta stats are shared code in the
top-level Data_Analysis modules.

PREREQ: modified_data/name_variants.jsonl (from rewrite_names.py).
"""

import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Data_Analysis/ on sys.path
from config import GEN_SUITE, ADAPTERS
from generations import generate
from judge import judge_answers
from paired_stats import paired_delta

HERE = os.path.dirname(os.path.abspath(__file__))   # Data_Analysis/gender_effect_exp
DA_ROOT = os.path.dirname(HERE)                      # Data_Analysis
REPO = os.path.dirname(DA_ROOT)                      # repo root (FinAITrainingData)
OUT_DIR = os.path.join(DA_ROOT, "results", "gender_effect")
os.makedirs(OUT_DIR, exist_ok=True)
GEN_PATH = os.path.join(OUT_DIR, "generations.jsonl")
JUD_PATH = os.path.join(OUT_DIR, "judgments.jsonl")
VARIANTS = os.path.join(REPO, "modified_data", "name_variants.jsonl")

KEY_FIELDS = ("gender", "type", "base_id", "demographic")


def load_variants():
    """{gender: {(type, base_id, demo): {"prompt","truth"}}}, paired across genders."""
    by_gender = defaultdict(dict)
    for line in open(VARIANTS):
        line = line.strip()
        if not line:
            continue
        v = json.loads(line)
        adapter = ADAPTERS[v["type"]]
        rec = v["record"]
        key = (v["type"], v["base_id"], v["demographic"])
        by_gender[v["gender"]][key] = {"prompt": adapter["prompt"](rec),
                                       "truth": adapter["truth"](rec)}
    male, female = by_gender.get("male", {}), by_gender.get("female", {})
    paired = set(male) & set(female)
    return {"male": {k: male[k] for k in paired},
            "female": {k: female[k] for k in paired}}


def _items(variants):
    for gender, recs in variants.items():
        for (t, bid, demo), d in recs.items():
            yield {"gender": gender, "type": t, "base_id": bid, "demographic": demo}, d["prompt"]


def _lookup(variants):
    def fn(g):
        d = variants[g["gender"]][(g["type"], g["base_id"], g["demographic"])]
        return d["prompt"], d["truth"]
    return fn


def aggregate():
    paired_delta(JUD_PATH, OUT_DIR, arm_col="gender", hi="male", lo="female",
                 case_cols=["type", "base_id", "demographic"], prefix="gender",
                 effect_name="Gender effect",
                 extra_group_tables=[("by_type", "type", "Gender effect by dataset type")])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "agg":
        aggregate()  # stats + figures only, from existing judgments (no API calls)
    else:
        variants = load_variants()
        print(f"paired (male & female) cases: {len(variants['male'])}")
        generate(_items(variants), GEN_PATH, key_fields=KEY_FIELDS, suite=GEN_SUITE)
        judge_answers(GEN_PATH, JUD_PATH, lookup=_lookup(variants), key_fields=KEY_FIELDS)
        aggregate()
