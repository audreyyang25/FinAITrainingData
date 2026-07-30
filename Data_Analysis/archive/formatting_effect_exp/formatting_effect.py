"""Formatting-effect experiment: does rendering the SAME case as a conversation
vs a narrative change how models' answers score?

    delta = score(conversation) - score(narrative)   (+ => conversation scored higher)

Unique to this experiment: pairing narrative (P12) with the verified conversation
rewrites. Generation, judging (with-question only), and the paired-delta stats are
all shared code in the top-level Data_Analysis modules.

PREREQ: modified_data/standard_conversation.jsonl (from rewrite_format.py).
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Data_Analysis/ on sys.path
from config import GEN_SUITE, DATA_DIR
from generations import generate
from judge import judge_answers
from paired_stats import paired_delta

HERE = os.path.dirname(os.path.abspath(__file__))   # Data_Analysis/formatting_effect_exp
DA_ROOT = os.path.dirname(HERE)                      # Data_Analysis
REPO = os.path.dirname(DA_ROOT)                      # repo root (FinAITrainingData)
OUT_DIR = os.path.join(DA_ROOT, "results", "convo_formatting_effect")
os.makedirs(OUT_DIR, exist_ok=True)
GEN_PATH = os.path.join(OUT_DIR, "generations.jsonl")
JUD_PATH = os.path.join(OUT_DIR, "judgments.jsonl")

STD_SRC = os.path.join(DATA_DIR, "suitability_only_P12.json")
CONVO_SRC = os.path.join(REPO, "modified_data", "standard_conversation.jsonl")

KEY_FIELDS = ("format", "id")


def _prompt(fact_pattern, question):
    return f"{fact_pattern}\n\nQuestion: {question}"


def load_formats():
    """{format: {id: {"prompt", "truth"}}}, restricted to ids present in BOTH."""
    narrative = {}
    for r in json.load(open(STD_SRC)):
        narrative[r["id"]] = {"prompt": _prompt(r["fact_pattern"], r["question"]),
                              "truth": r["answer"]}
    conversation = {}
    for line in open(CONVO_SRC):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("_verified"):  # skip fact-drifted rewrites; add them back once fixed
            continue
        conversation[r["id"]] = {"prompt": _prompt(r["fact_pattern"], r["question"]),
                                 "truth": r["answer"]}
    paired = set(narrative) & set(conversation)
    return {"narrative": {i: narrative[i] for i in paired},
            "conversation": {i: conversation[i] for i in paired}}


def _items(formats):
    for fmt, recs in formats.items():
        for rid, d in recs.items():
            yield {"format": fmt, "id": rid}, d["prompt"]


def _lookup(formats):
    def fn(g):
        d = formats[g["format"]][g["id"]]
        return d["prompt"], d["truth"]
    return fn


def aggregate():
    paired_delta(JUD_PATH, OUT_DIR, arm_col="format", hi="conversation", lo="narrative",
                 case_cols=["id"], prefix="formatting", effect_name="Formatting effect")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "agg":
        aggregate()  # stats + figures only, from existing judgments (no API calls)
    else:
        formats = load_formats()
        print(f"paired records: {len(formats['narrative'])}")
        generate(_items(formats), GEN_PATH, key_fields=KEY_FIELDS, suite=GEN_SUITE)
        judge_answers(GEN_PATH, JUD_PATH, lookup=_lookup(formats), key_fields=KEY_FIELDS)
        aggregate()
