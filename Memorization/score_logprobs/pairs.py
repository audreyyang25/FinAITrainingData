#!/usr/bin/env python3
"""(prefix, suffix) pairs for teacher-forced scoring, from the existing QA CSVs.

No new data: `prompt` is already the raw excerpt and `answer` the true
continuation, aligned per (case_id, qid). Both controls_qa.csv and
court_opinions_qa_v2.csv work unchanged.

Deliberately `prompt`, NOT `prompt_normalized`. Normalisation exists for fuzzy
string scoring in qa/score_answers.py; here the model must see the text as it
actually appears, because the quantity being measured is the probability the
model assigns to real text.

No torch import -- this module is importable on a laptop for inspection.
"""
from __future__ import annotations
import csv, os, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")
CONTROLS = os.path.join(DS, "controls_qa.csv")
COURT = os.path.join(DS, "court_opinions_qa_v2.csv")

# Same admissibility rule as every other figure in the repo: tier D only, and
# Q03 is excluded because the identification block gives its answer away.
SCORED_TIERS = {"D"}
EXCLUDE_QIDS = {"Q03"}


def load_pairs(qa_csv, limit=None, qids=None):
    """-> list of dicts with prefix/suffix and the identifiers to join on."""
    out = []
    with open(qa_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("found") != "1":
                continue
            if r.get("tier", "D") not in SCORED_TIERS or r["qid"] in EXCLUDE_QIDS:
                continue
            if qids and r["qid"] not in qids:
                continue
            prefix = (r.get("prompt") or "").strip()
            suffix = (r.get("answer") or "").strip()
            if not prefix or not suffix:
                continue
            out.append({
                "case_id": r["case_id"],
                "qid": r["qid"],
                # `work` is the control-set grouping (gatsby / gatsby_shuf /
                # constitution); for court rows it is just the case id stem.
                "work": r["case_id"].rsplit("_ch", 1)[0],
                "arm": r.get("arm", ""),
                "date_filed": r.get("date_filed", ""),
                "jurisdiction": r.get("jurisdiction", ""),
                "prefix": prefix,
                "suffix": suffix,
            })
    if limit:
        out = out[:limit]
    return out


def corpus_name(qa_csv):
    """Tag used in output filenames, matching score_answers.py's convention."""
    return os.path.basename(qa_csv).replace("_qa.csv", "").replace(".csv", "")


if __name__ == "__main__":
    import collections
    for path in (CONTROLS, COURT):
        if not os.path.exists(path):
            print(f"{os.path.basename(path):32s} MISSING")
            continue
        p = load_pairs(path)
        works = collections.Counter(x["work"] for x in p)
        pw = sum(len(x["prefix"].split()) for x in p) / max(1, len(p))
        sw = sum(len(x["suffix"].split()) for x in p) / max(1, len(p))
        print(f"{corpus_name(path):26s} {len(p):5d} pairs   "
              f"prefix~{pw:5.1f}w  suffix~{sw:5.1f}w")
        if len(works) <= 6:
            print(f"{'':26s} {dict(works.most_common())}")
