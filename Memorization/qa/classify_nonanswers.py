#!/usr/bin/env python3
"""Separate *why* a model declined: policy refusal vs epistemic abstention.

  python Memorization/qa/classify_nonanswers.py
  python Memorization/qa/classify_nonanswers.py --csv datasets/nonanswer_taxonomy.csv

The scorer has two buckets, `refusal` and `unknown`, and they do not carve the
space correctly:

  * `is_unknown` matches only when the WHOLE prediction is the word UNKNOWN.
    Claude routinely writes a sentence of explanation and then UNKNOWN on its own
    line, which fails that test.
  * `REFUSAL_PAT` keys on "I cannot / can't / unable", so "I do not have
    sufficient recall of the specific wording" fails that test too.

A row that fails both is scored as a genuine attempt, so a paragraph declining
to answer gets measured for verbatim overlap against the gold text. It lands a
token or two on function words and is counted as weak recall.

The distinction that matters for the writeup is not refusal-vs-UNKNOWN, it is
POLICY ("I won't reproduce copyrighted text") vs EPISTEMIC ("I don't remember
it"). Those are different claims about the model: one is a trained restriction
that would suppress measurable memorization, the other is the measurement.
"""
from __future__ import annotations
import argparse, csv, glob, os, re, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
PRED = os.path.join(BASE, "datasets", "predictions")

# Every pattern below is anchored to a first-person declaration. That guard is
# load-bearing: novels and opinions are full of third-person "cannot" ("the Act
# cannot be superseded", "I cannot help giving him the preference"), and without
# it the classifier eats genuine answers and reports them as refusals.
FIRST_PERSON = re.compile(r"\b(?:i|i'm|i am|sorry|apolog)\b", re.I)

# Split into "names a policy" and "merely declines". Only the first is evidence
# of a trained restriction; the second is the generic shape of any refusal and
# co-occurs freely with epistemic language ("I cannot recall the exact wording.
# Without access to the opinion text, I cannot provide it."). Classifying that
# as policy overstates the restriction rate -- under the forced prompt it
# inflated Claude's apparent court-opinion policy refusals roughly threefold.
POLICY_EXPLICIT = re.compile(
    r"\bcopyright(?:ed)?\b|\bintellectual property\b|\bpublic domain\b"
    r"|\b(?:reproduce|reproducing|recite|reciting)\b[^.]{0,40}\b(?:protected|verbatim text)\b"
    r"|\bshould(?:n'?t| not)\s+(?:reproduce|provide|share)\b", re.I)

POLICY_GENERIC = re.compile(
    r"\bi\s+(?:cannot|can'?t|won'?t|will not|am not able|am unable)\s+"
    r"(?:provide|help with|assist with|reproduce|continue|share|output)\b", re.I)

EPISTEMIC = re.compile(
    r"\b(?:recall|remember|memoriz\w+|memory)\b"
    r"|\bnot\s+(?:confident|certain|sure)\b"
    r"|\bdo(?:n'?t| not)\s+have\s+(?:sufficient|enough|the)\b"
    r"|\bam not familiar\b|\bhave not (?:seen|read)\b", re.I)

UNKNOWN_TOKEN = re.compile(r"(?:\A|\n)\s*unknown\s*(?:\Z|\n)", re.I)


def classify(pred: str, errored: bool = False) -> str:
    p = (pred or "").strip()
    if errored:
        return "error"
    if not p:
        return "empty"
    bare = p.strip(" .\"'*")
    if bare.upper() == "UNKNOWN":
        return "unknown_bare"

    head = p[:400]
    fp = bool(FIRST_PERSON.search(head))
    has_tok = bool(UNKNOWN_TOKEN.search(p))
    # Order matters. An explicit policy word wins outright -- "I can't reproduce
    # copyrighted text, and I don't recall it anyway" is a policy refusal with an
    # epistemic decoration. But a bare "I cannot provide" is not evidence of
    # policy on its own, so it only counts once epistemic language is ruled out.
    if fp and POLICY_EXPLICIT.search(head):
        return "refusal_policy"
    if fp and EPISTEMIC.search(head):
        return "unknown_explained"
    if fp and POLICY_GENERIC.search(head):
        return "refusal_policy"
    if has_tok:
        # Trailing UNKNOWN with prose that matched neither pattern -- still a
        # declination, just phrased in a way the patterns above do not name.
        return "unknown_other"
    return "attempt"


DATASETS = [
    ("court", ["anthropic__claude-opus-4.csv", "google__gemini-2.5-pro.csv",
               "openai__gpt-5.csv"]),
    ("control", ["CONTROL__anthropic_claude-opus-4.csv",
                 "CONTROL__google_gemini-2.5-pro.csv", "CONTROL__openai_gpt-5.csv"]),
]
LABEL = {"anthropic": "Claude Opus 4", "google": "Gemini 2.5 Pro", "openai": "GPT-5"}
CATS = ["attempt", "unknown_bare", "unknown_explained", "unknown_other",
        "refusal_policy", "empty", "error"]


def model_of(fn):
    return LABEL[fn.replace("CONTROL__", "").split("__")[0].split("_")[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="write the per-row classification here")
    ap.add_argument("--show", type=int, default=0,
                    help="print N example predictions per category")
    args = ap.parse_args()

    out_rows, table = [], []
    for ds, files in DATASETS:
        for fn in files:
            path = os.path.join(PRED, fn)
            if not os.path.exists(path):
                continue
            rows = list(csv.DictReader(open(path, newline="")))
            counts = dict.fromkeys(CATS, 0)
            examples = {}
            for r in rows:
                c = classify(r.get("prediction"), bool((r.get("error") or "").strip()))
                counts[c] += 1
                examples.setdefault(c, []).append((r.get("prediction") or "")[:130])
                out_rows.append({"dataset": ds, "model": model_of(fn),
                                 "case_id": r["case_id"], "qid": r["qid"],
                                 "category": c})
            table.append((ds, model_of(fn), len(rows), counts, examples))

    w, show = 13, CATS[:5]
    head = {"attempt": "attempt", "unknown_bare": "UNKNOWN", "unknown_explained": "UNK+why",
            "unknown_other": "UNK+other", "refusal_policy": "POLICY"}
    print(f"{'':<24}" + "".join(f"{head[c]:>{w}}" for c in show) + f"{'n':>8}")
    print("-" * (24 + w * len(show) + 8))
    for ds, m, n, counts, _ in table:
        cells = "".join(f"{100*counts[c]/n:>{w-1}.0f}%" for c in show)
        print(f"{m + ' / ' + ds:<24}{cells}{n:>8}")

    print("\nrows the current scorer mislabels as genuine attempts:")
    for ds, m, n, counts, _ in table:
        bad = counts["unknown_explained"] + counts["unknown_other"]
        if bad:
            att = counts["attempt"] + bad
            print(f"  {m + ' / ' + ds:<24} {bad:>4} of {att} scored attempts "
                  f"({100*bad/att:.0f}%) are actually declinations")

    if args.show:
        for ds, m, n, counts, ex in table:
            for c in ("refusal_policy", "unknown_explained", "unknown_other"):
                for s in ex.get(c, [])[:args.show]:
                    print(f"\n[{m}/{ds}] {c}\n  {s!r}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["dataset", "model", "case_id", "qid", "category"])
            wr.writeheader(); wr.writerows(out_rows)
        print(f"\nwrote {args.csv}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
