"""build_concept_spotcheck.py -- cross-family, same-tier (frontier) spot-check
cases, grouped by Llama semantic concept.

For a handful of concepts, find frontier cases where the four families SPLIT on
whether they engaged the concept (named >=1 member feature), and rank them for
reading. Three split types, each answering a different question:

    2v2            substance-vs-style: read a missing model against a naming one
    3v1_lone_miss  blind spot: 3 families + gold engaged it, one didn't
    1v3_lone_name  over/under-attention: one family raised it, three missed

Every case is gold-engaged (the reference also used the concept), and cases are
ranked by gold's importance on the concept -- a relevance proxy, so these are the
cases where the concept is MOST central. That biases toward high-stakes misses;
pass --random for an unbiased sample within each split type (seeded).

Each row carries its concept's grounded regex (built from the concept's member-
feature vocabulary, not guessed) and a ready-to-run read_cmd. Selection is
frequency-based (intensity carried no family signal); all data from outputs/llama.

    python build_concept_spotcheck.py
    python build_concept_spotcheck.py --random --seed 1
"""

import argparse
import csv
import json
import random
from collections import defaultdict

from coarsen_jsd import load, TIER
from config import output_path

FAMS = ["Anthropic", "OpenAI", "Gemini", "Qwen"]

CONCEPTS = [
    "Conflicts of Interest",
    "Regulatory Requirements",
    "Advisor Conduct",
    "Senior and Vulnerable Investors",
]

# Grounded in each concept's member-feature vocabulary (top terms), NOT guessed.
# A reading aid to jump to likely-relevant lines -- read the whole answer anyway.
CONCEPT_REGEX = {
    "Conflicts of Interest":
        "conflict|interest|self-deal|undisclos|compensat|commission|issuer|coerc",
    "Regulatory Requirements":
        "finra|regulat|rule|guidance|requirement|registrat|margin|scrutiny|securities",
    "Advisor Conduct":
        "advisor|churn|excessive|trading|misrepresent|mislead|switch|execution|conduct",
    "Senior and Vulnerable Investors":
        "elder|vulnerab|trusted|incapacit|exploit|diminish|heighten|capacity",
}

# how many of each split type per concept
TAKE = {"2v2": 3, "3v1_lone_miss": 2, "1v3_lone_name": 2}

sh = lambda m: m.split("/")[-1]


def concept_importance(extract_dir="outputs/llama"):
    """cimp[(model, case)][concept_id] = summed importance on that concept."""
    mass, named, total, ncase = load(extract_dir)
    groups = json.load(open(f"{extract_dir}/feature_groups.json"))
    f2c = groups["feature_to_concept"]
    clabel = {c["id"]: c["label"] for c in groups["concepts"]}
    lab2id = {v: k for k, v in clabel.items()}

    cases = json.load(open(f"{extract_dir}/case_feat.json"))
    gm = json.load(open(f"{extract_dir}/global_features.json"))["mapping"]

    cimp = defaultdict(lambda: defaultdict(float))
    cfeat = defaultdict(lambda: defaultdict(set))   # which member features were cited
    for k, c in cases.items():
        for e in c["features"].values():
            i = e["superset_index"]
            if i is None:
                continue
            g = gm.get(c["superset"][i])
            if g in f2c:
                cimp[(e["model"], k)][f2c[g]] += e["importance"]
                cfeat[(e["model"], k)][f2c[g]].add(g)
    return cimp, cfeat, lab2id, list(cases), mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true",
                    help="sample within each split type instead of ranking by gold importance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=output_path("crossfam_concept_spotcheck.csv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cimp, cfeat, lab2id, case_keys, mass = concept_importance()
    models = [m for m in mass if m != "gold"]
    front = {TIER[m][0]: m for m in models if TIER[m][1] == "frontier"}

    rows = []
    for cname in CONCEPTS:
        cid = lab2id[cname]
        rgx = CONCEPT_REGEX[cname]
        buckets = defaultdict(list)
        for k in case_keys:
            namedF = [f for f in FAMS if cimp[(front[f], k)].get(cid, 0) > 0]
            missF = [f for f in FAMS if cimp[(front[f], k)].get(cid, 0) == 0]
            gimp = cimp[("gold", k)].get(cid, 0)
            if gimp <= 0:                       # relevance anchor: reference used it
                continue
            n = len(namedF)
            if n == 2:
                buckets["2v2"].append((gimp, k, namedF, missF, front[missF[0]], "one_of_2_missing"))
            elif n == 3:
                buckets["3v1_lone_miss"].append((gimp, k, namedF, missF, front[missF[0]], "lone_misser"))
            elif n == 1:
                buckets["1v3_lone_name"].append((gimp, k, namedF, missF, front[namedF[0]], "lone_namer"))

        for stype, take in TAKE.items():
            pool = buckets[stype]
            chosen = rng.sample(pool, min(take, len(pool))) if args.random \
                else sorted(pool, key=lambda x: -x[0])[:take]
            for gimp, k, namedF, missF, rm, role in chosen:
                read = sh(rm)
                cmd = f'python read_case.py {k} {read} --feature "{rgx}"'
                # the specific member features the naming models cited -- tells you
                # what the split is actually about, and guards against concept-
                # boundary artifacts (a "miss" that named a sibling concept instead)
                cited = sorted({g for f in namedF
                                for g in cfeat[(front[f], k)].get(cid, ())})
                rows.append(dict(
                    concept=cname, case=k, split_type=stype,
                    gold_engaged="Y" if gimp > 0 else "n",
                    named=";".join(f[:3] for f in namedF),
                    missing=";".join(f[:3] for f in missF),
                    read_model=read, read_role=role,
                    named_features=" || ".join(g[:70] for g in cited),
                    regex=rgx, read_cmd=cmd, coding="", note=""))

    cols = ["concept", "case", "split_type", "gold_engaged", "named", "missing",
            "read_model", "read_role", "named_features", "regex", "read_cmd",
            "coding", "note"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows, "
          f"{'random' if args.random else 'gold-ranked'})")


if __name__ == "__main__":
    main()
