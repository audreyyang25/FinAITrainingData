"""group_features_llama.py -- semantic grouping of the global feature vocabulary,
as a check on the TF-IDF grouping used by coarsen_jsd.py.

TF-IDF merges LEXICALLY similar feature strings. That biases the coarsening test
toward finding family structure: two families phrasing one concept differently
("Reg BI's Care Obligation requires..." vs "the advisor must have a reasonable
basis...") share few words and survive as separate groups even at small K. A
semantic grouping collapses those, so if family separation still holds here, it
is not a vocabulary artifact.

The extractor model (Llama 3.3-70B) does the grouping, NOT the canonicalizer
(Opus). config.py flags the canonicalizer as carrying "a mild Anthropic lean" --
using it for a test designed to detect Anthropic-idiomatic vocabulary splitting
would be circular.

Two levels, so coarsen_jsd.py gets two granularities to check:
    concepts (~20)  -- e.g. "duty standard", "client profile", "product risk"
    domains  (~6)   -- concepts grouped again

Writes outputs/llama/feature_groups.json. Cached: rerunning is free unless you
pass --force.

    python group_features_llama.py
    python group_features_llama.py --n-concepts 20 --n-domains 6 --force
"""

import argparse
import json
import os

from extract_gold_features import parse_json
from config import EXTRACTORS, output_path
from llm import call_llm

GROUPER = EXTRACTORS[0]["extractor"]          # llama, same model as the extraction

TAXONOMY_SYSTEM = """You are organizing a vocabulary of reasoning features used in
US securities suitability / best-interest analysis (FINRA Rule 2111, Reg BI).

You will be given the full list of features. Propose exactly {n} concept
categories that partition them by SUBJECT MATTER -- what the feature is about,
not how it is phrased. Categories must be mutually exclusive, collectively
cover the list, and be roughly balanced in size.

Return JSON only:
{{"concepts": [{{"id": 0, "label": "short label", "gloss": "one line"}}, ...]}}"""

ASSIGN_SYSTEM = """Assign each feature to exactly one concept id.

Judge by subject matter, not wording. Two differently-phrased features about the
same underlying legal concept MUST get the same id.

CONCEPTS:
{concepts}

Return JSON only: {{"assignments": [{{"feature": "<verbatim>", "id": <int>}}, ...]}}
Every feature you were given must appear exactly once."""

DOMAIN_SYSTEM = """Group these concept categories into exactly {n} broader domains.

Return JSON only:
{{"domains": [{{"id": 0, "label": "short label", "concept_ids": [1, 4, 7]}}, ...]}}
Every concept id must appear in exactly one domain."""


def vocabulary(extract_dir):
    with open(os.path.join(extract_dir, "global_features.json")) as fh:
        mapping = json.load(fh)["mapping"]
    return sorted({v for v in mapping.values() if v})


def chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/llama")
    ap.add_argument("--n-concepts", type=int, default=20)
    ap.add_argument("--n-domains", type=int, default=6)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    out = os.path.join(args.dir, "feature_groups.json")
    if os.path.exists(out) and not args.force:
        print(f"{out} exists -- use --force to regenerate")
        return

    feats = vocabulary(args.dir)
    print(f"{len(feats)} global features -> {args.n_concepts} concepts "
          f"-> {args.n_domains} domains   (grouper: {GROUPER})")

    # 1. induce the concept taxonomy from the whole vocabulary at once
    raw = call_llm(model=GROUPER, max_tokens=4000, temperature=0.0,
                   system=TAXONOMY_SYSTEM.format(n=args.n_concepts),
                   user="\n".join(f"- {f}" for f in feats))
    concepts = parse_json(raw)["concepts"]
    print(f"  taxonomy: {len(concepts)} concepts")
    for c in concepts:
        print(f"    {c['id']:>2}  {c['label']}")

    listing = "\n".join(f"{c['id']}: {c['label']} -- {c['gloss']}" for c in concepts)
    valid = {c["id"] for c in concepts}

    # 2. assign every feature, in batches
    assign, missing = {}, []
    for i, batch in enumerate(chunk(feats, args.batch), 1):
        raw = call_llm(model=GROUPER, max_tokens=4000, temperature=0.0,
                       system=ASSIGN_SYSTEM.format(concepts=listing),
                       user="\n".join(f"- {f}" for f in batch))
        got = {a["feature"]: a["id"] for a in parse_json(raw)["assignments"]
               if a.get("id") in valid}
        for f in batch:
            if f in got:
                assign[f] = got[f]
            else:
                missing.append(f)
        print(f"  batch {i}: {len(got)}/{len(batch)} assigned")

    if missing:
        # Unassigned features get their own bucket rather than being dropped --
        # dropping them would silently shrink the distributions.
        spare = max(valid) + 1
        for f in missing:
            assign[f] = spare
        concepts.append({"id": spare, "label": "UNASSIGNED", "gloss": "grouper failed"})
        print(f"  {len(missing)} features unassigned -> bucket {spare}")

    # 3. concepts -> domains
    raw = call_llm(model=GROUPER, max_tokens=2000, temperature=0.0,
                   system=DOMAIN_SYSTEM.format(n=args.n_domains),
                   user=listing)
    domains = parse_json(raw)["domains"]
    c2d = {cid: d["id"] for d in domains for cid in d["concept_ids"]}
    for c in concepts:
        c2d.setdefault(c["id"], max(c2d.values(), default=0) + 1)

    payload = {
        "grouper": GROUPER,
        "concepts": concepts,
        "domains": domains,
        "feature_to_concept": assign,
        "feature_to_domain": {f: c2d[cid] for f, cid in assign.items()},
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    n_c = len(set(assign.values()))
    n_d = len(set(payload["feature_to_domain"].values()))
    print(f"\nwrote {out}  ({n_c} concepts, {n_d} domains in use)")


if __name__ == "__main__":
    main()
