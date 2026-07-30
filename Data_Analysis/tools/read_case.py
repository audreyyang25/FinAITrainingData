"""read_case.py -- read one model's full answer to one case, with the case in
front of you and the extracted features beside it.

Built for the structure spot-check: when a model scores 0 on a feature, you need
to decide whether the concept is ABSENT, PRESENT-AS-FRAME (invoked in passing,
not itemized), or PRESENT-AS-ELEMENT (itemized but the extractor missed it).
That judgment needs the fact pattern, the full answer, and the extraction --
which otherwise live in three different files.

  python read_case.py standard:30 haiku
  python read_case.py standard:30 haiku --feature fiduciary   # highlight matches
  python read_case.py standard:30 gold                        # the gold answer
"""

import argparse
import json
import os
import re

from shared.config import load_dataset, GENERATIONS_JSON
from shared.utils import load_json


def resolve(candidates, query, what):
    """Unique substring match, or list the alternatives and bail."""
    hits = [c for c in candidates if query.lower() in c.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"no {what} matches {query!r}")
    print(f"'{query}' matched several {what}s -- be more specific:")
    for h in sorted(hits)[:10]:
        print("   ", h)
    raise SystemExit(1)


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def highlight(text, pattern):
    """Mark lines containing the pattern with a leading '>'."""
    if not pattern:
        return text
    rx = re.compile(pattern, re.I)
    return "\n".join(("> " if rx.search(ln) else "  ") + ln
                     for ln in text.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", help="dataset:id, e.g. standard:30 (as printed by examine_feature.py)")
    ap.add_argument("model", help="model substring, or 'gold' for the reference answer")
    ap.add_argument("--feature", help="regex; answer lines matching it are marked with '>'")
    ap.add_argument("--dir", default="outputs/part2_coverage/llama",
                    help="extraction dir holding case_feat.json (default outputs/part2_coverage/llama)")
    ap.add_argument("--no-case", action="store_true", help="skip the fact pattern")
    args = ap.parse_args()

    if ":" not in args.case:
        raise SystemExit("case must look like standard:30")
    ds, cid = args.case.split(":", 1)

    prompt = truth = None
    for rid, p, t in load_dataset(ds):
        if str(rid) == cid:
            prompt, truth = p, t
            break
    if prompt is None:
        raise SystemExit(f"case {args.case} not found in dataset {ds!r}")

    if not args.no_case:
        rule(f"CASE  {args.case}")
        print(prompt)

    if args.model.lower() == "gold":
        rule("GOLD ANSWER")
        print(highlight(truth, args.feature))
        return

    records = {}
    for r in load_json(GENERATIONS_JSON, default=[]):
        if r["dataset"] == ds and str(r["case_id"]) == cid:
            records[r["model"]] = r
    if not records:
        raise SystemExit(f"no generations found for {args.case}")

    model = resolve(sorted(records), args.model, "model")
    rule(f"ANSWER  {model}")
    print(highlight(records[model]["answer"], args.feature))

    # Extracted features for this model on this case, as the pipeline saw them.
    path = os.path.join(args.dir, "case_feat.json")
    if os.path.exists(path):
        with open(path) as fh:
            case = json.load(fh).get(args.case)
        if case:
            rows = [e for e in case["features"].values() if e["model"] == model]
            rule(f"EXTRACTED FEATURES  ({len(rows)}, importance sums to ~100)")
            for e in sorted(rows, key=lambda e: -e["importance"]):
                print(f"  {e['importance']:>3}  {e['feature']}")

    print()


if __name__ == "__main__":
    main()
