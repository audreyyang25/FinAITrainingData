"""examine_feature.py -- contrastive view of ONE global feature across two models.

For a global feature (matched by substring) and two models, shows per case whether
each model named a raw feature mapping to it, with what importance and raw text,
ranked by the importance gap -- so you read the cases where the two models most
DISAGREE, instead of browsing. Also reports SELECTION counts (how many cases each
names it), which are sturdier than the importance weights.

  python examine_feature.py --feature "fiduciary" \
         --a claude-opus-4.8 --b claude-haiku-4.5

  # add --show-answers to print answer excerpts for the top cases
"""

import argparse
import json
import os


def load(d):
    with open(os.path.join(d, "case_feat.json")) as fh:
        cases = json.load(fh)
    with open(os.path.join(d, "global_features.json")) as fh:
        mapping = json.load(fh)["mapping"]
    return cases, mapping


def _one(candidates, query, what):
    hits = [c for c in candidates if query.lower() in c.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"no {what} matches {query!r}")
    print(f"'{query}' matched several {what}s -- be more specific:")
    for h in hits[:10]:
        print("   ", h)
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/llama")
    ap.add_argument("--feature", required=True, help="substring of the global feature")
    ap.add_argument("--a", required=True, help="model A (substring ok)")
    ap.add_argument("--b", required=True, help="model B (substring ok)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--show-answers", action="store_true",
                    help="print answer excerpts for the top cases (reads generations.jsonl)")
    args = ap.parse_args()

    cases, mapping = load(args.dir)
    models = sorted({e["model"] for c in cases.values()
                     for e in c["features"].values()})
    A = _one(models, args.a, "model")
    B = _one(models, args.b, "model")
    target = _one(sorted({v for v in mapping.values() if v}), args.feature, "global feature")
    print(f"\nfeature: {target}\n  A = {A}\n  B = {B}")

    rows = []
    for key, c in cases.items():
        sup = c["superset"]
        per = {A: [], B: []}
        for e in c["features"].values():
            if e["model"] not in per:
                continue
            i = e["superset_index"]
            if i is not None and mapping.get(sup[i]) == target:
                per[e["model"]].append((e["importance"], e["feature"]))
        ia = sum(x for x, _ in per[A])
        ib = sum(x for x, _ in per[B])
        if ia or ib:
            rows.append({"case": key, "ia": ia, "ib": ib,
                         "ta": " | ".join(t for _, t in per[A]),
                         "tb": " | ".join(t for _, t in per[B])})

    na = [r["ia"] for r in rows if r["ia"]]
    nb = [r["ib"] for r in rows if r["ib"]]
    print(f"\nSELECTION: A names it in {len(na)} cases, B in {len(nb)} "
          f"(of {len(cases)} total)")
    if na and nb:
        print(f"MEAN IMPORTANCE when named: A={sum(na)/len(na):.1f}  "
              f"B={sum(nb)/len(nb):.1f}")

    answers = {}
    if args.show_answers:
        gpath = os.path.join(os.path.dirname(args.dir.rstrip("/")) or ".",
                             "generations.jsonl")
        gpath = gpath if os.path.exists(gpath) else "outputs/generations.jsonl"
        with open(gpath) as fh:
            for line in fh:
                r = json.loads(line)
                answers[(f"{r['dataset']}:{r['case_id']}", r["model"])] = r["answer"]

    rows.sort(key=lambda r: -abs(r["ia"] - r["ib"]))
    print("\nMost contrastive cases (grep generations.jsonl for the case + model):")
    for r in rows[:args.top]:
        lead = A if r["ia"] > r["ib"] else B
        print(f"\n  {r['case']}   A={r['ia']:>3}  B={r['ib']:>3}   (+{lead.split('/')[-1]})")
        if r["ta"]:
            print(f"     A raw: {r['ta'][:120]}")
        if r["tb"]:
            print(f"     B raw: {r['tb'][:120]}")
        if args.show_answers:
            for tag, mdl in (("A", A), ("B", B)):
                ans = answers.get((r["case"], mdl), "")
                if ans:
                    print(f"     {tag} answer: {ans[:200].strip()}...")


if __name__ == "__main__":
    main()
