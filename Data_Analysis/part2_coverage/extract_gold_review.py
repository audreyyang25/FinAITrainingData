"""extract_gold_review.py -- pull the N cases most worth a human gold-standard audit.

Ranking.  The obvious sort -- lowest gold coverage -- is a noisy target: coverage
is gold_found / |superset|, and the superset is the UNION of every feature any
model named, so a case where 12 models each invent a different spurious feature
inflates the denominator and tanks gold's coverage even when gold is perfect.
Low coverage conflates "gold missed things" with "the models were noisy".

We therefore rank by CONSENSUS MISSES: superset features gold did not name, that
at least `--support` of the 12 contestant models did.  A feature 10/12 independent
models flag and gold omits is a real candidate gap; a feature 1/12 flags is
probably a hallucination and says nothing about gold.  Raw gold_coverage is kept
as a column so the alternative sort is one click away.

Output: one row per case, holding the complete original record (fact pattern,
question, gold answer/analysis, and every other source field) plus the coverage
counts and the actual feature text on both sides of the ledger.

  python extract_gold_review.py --n 100 --suffix _100
"""

import argparse
import json
from collections import defaultdict

import pandas as pd

from shared.config import ADAPTERS, DATA_DIR, part_output

SEP = "  ||  "          # multi-value cell separator; readable in Excel


def load_source_records():
    """(dataset, case_id) -> the complete original record, all fields intact."""
    records = {}
    for name, adapter in ADAPTERS.items():
        with open(f"{DATA_DIR}/{adapter['file']}") as fh:
            for r in json.load(fh):
                records[(name, r["id"])] = r
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--support", type=float, default=0.5,
                    help="fraction of models that must name a gold-missed "
                         "feature for it to count as a consensus miss")
    args = ap.parse_args()

    # Independent-Llama extraction (Part 2), not the legacy self-report file.
    with open(part_output("part2_coverage", "llama/case_feat.json")) as fh:
        cases = json.load(fh)
    source = load_source_records()

    rows = []
    no_gold = []      # gold absent entirely -- an extraction bug, not bad gold

    for case in cases.values():

        superset = case["superset"]
        n_features = len(superset)
        if not n_features:
            continue

        gold_idx = set()
        support = defaultdict(set)      # superset slot -> models naming it

        for e in case["features"].values():
            idx = e["superset_index"]
            if idx is None:
                continue
            if e["model_family"] == "gold":
                gold_idx.add(idx)
            else:
                support[idx].add(e["model"])

        n_models = len({e["model"] for e in case["features"].values()
                        if e["model_family"] != "gold"})
        if not n_models:
            continue

        # Gold contributed no features at all -> gold_coverage is 0 because gold
        # is MISSING, not because it is bad.  These would otherwise sort to the
        # top of the audit list and give the reviewer nothing to review.
        if not gold_idx:
            no_gold.append((case["dataset"], case["case_id"], n_features))
            continue

        threshold = args.support * n_models

        # Features gold missed, most-corroborated first -- this is the audit list.
        missed = sorted(
            (i for i in range(n_features) if i not in gold_idx),
            key=lambda i: -len(support[i]),
        )
        consensus = [i for i in missed if len(support[i]) >= threshold]

        record = source.get((case["dataset"], case["case_id"]), {})

        row = {
            "dataset": case["dataset"],
            "case_id": case["case_id"],

            # --- the audit signal
            "consensus_misses": len(consensus),
            "max_missed_support": (max((len(support[i]) for i in missed), default=0)),
            "gold_features_found": len(gold_idx),
            "total_features": n_features,
            "gold_coverage": len(gold_idx) / n_features,
            "n_models": n_models,

            # --- what gold missed, and how many models corroborate each
            "consensus_missed_features": SEP.join(
                f"[{len(support[i])}/{n_models}] {superset[i]}" for i in consensus),
            "all_missed_features": SEP.join(
                f"[{len(support[i])}/{n_models}] {superset[i]}" for i in missed),

            # --- what gold DID name; the tail of it is gold's unique contribution
            "gold_features": SEP.join(superset[i] for i in sorted(gold_idx)),
            "gold_only_features": SEP.join(
                superset[i] for i in sorted(gold_idx) if not support[i]),
        }

        # Every field of the original record, verbatim.
        for k, v in record.items():
            row[f"src_{k}"] = v if isinstance(v, (str, int, float, bool)) \
                else json.dumps(v, ensure_ascii=False)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["consensus_misses", "max_missed_support", "gold_coverage"],
        ascending=[False, False, True],
    )

    top = df.head(args.n)
    path = part_output("part2_coverage", f"gold_review{args.suffix}.csv")
    top.to_csv(path, index=False)

    print(f"{len(df)} cases scored -> wrote top {len(top)} to {path}\n")

    if no_gold:
        print(f"!! EXCLUDED {len(no_gold)} cases with NO gold features extracted "
              f"(pipeline bug, not a gold-quality signal):")
        for ds, cid, n in no_gold:
            print(f"     {ds}:{cid}  ({n} features from the models, 0 from gold)")
        print()

    print(f"consensus threshold: a gold-missed feature must be named by "
          f">= {args.support:.0%} of models\n")
    print("selected set:")
    print(f"  consensus_misses  mean {top.consensus_misses.mean():.1f}  "
          f"range {top.consensus_misses.min()}-{top.consensus_misses.max()}")
    print(f"  gold_coverage     mean {top.gold_coverage.mean():.2f}  "
          f"range {top.gold_coverage.min():.2f}-{top.gold_coverage.max():.2f}")
    print(f"  by dataset: {dict(top.dataset.value_counts())}")

    # How much does the smarter ranking actually change the sample?
    naive = set(map(tuple, df.nsmallest(args.n, "gold_coverage")[
        ["dataset", "case_id"]].values))
    chosen = set(map(tuple, top[["dataset", "case_id"]].values))
    print(f"\n  overlap with a naive lowest-gold-coverage top-{args.n}: "
          f"{len(naive & chosen)}/{args.n}")


if __name__ == "__main__":
    main()
