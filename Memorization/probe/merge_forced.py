#!/usr/bin/env python3
"""Combine a default-prompt run with its forced-attempt follow-up.

  python Memorization/probe/merge_forced.py \
      --base       datasets/predictions/openai__gpt-5.csv \
      --forced     datasets/predictions/FORCED__openai__gpt-5.csv \
      --out        datasets/predictions/MERGED__openai__gpt-5.csv

The forced run only covers rows the base run declined, so neither file alone is
a complete probe: the base is missing the abstained rows' true content, and the
forced file is missing every row that was answered first time. Merging gives one
row per (case_id, qid) with the abstention replaced by an actual attempt.

Row provenance is kept in a `prompt_mode` column so downstream analysis can
still split the two populations -- a merged mean that cannot be decomposed back
into "answered freely" and "answered under duress" hides the very effect this
whole exercise is measuring. Scorers read by column name, so the extra field is
inert for everything that does not ask for it.
"""
from __future__ import annotations
import argparse, csv, os, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the default-prompt run")
    ap.add_argument("--forced", required=True, help="the --force-attempt follow-up")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base, forced = read(args.base), read(args.forced)
    cols = list(base[0].keys())
    if "prompt_mode" not in cols:
        cols.append("prompt_mode")

    merged = {}
    for r in base:
        r["prompt_mode"] = "default"
        merged[(r["case_id"], r["qid"])] = r

    replaced = added = 0
    for r in forced:
        k = (r["case_id"], r["qid"])
        # A forced row that itself errored is not an improvement on the
        # abstention it was meant to replace -- keep the original.
        if (r.get("error") or "").strip():
            continue
        r["prompt_mode"] = "forced"
        if k in merged:
            replaced += 1
        else:
            added += 1
        merged[k] = r

    rows = sorted(merged.values(), key=lambda r: (r["case_id"], r["qid"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"base    {len(base):>6} rows   {os.path.basename(args.base)}")
    print(f"forced  {len(forced):>6} rows   {os.path.basename(args.forced)}")
    print(f"        {replaced:>6} abstentions replaced by a forced attempt")
    if added:
        print(f"        {added:>6} forced rows had no counterpart in base (unexpected)")
    print(f"merged  {len(rows):>6} rows -> {args.out}")


if __name__ == "__main__":
    main()
