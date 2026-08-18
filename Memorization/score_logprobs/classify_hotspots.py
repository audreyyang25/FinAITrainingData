#!/usr/bin/env python3
"""Separate genuine memorisation from recurring legal furniture.

  python classify_hotspots.py
  python classify_hotspots.py --top-n 60 --out-candidates candidates.csv

THE ARM IS THE CLASSIFIER, not a regex. A hot spot in a POST-cutoff opinion
cannot be memorisation of that opinion -- the model never saw it. So any span
whose wording also occurs in the post-cutoff corpus is recurring furniture by
construction, whichever arm it was found in. That is a definitional test, and it
does not depend on anyone's judgement about what "boilerplate" looks like.

Recurrence is measured with word shingles rather than exact match, because
citation strings vary in punctuation and party names while staying formulaic.

The regex CATEGORIES are for reading, not for deciding. They label what kind of
furniture a span is (statutory citation / SCOTUS syllabus / pleading template)
once recurrence has already established that it IS furniture. A span that
recurs nowhere and matches no category is the interesting residual: a candidate
for real memorisation, and the input to a fixed-prefix sliding-window pass.
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, re, statistics, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")
TEXT_DIR = os.path.join(DS, "court_opinions_courtlistener")
MANIFEST = os.path.join(REPO, "Data Collection and Training Material Generation",
                        "crawl_manifest", "case_ids.json")

SHINGLE = 8            # words; long enough that ordinary phrasing does not collide

CATEGORIES = [
    ("statutory_citation",
     re.compile(r"\b\d+\s*U\.?S\.?C\.?|\bC\.?F\.?R\.?\s*§|§\s*\d|"
                r"\bRule\s+\d+[a-z]?[-–]?\d|\bSection\s+\d+\([a-z]\)", re.I)),
    ("scotus_boilerplate",
     re.compile(r"syllabus|Reporter of Decisions|Detroit Timber|"
                r"constitutes no part of the opinion", re.I)),
    ("pleading_template",
     re.compile(r"unless enjoined|reasonably likely to continue|"
                r"substantial assistance|in connection with the purchase or sale|"
                r"knowingly or recklessly|for the foregoing reasons|"
                r"failed to state a claim|upon which relief can be granted", re.I)),
    # Quoted PRECEDENT: a reporter citation or a quotation is a span the model
    # could have learned from any of the hundreds of other opinions quoting the
    # same case -- not from this opinion. The post-cutoff reference corpus is
    # 115 securities opinions, so APA/immigration/jurisdiction boilerplate does
    # not appear in it and survives the recurrence test on a technicality.
    ("quoted_precedent",
     re.compile(r"\b\d+\s+F\.\s?(?:2d|3d|4th|Supp\.?)|\b\d+\s+U\.\s?S\.\s+\d+|"
                r"\([A-Z][a-z]*\.?\s*Cir\.\s*\d{4}\)|\bv\.\s+[A-Z][\w.]+.{0,40}\d{3}|"
                r"[“\"][^”\"]{40,}[”\"]", re.I)),
    ("court_furniture",
     re.compile(r"UNITED STATES COURT OF APPEALS|FOR PUBLICATION|"
                r"Before:.*Circuit Judges|Argued|Submitted|Decided:|"
                r"NOT FOR PUBLICATION|PER CURIAM", re.I)),
]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s§]", " ", t.lower())).strip()


def shingles(t, n=SHINGLE):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotspots", default="")
    ap.add_argument("--text-dir", default=TEXT_DIR)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--self-rep-threshold", type=float, default=0.25,
                    help="fraction of a span's shingles already present EARLIER "
                         "in the same document; above this it is copying from "
                         "context, not memorisation")
    ap.add_argument("--recur-threshold", type=float, default=0.25,
                    help="fraction of a span's shingles that must appear elsewhere "
                         "for it to count as recurring furniture")
    ap.add_argument("--ctx", type=int, default=400,
                    help="characters of surrounding text used to detect a "
                         "quotation whose citation fell outside the window")
    ap.add_argument("--top-n", type=int, default=50,
                    help="how many candidates to print")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-candidates", default="")
    args = ap.parse_args()

    hp = args.hotspots or sorted(glob.glob(os.path.join(DS, "logprobs", "hotspots__*.csv")))[0]
    hot = list(csv.DictReader(open(hp, newline="")))
    print(f"{len(hot)} hot spots from {os.path.basename(hp)}")

    man = json.load(open(args.manifest))
    arm = {c["case_id"]: c.get("arm", "") for c in man["court_opinions"]["cases"]}

    # Shingle index of the POST-cutoff corpus. The model has not seen any of it,
    # so anything appearing here is language that recurs across the genre.
    # Kept PER DOCUMENT, not as a flat union, so a post-cutoff span can be tested
    # against the post-cutoff corpus MINUS its own document. Flattening here made
    # the test circular for the post arm -- the span's own text was in the index,
    # so recur_in_post came back 1.0 for every post-cutoff span and the whole arm
    # was declared "recurring furniture" by construction. All 63 post-cutoff
    # recurring_furniture verdicts were that artefact. The pre arm was unaffected,
    # since a pre-cutoff document never enters this index.
    post_per, other_pre = {}, collections.defaultdict(set)
    for f in glob.glob(os.path.join(args.text_dir, "*.txt")):
        cid = os.path.basename(f)[:-4]
        g = shingles(open(f, errors="ignore").read())
        if "pre" in arm.get(cid, ""):
            other_pre[cid] = g
        else:
            post_per[cid] = g
    post_grams = set().union(*post_per.values()) if post_per else set()
    print(f"post-cutoff shingle index: {len(post_grams):,} {SHINGLE}-grams "
          f"from opinions the model never saw")
    all_pre = set()
    for g in other_pre.values():
        all_pre |= g

    rows = []
    for r in hot:
        g = shingles(r["text"])
        if not g:
            continue
        cid = r["case_id"]
        # Exclude the span's own document. For a pre-cutoff span this is a no-op;
        # for a post-cutoff span it is the difference between a real test and a
        # tautology.
        ref_post = post_grams - post_per[cid] if cid in post_per else post_grams
        in_post = len(g & ref_post) / len(g)
        # Other PRE-cutoff cases, excluding this one: recurrence across the
        # training-visible corpus is weaker evidence than post-cutoff recurrence
        # (the model could have memorised the same passage from several sources),
        # so it is reported but never used to disqualify on its own.
        others = set()
        for c2, g2 in other_pre.items():
            if c2 != cid:
                others |= g2
        in_other_pre = len(g & others) / len(g)
        # IN-DOCUMENT REPETITION. The profiler gives every token up to ~2k tokens
        # of preceding context, and courts restate their issues and facts several
        # times per opinion. A span the model "predicts perfectly" may simply be
        # a restatement of something 800 tokens earlier -- induction from context,
        # not recall from training. Checked against the document text BEFORE this
        # span only.
        prior = shingles(open(os.path.join(args.text_dir, cid + ".txt"),
                              errors="ignore").read()[:int(r["char_start"])])
        self_rep = len(g & prior) / len(g)
        cat = next((n for n, rx in CATEGORIES if rx.search(r["text"])), "")
        # NEIGHBOURHOOD CHECK. A 50-token window can slice a quoted statute or
        # multi-factor test away from the citation that marks it, leaving the
        # quoted BODY looking case-specific. 8 U.S.C. 1182(f) surfaced exactly
        # this way. So re-test with +/- `ctx` characters of surrounding text
        # before calling anything a memorisation candidate.
        doc = open(os.path.join(args.text_dir, cid + ".txt"), errors="ignore").read()
        near = doc[max(0, int(r["char_start"]) - args.ctx):
                   int(r["char_end"]) + args.ctx]
        cat_near = next((n for n, rx in CATEGORIES if rx.search(near)), "")
        if self_rep >= args.self_rep_threshold:
            verdict = "in_document_repetition"
        elif in_post >= args.recur_threshold:
            verdict = "recurring_furniture"
        elif "post" in arm.get(cid, ""):
            # Cannot be memorisation of this case; if it does not recur either,
            # it is genre-typical prose the model predicts from style alone.
            verdict = "post_cutoff_predictable"
        elif cat:
            verdict = "furniture_by_pattern"
        elif cat_near:
            verdict = "quoted_in_context"
        else:
            verdict = "MEMORISATION_CANDIDATE"
        rows.append({**r, "arm_manifest": arm.get(cid, ""), "category": cat or "-",
                     "category_near": cat_near or "-",
                     "self_repetition": round(self_rep, 3),
                     "recur_in_post": round(in_post, 3),
                     "recur_in_other_pre": round(in_other_pre, 3),
                     "verdict": verdict})

    print("\nVERDICTS")
    vc = collections.Counter(x["verdict"] for x in rows)
    for k, v in vc.most_common():
        print(f"   {k:26s} {v:4d}  ({100*v/len(rows):4.1f}%)")
    print("\nCATEGORY (what kind of furniture, where one matched)")
    for k, v in collections.Counter(x["category"] for x in rows).most_common():
        print(f"   {k:26s} {v:4d}")

    cands = [x for x in rows if x["verdict"] == "MEMORISATION_CANDIDATE"]
    cands.sort(key=lambda x: float(x["mean_logp"]), reverse=True)
    print(f"\n{len(cands)} MEMORISATION CANDIDATES "
          f"(pre-cutoff, no post-cutoff recurrence, no furniture pattern)\n")
    import textwrap
    for x in cands[:args.top_n]:
        print(f"[{x['rank']:>3}] logp={float(x['mean_logp']):+.4f}  "
              f"selfrep={x['self_repetition']:.2f} recur_post={x['recur_in_post']:.2f}  "
              f"{x['case_id'][:44]}")
        print(textwrap.fill(x["text"][:200], 96, initial_indent="      ",
                            subsequent_indent="      "))
    out = args.out or hp.replace("hotspots__", "classified__")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    if cands:
        oc = args.out_candidates or hp.replace("hotspots__", "candidates__")
        with open(oc, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cands[0].keys()))
            w.writeheader(); w.writerows(cands)
        print(f"wrote {oc}  ({len(cands)} rows) — the sliding-window input")


if __name__ == "__main__":
    main()
