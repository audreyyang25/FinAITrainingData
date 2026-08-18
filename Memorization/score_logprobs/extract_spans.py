#!/usr/bin/env python3
"""Stage 2: turn profile hot spots into short-prefix extraction tests.

  python extract_spans.py                       # writes a QA-format csv
  python extract_spans.py --prefix-words 50 --n-control 3

WHY THIS EXISTS. The profile gives every token up to ~2k tokens of preceding
context, so a span that restates something 800 tokens earlier scores well
whether or not the model memorised it -- the answer is sitting in the context
window. That is why 197 of 400 hot spots came back `in_document_repetition`.

But the profile being blind is NOT evidence of absence. In the extraction
setting the model gets a ~50-word prefix and nothing else; the earlier
restatement is unavailable. If it can still produce the span there, that IS
memorisation. Those spans are unadjudicated, not disqualified, and this script
builds the test that adjudicates them.

THREE GROUPS, and the comparison between them is the whole design:

  TGT  pre-cutoff in-document-repetition spans. Can be memorised; unmeasured.
  CTL  random spans from the SAME opinions, matched on length. The targets were
       selected for scoring well WITH context, so regression to the mean alone
       will pull them down under a short prefix. Without this control a drop
       proves nothing.
  FLR  the same in-document-repetition spans from POST-cutoff opinions. The
       model never saw those documents, so whatever they score is the floor.

TGT above both CTL and FLR is memorisation. TGT level with FLR is not.

Output is in the QA schema, so score_logprobs.py consumes it unchanged:
    python score_logprobs.py --model ... --qa spans_stage2.csv
The group is encoded in the qid (TGT0001 / CTL0001 / FLR0001) because
pairs.load_pairs does not carry extra columns through.
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, random, re, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
DS = os.path.join(BASE, "datasets")
TEXT_DIR = os.path.join(DS, "court_opinions_courtlistener")

# The QA schema pairs.load_pairs expects. Extra columns are ignored downstream,
# so the group has to ride in the qid.
QA_COLS = ["case_id", "arm", "date_filed", "caption", "jurisdiction", "court",
           "court_level", "text_source", "qid", "tier", "question", "prompt",
           "prompt_normalized", "prompt_words", "answer", "answer_normalized",
           "found", "evidence", "n_sentences", "n_headings", "n_footnotes",
           "group", "src_rank", "src_mean_logp", "char_start", "char_end"]


def words_before(text, char_start, n):
    """The n whitespace-delimited words immediately preceding char_start."""
    return " ".join(text[:char_start].split()[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", default="")
    ap.add_argument("--text-dir", default=TEXT_DIR)
    ap.add_argument("--out", default=os.path.join(DS, "spans_stage2_qa.csv"))
    ap.add_argument("--verdict", default="in_document_repetition",
                    help="which profile verdict to promote to a prefix test")
    ap.add_argument("--prefix-words", type=int, default=50,
                    help="~50 words is ~65 tokens, comparable to Cooper et al. "
                         "and past the 48-word point where Gatsby crossed the "
                         "memorisation threshold")
    ap.add_argument("--n-control", type=int, default=3,
                    help="random control spans per target, from the same opinion")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cp = args.classified or sorted(glob.glob(os.path.join(DS, "logprobs", "classified__*.csv")))[0]
    rows = list(csv.DictReader(open(cp, newline="")))
    print(f"{len(rows)} classified windows from {os.path.basename(cp)}")

    sel = [r for r in rows if r["verdict"] == args.verdict]
    tgt = [r for r in sel if "pre" in r["arm_manifest"]]
    flr = [r for r in sel if "post" in r["arm_manifest"]]
    print(f"  verdict={args.verdict}: {len(sel)}  ->  "
          f"{len(tgt)} pre-cutoff TARGETS, {len(flr)} post-cutoff FLOOR")
    if not tgt:
        sys.exit("no targets")

    rng = random.Random(args.seed)
    texts, out, n = {}, [], collections.Counter()

    def emit(group, r, text, cs, ce):
        pre = words_before(text, cs, args.prefix_words)
        suf = " ".join(text[cs:ce].split())
        if len(pre.split()) < args.prefix_words * 0.6 or len(suf.split()) < 8:
            return False                     # too near the document start
        n[group] += 1
        out.append({
            "case_id": r["case_id"], "arm": r.get("arm_manifest", ""),
            "date_filed": r.get("date_filed", ""), "caption": r["case_id"],
            "jurisdiction": r.get("jurisdiction", ""), "court": "", "court_level": "",
            "text_source": "courtlistener",
            "qid": f"{group}{n[group]:04d}", "tier": "D",
            "question": f"Continue the passage verbatim ({group}).",
            "prompt": pre, "prompt_normalized": pre, "prompt_words": len(pre.split()),
            "answer": suf, "answer_normalized": suf, "found": 1,
            "evidence": f"{args.verdict} span, profile rank {r.get('rank','')}",
            "n_sentences": 0, "n_headings": 0, "n_footnotes": 0,
            "group": group, "src_rank": r.get("rank", ""),
            "src_mean_logp": r.get("mean_logp", ""),
            "char_start": cs, "char_end": ce})
        return True

    def text_of(cid):
        if cid not in texts:
            p = os.path.join(args.text_dir, cid + ".txt")
            texts[cid] = open(p, errors="ignore").read() if os.path.exists(p) else ""
        return texts[cid]

    used = collections.defaultdict(list)     # case_id -> [(cs, ce)] already taken
    for r in tgt + flr:
        t = text_of(r["case_id"])
        if not t:
            continue
        cs, ce = int(r["char_start"]), int(r["char_end"])
        grp = "TGT" if r in tgt else "FLR"
        if emit(grp, r, t, cs, ce):
            used[r["case_id"]].append((cs, ce))

    # Matched controls: same opinion, same span length in words, random position,
    # never overlapping a target. Matching on LENGTH matters because logp_per_token
    # is length-sensitive and the targets are not a random length distribution.
    for r in tgt:
        t = text_of(r["case_id"])
        if not t:
            continue
        # Real character offsets for every word. An earlier version derived them
        # from " ".join(text.split()), which collapses the newlines and double
        # spaces the opinions are full of -- so the offsets drifted and control
        # spans began mid-word ("ecial attention" instead of "special").
        wspans = [(m.start(), m.end()) for m in re.finditer(r"\S+", t)]
        span_w = len(" ".join(t[int(r["char_start"]):int(r["char_end"])].split()).split())
        if len(wspans) < args.prefix_words + span_w + 40:
            continue
        placed = 0
        for _ in range(args.n_control * 30):
            if placed >= args.n_control:
                break
            wi = rng.randint(args.prefix_words + 5, len(wspans) - span_w - 5)
            cs = wspans[wi][0]
            ce = wspans[wi + span_w - 1][1]
            if any(cs < b and ce > a for a, b in used[r["case_id"]]):
                continue                      # overlaps a target or another control
            if emit("CTL", r, t, cs, ce):
                used[r["case_id"]].append((cs, ce))
                placed += 1

    out.sort(key=lambda x: x["qid"])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=QA_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(out)

    print(f"\n{'group':6s} {'n':>5s}  {'cases':>6s}  {'median suffix words':>20s}")
    for g in ("TGT", "CTL", "FLR"):
        s = [x for x in out if x["group"] == g]
        if not s:
            continue
        ws = sorted(len(x["answer"].split()) for x in s)
        print(f"{g:6s} {len(s):5d}  {len({x['case_id'] for x in s}):6d}  "
              f"{ws[len(ws)//2]:20d}")
    print(f"\nwrote {args.out}")
    print("\nnext, on the pod:")
    print(f"  python3 score_logprobs.py --model meta-llama/Llama-3.1-70B \\")
    print(f"      --qa '{args.out}' --truncate-n 25")
    print("\nthen compare logp_per_token across TGT / CTL / FLR (qid prefix).")
    print("TGT above BOTH CTL and FLR is memorisation; TGT level with FLR is not.")


if __name__ == "__main__":
    main()
