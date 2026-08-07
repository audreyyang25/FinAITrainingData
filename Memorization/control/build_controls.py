#!/usr/bin/env python3
"""Positive and negative controls, built through the identical QA pipeline.

Claude declines 89% of court-opinion questions and scores ~3 tokens of verbatim
overlap when it answers, against a 0.9-token null floor. That is either a real
finding (frontier models recognize these opinions but have not memorized their
text) or a broken harness, and nothing in the court-opinion data can tell the
two apart.

  positive  The Great Gatsby (Project Gutenberg #64317). Public domain in the US
            since 2021, AND named in Cooper et al. as substantially memorized by
            Llama 3.1 70B — the only one of that paper's headline books that can
            be obtained cleanly. If the pipeline cannot detect memorization here,
            it cannot detect it anywhere.

  negative  The same passages with word order shuffled inside each one. Same
            vocabulary, same length, same prose register — but no reproducible
            sequence. Anything above the floor here is the metric rewarding
            style rather than recall.

Output matches court_opinions_qa_v2.csv column-for-column, so run_probe.py and
score_answers.py consume it with no changes.

  python Memorization/control/build_controls.py --gatsby /path/to/pg64317.txt
"""
from __future__ import annotations
import argparse, csv, os, random, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qa"))
from build_questions import (ANCHOR_TARGET, ANCHOR_PREFIXES, clean_sentences,
                             strip_furniture, split_prompt)
from score_answers import normalize

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
OUT = os.path.join(BASE, "datasets", "controls_qa.csv")

GUT_START = re.compile(r"\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
GUT_END = re.compile(r"\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)


def strip_gutenberg(t: str) -> str:
    """Drop the license header/footer — it is boilerplate that appears in
    thousands of books and would be memorized independently of the novel."""
    m = GUT_START.search(t)
    if m:
        t = t[m.end():]
    m = GUT_END.search(t)
    if m:
        t = t[:m.start()]
    return t.strip()


def chunk(words: list[str], n_chunks: int, min_words: int) -> list[list[str]]:
    size = max(min_words, len(words) // n_chunks)
    return [words[i:i + size] for i in range(0, len(words), size)
            if len(words[i:i + size]) >= min_words]


def shuffle_within(words: list[str], seed: int) -> list[str]:
    """Negative control: same words, destroyed sequence. Shuffling in windows
    rather than globally keeps local vocabulary drift realistic."""
    rng = random.Random(seed)
    out, W = [], 40
    for i in range(0, len(words), W):
        blk = words[i:i + W]
        rng.shuffle(blk)
        out.extend(blk)
    return out


def questions_for(words: list[str]) -> list[tuple]:
    """The same question shapes the court opinions get: depth-sampled sentence
    completions plus the Q21-Q24 nested-prefix family sharing one break."""
    body = " ".join(words)
    sents = clean_sentences(body)
    out = []

    for qid, depth in [("Q07", 0.02), ("Q08", 0.15), ("Q09", 0.30), ("Q10", 0.45),
                       ("Q11", 0.60), ("Q12", 0.75), ("Q13", 0.90)]:
        if len(sents) < 6:
            continue
        i = min(len(sents) - 1, max(0, int(len(sents) * depth)))
        for j in list(range(i, len(sents))) + list(range(i - 1, -1, -1)):
            p, a = split_prompt(sents[j], 12)
            if p:
                out.append((qid, "D", f"Complete this sentence (~{int(depth*100)}% into the text).",
                            p, a, f"sentence {j+1}/{len(sents)}"))
                break

    # Anchored family: fixed break, prefix grows backward.
    lo, hi = max(ANCHOR_PREFIXES), len(words) - ANCHOR_TARGET
    if hi > lo:
        anchor = None
        start = min(max(len(words) // 2, lo), hi - 1)
        for i in list(range(start, hi)) + list(range(start - 1, lo - 1, -1)):
            if words[i][:1].isupper() and words[i - 1].endswith((".", '."', ".”")):
                anchor = i
                break
        if anchor:
            ans = " ".join(words[anchor: anchor + ANCHOR_TARGET])
            for qid, n in zip(("Q21", "Q22", "Q23", "Q24"), ANCHOR_PREFIXES):
                out.append((qid, "D",
                            f"Complete the next {ANCHOR_TARGET} words of this passage, "
                            f"verbatim from the text.",
                            " ".join(words[anchor - n: anchor]), ans,
                            f"anchor {anchor}, prefix={n}w — break identical across Q21-Q24"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", required=True, metavar="SLUG:TITLE:PATH",
                    help="repeatable, e.g. mobydick:'Moby-Dick by Herman Melville':/path.txt")
    ap.add_argument("--chunks", type=int, default=8, help="pseudo-cases per work")
    ap.add_argument("--min-words", type=int, default=1200)
    ap.add_argument("--negative-from", default="",
                    help="slug whose passages get word-scrambled as the negative control")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    works = []
    for spec in args.text:
        slug, title, path = spec.split(":", 2)
        raw = strip_gutenberg(open(path, encoding="utf-8").read())
        w = re.sub(r"\s+", " ", strip_furniture(raw)).split()
        blocks = chunk(w, args.chunks, args.min_words)
        works.append((slug, title, blocks))
        print(f"  {slug:<12} {len(w):>7} words -> {len(blocks)} chunks")
    print()

    rows = []
    conds = [(slug, title, blocks, "positive", lambda w, i: w) for slug, title, blocks in works]
    for slug, title, blocks in works:
        if slug == args.negative_from:
            conds.append((f"{slug}_shuf", f"{title} (word order scrambled)",
                          blocks, "negative", shuffle_within))

    for prefix, cap, blocks, cond, xform in conds:
        for ci, blk in enumerate(blocks, 1):
            w = xform(list(blk), ci)
            cid = f"{prefix}_ch{ci:02d}"
            for qid, tier, question, prompt, answer, ev in questions_for(w):
                rows.append({
                    "case_id": cid, "arm": cond, "date_filed": "1800-01-01",
                    "caption": f"{cap} — passage {ci}",
                    "jurisdiction": "control", "court": cond, "court_level": cond,
                    "text_source": "gutenberg", "qid": qid, "tier": tier,
                    "question": question,
                    "prompt": prompt, "prompt_normalized": normalize(prompt, lower=False),
                    "prompt_words": len(prompt.split()),
                    "answer": answer, "answer_normalized": normalize(answer, lower=False),
                    "found": 1, "evidence": ev,
                    "n_sentences": 0, "n_headings": 0, "n_footnotes": 0})

    cols = ["case_id", "arm", "date_filed", "caption", "jurisdiction", "court",
            "court_level", "text_source", "qid", "tier", "question", "prompt",
            "prompt_normalized", "prompt_words", "answer", "answer_normalized",
            "found", "evidence", "n_sentences", "n_headings", "n_footnotes"]
    rows.sort(key=lambda r: (r["case_id"], r["qid"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    import collections
    c = collections.Counter(r["arm"] for r in rows)
    w = collections.Counter(r["case_id"].rsplit("_ch", 1)[0] for r in rows)
    print(f"{len(rows)} rows: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print("  per work: " + ", ".join(f"{k}={v}" for k, v in sorted(w.items())))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
