#!/usr/bin/env python3
"""Build 20 ground-truth questions per court opinion, weighted toward verbatim recall.

  Tier A (4)   identity facts - calibration floor; tells you the model knows the case exists
  Tier D (16)  verbatim text  - the construct that actually matches Cooper et al.

Tier D answers are *slices of the document*, not facts about it, so extraction is close
to deterministic. Completions are sampled at seven depths because memorization decays
from the opening -- a depth gradient is more informative than a single probe.

Output: long CSV, one row per (case, question), with a `prompt` column for completion
questions. `found=0` marks rows needing a hand-written answer.

  python Memorization/qa/build_questions.py
  python Memorization/qa/build_questions.py --arm post_cutoff_control
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_answers import normalize   # unicode/punctuation folding, shared with scoring

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
MANIFEST = os.path.join(BASE, "crawl_manifest", "case_ids.json")
SRC = [("courtlistener", os.path.join(BASE, "datasets", "court_opinions_courtlistener")),
       ("pdf_clean", os.path.join(BASE, "datasets", "court_opinions_clean")),
       ("pdf_raw", os.path.join(BASE, "datasets", "court_opinions"))]
OUT = os.path.join(BASE, "datasets", "court_opinions_qa.csv")

PREFIX_WORDS = 12
LONG_PREFIX, LONG_TARGET = 30, 60

# --------------------------------------------------------------- text hygiene
FURNITURE = [
    re.compile(r"^\s*\d{1,4}\s+No[s]?\.\s*\d{1,2}[-‐-―]\d{3,5}.*$", re.M),
    re.compile(r"^\s*No[s]?\.\s*\d{1,2}[-‐-―]\d{3,5}\s+\d{1,4}\s*$", re.M),
    re.compile(r"^\s*\d{1,4}\s*$", re.M),
    re.compile(r"^\s*-\s*\d{1,4}\s*-\s*$", re.M),
    re.compile(r"^\s*Page \d+ of \d+\s*$", re.M | re.I),
]


def strip_furniture(t: str) -> str:
    for p in FURNITURE:
        t = p.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t)


SENT = re.compile(r"(?<=[.!?”])\s+(?=[A-Z“])")


def clean_sentences(t: str) -> list[str]:
    """Sentences usable as verbatim targets.

    CourtListener's plain_text interleaves footnote bodies with prose, so a sentence can
    absorb `25 Id. at 263` mid-clause. Rather than unpick that, exclude candidates
    carrying digits or citation markers: costs coverage, buys prompts that read as the
    court actually wrote them.
    """
    out = []
    for s in SENT.split(strip_furniture(t)):
        s = re.sub(r"\s+", " ", s).strip()
        w = s.split()
        if not (22 <= len(w) <= 70) or not s or not s[0].isalpha() or not s[0].isupper():
            continue
        if any(c.isdigit() for c in s):
            continue
        if re.search(r"\b(Id\.|supra|infra|Cir\.|U\.S\.C\.|§|¶|F\.\s?(?:2d|3d|4th))", s):
            continue
        if s.count("(") != s.count(")") or s.count("“") != s.count("”"):
            continue
        out.append(s)
    return out


def find_footnotes(t: str) -> dict:
    """{n: text} only for a genuinely sequential footnote block.

    Any line starting with a digit could be a page number, paragraph number, or list
    item. Require a line-initial marker, real sentence text, and an unbroken ascending
    run from 1 that advances through the document. Otherwise return nothing -- a blank
    is honest, a wrong footnote silently corrupts the accuracy numbers.
    """
    cand = {}
    for m in re.finditer(r"^[ \t]*\[?(\d{1,3})\]?[ \t]+([A-Z\"'“][^\n]{30,})$", t, re.M):
        n = int(m.group(1))
        cand.setdefault(n, (re.sub(r"\s+", " ", m.group(2)).strip(), m.start()))
    if 1 not in cand:
        return {}
    run, pos = {}, -1
    for i in range(1, 200):
        if i not in cand:
            break
        txt, at = cand[i]
        if at < pos:
            break
        run[i], pos = txt, at
    return run if len(run) >= 3 else {}


HEAD_PATS = [
    re.compile(r"^[ \t]*([IVXL]{1,5}\.(?:\s+[A-Z][^\n]{0,58})?)[ \t]*$", re.M),
    re.compile(r"^[ \t]*([A-Z][A-Z \.\-'&]{5,50})[ \t]*$", re.M),
    re.compile(r"^[ \t]*([A-Z]\.\s+[A-Z][^\n]{2,58})[ \t]*$", re.M),
]


def find_headings(t: str) -> list[tuple[str, int]]:
    """Ordered (heading, offset). Skips the caption block, where all-caps party names
    would otherwise read as section headings."""
    start = int(len(t) * 0.05)
    seen, out = set(), []
    for p in HEAD_PATS:
        for m in p.finditer(t):
            if m.start() < start:
                continue
            h = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(h) < 4 or h.lower().startswith(("in the united states", "the united states")):
                continue
            if (h, m.start()) not in seen:
                seen.add((h, m.start()))
                out.append((h, m.start()))
    out.sort(key=lambda x: x[1])
    dedup = []
    for h, at in out:
        if dedup and at - dedup[-1][1] < 20:
            continue
        dedup.append((h, at))
    return dedup


def sentence_after(t: str, offset: int) -> str | None:
    tail = strip_furniture(t[offset:])
    tail = tail.split("\n", 1)[1] if "\n" in tail else tail
    for s in SENT.split(tail):
        s = re.sub(r"\s+", " ", s).strip()
        if len(s.split()) >= 8 and s[:1].isupper():
            return s
    return None


def split_prompt(sentence: str, nwords: int = PREFIX_WORDS):
    w = sentence.split()
    if len(w) <= nwords + 6:
        return None, None
    return " ".join(w[:nwords]), " ".join(w[nwords:])


# ------------------------------------------------------------------ questions
def q_court(t, r, ctx):
    m = re.search(r"(United States Court of Appeals[^\n]*)", t)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(), "", "from header"
    return (r.get("court_name") or r.get("court")), "", "from manifest"


def q_year(t, r, ctx):
    d = r.get("date_filed") or ""
    return (d[:4] or None), "", f"date_filed={d}"


def q_parties(t, r, ctx):
    # Lowercased: captions arrive inconsistently cased (slug-derived for the
    # original scrape, title-cased for fetched ones), and scoring is already
    # case-insensitive, so store it the way it scores.
    cap = (r.get("caption") or "").strip().lower()
    return (cap or None), "", "manifest caption (lowercased)"


def q_date(t, r, ctx):
    return (r.get("date_filed") or None), "", f"source={r.get('date_source')}"


def q_first_sentence(t, r, ctx):
    s = ctx["sents"]
    return (s[0] if s else None), "", "first clean sentence"


def q_last_sentence(t, r, ctx):
    s = ctx["sents"]
    return (s[-1] if s else None), "", "last clean sentence"


PROMPT_LENGTHS = [6, 12, 24, 48]


def make_completion(depth: float, fixed_words: int | None = None):
    """Sample a completion at `depth` through the document.

    Prompt length rotates across cases unless pinned. A full depth x length factorial
    would be 28 probes per document; rotating instead gives each document 7 probes while
    every (depth, length) cell still accumulates ~30 observations corpus-wide. The unit
    of analysis is the corpus curve, not the individual document, so the within-document
    grid is not needed -- Q19/Q20 pin two lengths at mid-depth for cases where a
    within-document contrast is wanted.
    """
    def fn(t, r, ctx):
        s = ctx["sents"]
        if len(s) < 6:
            return None, "", "too few clean sentences"
        nw = fixed_words or PROMPT_LENGTHS[ctx["rotation"] % len(PROMPT_LENGTHS)]
        i = min(len(s) - 1, max(0, int(len(s) * depth)))
        for j in list(range(i, len(s))) + list(range(i - 1, -1, -1)):
            p, a = split_prompt(s[j], nw)
            if p:
                return a, p, (f"sentence {j+1}/{len(s)} (~{int(depth*100)}% depth), "
                              f"prompt={nw}w")
        return None, "", f"no sentence long enough for a {nw}-word prompt"
    return fn


def q_long_completion(t, r, ctx):
    """A longer span than one sentence -- closer to the paper's 50-token suffix.

    Must come from *contiguous* document text. Building it from the filtered sentence
    list would silently splice across dropped sentences, so the "continuation" would not
    actually follow the prompt in the opinion.
    """
    body = re.sub(r"\s+", " ", strip_furniture(t)).strip()
    w = body.split()
    need = LONG_PREFIX + LONG_TARGET
    if len(w) < need + 200:
        return None, "", "document too short"
    # Scan from one third in for a window free of citation and footnote debris.
    for start in range(len(w) // 3, min(len(w) - need, int(len(w) * 0.8)), 10):
        win = " ".join(w[start:start + need])
        if any(c.isdigit() for c in win):
            continue
        if re.search(r"\b(Id\.|supra|infra|Cir\.|U\.S\.C\.|§|¶)", win):
            continue
        if not w[start][:1].isupper():
            continue
        return (" ".join(w[start + LONG_PREFIX:start + need]),
                " ".join(w[start:start + LONG_PREFIX]),
                f"contiguous words {start}-{start+need} of {len(w)}")
    return None, "", "no clean contiguous window"


ANCHOR_TARGET = 20          # suffix length, identical across the family
ANCHOR_PREFIXES = (6, 12, 24, 48)


def _anchor_index(words: list[str], depth: float) -> int | None:
    """Word index of a clean sentence start near `depth`, with room either side.

    The whole point of the anchored family is that the *break* stays fixed while
    the prefix lengthens. Growing a prefix forward (what split_prompt does) moves
    the break too, so prefix length and target text change together and the
    comparison means nothing.
    """
    lo, hi = max(ANCHOR_PREFIXES), len(words) - ANCHOR_TARGET
    if hi <= lo:
        return None
    start = min(max(int(len(words) * depth), lo), hi - 1)
    for i in list(range(start, hi)) + list(range(start - 1, lo - 1, -1)):
        if not words[i][:1].isupper() or not words[i - 1].endswith((".", '."', ".\u201d")):
            continue
        win = " ".join(words[i - max(ANCHOR_PREFIXES): i + ANCHOR_TARGET])
        if any(c.isdigit() for c in win):
            continue
        if re.search(r"\b(Id\.|supra|infra|Cir\.|U\.S\.C\.|\u00a7|\u00b6)", win):
            continue
        return i
    return None


def make_anchored_completion(depth: float, prefix_words: int):
    """One member of a nested-prefix family: same break, same answer, longer prompt."""
    def fn(t, r, ctx):
        w = ctx.get("body_words") or []
        i = _anchor_index(w, depth)
        if i is None:
            return None, "", "no clean anchor with room for a 48-word prefix"
        return (" ".join(w[i: i + ANCHOR_TARGET]),
                " ".join(w[i - prefix_words: i]),
                f"anchor word {i}/{len(w)} (~{int(depth*100)}%), prefix={prefix_words}w, "
                f"target={ANCHOR_TARGET}w — break identical across Q21-Q24")
    return fn


def make_heading_name(n: int):
    def fn(t, r, ctx):
        h = ctx["heads"]
        return (h[n - 1][0] if len(h) >= n else None), "", f"{len(h)} headings detected"
    return fn


def q_heading_list(t, r, ctx):
    h = [x[0] for x in ctx["heads"][:8]]
    return (" | ".join(h) if len(h) >= 2 else None), "", f"{len(ctx['heads'])} headings total"


def make_heading_sentence(n: int):
    def fn(t, r, ctx):
        h = ctx["heads"]
        if len(h) < n:
            return None, "", f"only {len(h)} headings found"
        name, at = h[n - 1]
        return sentence_after(t, at), "", f"heading {n} = {name!r}"
    return fn


def make_footnote(n: int):
    def fn(t, r, ctx):
        f = ctx["fnotes"]
        if not f:
            return None, "", "no validated footnote block"
        k = max(f) if n == -1 else n
        if k not in f:
            return None, "", f"no validated footnote {k}"
        return f[k][:600], "", f"footnote {k} of {max(f)}"
    return fn


QUESTIONS = [
    ("Q01", "A", "Which court decided this case?", q_court),
    ("Q02", "A", "In what year was this case decided?", q_year),
    ("Q03", "A", "Who were the parties (case caption)?", q_parties),
    ("Q04", "A", "On what exact date was the opinion decided?", q_date),

    ("Q05", "D", "What is the first sentence of the opinion?", q_first_sentence),
    ("Q06", "D", "What is the final sentence of the opinion?", q_last_sentence),

    ("Q07", "D", "Complete this sentence (opening of the opinion).", make_completion(0.02)),
    ("Q08", "D", "Complete this sentence (~15% into the opinion).", make_completion(0.15)),
    ("Q09", "D", "Complete this sentence (~30% into the opinion).", make_completion(0.30)),
    ("Q10", "D", "Complete this sentence (~45% into the opinion).", make_completion(0.45)),
    ("Q11", "D", "Complete this sentence (~60% into the opinion).", make_completion(0.60)),
    ("Q12", "D", "Complete this sentence (~75% into the opinion).", make_completion(0.75)),
    ("Q13", "D", "Complete this sentence (~90% into the opinion).", make_completion(0.90)),
    ("Q14", "D", "Continue this passage for roughly 60 words.", q_long_completion),

    ("Q15", "D", "What is the first section heading in the opinion?", make_heading_name(1)),
    ("Q16", "D", "List the section headings of the opinion in order.", q_heading_list),
    ("Q17", "D", "What is the first sentence after the first section heading?", make_heading_sentence(1)),
    ("Q18", "D", "What is the first sentence after the second section heading?", make_heading_sentence(2)),

    ("Q19", "D", "Complete this sentence (mid-document, short 6-word prompt).",
     make_completion(0.50, fixed_words=6)),
    ("Q20", "D", "Complete this sentence (mid-document, long 48-word prompt).",
     make_completion(0.50, fixed_words=48)),
]


# v2: drops Q05/Q06/Q15/Q18 (regex extractions the author reviewed and rejected)
# and replaces them with a nested-prefix family sharing one break point.
DROPPED_V2 = {"Q05", "Q06", "Q15", "Q18"}
QUESTIONS_V2 = [q for q in QUESTIONS if q[0] not in DROPPED_V2] + [
    # The break often falls at a sentence boundary, which reads as "write the
    # next sentence" unless the instruction is explicit. Stating the exact word
    # count and the verbatim requirement makes the task unambiguous either way.
    ("Q21", "D", f"Complete the next {ANCHOR_TARGET} words of this passage, verbatim from the opinion.",
     make_anchored_completion(0.45, 6)),
    ("Q22", "D", f"Complete the next {ANCHOR_TARGET} words of this passage, verbatim from the opinion.",
     make_anchored_completion(0.45, 12)),
    ("Q23", "D", f"Complete the next {ANCHOR_TARGET} words of this passage, verbatim from the opinion.",
     make_anchored_completion(0.45, 24)),
    ("Q24", "D", f"Complete the next {ANCHOR_TARGET} words of this passage, verbatim from the opinion.",
     make_anchored_completion(0.45, 48)),
]


def load_text(case_id):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in case_id)
    for name, d in SRC:
        p = os.path.join(d, safe + ".txt")
        if os.path.exists(p) and os.path.getsize(p) > 200:
            return open(p).read(), name
    return "", "missing"


def bad_matches() -> set:
    """Cases where the CourtListener fetch did not land on the right opinion."""
    out = set()
    for f in glob.glob(os.path.join(BASE, "crawl_manifest", "raw", "court_opinions", "*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        # "fetched_by_id" cases came straight from a CourtListener opinion ID —
        # provenance is certain and there is no local PDF to verify against, so
        # the absence of a match verdict is not a failed match.
        if j.get("verdict") not in ("exact", "match", "fetched_by_id"):
            cid = (j.get("manifest_record") or {}).get("case_id")
            if cid:
                out.add(cid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["pre_cutoff", "post_cutoff_control"])
    ap.add_argument("--jurisdiction", choices=["federal", "state"])
    ap.add_argument("--min-coverage", type=int, default=13,
                    help="drop cases with fewer than N of 20 questions extracted")
    ap.add_argument("--keep-mismatched", action="store_true",
                    help="keep cases whose CourtListener match failed verification")
    ap.add_argument("--out", default=None)
    ap.add_argument("--v2", action="store_true",
                    help="build the v2 set: drop Q05/Q06/Q15/Q18, add the Q21-Q24 "
                         "nested-prefix family")
    args = ap.parse_args()
    questions = QUESTIONS_V2 if args.v2 else QUESTIONS
    if args.out is None:
        args.out = OUT.replace(".csv", "_v2.csv") if args.v2 else OUT

    man = json.load(open(MANIFEST))
    cases = man["court_opinions"]["cases"]
    if args.arm:
        cases = [c for c in cases if c.get("arm") == args.arm]
    if args.jurisdiction:
        cases = [c for c in cases if c.get("jurisdiction") == args.jurisdiction]

    dropped_mismatch = []
    if not args.keep_mismatched:
        bad = bad_matches()
        dropped_mismatch = [c["case_id"] for c in cases if c["case_id"] in bad]
        cases = [c for c in cases if c["case_id"] not in bad]

    rows, cov, srcs = [], collections.Counter(), collections.Counter()
    for idx, rec in enumerate(cases):
        text, src = load_text(rec["case_id"])
        srcs[src] += 1
        ctx = {"rotation": idx,
               "sents": clean_sentences(text) if text else [],
               "heads": find_headings(text) if text else [],
               "fnotes": find_footnotes(text) if text else {},
               "body_words": re.sub(r"\s+", " ", strip_furniture(text)).split() if text else []}
        for qid, tier, question, fn in questions:
            try:
                ans, prompt, ev = fn(text, rec, ctx) if text else (None, "", "no text")
            except Exception as e:
                ans, prompt, ev = None, "", f"EXTRACTOR ERROR: {type(e).__name__}"
            found = 1 if ans not in (None, "") else 0
            cov[qid] += found
            rows.append({"case_id": rec["case_id"], "arm": rec.get("arm"),
                         # date_filed travels with every row so arms can be
                         # recomputed per model. The `arm` column is only the
                         # 2023-12-31 (Llama) split — it is wrong for any model
                         # with a different cutoff, and date_filed is the fix.
                         "date_filed": rec.get("date_filed"),
                         "caption": rec.get("caption"),
                         "jurisdiction": rec.get("jurisdiction"), "court": rec.get("court"),
                         "court_level": rec.get("court_level"), "text_source": src,
                         "qid": qid, "tier": tier, "question": question,
                         "prompt": prompt or "",
                         "prompt_normalized": normalize(prompt or "", lower=False),
                         "prompt_words": len((prompt or "").split()),
                         "answer": ans or "",
                         "answer_normalized": normalize(ans or "", lower=False),
                         "found": found, "evidence": ev,
                         "n_sentences": len(ctx["sents"]), "n_headings": len(ctx["heads"]),
                         "n_footnotes": len(ctx["fnotes"])})

    # Coverage filter: a case with almost nothing extracted contributes noise, and the
    # cause is systematic (citation-dense opinions leave too few clean sentences), so
    # those cases are not missing at random.
    per = collections.Counter()
    for r in rows:
        per[r["case_id"]] += r["found"]
    low = sorted(c for c, v in per.items() if v < args.min_coverage)
    rows = [r for r in rows if per[r["case_id"]] >= args.min_coverage]
    kept_cases = {r["case_id"] for r in rows}

    cols = ["case_id", "arm", "date_filed", "caption", "jurisdiction", "court",
            "court_level", "text_source",
            "qid", "tier", "question", "prompt", "prompt_normalized", "prompt_words",
            "answer", "answer_normalized", "found", "evidence",
            "n_sentences", "n_headings", "n_footnotes"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    cov = collections.Counter()
    for r in rows:
        cov[r["qid"]] += r["found"]

    if dropped_mismatch:
        print(f"excluded {len(dropped_mismatch)} unverified match(es): "
              + ", ".join(c[:44] for c in dropped_mismatch))
    if low:
        print(f"excluded {len(low)} cases below {args.min_coverage}/20 coverage:")
        for c in low:
            print(f"   {per[c]:>2}/20  {c[:60]}")
    n = len(kept_cases)
    print(f"\n{n} cases x {len(questions)} questions = {len(rows)} rows")
    print(f"text sources: {dict(srcs)}\n")
    print(f"{'qid':<5}{'tier':<6}{'found':>7}{'cov':>7}   question")
    for qid, tier, question, _ in questions:
        print(f"{qid:<5}{tier:<6}{cov[qid]:>7}{100*cov[qid]//max(1,n):>6}%   {question[:56]}")
    tot = sum(cov.values())
    print(f"\noverall extraction: {tot}/{len(rows)} ({100*tot//max(1,len(rows))}%)")
    print(f"blanks to fill by hand: {len(rows)-tot}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
