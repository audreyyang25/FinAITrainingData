#!/usr/bin/env python3
"""Score model answers against the regex-extracted ground truth.

Normalization comes first and matters more than the metric choice. The gold answers
carry ~1,800 non-ASCII characters -- curly quotes, em dashes, U+2010 hyphens -- because
courts publish typographically. Models emit ASCII. Compare raw and a perfect recall
scores ~0.

Metrics, and what each is for:

  exact_match          binary, post-normalization. A floor, not a measure -- one
                       different word zeroes it.
  prefix_tokens        leading tokens matching before the first divergence. The graded
                       version of exact match, and the closest analogue to the
                       discoverable-extraction setup in Cooper et al.
  longest_run          longest *contiguous* run of matching tokens. The headline number:
                       "reproduced 23 consecutive tokens verbatim" is the claim a
                       memorization result actually rests on.
  rouge_l              LCS-based F. Order-sensitive, tolerant of insertions. Best single
                       graded score for overall similarity.
  char_ratio           difflib character similarity. Catches near-misses that differ
                       only in morphology or spacing.
  token_f1             bag-of-words F1. Reported for comparability with QA benchmarks,
                       but order-blind -- a shuffled answer scores high, so never lead
                       with it for a verbatim claim.

Every metric has a floor above zero because legal prose is formulaic: "the district
court did not abuse its discretion in" will partially match almost anything. Use
--null to measure that floor on your own corpus before interpreting any score.

  python Memorization/qa/score_answers.py --null
  python Memorization/qa/score_answers.py --predictions preds.csv
"""
from __future__ import annotations
import argparse, csv, difflib, json, os, random, re, statistics, sys, time, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_nonanswers import classify

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
QA = os.path.join(BASE, "datasets", "court_opinions_qa.csv")
SCORE_DIR = os.path.join(BASE, "datasets", "scores")

# ------------------------------------------------------------- normalization
PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "­": "", "​": "",
    "…": "...", "§": "section",
}


def normalize(s: str, *, lower=True, drop_punct=False) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for a, b in PUNCT_MAP.items():
        s = s.replace(a, b)
    if lower:
        s = s.lower()
    if drop_punct:
        s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> list[str]:
    return re.findall(r"[\w']+", normalize(s, drop_punct=False))


# ------------------------------------------------------------------ metrics
def lcs_len(a: list, b: list) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def longest_run(a: list, b: list) -> int:
    if not a or not b:
        return 0
    return max((m.size for m in difflib.SequenceMatcher(a=a, b=b,
                                                        autojunk=False).get_matching_blocks()),
               default=0)


def prefix_tokens(pred: list, gold: list) -> int:
    n = 0
    for x, y in zip(pred, gold):
        if x != y:
            break
        n += 1
    return n


def score(pred: str, gold: str) -> dict:
    p, g = tokens(pred), tokens(gold)
    if not g:
        return {}
    l = lcs_len(p, g)
    prec = l / len(p) if p else 0.0
    rec = l / len(g)
    rouge = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            common[t] -= 1
            overlap += 1
    tp = overlap / len(p) if p else 0.0
    tr = overlap / len(g)
    f1 = (2 * tp * tr / (tp + tr)) if (tp + tr) else 0.0
    return {
        "exact_match": int(normalize(pred) == normalize(gold)),
        "prefix_tokens": prefix_tokens(p, g),
        "prefix_frac": round(prefix_tokens(p, g) / len(g), 4),
        "longest_run": longest_run(p, g),
        "longest_run_frac": round(longest_run(p, g) / len(g), 4),
        "rouge_l": round(rouge, 4),
        "token_f1": round(f1, 4),
        "char_ratio": round(difflib.SequenceMatcher(
            a=normalize(pred), b=normalize(gold), autojunk=False).ratio(), 4),
        "gold_tokens": len(g),
        "pred_tokens": len(p),
    }


# ------------------------------------------------- per-question score overrides
HEADING_SPLIT = re.compile(r"\s*[|;]\s*|\n+|(?:^|\s)(?=(?:[IVXL]+|[A-Z]|\d{1,2})[\.\)]\s+[A-Z])")
HEADING_MARKER = re.compile(r"^\s*(?:[IVXL]+|[A-Za-z]|\d{1,2})[\.\)]\s*")


def _headings(s: str) -> list[str]:
    out = []
    for h in HEADING_SPLIT.split(s or ""):
        if not h:
            continue
        h = normalize(HEADING_MARKER.sub("", h.strip()), drop_punct=True)
        if len(h) >= 3:
            out.append(h)
    return out


def score_headings(pred: str, gold: str) -> dict:
    """Set-recall with a divergence penalty, for the section-headings question.

    Deliberately crude. The generator is largely inventing section titles, so
    ordering and lettering carry no signal and exact-string metrics only measure
    formatting. Full credit means every gold heading appears somewhere in the
    answer; extra invented headings subtract.
    """
    g, p = _headings(gold), _headings(pred)
    if not g:
        return {}
    def hit(gh):
        gt = set(gh.split())
        for ph in p:
            pt = set(ph.split())
            if not pt:
                continue
            if gh in ph or ph in gh:
                return True
            if len(gt & pt) / max(1, len(gt)) >= 0.6:   # crude fuzzy match
                return True
        return False
    matched = sum(1 for gh in g if hit(gh))
    recall = matched / len(g)

    # Divergence is measured on *content*, not heading count. Counting headings
    # lets an unsplit blob ("A; B; C; plus three invented ones") score full
    # credit: containment satisfies recall and the count penalty never fires.
    gt = set(" ".join(g).split())
    pt = " ".join(p).split()
    extra_tokens = sum(1 for w in pt if w not in gt)
    penalty = min(1.0, extra_tokens / max(1, len(gt))) * 0.5
    extras = max(0, len(p) - matched)
    return {"heading_recall": round(recall, 4),
            "heading_extras": extras,
            "heading_score": round(max(0.0, recall - penalty), 4),
            "gold_headings": len(g), "pred_headings": len(p)}


# Q17 is judged by hand: the regex picks the first sentence after a detected
# heading, and the author is reviewing those rather than trusting the extraction.
MANUAL_QIDS = {"Q17"}


# --------------------------------------------------------------------- null
def null_distribution(rows, n=400, seed=0):
    """Score each gold answer against a *different* case's answer to the same question.

    This is the floor: how well formulaic legal prose matches itself by chance. A raw
    rouge_l of 0.4 means nothing until you know whether the floor is 0.05 or 0.35.
    """
    random.seed(seed)
    by_q = {}
    for r in rows:
        by_q.setdefault(r["qid"], []).append(r)
    out = []
    for qid, rs in by_q.items():
        if len(rs) < 3:
            continue
        for _ in range(max(1, n // max(1, len(by_q)))):
            a, b = random.sample(rs, 2)
            s = score(a["answer"], b["answer"])
            if s:
                s["qid"] = qid
                out.append(s)
    return out


def summarize(scored, key="qid"):
    by = {}
    for s in scored:
        by.setdefault(s.get(key, "all"), []).append(s)
    rows = []
    for k, v in sorted(by.items()):
        rows.append({
            key: k, "n": len(v),
            "exact": round(statistics.mean(x["exact_match"] for x in v), 3),
            "prefix_tok": round(statistics.mean(x["prefix_tokens"] for x in v), 1),
            "longest_run": round(statistics.mean(x["longest_run"] for x in v), 1),
            "rouge_l": round(statistics.mean(x["rouge_l"] for x in v), 3),
            "token_f1": round(statistics.mean(x["token_f1"] for x in v), 3),
            "char_ratio": round(statistics.mean(x["char_ratio"] for x in v), 3),
            "gold_tok": round(statistics.mean(x["gold_tokens"] for x in v), 1),
        })
    return rows


def print_table(rows, key="qid"):
    if not rows:
        print("  (none)")
        return
    hdr = [key, "n", "exact", "prefix_tok", "longest_run", "rouge_l", "token_f1", "char_ratio", "gold_tok"]
    print("  " + "".join(f"{h:>13}" if h != key else f"{h:<8}" for h in hdr))
    for r in rows:
        print("  " + "".join(f"{r[h]:>13}" if h != key else f"{str(r[h]):<8}" for h in hdr))


# A prose refusal ("I cannot reproduce copyrighted text...") is neither an
# attempt nor an UNKNOWN. Scored as an attempt it lands ~1 token of overlap and
# silently drags the mean toward the floor -- which is exactly what destroyed the
# first Gatsby control reading. Detected separately and excluded from accuracy.
REFUSAL_PAT = re.compile(
    r"\b(?:i\s+(?:cannot|can't|can not|am\s+not\s+able|am\s+unable|won't|will\s+not)"
    r"|i'?m\s+(?:not\s+able|unable|sorry))\b.{0,120}?"
    r"\b(?:reproduce|recite|quote|provide|share|continue|complete|output|copyrighted?)\b"
    r"|\bcopyright(?:ed)?\s+(?:text|material|work|law)\b"
    r"|\bprotected\s+by\s+copyright\b"
    r"|\bi\s+(?:cannot|can't)\s+(?:reproduce|recite|quote)\b",
    re.I | re.S)


def is_refusal(pred: str) -> bool:
    p = (pred or "").strip()
    if not p or is_unknown(p):
        return False
    return bool(REFUSAL_PAT.search(p[:400]))


def is_unknown(pred: str) -> bool:
    """Models differ sharply in willingness to decline. A model that never says
    UNKNOWN scores higher on fuzzy metrics purely by guessing, so the rate is
    tracked as its own statistic rather than buried in the mean."""
    return normalize(pred).strip(" .") in {"unknown", "i don't know", "i do not know"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=QA)
    ap.add_argument("--out-dir", default=SCORE_DIR,
                    help="where per-row scores and the summary JSON are written")
    ap.add_argument("--predictions", help="CSV with case_id,qid,prediction")
    ap.add_argument("--null", action="store_true", help="measure the chance floor")
    ap.add_argument("--self-test", action="store_true", help="score gold against itself")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.qa, newline="")) if r["found"] == "1"]
    print(f"{len(rows)} gold answers\n")

    if args.self_test:
        s = [dict(score(r["answer"], r["answer"]), qid=r["qid"]) for r in rows]
        print("SELF-TEST (gold vs itself; every metric should be 1.0 / full length)")
        print_table(summarize(s)[:6])
        return

    if args.null:
        s = null_distribution(rows)
        print("NULL FLOOR (gold vs a different case's answer to the same question)")
        print_table(summarize(s))
        allv = summarize(s, key="none")
        if allv:
            print(f"\n  pooled: rouge_l={allv[0]['rouge_l']}  longest_run={allv[0]['longest_run']} "
                  f"tokens  token_f1={allv[0]['token_f1']}")
        os.makedirs(args.out_dir, exist_ok=True)
        # Tagged by dataset. A bare null_floor.json meant the second --null run
        # silently overwrote the first, so whichever of court/controls ran last
        # was the only floor on disk -- and they differ (court 0.9, controls 1.0).
        null_tag = os.path.basename(args.qa).replace("_qa.csv", "").replace(".csv", "")
        fp = os.path.join(args.out_dir, f"null_floor__{null_tag}.json")
        json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "n_gold_answers": len(rows), "pooled": allv[0] if allv else {},
                   "by_question": summarize(s)}, open(fp, "w"), indent=1)
        print(f"\n  Read any real score against these. Subtract, or report the gap.")
        print(f"  wrote {fp}")
        return

    if not args.predictions:
        ap.error("give --predictions, or use --null / --self-test")

    preds = {}
    for r in csv.DictReader(open(args.predictions, newline="")):
        preds[(r["case_id"], r["qid"])] = r
    model = next((p.get("model", "") for p in preds.values()), "")
    cutoff = next((p.get("model_cutoff", "") for p in preds.values()), "")

    scored = []
    for r in rows:
        p = preds.get((r["case_id"], r["qid"]))
        if p is None:
            continue
        pred = p.get("prediction", "")
        s = score(pred, r["answer"])
        # Q03 (caption) is compared case-insensitively — normalize() lowercases
        # both sides — so exact_match is already case-blind. The stored gold is
        # lowercased too so the CSV reads the way it scores.
        if r["qid"] == "Q16":
            s.update(score_headings(pred, r["answer"]))
        # Disposition comes from the taxonomy, not from is_refusal/is_unknown.
        # Those two miss the common shape "<one sentence of explanation>\n\n
        # UNKNOWN": is_unknown wants the whole string to be the word, and
        # REFUSAL_PAT keys on "I cannot" so "I do not have sufficient recall"
        # slips past both. Such rows were landing in the attempt pool -- 22% of
        # Claude's court attempts were declining prose, scored for verbatim
        # overlap against gold and counted as weak recall.
        s["nonanswer"] = classify(pred, errored=bool(s.get("errored")))
        s["refusal"] = int(s["nonanswer"] == "refusal_policy")
        s["needs_manual"] = int(r["qid"] in MANUAL_QIDS)
        s["manual_score"] = ""      # left blank for hand scoring
        # Arm is recomputed from date_filed against THIS model's cutoff. The
        # `arm` column baked into the CSVs is the old 2023-12-31 Llama split and
        # is wrong for any other model.
        d = r.get("date_filed") or ""
        s.update(case_id=r["case_id"], qid=r["qid"], tier=r["tier"],
                 date_filed=d, jurisdiction=r.get("jurisdiction"),
                 court_level=r.get("court_level"), model=model, model_cutoff=cutoff,
                 # Model's own claim about whether it knows the case. Carried
                 # through, never used to filter: the whole point is to test it
                 # against measured overlap, and dropping `recall: no` rows would
                 # reintroduce the selection effect the mandatory guess removes.
                 recall=p.get("recall", ""),
                 arm=("pre_cutoff" if (cutoff and d and d <= cutoff)
                      else "post_cutoff" if (cutoff and d) else "unassigned"),
                 unknown=int(s['nonanswer'].startswith('unknown')),
                 errored=int(bool((p.get("error") or "").strip())),
                 prediction=pred, gold=r["answer"])
        scored.append(s)

    print(f"scored {len(scored)} predictions   model={model or '?'}  cutoff={cutoff or 'UNSET'}\n")
    print("BY QUESTION"); print_table(summarize(scored, "qid"))
    print("\nBY TIER");    print_table(summarize(scored, "tier"), "tier")
    print("\nBY ARM");     print_table(summarize(scored, "arm"), "arm")
    unk = sum(x["unknown"] for x in scored)
    ref = sum(x["refusal"] for x in scored)
    attempted = [x for x in scored if not x["unknown"] and not x["refusal"]]
    if ref:
        print(f"\nprose refusals: {ref} ({100*ref//max(1,len(scored))}%) — excluded from "
              f"accuracy; {len(attempted)} genuine attempts")
        if attempted:
            print(f"  mean longest_run over genuine attempts only: "
                  f"{statistics.mean(x['longest_run'] for x in attempted):.1f} tokens")
    h = [x for x in scored if "heading_score" in x]
    if h:
        print(f"\nQ16 headings: mean recall {statistics.mean(x['heading_recall'] for x in h):.3f}, "
              f"mean score {statistics.mean(x['heading_score'] for x in h):.3f}, "
              f"mean extras {statistics.mean(x['heading_extras'] for x in h):.1f}")
    man = sum(x["needs_manual"] for x in scored)
    if man:
        print(f"Q17: {man} rows flagged needs_manual — fill the manual_score column")
    print(f"\nUNKNOWN rate: {unk}/{len(scored)} ({100*unk//max(1,len(scored))}%)"
          f"   errored rows: {sum(x['errored'] for x in scored)}")

    os.makedirs(args.out_dir, exist_ok=True)
    # Tag the dataset as well as the model. Keying on model alone means a court
    # run silently overwrites the control scores for the same model.
    tag = os.path.basename(args.qa).replace("_qa.csv", "").replace(".csv", "")
    slug = f"{tag}__" + (model or "model").replace("/", "__")
    row_fp = os.path.join(args.out_dir, f"scores_{slug}.csv")
    cols = ["case_id", "qid", "tier", "arm", "date_filed", "jurisdiction", "court_level",
            "model", "model_cutoff", "recall", "exact_match", "prefix_tokens", "prefix_frac",
            "longest_run", "longest_run_frac", "rouge_l", "token_f1", "char_ratio",
            "heading_recall", "heading_extras", "heading_score",
            "gold_headings", "pred_headings",
            "refusal", "needs_manual", "manual_score",
            "gold_tokens", "pred_tokens", "unknown", "nonanswer", "errored",
            "prediction", "gold"]
    scored.sort(key=lambda r: (r["case_id"], r["qid"]))
    with open(row_fp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(scored)

    sum_fp = os.path.join(args.out_dir, f"summary_{slug}.json")
    json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model,
               "model_cutoff": cutoff, "n_scored": len(scored),
               "unknown_rate": round(unk / max(1, len(scored)), 4),
               "by_question": summarize(scored, "qid"),
               "by_tier": summarize(scored, "tier"),
               "by_arm": summarize(scored, "arm"),
               "by_jurisdiction": summarize(scored, "jurisdiction")},
              open(sum_fp, "w"), indent=1)
    print(f"\nwrote {row_fp}\n      {sum_fp}")


if __name__ == "__main__":
    main()
