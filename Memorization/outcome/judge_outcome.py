#!/usr/bin/env python3
"""Grade each target answer against the full opinion, with an LLM judge.

  python Memorization/outcome/judge_outcome.py --all --workers 6
  python Memorization/outcome/judge_outcome.py --all --sample 200 --web --out-suffix _web
  python Memorization/outcome/judge_outcome.py --all --judge google/gemini-3.1-pro-preview \
      --sample 200 --out-suffix _altjudge      # the self-preference check

Design notes:

  * **Anchored levels, not a free 0-1.** Judges asked for "a number between 0 and
    1" cluster at 0/0.5/0.8/1.0 and are unreliable between the modes. The rubric
    below gives five described anchors that happen to be numeric, which is a
    classification task rather than an estimation one.

  * **Outcome and reasoning scored separately.** A model that names the right
    disposition and invents the rationale is the exact signature of knowledge
    acquired from a summary rather than the opinion. A single blended score
    would hide it. They are combined afterwards, not by the judge.

  * **Web search is OFF by default**, and `--web` opts in. The judge already
    holds the full opinion, which *is* the ground truth for what the court held,
    so retrieval adds no information about the thing being measured -- but it
    does add three problems. Retrieval surfaces secondary summaries (Justia,
    Casetext, firm client alerts), which are plausibly the same sources a target
    model learned from; grading against those instead of the opinion produces
    spurious agreement that is strongest for well-covered cases, i.e. bias
    correlated with the variable under study. It can also surface subsequent
    history and penalize a candidate for correctly describing the opinion it was
    given. And it makes verdicts irreproducible as the web changes. Run `--web`
    on a subset as a sensitivity check rather than as the primary grade.

  * The judge never sees which model produced an answer, or its arm.
"""
from __future__ import annotations
import argparse, csv, datetime, glob, json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "probe"))
import providers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import JUDGE, LABEL

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
DS = os.path.join(BASE, "datasets")
# The complete set. `court_opinions/` covers only 105 of 136 cases and is missing
# the entire 2026 post-cutoff arm, which is the control this design depends on.
TEXT_DIR = os.path.join(DS, "court_opinions_courtlistener")
PRED_DIR = os.path.join(DS, "outcomes")
OUT_DIR = os.path.join(DS, "outcome_scores")

SYSTEM = """You are grading whether a description of a US court opinion is accurate.

You will be given the FULL TEXT of an opinion and a CANDIDATE ANSWER describing
its outcome and reasoning. The opinion is the ground truth. Grade the candidate
against it.

Score two things independently.

OUTCOME — did the candidate get the disposition and the prevailing party right?
  1.00  disposition and prevailing party both correct
  0.75  disposition correct; prevailing party unclear or partly wrong
  0.50  partially correct (e.g. right on some claims, wrong on others, or
        correct direction but wrong procedural posture)
  0.25  substantially wrong but touching something real in the case
  0.00  wrong, or no outcome stated

REASONING — did the candidate capture the court's actual rationale?
  1.00  identifies the legal question the case turned on and the court's reason
  0.75  broadly right rationale, missing a material element
  0.50  generic but not wrong (recites the area of law without the actual holding)
  0.25  mostly wrong; a plausible-sounding rationale the court did not give
  0.00  wrong, fabricated, or absent

Judge only against what the opinion says. A candidate is not credited for
statements that are true of the case generally but absent from the court's
actual holding. Do not reward fluency or hedging.

If the candidate declines to answer or says it does not know, set
"declined": true and score both 0.

Reply with ONLY a JSON object, no prose and no code fence:
{"outcome_score": <float>, "reasoning_score": <float>, "declined": <bool>,
 "disposition_actual": "<affirmed|reversed|vacated|remanded|dismissed|mixed|other>",
 "disposition_claimed": "<same vocabulary, or none>",
 "justification": "<one or two sentences>"}"""

JSON_RE = re.compile(r"\{.*\}", re.S)
COLS = ["case_id", "model", "arm", "date_filed", "court_level", "judge", "web",
        "outcome_score", "reasoning_score", "combined", "declined",
        "disposition_actual", "disposition_claimed", "justification",
        "parse_error", "error", "in_tokens", "out_tokens", "latency_s", "ts"]


def load_text(case_id: str) -> str:
    p = os.path.join(TEXT_DIR, f"{case_id}.txt")
    if not os.path.exists(p):
        return ""
    return open(p, encoding="utf-8", errors="ignore").read()


def build_user(opinion: str, answer: str) -> str:
    return (f"=== FULL OPINION TEXT ===\n{opinion}\n\n"
            f"=== CANDIDATE ANSWER ===\n{answer}\n\n"
            f"Grade the candidate answer against the opinion. JSON only.")


def parse(txt: str) -> tuple[dict, str]:
    m = JSON_RE.search(txt or "")
    if not m:
        return {}, "no JSON found"
    try:
        d = json.loads(m.group(0))
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"[:120]
    return d, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="append",
                    help="outcome CSV(s); repeatable. Default: all in datasets/outcomes/")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--web", action="store_true",
                    help="enable judge web search (sensitivity check only -- see "
                         "module docstring; the primary grade should be opinion-only)")
    ap.add_argument("--sample", type=int, help="grade a random N rows only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = args.predictions or sorted(glob.glob(os.path.join(PRED_DIR, "*.csv")))
    files = [f for f in files if "__judged" not in os.path.basename(f)]
    if not files:
        sys.exit(f"no prediction files in {PRED_DIR} — run run_outcome.py first")

    rows = []
    for f in files:
        for r in csv.DictReader(open(f, newline="")):
            if (r.get("error") or "").strip():
                continue
            rows.append(r)
    if args.sample and args.sample < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]

    judge_slug = args.judge.replace("/", "__").replace(":", "_")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"judged__{judge_slug}{args.out_suffix}.csv")

    done = set()
    if os.path.exists(out_path) and not args.force:
        with open(out_path, newline="") as fh:
            prior = [r for r in csv.DictReader(fh) if not (r.get("error") or "").strip()]
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader(); w.writerows(prior)
        done = {(r["case_id"], r["model"]) for r in prior}
    todo = [r for r in rows if (r["case_id"], r["model"]) not in done]

    missing = sorted({r["case_id"] for r in todo if not load_text(r["case_id"])})
    if missing:
        print(f"WARNING: {len(missing)} cases have no text and will be skipped: {missing[:4]}")
        todo = [r for r in todo if load_text(r["case_id"])]

    web = args.web
    print(f"judge     {args.judge}   web_search={'ON' if web else 'off'}")
    print(f"rows      {len(rows)} total, {len(done)} done, {len(todo)} to grade")
    print(f"out       {out_path}\n")
    if args.dry_run:
        r = todo[0]
        t = load_text(r["case_id"])
        print(f"[{r['case_id']} / {r['model']}]  opinion {len(t.split()):,} words")
        print(build_user(t[:600] + "\n...[truncated for preview]...", r["prediction"])[:1400])
        print(f"\n(dry run — {len(todo)} rows would be graded)")
        return

    mode = "w" if (args.force or not os.path.exists(out_path)) else "a"
    fh = open(out_path, mode, newline="")
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    if mode == "w":
        w.writeheader()
    lock, n, bad = threading.Lock(), 0, 0

    def work(r):
        nonlocal n, bad
        res = providers.call(args.judge, SYSTEM,
                             build_user(load_text(r["case_id"]), r["prediction"]),
                             max_tokens=args.max_tokens, temperature=0.0,
                             web_search=web)
        d, perr = ({}, res.error) if res.error else parse(res.text)
        os_, rs = d.get("outcome_score"), d.get("reasoning_score")
        try:
            os_, rs = float(os_), float(rs)
            comb = round((os_ + rs) / 2, 4)
        except (TypeError, ValueError):
            os_ = rs = comb = ""
        row = dict(case_id=r["case_id"], model=r["model"], arm=r.get("arm", ""),
                   date_filed=r.get("date_filed", ""), court_level=r.get("court_level", ""),
                   judge=args.judge, web=int(web),
                   outcome_score=os_, reasoning_score=rs, combined=comb,
                   declined=int(bool(d.get("declined"))),
                   disposition_actual=d.get("disposition_actual", ""),
                   disposition_claimed=d.get("disposition_claimed", ""),
                   justification=(d.get("justification") or "")[:400],
                   parse_error="" if d else (perr or "")[:160],
                   error=res.error[:200], in_tokens=res.in_tokens,
                   out_tokens=res.out_tokens, latency_s=round(res.latency_s, 2),
                   ts=datetime.datetime.now().isoformat(timespec="seconds"))
        with lock:
            w.writerow(row); fh.flush()
            n += 1
            if not d:
                bad += 1
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}  unparsed={bad}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"\ndone: {n} graded, {bad} unparsed -> {out_path}")


if __name__ == "__main__":
    main()
