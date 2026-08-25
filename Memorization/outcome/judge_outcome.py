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
import argparse, collections, csv, datetime, glob, json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "probe"))
import providers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import JUDGE, LABEL, FED_APPEAL_DIR

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
 "disposition_actual": "<affirmed|reversed|vacated|remanded|dismissed|granted|denied|judgment|mixed>",
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


# --- stable ground truth ----------------------------------------------------
# WHY THIS PASS EXISTS.
# `disposition_actual` is a fact about the opinion, but the grading prompt
# re-derives it on every call -- six times per case, each time in the presence
# of a different candidate answer. Measured on the pre-cutoff run, 16 of 136
# cases (12%) came back with DIFFERENT actual labels depending on which model
# was being graded (`mixed` vs `other`, `vacated` vs `other`, `affirmed` vs
# blank). That is a contamination path: ground truth must not be able to drift
# toward the answer being scored.
#
# So determine it once per case, from the opinion alone, with no candidate in
# the prompt. 136 calls instead of 576 -- cheaper than what it replaces.
#
# TAXONOMY v1 kept `mixed` as one bucket, on the grounds that splitting it was a
# stricter bar and the single label kept the numbers comparable to figures
# already built. v2 below reverses that call.
#
# TAXONOMY v2, 2026-08-25. The corpus is now federal courts of appeals only, so
# the trial-court labels (`granted`, `judgment`) no longer carry their weight,
# and the single `mixed` bucket was doing too much: it held 33 of the 86 labelled
# federal appellate cases -- the second-largest class -- while telling you
# nothing about WHICH combination occurred. "Affirmed in part, reversed in part"
# and "reversed and remanded" are different outcomes with different base rates,
# and collapsing them made the largest error class in figure o3 uninterpretable.
#
# v2 splits it into the three combinations that actually occur in appellate
# practice and keeps `other` as a genuine catch-all rather than a hidden mixed.
#
# NOTE: `granted` is deliberately absent while `denied` is kept, per the study
# design. A granted petition (mandamus, rehearing, petition for review) therefore
# falls to `other`. Watch the `other` count -- if it is large and mostly grants,
# the taxonomy needs a `granted` label back.
TRUTH_LABELS = ["affirmed", "reversed", "vacated", "remanded", "dismissed",
                "denied", "affirmed_and_reversed", "reversed_and_remanded",
                "vacated_and_remanded", "other"]

SYSTEM_TRUTH = """You are reading a US court opinion and recording its disposition.

You will be given the FULL TEXT of an opinion. Report what THIS court did to the
judgment or matter before it -- not what the court below did, and not what any
cited case did. Read the mandate, which is usually the final paragraph.

Choose exactly one label, and choose the MOST SPECIFIC one that is accurate.
The compound labels are not a last resort: if the court both reversed and
remanded, the answer is reversed_and_remanded, NOT reversed.

  affirmed               upheld the decision below in full, and did nothing else
  reversed               overturned it in full, without remanding
  vacated                set it aside in full, without remanding
  remanded               sent it back without otherwise disturbing the decision
  dismissed              disposed of the appeal or action without reaching the
                         merits (including dismissal for want of jurisdiction)
  denied                 denied the petition, motion or relief sought
  affirmed_and_reversed  affirmed as to some parts and reversed or vacated as to
                         others ("affirmed in part, reversed in part"). Use this
                         even if the court also remanded.
  reversed_and_remanded  reversed and sent the case back for further proceedings
  vacated_and_remanded   vacated and sent the case back for further proceedings
  other                  none of the above fits -- for example the court GRANTED
                         a petition or the relief sought, certified a question,
                         or entered some disposition not listed here

Rules for choosing between them:
  * Prefer a compound label over a simple one whenever both halves apply.
  * If the court affirmed some parts and reversed or vacated others, use
    affirmed_and_reversed regardless of whether it also remanded -- the mixed
    direction is the more informative fact.
  * Reverse/vacate plus remand is reversed_and_remanded / vacated_and_remanded,
    never plain remanded.
  * Use other only when nothing above is accurate. Do not use it for a
    disposition you are merely unsure how to phrase.

Reply with ONLY a JSON object, no prose and no code fence:
{"disposition_actual": "<affirmed|reversed|vacated|remanded|dismissed|denied|affirmed_and_reversed|reversed_and_remanded|vacated_and_remanded|other>",
 "prevailing_party": "<short phrase, or unclear>",
 "justification": "<one sentence quoting or paraphrasing the mandate>"}"""

TRUTH_COLS = ["case_id", "disposition_actual", "prevailing_party",
              "justification", "judge", "parse_error", "error", "ts"]


def build_truth_user(opinion: str) -> str:
    return (f"=== FULL OPINION TEXT ===\n{opinion}\n\n"
            f"What was this court's disposition? JSON only.")


# --- regrading against a fixed disposition ----------------------------------
# The OUTCOME anchors are phrased "disposition correct / partly correct", so the
# judge has to commit to a disposition in order to score. That means a score
# carries the judge's own reading of the label, and when the stable truth pass
# disagreed, the two stopped matching: on the 38 attempted rows whose label
# moved, corr(disposition hit, outcome_score) is +0.435 against the OLD label
# and -0.393 against the new one. The sign flip is the contamination, measured.
#
# Fix: supply the established disposition and grade against it, so the judge is
# scoring the candidate rather than re-litigating ground truth. Used only for
# the affected rows -- regrading all 815 would cost ~$126 to change nothing on
# the ~90% whose labels never moved.
SYSTEM_REGRADE = SYSTEM.replace(
    "You will be given the FULL TEXT of an opinion and a CANDIDATE ANSWER describing\n"
    "its outcome and reasoning. The opinion is the ground truth. Grade the candidate\n"
    "against it.",
    "You will be given the FULL TEXT of an opinion, the court's ESTABLISHED\n"
    "DISPOSITION (already determined from the opinion in a separate reading; treat\n"
    "it as correct and do not re-derive it), and a CANDIDATE ANSWER describing the\n"
    "outcome and reasoning. Grade the candidate against the opinion, scoring the\n"
    "OUTCOME half relative to the established disposition.")


def build_regrade_user(opinion: str, answer: str, disposition: str) -> str:
    return (f"=== FULL OPINION TEXT ===\n{opinion}\n\n"
            f"=== ESTABLISHED DISPOSITION ===\n{disposition}\n\n"
            f"=== CANDIDATE ANSWER ===\n{answer}\n\n"
            f"Grade the candidate answer. JSON only.")


def parse(txt: str) -> tuple[dict, str]:
    m = JSON_RE.search(txt or "")
    if not m:
        return {}, "no JSON found"
    try:
        d = json.loads(m.group(0))
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"[:120]
    return d, ""


def run_truth_pass(args):
    """One judge call per case, opinion only. Order-independent and idempotent."""
    judge_slug = args.judge.replace("/", "__").replace(":", "_")
    out_path = os.path.join(OUT_DIR, f"truth__{judge_slug}{args.out_suffix}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.fed_appeal:
        # Label the CORPUS, not the probed set. Ground truth is a property of the
        # opinion, so it can be established before any model has been asked --
        # which is the point here: knowing the disposition mix of each arm is how
        # you find out whether the arms are comparable, and that has to be
        # answerable before the expensive generation run, not after.
        idx = os.path.join(DS, FED_APPEAL_DIR, "_index.csv")
        rows = list(csv.DictReader(open(idx, newline="")))
        case_ids = [r["case_id"] for r in rows]
        print(f"cases from {FED_APPEAL_DIR}/_index.csv: {len(case_ids)}")
    else:
        # Case list comes from the predictions, so truth covers exactly the cases
        # that were actually probed -- no more, no fewer.
        pred_dir = args.pred_dir or PRED_DIR
        case_ids = []
        for f in sorted(glob.glob(os.path.join(pred_dir, "*.csv"))):
            for r in csv.DictReader(open(f, newline="")):
                if r["case_id"] not in case_ids:
                    case_ids.append(r["case_id"])

    done = set()
    if os.path.exists(out_path) and not args.force:
        prior = [r for r in csv.DictReader(open(out_path, newline=""))
                 if not (r.get("error") or "").strip()]
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=TRUTH_COLS, extrasaction="ignore")
            w.writeheader(); w.writerows(prior)
        done = {r["case_id"] for r in prior}
    todo = [c for c in case_ids if c not in done]
    print(f"truth pass: {len(case_ids)} cases, {len(done)} done, {len(todo)} to judge")
    print(f"  judge {args.judge} -> {out_path}")
    if args.dry_run:
        print(build_truth_user(load_text(todo[0]))[:600] if todo else "(nothing to do)")
        return

    mode = "a" if done else "w"
    fh = open(out_path, mode, newline="")
    w = csv.DictWriter(fh, fieldnames=TRUTH_COLS, extrasaction="ignore")
    if not done:
        w.writeheader()
    lock, n = threading.Lock(), [0]

    def work(cid):
        txt = load_text(cid)
        if not txt:
            row = dict(case_id=cid, error="no opinion text on disk")
        else:
            res = providers.call(args.judge, SYSTEM_TRUTH, build_truth_user(txt),
                                 max_tokens=args.max_tokens, temperature=0.0,
                                 web_search=False)
            d, perr = ({}, res.error) if res.error else parse(res.text)
            row = dict(case_id=cid,
                       disposition_actual=(d.get("disposition_actual") or "").strip().lower(),
                       prevailing_party=d.get("prevailing_party", ""),
                       justification=(d.get("justification") or "")[:400],
                       parse_error="" if d else (perr or "")[:160], error=res.error[:200])
        row.update(judge=args.judge,
                   ts=datetime.datetime.now().isoformat(timespec="seconds"))
        with lock:
            w.writerow(row); fh.flush()
            n[0] += 1
            if n[0] % 20 == 0:
                print(f"    {n[0]}/{len(todo)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    rows = list(csv.DictReader(open(out_path, newline="")))
    print(f"  done: {len(rows)} cases")
    print(f"  {dict(collections.Counter(r['disposition_actual'] for r in rows))}")


def judged_files(args):
    """The judged CSVs these maintenance passes act on.

    Both passes are destructive and both used to take every judged file in the
    directory. That is wrong once a second experiment exists: regrading is
    driven by the gap between a file and its .unstable baseline, and that gap
    does not close (apply_truth rewrites the label, regrade deliberately leaves
    it), so an already-regraded file is eligible forever and gets re-scored, and
    re-paid for, on every later run over a different experiment. --only scopes
    a pass to the experiment being worked on.
    """
    files = [f for f in sorted(glob.glob(os.path.join(OUT_DIR, "judged__*.csv")))
             if not f.endswith(".unstable.csv")]
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
        if not files:
            sys.exit(f"--only {args.only!r} matched no judged CSV in {OUT_DIR}")
    return files


def apply_truth(args):
    """Overwrite disposition_actual in judged CSVs from the stable truth table.

    Only the LABEL is replaced. outcome_score and reasoning_score are graded
    against the opinion's substance, not against the label, so they stand.
    """
    judge_slug = args.judge.replace("/", "__").replace(":", "_")
    tpath = (os.path.join(OUT_DIR, f"truth__{judge_slug}.csv")
             if args.apply_truth == "AUTO" else args.apply_truth)
    truth = {r["case_id"]: r["disposition_actual"]
             for r in csv.DictReader(open(tpath, newline=""))
             if r.get("disposition_actual")}
    print(f"truth table: {len(truth)} cases from {os.path.basename(tpath)}")

    for jp in judged_files(args):
        rows = list(csv.DictReader(open(jp, newline="")))
        cols = list(rows[0].keys())
        changed = miss = 0
        for r in rows:
            t = truth.get(r["case_id"])
            if t is None:
                miss += 1
                continue
            if r["disposition_actual"].strip().lower() != t:
                changed += 1
            r["disposition_actual"] = t
        bak = jp[:-4] + ".unstable.csv"
        if not os.path.exists(bak):
            os.rename(jp, bak)
        with open(jp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"  {os.path.basename(jp):52s} {len(rows):4d} rows, "
              f"{changed:3d} labels changed, {miss:3d} unmatched   (backup {os.path.basename(bak)})")


def regrade_changed(args):
    """Rescore only the attempted rows whose disposition_actual moved.

    Declined rows are skipped: the rubric scores them 0 regardless of label, so
    the disposition they were graded against cannot have mattered.
    """
    preds = {}
    for d in (PRED_DIR, os.path.join(DS, "outcomes_predict")):
        for f in glob.glob(os.path.join(d, "*.csv")):
            for r in csv.DictReader(open(f, newline="")):
                preds[(r["case_id"], r["model"])] = r.get("prediction", "")

    for jp in judged_files(args):
        bak = jp[:-4] + ".unstable.csv"
        if not os.path.exists(bak):
            print(f"  {os.path.basename(jp)}: no .unstable baseline, skipped")
            continue
        old = {(r["case_id"], r["model"]): r
               for r in csv.DictReader(open(bak, newline=""))}
        rows = list(csv.DictReader(open(jp, newline="")))
        cols = list(rows[0].keys())
        todo = [r for r in rows
                if not int(r.get("declined") or 0)
                and (r["case_id"], r["model"]) in old
                and old[(r["case_id"], r["model"])]["disposition_actual"].strip().lower()
                != r["disposition_actual"].strip().lower()]
        print(f"\n{os.path.basename(jp)}: {len(todo)} attempted rows to regrade")
        if args.dry_run or not todo:
            continue

        lock, n = threading.Lock(), [0]

        def work(r):
            txt = load_text(r["case_id"])
            ans = preds.get((r["case_id"], r["model"]), "")
            if not txt or not ans:
                with lock:
                    print(f"    skip {r['case_id'][:40]} (missing text or prediction)")
                return
            res = providers.call(args.judge, SYSTEM_REGRADE,
                                 build_regrade_user(txt, ans, r["disposition_actual"]),
                                 max_tokens=args.max_tokens, temperature=0.0,
                                 web_search=False)
            d, _ = ({}, None) if res.error else parse(res.text)
            try:
                os_, rs = float(d["outcome_score"]), float(d["reasoning_score"])
            except (KeyError, TypeError, ValueError):
                with lock:
                    print(f"    parse fail {r['case_id'][:40]} — left unchanged")
                return
            with lock:
                r["outcome_score"] = os_
                r["reasoning_score"] = rs
                r["combined"] = round((os_ + rs) / 2, 4)
                # disposition_actual stays as the truth pass set it; only what the
                # candidate claimed is re-read.
                r["disposition_claimed"] = d.get("disposition_claimed", r["disposition_claimed"])
                r["justification"] = (d.get("justification") or "")[:400]
                n[0] += 1

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        with open(jp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"  regraded {n[0]}/{len(todo)} rows -> {os.path.basename(jp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="append",
                    help="outcome CSV(s); repeatable. Default: all in datasets/outcomes/")
    ap.add_argument("--pred-dir",
                    help="directory to glob instead of datasets/outcomes -- use with "
                         "datasets/outcomes_predict for the --predict run. Keep the two "
                         "experiments in separate judged files via --out-suffix.")
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
    ap.add_argument("--regrade-changed", action="store_true",
                    help="rescore only the attempted rows whose disposition_actual "
                         "moved in --apply-truth, grading against the fixed label")
    ap.add_argument("--fed-appeal", action="store_true",
                    help="with --truth-only: label every case in the federal\n"
                         "appellate corpus index rather than only those already\n"
                         "probed, so arm balance can be checked before spending\n"
                         "anything on generation")
    ap.add_argument("--truth-only", action="store_true",
                    help="one pass per CASE, opinion only, no candidate answer -- "
                         "writes outcome_scores/truth__<judge>.csv")
    ap.add_argument("--apply-truth", metavar="TRUTH_CSV", nargs="?", const="AUTO",
                    help="rewrite disposition_actual in the judged CSVs from a "
                         "truth table; originals are backed up to *.unstable.csv")
    ap.add_argument("--only", metavar="SUBSTRING",
                    help="restrict --apply-truth / --regrade-changed to judged CSVs "
                         "whose filename contains SUBSTRING (e.g. '__predict'). Both "
                         "passes rewrite files in place and default to every judged "
                         "CSV present, which re-scores other experiments.")
    args = ap.parse_args()

    if args.truth_only:
        return run_truth_pass(args)
    if args.regrade_changed:
        return regrade_changed(args)
    if args.apply_truth:
        return apply_truth(args)

    pred_dir = args.pred_dir or PRED_DIR
    files = args.predictions or sorted(glob.glob(os.path.join(pred_dir, "*.csv")))
    files = [f for f in files if "__judged" not in os.path.basename(f)]
    if not files:
        sys.exit(f"no prediction files in {pred_dir} — run run_outcome.py first")

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
