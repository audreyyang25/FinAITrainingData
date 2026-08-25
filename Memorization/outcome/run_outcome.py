#!/usr/bin/env python3
"""Ask each target model what happened in each case. Outcome *and* reasoning.

  python Memorization/outcome/run_outcome.py --model openai/gpt-5 --workers 8
  python Memorization/outcome/run_outcome.py --all --workers 8
  python Memorization/outcome/run_outcome.py --model ... --dry-run

This measures something different from the memorization probe. Reproducing an
opinion's wording is verbatim recall; knowing that the Ninth Circuit affirmed
dismissal on loss-causation grounds is semantic knowledge, obtainable from a law
review, a news summary, or a docket entry without ever seeing the opinion. The
two can dissociate, and the point of running both is to see whether they do.

Why the prompt asks for reasoning as well as outcome: federal appellate courts
affirm roughly 75-80% of the time, so "affirmed" alone is guessable and a model
that always says it scores high while knowing nothing. The court's *rationale*
is not guessable from a base rate. Outcome and reasoning are scored separately
downstream for the same reason.

Web search is off, enforced in the provider layer. Temperature is 0.
"""
from __future__ import annotations
import argparse, collections, csv, datetime, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "probe"))
import providers
from run_probe import load_meta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (TARGETS, CUTOFF, arm, common_arm, COMMON_PRE_MAX,
                    COMMON_POST_MIN, FED_APPEAL_DIR)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
QA = os.path.join(BASE, "datasets", "court_opinions_qa_v2.csv")
OUT_DIR = os.path.join(BASE, "datasets", "outcomes")
PREDICT_DIR = os.path.join(BASE, "datasets", "outcomes_predict")

SYSTEM = (
    "You are being tested on your knowledge of specific published US court opinions. "
    "Answer only from memory. Do not browse, search, or use tools.\n\n"
    "Give two things, briefly:\n"
    "1. THE OUTCOME — the court's disposition (affirmed, reversed, vacated, remanded, "
    "dismissed, or a mix) and which party prevailed.\n"
    "2. THE REASONING — the court's principal rationale for that disposition: the legal "
    "question it turned on and why it came out the way it did.\n\n"
    "Describe what the court actually held, not what it plausibly might have held. "
    "Do not pad the answer with general background about the area of law.\n\n"
    "If you do not know this specific case, you may reply with: UNKNOWN\n"
    "If you are unsure, you can state your lack of certainty but still give it your best attempt."
)

# --predict swaps the escape hatch for a forecast. Built from SYSTEM by dropping
# its last paragraph, so the task definition, the two-part output shape and the
# "what the court actually held" instruction stay byte-identical -- only the
# instruction about not knowing changes, which is the whole manipulation.
#
# Paired with --arm post_cutoff this measures something the default run cannot:
# every case is provably outside training data, so a score here is not degraded
# recall, it is prediction from the caption, court, posture and base rates. US
# appellate courts affirm 75-80% of the time, so the OUTCOME half has a high
# guessable floor -- the REASONING half is where a real signal would show, which
# is why the judge scores them separately.
SYSTEM_PREDICT = SYSTEM.rsplit("\n\n", 1)[0] + (
    "\n\nYou will not recognize many of these cases: some were decided after your "
    "training data ends. Do not reply UNKNOWN and do not decline. If you do not "
    "remember the case, or doubt it exists, predict what the court would most likely "
    "have held and why — reason from the caption, the court, the procedural posture, "
    "and how cases of this kind usually come out. Give the same two things in the "
    "same form either way."
)


def build_user(case_id: str, m: dict) -> str:
    """Identify the case precisely enough that a wrong answer means ignorance
    rather than a name collision. `SEC v. Smith` is not unique; caption plus
    court plus year plus docket is."""
    cap = m.get("caption") or case_id
    bits = [f"Case: {cap}"]
    if m.get("court"):
        bits.append(f"Court: {m['court']}")
    if m.get("date"):
        bits.append(f"Decided: {m['date']}")
    if m.get("docket"):
        bits.append(f"Docket: {m['docket']}")
    return "\n".join(bits) + "\n\nWhat was the outcome of this case, and what was the court's reasoning?"


COLS = ["case_id", "model", "model_cutoff", "arm", "date_filed", "caption",
        "court_level", "prediction", "refused", "finish", "error", "temp_dropped",
        "latency_s", "in_tokens", "out_tokens", "served_by", "ts"]


def run_one(model: str, cases: list, meta: dict, args) -> None:
    slug = model.replace("/", "__").replace(":", "_")
    # Arm is filtered HERE, not in main(), because the cutoffs differ per model:
    # a case filed 2026-01-15 is post-cutoff for GPT-5 and pre-cutoff for Fable 5.
    # Filtering once against a single cutoff would silently mix the arms.
    if args.arm:
        cases = [c for c in cases if arm(c["date_filed"], model) == args.arm]
    # Predictions land in their own directory so the judge's default glob keeps
    # the two experiments apart -- one judged CSV per experiment, never mixed.
    out_dir = PREDICT_DIR if args.predict else OUT_DIR
    prefix = "PREDICT__" if args.predict else ""
    out_path = args.out or os.path.join(out_dir, f"{prefix}{slug}.csv")
    os.makedirs(out_dir, exist_ok=True)

    done = set()
    if os.path.exists(out_path) and not args.force:
        with open(out_path, newline="") as fh:
            prior = list(csv.DictReader(fh))
        keep = [r for r in prior if not (r.get("error") or "").strip()]
        if len(keep) != len(prior):
            with open(out_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
                w.writeheader(); w.writerows(keep)
        done = {r["case_id"] for r in keep}
    todo = [c for c in cases if c["case_id"] not in done]

    print(f"\n=== {model}   cutoff {CUTOFF.get(model,'?')}"
          f"{'   [PREDICT]' if args.predict else ''}"
          f"{f'   arm={args.arm}' if args.arm else ''}")
    print(f"    {len(cases)} cases, {len(done)} done, {len(todo)} to query -> {out_path}")
    if args.dry_run:
        for c in todo[:2]:
            print("-" * 68)
            print(build_user(c["case_id"], meta.get(c["case_id"], {})))
        return

    mode = "w" if args.force else "a"
    new = args.force or not os.path.exists(out_path)
    fh = open(out_path, mode, newline="")
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    if new:
        w.writeheader()
    lock, n_ok, n_err = threading.Lock(), 0, 0

    def work(c):
        nonlocal n_ok, n_err
        cid = c["case_id"]
        m = meta.get(cid, {})
        res = providers.call(model, SYSTEM_PREDICT if args.predict else SYSTEM,
                             build_user(cid, m),
                             max_tokens=args.max_tokens, temperature=0.0,
                             reasoning={"enabled": False} if args.reasoning_off else None)
        row = dict(case_id=cid, model=model, model_cutoff=CUTOFF.get(model, ""),
                   arm=arm(c["date_filed"], model), date_filed=c["date_filed"],
                   caption=m.get("caption", ""), court_level=c["court_level"],
                   prediction=res.text, refused=int(res.refused), finish=res.finish,
                   error=res.error, temp_dropped=int(res.temp_dropped),
                   latency_s=round(res.latency_s, 2), in_tokens=res.in_tokens,
                   out_tokens=res.out_tokens, served_by=res.served_by,
                   ts=datetime.datetime.now().isoformat(timespec="seconds"))
        with lock:
            w.writerow(row); fh.flush()
            if res.error:
                n_err += 1
            else:
                n_ok += 1
            if (n_ok + n_err) % 20 == 0:
                print(f"    {n_ok+n_err}/{len(todo)}  ok={n_ok} err={n_err}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"    done: ok={n_ok} errors={n_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="one OpenRouter model id")
    ap.add_argument("--all", action="store_true", help="every model in models.TARGETS")
    ap.add_argument("--qa", default=QA)
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    # 4096, not 1024. GPT-5 emits reasoning tokens that count against max_tokens
    # even with reasoning disabled, and at 1024 it spent the entire budget before
    # producing any visible text -- every row came back `finish=length` and blank.
    # An answer here is ~150-300 tokens; the rest is headroom for that overhead.
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--reasoning-off", action="store_true", default=True,
                    help="disable test-time reasoning (default; keeps budgets equal "
                         "across models, which is otherwise a confound)")
    ap.add_argument("--predict", action="store_true",
                    help="swap the UNKNOWN escape hatch for a forecast; writes to\n"
                         "datasets/outcomes_predict/PREDICT__<model>.csv")
    ap.add_argument("--arm", choices=["pre_cutoff", "post_cutoff"],
                    help="restrict to one arm, resolved per model against that\n"
                         "model's own cutoff")
    ap.add_argument("--common-arm", choices=["pre", "post", "both"],
                    help="use the three-model COMMON arm split instead of a per-model\n"
                         "one: pre means before every cutoff in models.NEWER, post means\n"
                         "after all of them, and cases between the two are dropped as\n"
                         "ambiguous. Makes pre/post deltas comparable across models.")
    ap.add_argument("--fed-appeal", action="store_true",
                    help="take the case list from the federal courts of appeals corpus\n"
                         "(datasets/fed_appeal_court_opinions/_index.csv) instead of the\n"
                         "QA CSV. Court level stops being a second variable alongside the\n"
                         "arm, and newly crawled cases are picked up without a QA rebuild.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.model and not args.all:
        ap.error("pass --model or --all")

    # WHERE THE CASE LIST COMES FROM.
    # The QA CSV is built for the verbatim-memorization probe, where the rows are
    # question spans. This experiment needs none of that -- only a case id, a
    # filing date and a court level, with captions and dockets coming from the
    # manifest via load_meta(). Sourcing from the corpus index instead means the
    # outcome probe does not depend on a QA rebuild to see newly crawled cases,
    # and nothing here can force a regeneration of files other experiments were
    # built against.
    seen, cases = set(), []
    if args.fed_appeal:
        src = os.path.join(BASE, "datasets", FED_APPEAL_DIR, "_index.csv")
        rows = list(csv.DictReader(open(src, newline="")))
        print(f"cases from {FED_APPEAL_DIR}/_index.csv: {len(rows)}")
    else:
        src = args.qa
        rows = list(csv.DictReader(open(src, newline="")))
    for r in rows:
        if r["case_id"] in seen:
            continue
        seen.add(r["case_id"])
        cases.append({"case_id": r["case_id"], "date_filed": r.get("date_filed", ""),
                      "court_level": r.get("court_level", "")})
    # --- the common-arm design, applied before anything is paid for ----------
    # Both filters drop cases, so both run here rather than at analysis time:
    # a case excluded after generation and judging has already cost ~$0.75, and
    # one excluded only at analysis time tends to get silently re-included the
    # next time someone writes a fresh script against the judged CSV.
    if args.common_arm:
        buckets = collections.Counter()
        kept = []
        for c in cases:
            a = common_arm(c["date_filed"])
            buckets[a or "ambiguous"] += 1
            if a and (args.common_arm == "both" or a == f"{args.common_arm}_cutoff"):
                c["common_arm"] = a
                kept.append(c)
        print(f"common-arm filter ({args.common_arm}): {dict(buckets)} "
              f"-> {len(kept)} cases")
        print(f"  pre <= {COMMON_PRE_MAX}, post > {COMMON_POST_MIN}; "
              f"{buckets['ambiguous']} ambiguous case(s) excluded by design")
        cases = kept

    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        sys.exit("no cases left after filtering — check --common-arm / --fed-appeal")
    meta = load_meta()

    models = [m for m, _, _ in TARGETS] if args.all else [args.model]
    print(f"{len(cases)} cases x {len(models)} model(s)")
    for m in models:
        run_one(m, cases, meta, args)


if __name__ == "__main__":
    main()
