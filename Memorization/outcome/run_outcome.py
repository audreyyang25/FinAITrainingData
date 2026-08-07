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
import argparse, csv, datetime, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "probe"))
import providers
from run_probe import load_meta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import TARGETS, CUTOFF, arm

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
QA = os.path.join(BASE, "datasets", "court_opinions_qa_v2.csv")
OUT_DIR = os.path.join(BASE, "datasets", "outcomes")

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
    out_path = args.out or os.path.join(OUT_DIR, f"{slug}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)

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

    print(f"\n=== {model}   cutoff {CUTOFF.get(model,'?')}")
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
        res = providers.call(model, SYSTEM, build_user(cid, m),
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
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.model and not args.all:
        ap.error("pass --model or --all")

    # One row per case, not per question: dedupe the QA file.
    seen, cases = set(), []
    for r in csv.DictReader(open(args.qa, newline="")):
        if r["case_id"] in seen:
            continue
        seen.add(r["case_id"])
        cases.append({"case_id": r["case_id"], "date_filed": r.get("date_filed", ""),
                      "court_level": r.get("court_level", "")})
    if args.limit:
        cases = cases[: args.limit]
    meta = load_meta()

    models = [m for m, _, _ in TARGETS] if args.all else [args.model]
    print(f"{len(cases)} cases x {len(models)} model(s)")
    for m in models:
        run_one(m, cases, meta, args)


if __name__ == "__main__":
    main()
