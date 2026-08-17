#!/usr/bin/env python3
"""Query closed-weight models with the court-opinion QA set.

Deliberately arm-blind: `arm` is read from the QA CSV and written straight to the
output, never branched on. Treatment (pre-cutoff) and control (post-cutoff) rows
take byte-identical code paths, which is what makes the comparison mean anything.

Writes predictions only. No scoring, no analysis -- score with
Memorization/qa/score_answers.py once you've decided how to analyze.

  export OPENROUTER_API_KEY=...
  python Memorization/probe/run_probe.py --model anthropic/claude-opus-4.1 --dry-run
  python Memorization/probe/run_probe.py --model openai/gpt-4o --limit 20
  python Memorization/probe/run_probe.py --model google/gemini-2.5-pro --cutoff 2025-01-31
"""
from __future__ import annotations
import argparse, csv, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import providers

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
QA = os.path.join(BASE, "datasets", "court_opinions_qa.csv")
OUT_DIR = os.path.join(BASE, "datasets", "predictions")

# Recall, not reasoning. No browsing, no hedging. There is NO escape hatch: the
# answer is compulsory, and a model that does not know the case says so in a
# separate RECALL field instead of withholding the guess. See the note below the
# prompt for why the guess has to stay mandatory.
SYSTEM = (
    "You are being tested on your verbatim recall of specific published texts. "
    "Answer only from memory. Do not browse, search, or use tools.\n\n"
    "When a question asks you to complete or continue a passage, reproduce the opinion's "
    "EXACT original wording — the same words, in the same order, with the same punctuation "
    "and capitalization. Do not paraphrase, summarize, modernize, correct, or improve the "
    "text. Do not write what the court plausibly might have said; reproduce only what you "
    "actually remember it saying. A partial verbatim answer is better than a complete "
    "paraphrase.\n\n"
    "Reply in exactly two lines, with no preamble, no explanation, and no surrounding "
    "quotation marks unless they appear in the text itself:\n\n"
    "ANSWER: <the continuation>\n"
    "SOURCE: recalled   (the words you just wrote are ones you have actually seen in "
    "this document)\n"
    "SOURCE: inferred   (you reconstructed them from how documents like this usually "
    "read)\n\n"
    "The ANSWER line is MANDATORY on every reply, including when SOURCE is inferred. "
    "Never reply UNKNOWN, never say you are unsure in the ANSWER line, and never explain "
    "that you cannot recall the text. If you do not remember the exact wording, give your "
    "single best guess at what the original words are — guess the actual wording rather "
    "than writing a paraphrase or a summary. SOURCE is a description of the answer you "
    "just gave, not a prediction about whether you can answer."
)

# WHY BOTH A SELF-REPORT AND A MANDATORY GUESS.
#
# The post-cutoff arm is the floor this study rests on: a model cannot have
# memorized an opinion filed after its training cutoff, so whatever it scores
# there is what verbatim text it can produce with zero knowledge of the case --
# mostly statutes and boilerplate QUOTED inside the opinion, which it learned
# elsewhere. That only works if EVERY post-cutoff row yields a scoreable guess.
#
# Giving the model an escape hatch destroys exactly that. It would decline
# precisely on post-cutoff cases, the arm would collapse to a self-selected
# handful (Claude Opus 4 answered 15 of 656 under the voluntary prompt), and the
# floor would stop being a floor. So the guess stays compulsory.
#
# Asking for the self-report separately recovers the abstention signal without
# paying that price, and turns it into a better variable: whether the report
# actually predicts higher verbatim overlap.
#
# WHY IT IS *AFTER* THE ANSWER, AND WHY IT ASKS ABOUT PROVENANCE.
# The first version put `RECALL: yes|no` first and asked whether the model
# "recognized this opinion and remembered its wording". GPT-5 answered `no` on
# 10 of 10 pre-cutoff rows -- including ones where it then reproduced a
# plausible 27-word continuation -- and its single `yes` was on a POST-cutoff
# case it cannot have seen. Two causes, both fixed here:
#   * ORDER. Asked before answering, the model reports a prior about whether it
#     will succeed, not an observation about what it produced. It has not tried
#     yet, so it cannot know. Moving the field after ANSWER makes it a posterior.
#   * FRAMING. "Do you remember the wording" is introspective, and models have
#     poor access to whether their own output came from memorization or from
#     fluent pattern completion; it also reads as an overclaim, which RLHF
#     pushes against, so `no` is the safe token regardless of capability.
#     `recalled` vs `inferred` asks about the provenance of text already on the
#     page -- a comparison between two concrete alternatives, not a confidence
#     confession.
# Column stays `recall` so score_answers and the scores CSV do not churn; the
# VALUES are now 'recalled' / 'inferred' / '' (format not followed).
RECALL_RE = re.compile(r"^\s*(?:SOURCE|RECALL)\s*:\s*(recalled|inferred|yes|no)\b",
                       re.I | re.M)
ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*", re.I | re.M)
_ALIAS = {"yes": "recalled", "no": "inferred"}   # tolerate the old vocabulary


def split_recall(pred: str) -> tuple[str, str]:
    """Raw completion -> (source_flag, answer_text).

    source_flag is 'recalled' / 'inferred' / '' when the model ignored the
    format. The ANSWER text is what gets scored; leaving the SOURCE line in
    would put its tokens into pred and corrupt char_ratio and token_f1. A reply
    that skips the header entirely is treated as all-answer rather than dropped
    -- format non-compliance is not the same as a non-answer.
    """
    p = pred or ""
    m = RECALL_RE.search(p)
    flag = _ALIAS.get(m.group(1).lower(), m.group(1).lower()) if m else ""
    a = ANSWER_RE.search(p)
    if not a:
        # No ANSWER: header. Everything except the SOURCE line is the answer.
        return flag, (RECALL_RE.sub("", p).strip() if m else p.strip())
    # SOURCE now trails ANSWER, so the answer ends where the SOURCE line begins.
    body = p[a.end():]
    m2 = RECALL_RE.search(body)
    return flag, (body[:m2.start()] if m2 else body).strip()


def load_meta() -> dict:
    """case_id -> caption/docket/court/year, from the crawl manifest.

    The QA CSV carries only identifiers, so without this the prompt would name
    the case by its filename slug ("cl_2014_ca2_city_of_pontiac_v._ubs_ag")
    rather than a caption a model could plausibly recognize.
    """
    import json
    man = json.load(open(os.path.join(BASE, "crawl_manifest", "case_ids.json")))
    out = {}
    for c in man["court_opinions"]["cases"]:
        # Captions come from CourtListener's caseName and are already correctly
        # cased. The old per-word capitalize() existed to tidy filename slugs and
        # now corrupts real names ("City Of Pontiac V. UBS Ag"), so it is gone.
        cap = (c.get("caption") or "").strip()
        out[c["case_id"]] = {"caption": cap, "docket": c.get("docket"),
                             "court": c.get("court_name") or c.get("court"),
                             "year": c.get("year"), "date": c.get("date_filed")}
    return out


# Q03 asks for the case caption, which is also how we identify the case to the
# model -- structurally unanswerable as a probe. Excluded by default.
LEAKED_QIDS = {"Q03"}


def build_user(row: dict, meta: dict) -> str:
    """Identify the case, then ask. Identical shape for every arm.

    The identification block deliberately carries caption + docket only. Adding
    court or year would hand Q01 and Q02 their own answers.
    """
    # The row's own caption wins: control sets (Gatsby) are not in the court
    # manifest, and the QA CSV now carries caption directly.
    m = meta.get(row["case_id"], {})
    cap = (m.get("caption") or "").strip() or (row.get("caption") or "").strip() or row["case_id"]
    bits = [f"Case: {cap}"]
    if m.get("docket"):
        bits.append(f"Docket No. {m['docket']}")
    head = "\n".join(bits)

    prompt = (row.get("prompt_normalized") or row.get("prompt") or "").strip()
    if prompt:
        # Control sets are books, not opinions. The court-opinion wording is left
        # byte-identical so runs already completed under it stay comparable.
        noun = "text" if row.get("jurisdiction") == "control" else "opinion"
        # State the target length. Without it GPT-5 answered a 26-word gold with
        # the single word "affirmed." on 5 of 14 smoke rows -- and a truncated
        # guess cannot produce a long verbatim run, so it deflates the
        # post-cutoff floor and flatters pre-cutoff recall by comparison. The
        # word COUNT is not a leak: it is a property of the excerpt's
        # continuation, not of its content, and it is what "guess at the
        # required length" was already asking for, just unenforceably.
        n = len(str(row.get("answer") or "").split())
        length = (f" The continuation is about {n} words long; produce roughly "
                  f"that many.") if n else ""
        return (f"{head}\n\n{row['question']}\n\n"
                f"The passage below is an exact excerpt from this {noun}. Continue it "
                f"word-for-word as the {noun} actually reads. Output only the "
                f"continuation — do not repeat the excerpt.{length}\n\n"
                f"EXCERPT: \"{prompt}\"")
    return f"{head}\n\n{row['question']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="OpenRouter model id, e.g. anthropic/claude-opus-4.1")
    ap.add_argument("--cutoff", default="",
                    help="this model's training cutoff (YYYY-MM-DD), recorded on every "
                         "row so arms can be assigned per model at analysis time")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="pass --temperature=-1 to omit it entirely")
    ap.add_argument("--reasoning-effort", default="off",
                    choices=["minimal", "low", "medium", "high", "off", "default"],
                    help="minimize test-time deliberation — a reasoning model can "
                         "reconstruct a plausible answer instead of recalling one. "
                         "'off' disables entirely; 'default' sends nothing.")
    ap.add_argument("--qa", default=QA)
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, help="cap rows (for a smoke run)")
    ap.add_argument("--qid", action="append", help="restrict to these question ids")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="reasoning models bill thinking against max_tokens; too low a "
                         "cap returns an EMPTY answer with the budget fully consumed")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="seconds between calls (per worker)")
    ap.add_argument("--no-sort", action="store_true",
                    help="leave rows in completion order instead of sorting by "
                         "(case_id, qid) at the end")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel requests. Calls are I/O-bound so this scales nearly "
                         "linearly; 4-8 is a sane range. Too many invites 429s from "
                         "OpenRouter — errored rows are retried on the next run, so an "
                         "over-aggressive setting costs time rather than data.")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, call nothing")
    ap.add_argument("--force", action="store_true", help="re-query rows already present")
    ap.add_argument("--keep-errors", action="store_true",
                    help="treat previously-errored rows as done instead of retrying them")
    ap.add_argument("--include-leaked", action="store_true",
                    help="also ask Q03, whose answer the identification block gives away")
    args = ap.parse_args()

    reasoning = (None if args.reasoning_effort == "default"
                 else {"enabled": False} if args.reasoning_effort == "off"
                 else {"effort": args.reasoning_effort})
    meta = load_meta()
    rows = [r for r in csv.DictReader(open(args.qa, newline="")) if r["found"] == "1"]
    if not args.include_leaked:
        rows = [r for r in rows if r["qid"] not in LEAKED_QIDS]
    if args.qid:
        rows = [r for r in rows if r["qid"] in args.qid]
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    slug = args.model.replace("/", "__").replace(":", "_")
    out_path = args.out or os.path.join(OUT_DIR, f"{slug}.csv")
    # Resume: keep completed rows, drop errored ones so they are re-queried.
    # A row that failed on a transient API error is not "done" -- leaving it in
    # the skip set would silently bake the failure into the results. The file is
    # rewritten without those rows first, so a retry does not append a duplicate
    # for the same (case_id, qid).
    done, retrying = set(), 0
    if os.path.exists(out_path) and not args.force:
        with open(out_path, newline="") as fh:
            prior = list(csv.DictReader(fh))
        keep = [r for r in prior if not (r.get("error") or "").strip()]
        retrying = len(prior) - len(keep)
        if retrying and not args.keep_errors:
            with open(out_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=prior[0].keys())
                w.writeheader()
                w.writerows(keep)
            done = {(r["case_id"], r["qid"]) for r in keep}
        else:
            done = {(r["case_id"], r["qid"]) for r in prior}
            retrying = 0
    todo = [r for r in rows if (r["case_id"], r["qid"]) not in done]

    system = SYSTEM

    print(f"model     {args.model}   (via OpenRouter)")
    print(f"cutoff    {args.cutoff or 'UNSET — pass --cutoff to record it'}")
    print(f"reasoning {args.reasoning_effort}   (minimize: recall, not deliberation)")
    print(f"workers   {args.workers}")
    print(f"rows      {len(rows)} total, {len(done)} already done, {len(todo)} to query"
          + (f"  ({retrying} previously errored — will retry)" if retrying else ""))
    print(f"out       {out_path}\n")

    if args.dry_run:
        for r in todo[:3]:
            print("=" * 72)
            print(f"[{r['case_id']} {r['qid']} arm={r['arm']}]")
            print(build_user(r, meta))
        print(f"\n(dry run — {len(todo)} rows would be queried)")
        return

    cols = ["case_id", "qid", "arm", "jurisdiction", "court_level", "tier",
            "model", "model_cutoff", "served_by", "recall", "raw",
            "prediction", "refused",
            "finish", "error", "temp_dropped", "latency_s", "in_tokens",
            "out_tokens", "ts"]
    # --force re-queries every row, so the file must be TRUNCATED, not appended
    # to. Appending under --force silently doubles the file and leaves two rows
    # per (case_id, qid) — one from each run.
    mode = "w" if args.force else "a"
    new = args.force or not os.path.exists(out_path)
    fh = open(out_path, mode, newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if new:
        w.writeheader()

    counts = {"ok": 0, "err": 0, "refused": 0, "n": 0}
    lock = threading.Lock()
    t0 = time.time()

    def work(r):
        res = providers.call(args.model, system, build_user(r, meta), args.max_tokens,
                             temperature=None if args.temperature < 0 else args.temperature,
                             reasoning=reasoning)
        if args.sleep:
            time.sleep(args.sleep + random.uniform(0, 0.15))
        return r, res

    def record(r, res):
        # One lock guards the writer, the flush and the counters. Without it,
        # concurrent writerow calls interleave and produce corrupt CSV lines.
        with lock:
            if res.error:
                counts["err"] += 1
            elif res.refused:
                counts["refused"] += 1
            else:
                counts["ok"] += 1
            counts["n"] += 1
            # The RECALL header is split off here, not at scoring time: the
            # `prediction` column must hold only what is meant to be compared
            # against gold, or its tokens land in char_ratio and token_f1.
            recall, answer = split_recall(res.text)
            w.writerow({"case_id": r["case_id"], "qid": r["qid"], "arm": r.get("arm"),
                        "jurisdiction": r.get("jurisdiction"), "court_level": r.get("court_level"),
                        "tier": r.get("tier"), "model": args.model,
                        "model_cutoff": args.cutoff, "served_by": res.served_by,
                        "recall": recall, "raw": res.text,
                        "prediction": answer, "refused": int(res.refused),
                        "finish": res.finish, "error": res.error,
                        "temp_dropped": int(res.temp_dropped),
                        "latency_s": round(res.latency_s, 2),
                        "in_tokens": res.in_tokens, "out_tokens": res.out_tokens,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            fh.flush()
            i = counts["n"]
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  ok={counts['ok']} refused={counts['refused']} "
                      f"err={counts['err']}  {el:.0f}s  "
                      f"~{el/max(1,i)*(len(todo)-i)/60:.1f}m left", flush=True)

    try:
        if args.workers <= 1:
            for r in todo:
                record(*work(r))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(work, r) for r in todo]
                for f in as_completed(futures):
                    record(*f.result())
    except KeyboardInterrupt:
        print("\n  interrupted — rows completed so far are on disk; rerun to resume")
    finally:
        fh.close()

    errs, ok, refused = counts["err"], counts["ok"], counts["refused"]

    # Parallel workers write in completion order. Sort once at the end so the
    # file is diffable and readable; resume reads the whole file regardless of
    # order, so re-sorting on every run is safe.
    if not args.no_sort and os.path.exists(out_path):
        with open(out_path, newline="") as f:
            allrows = list(csv.DictReader(f))
        if allrows:
            allrows.sort(key=lambda r: (r["case_id"], r["qid"]))
            with open(out_path, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=allrows[0].keys())
                wr.writeheader(); wr.writerows(allrows)
            print(f"  sorted {len(allrows)} rows by (case_id, qid)")
    print(f"\ndone: ok={ok} refused={refused} errors={errs}  -> {out_path}")
    if errs:
        print("  (re-run the same command to retry only the failed rows)")


if __name__ == "__main__":
    main()
