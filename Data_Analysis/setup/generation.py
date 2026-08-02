"""setup/generation.py -- SETUP stage: collect one answer per (case, model).

Each of the 12 models in config.EVALUATION_MODELS answers every case across the
five suitability datasets. ANSWER ONLY -- no self-reported features (feature
extraction happens independently in Part 2). A "gold" row per case carries the
dataset's reference answer (truth), so Part 2 can extract gold features exactly
the way it extracts every model's. Output is a single JSON array at the
Data_Analysis root: generations.json.

Two ways to build it, both emitting the identical schema:
    {dataset, case_id, model_family, model_key, model, answer}

    # fresh answer-only API run (resumable; --limit takes first N cases/dataset)
    python -m setup.generation
    python -m setup.generation --limit 100

    # replicate the paper's original answers by stripping the self-reported
    # features off a prior self-report run -- no API calls
    python -m setup.generation --from-jsonl archive/selfreport_generations.jsonl

Fresh runs are resumable: existing generations.json is loaded first and only
missing (dataset, case_id, model) cells are (re)run; the array is flushed every
FLUSH_EVERY completions and once at the end.
"""

import argparse
import re

from tqdm import tqdm

from shared.config import (
    ADAPTERS,
    load_dataset,
    part_output,
    GENERATIONS_JSON,
    EVALUATION_MODELS,
    MAX_WORKERS,
    GENERATION_CONFIG,
)
from shared.llm import call_llm
from shared.prompts import ANSWER_ONLY_SYSTEM
from shared.utils import (
    load_json,
    save_json,
    load_jsonl,
    append_jsonl,
    parallel_yield,
)

FAILURES = part_output("setup", "generation_failures.jsonl")
FLUSH_EVERY = 200
RECORD_FIELDS = ("dataset", "case_id", "model_family", "model_key", "model", "answer")

# Reasoning models sometimes put the analysis in the thinking block and emit only
# a pointer to it ("See full analysis above"), or collapse to a bare verdict with
# no reasoning. Both pass call_llm's guards -- finish_reason is "stop" and the
# content is non-empty -- so they need catching here or they land in
# generations.json looking valid and drag the model's scores down.
#
# 300 chars is deliberately conservative: below it answers are bare verdicts with
# no supporting reasoning, while the 300-500 band is terse-but-genuine analysis
# from the small qwen models. Regenerating those would bias the corpus toward
# verbosity, which Part 2 treats as a variable rather than a defect.
MIN_ANSWER_CHARS = 300

# Anchored to the opening only: a full analysis may legitimately say "as noted
# above" partway through, but one that *starts* by deferring is a stub.
POINTER_RE = re.compile(
    r"^\W*(see|refer to|as (shown|stated|noted|described))\b[^.]{0,80}\babove\b"
    r"|^\W*(the )?(full |complete |detailed )?(analysis|assessment|response|answer)\b"
    r"[^.]{0,80}\b(above|previously provided)\b",
    re.I,
)


def invalid_reason(answer):
    """Why `answer` isn't a usable analysis, or None if it's fine. Used both on
    fresh completions and on reload, so a bad row already in generations.json is
    dropped and regenerated rather than skipped as done."""
    answer = (answer or "").strip()
    if not answer:
        return "empty answer"
    if len(answer) < MIN_ANSWER_CHARS:
        return f"answer too short ({len(answer)} < {MIN_ANSWER_CHARS} chars)"
    if POINTER_RE.search(answer[:200]):
        return "answer defers to the reasoning block instead of containing the analysis"
    return None


def _key(rec):
    return (rec["dataset"], rec["case_id"], rec["model"])


def clean_from_jsonl(src, out=GENERATIONS_JSON):
    """Build answer-only generations.json from a prior self-report run by
    keeping only RECORD_FIELDS (i.e. dropping the 'features' list). Gold rows in
    the source are preserved."""
    records = [{k: r[k] for k in RECORD_FIELDS} for r in load_jsonl(src)]
    save_json(out, records)
    return len(records), out


def _generate_one(task):
    """Worker: one (case x model) answer. No file I/O -- returns ("ok", record)
    or ("fail", failure_row) for the main thread to write."""
    dataset_name, case_id, prompt, model_info = task

    raw = None
    try:
        raw = call_llm(
            model=model_info["model"],
            system=ANSWER_ONLY_SYSTEM,
            user=prompt,
            **GENERATION_CONFIG,
        )
        answer = (raw or "").strip()
        bad = invalid_reason(answer)
        if bad:
            raise ValueError(bad)
    except Exception as e:
        return ("fail", {
            "dataset": dataset_name,
            "case_id": case_id,
            "model": model_info["model"],
            "error": str(e),
            "raw": (raw or "")[:2000],
        })

    return ("ok", {
        "dataset": dataset_name,
        "case_id": case_id,
        "model_family": model_info["model"].split("/")[0],
        "model_key": model_info["key"],
        "model": model_info["model"],
        "answer": answer,
    })


def generate_all(limit=None, max_workers=MAX_WORKERS, out_path=GENERATIONS_JSON):
    records = load_json(out_path, default=[])

    # Drop previously-saved rows that fail the validity check so they re-run.
    # Gold rows are the dataset's reference answer, not a completion -- never
    # judge them by these rules. Rows missing their identifying fields are dropped
    # too: they can't be keyed, so they'd crash _key() below and can never be
    # matched to a case again.
    kept, dropped, malformed = [], [], 0
    for r in records:
        if not all(k in r for k in ("dataset", "case_id", "model")):
            malformed += 1
            continue
        if r.get("model") == "gold" or not invalid_reason(r.get("answer")):
            kept.append(r)
        else:
            dropped.append(r)
    if malformed:
        print(f"dropped {malformed} record(s) missing dataset/case_id/model")
    if dropped:
        by_model = {}
        for r in dropped:
            by_model[r["model"]] = by_model.get(r["model"], 0) + 1
        print(f"regenerating {len(dropped)} invalid answer(s) already in {out_path}:")
        for model, n in sorted(by_model.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4}  {model}")
    records = kept

    completed = {_key(r) for r in records}

    # Build the task list, skipping done cells. Gold rows are the reference
    # answer (truth) -- no API call needed, so they're added inline here.
    tasks = []
    for dataset_name in ADAPTERS:
        for case_id, prompt, truth in load_dataset(dataset_name, limit=limit):
            if (dataset_name, case_id, "gold") not in completed:
                records.append({
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "model_family": "gold",
                    "model_key": "gold",
                    "model": "gold",
                    "answer": truth,
                })
                completed.add((dataset_name, case_id, "gold"))
            for model_info in EVALUATION_MODELS:
                if (dataset_name, case_id, model_info["model"]) in completed:
                    continue
                tasks.append((dataset_name, case_id, prompt, model_info))

    print(f"{len(tasks)} model generations to run ({max_workers} concurrent) -> {out_path}")

    # Fan out the calls; the main loop is the sole writer.
    since_flush = 0
    for status, payload in tqdm(
        parallel_yield(_generate_one, tasks, max_workers),
        total=len(tasks),
    ):
        if status == "ok":
            records.append(payload)
            since_flush += 1
            if since_flush >= FLUSH_EVERY:
                save_json(out_path, records)
                since_flush = 0
        else:
            append_jsonl(FAILURES, payload)
            print(f"SKIP {payload['dataset']}/{payload['case_id']}/"
                  f"{payload['model']}: {payload['error']}")

    save_json(out_path, records)
    print(f"wrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-jsonl", default=None,
                    help="build from a prior self-report generations.jsonl "
                         "(strip features); makes no API calls")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N cases per dataset (fresh runs only)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    if args.from_jsonl:
        n, path = clean_from_jsonl(args.from_jsonl)
        print(f"cleaned {n} records -> {path}")
    else:
        generate_all(limit=args.limit, max_workers=args.workers)
