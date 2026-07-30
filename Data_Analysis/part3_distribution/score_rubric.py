"""score_rubric.py -- binary rubric scoring of every generation against rubric.py.

One labeler call per answer returns 20 booleans (does the answer engage each
criterion). Resumable and parallel, mirroring indep_extractor_pipeline. Output:
outputs/rubric/{judge_slug}.jsonl, one record per (dataset, case_id, model).

  python score_rubric.py                                   # full run, llama judge
  python score_rubric.py --models opus,gpt-5.5,gemini-3.1-pro,qwen3.5-397b --limit-cases 50
  python score_rubric.py --judge openai/gpt-5.4-mini --limit-cases 30   # 2nd judge for agreement
"""

import argparse
import os

from tqdm import tqdm

from shared.config import part_output, GENERATIONS_JSON, MAX_WORKERS, EXTRACTORS, EXTRACTOR_PROVIDER
from shared.utils import parse_json
from shared.llm import call_llm
from shared.utils import load_json, append_jsonl, existing_keys, parallel_yield, ensure_dir
from part3_distribution.rubric import RUBRIC_SYSTEM, user_prompt, IDS

DEFAULT_JUDGE = EXTRACTORS[0]["extractor"]      # llama, same as the extractor
KEY = ["dataset", "case_id", "model"]


def _score_one(task):
    judge, dataset, case_id, model, answer = task
    raw = None
    try:
        raw = call_llm(
            model=judge,
            system=RUBRIC_SYSTEM,
            user=user_prompt(answer),
            max_tokens=3000,
            temperature=0,
            provider=EXTRACTOR_PROVIDER.get(judge),
        )
        parsed = parse_json(raw)
        scores = parsed["scores"]
        # normalize to {id: bool}; require all 20 ids present
        out = {}
        for i in IDS:
            cell = scores[str(i)]
            out[i] = bool(cell["present"] if isinstance(cell, dict) else cell)
        ev = {i: (scores[str(i)].get("evidence", "") if isinstance(scores[str(i)], dict) else "")
              for i in IDS}
    except Exception as e:
        return ("fail", {"dataset": dataset, "case_id": case_id, "model": model,
                         "error": str(e), "raw": (raw or "")[:1500]})
    return ("ok", {"dataset": dataset, "case_id": case_id, "model": model,
                   "judge": judge, "present": out, "evidence": ev})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--models", default="", help="comma-separated model substrings (default: all)")
    ap.add_argument("--limit-cases", type=int, default=0, help="max distinct cases per model (0=all)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    slug = args.judge.split("/")[-1].replace(".", "_")
    outdir = part_output("part3_distribution", "rubric")
    ensure_dir(outdir)
    out = os.path.join(outdir, f"{slug}.jsonl")
    fails = os.path.join(outdir, f"{slug}.failures.jsonl")
    done = existing_keys(out, KEY)

    filt = [s.strip() for s in args.models.split(",") if s.strip()]
    gens = load_json(GENERATIONS_JSON, default=[])

    tasks, per_model = [], {}
    for g in gens:
        m = g["model"]
        if filt and not any(s in m for s in filt):
            continue
        if args.limit_cases:
            per_model.setdefault(m, 0)
            if per_model[m] >= args.limit_cases:
                continue
            per_model[m] += 1
        if (g["dataset"], g["case_id"], m) in done:
            continue
        tasks.append((args.judge, g["dataset"], g["case_id"], m, g["answer"]))

    print(f"judge={args.judge}  {len(tasks)} answers to score ({args.workers} concurrent) -> {out}")
    ok = 0
    for status, payload in tqdm(parallel_yield(_score_one, tasks, args.workers), total=len(tasks)):
        if status == "ok":
            append_jsonl(out, payload); ok += 1
        else:
            append_jsonl(fails, payload)
    print(f"done: {ok} scored, {len(tasks)-ok} failed -> {out}")


if __name__ == "__main__":
    main()
