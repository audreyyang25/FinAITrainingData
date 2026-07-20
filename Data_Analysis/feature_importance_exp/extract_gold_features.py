import json
import re
from tqdm import tqdm

from config import (
    ADAPTERS,
    load_dataset,
    output_path,
    MAX_WORKERS,
    CANONICALIZER_MODEL,
)

from llm import call_llm
from schemas import GenerationOutput

from prompts import GENERATION_SYSTEM

from utils import (
    append_jsonl,
    existing_keys,
    parallel_yield,
)

OUTPUT = output_path("generations.jsonl")
FAILURES = output_path("gold_failures.jsonl")

GOLD_MODEL = "gold"

def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = (
            raw
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    # raw_decode parses the first complete JSON object and ignores anything
    # after it, so a valid object followed by trailing content (a second
    # object, a stray note, an extra code fence) no longer raises "Extra data".
    obj, _ = json.JSONDecoder().raw_decode(raw, start)
    return obj

def _extract_one(task):
    """Worker: extract gold features for one case. No file I/O -- returns
    ("ok", record) or ("fail", failure_row) for the main thread to write."""

    dataset_name, case_id, truth = task

    extraction_prompt = f"""
Below is the reference answer to a legal/compliance analysis question.

Extract the reasoning features used in this answer.

REFERENCE ANSWER:

{truth}
"""

    raw = None
    try:
        raw = call_llm(
            model=CANONICALIZER_MODEL,
            system=GENERATION_SYSTEM,
            user=extraction_prompt,
            max_tokens=10000,
            temperature=0,
            reasoning_effort="medium",
        )
        parsed = parse_json(raw)
        validated = GenerationOutput(**parsed)

    except Exception as e:
        return ("fail", {
            "dataset": dataset_name,
            "case_id": case_id,
            "model": GOLD_MODEL,
            "error": str(e),
            "raw": (raw or "")[:2000],
        })

    return ("ok", {
        "dataset": dataset_name,
        "case_id": case_id,
        "model_family": "gold",
        "model_key": "gold",
        "model": "gold",
        "answer": truth,
        "features": [f.model_dump() for f in validated.features],
    })

def extract_gold_features(limit=None, max_workers=MAX_WORKERS):
    completed = existing_keys(
        OUTPUT,
        ["dataset", "case_id", "model"],
    )

    tasks = []
    for dataset_name in ADAPTERS:
        for case_id, _prompt, truth in load_dataset(dataset_name, limit=limit):
            if (dataset_name, case_id, GOLD_MODEL) in completed:
                continue
            tasks.append((dataset_name, case_id, truth))

    print(f"{len(tasks)} gold extractions to run ({max_workers} concurrent)")

    for status, payload in tqdm(
        parallel_yield(_extract_one, tasks, max_workers),
        total=len(tasks),
    ):
        if status == "ok":
            append_jsonl(OUTPUT, payload)
        else:
            append_jsonl(FAILURES, payload)
            print(f"SKIP gold {payload['dataset']}/{payload['case_id']}: "
                  f"{payload['error']}")

if __name__ == "__main__":
    extract_gold_features()