import json
import re
from collections import defaultdict

from tqdm import tqdm

from llm import call_llm
from schemas import CaseCanonicalization
from prompts import CASE_CANONICALIZATION_SYSTEM

from utils import (
    load_json,
    load_jsonl,
    save_json,
    append_jsonl,
    parallel_yield,
)
from config import (
    CANONICALIZER_MODEL,
    CANONICALIZATION_CONFIG,
    output_path,
    MAX_WORKERS,
)


INPUT = output_path("generations.jsonl")
OUTPUT = output_path("case_features.json")
FAILURES = output_path("case_canonicalization_failures.jsonl")


def group_cases(records):

    cases = defaultdict(list)

    for r in records:
        cases[(r["dataset"], r["case_id"])].append(r)

    return cases



def build_registry(case_records):
    """Assign every raw feature (across all models, incl. gold) an integer id.

    Returns an insertion-ordered dict:
        {"<id>": {model, model_family, feature, importance}}

    Gold is included so its reasoning contributes to the superset; it is
    excluded from coverage scoring downstream (model_family == "gold").
    """

    registry = {}
    fid = 0

    for r in case_records:

        if not r.get("features"):
            continue

        for f in r["features"]:
            fid += 1
            registry[str(fid)] = {
                "model": r["model"],
                "model_family": r["model_family"],
                "feature": f["feature"],
                "importance": f["importance"],
            }

    return registry



def build_prompt(registry):

    numbered = "\n".join(
        f'{fid}. {e["feature"]}'
        for fid, e in registry.items()
    )

    return (
        "Numbered reasoning features extracted from several answers to the "
        "same question:\n\n" + numbered
    )



def parse_json(raw):

    raw = raw.strip()

    if raw.startswith("```"):
        raw = (
            raw
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError(
                f"no JSON object in response: {raw[:200]!r}"
            )
        raw = m.group(0)

    return json.loads(raw)



def _canonicalize_one(task):
    """Worker: canonicalize one case. No file I/O -- returns
    ("ok", (key, case_dict)) / ("fail", failure_row) / ("skip", None)."""

    dataset, case_id, case_records = task

    registry = build_registry(case_records)

    if not registry:
        # No model produced features for this case; nothing to merge.
        return ("skip", None)

    try:
        raw = call_llm(
            model=CANONICALIZER_MODEL,
            system=CASE_CANONICALIZATION_SYSTEM,
            user=build_prompt(registry),
            max_tokens=CANONICALIZATION_CONFIG["max_tokens"],
            temperature=CANONICALIZATION_CONFIG["temperature"],
            reasoning_effort=CANONICALIZATION_CONFIG["reasoning_effort"],
        )
        parsed = parse_json(raw)
        validated = CaseCanonicalization(**parsed)

    except Exception as e:
        return ("fail", {
            "dataset": dataset,
            "case_id": case_id,
            "error": str(e),
        })

    superset = validated.superset
    n = len(superset)

    # Attach each raw feature's canonical superset index (or None). Guard
    # against the model returning an out-of-range index or omitting an id.
    for fid, entry in registry.items():
        idx = validated.mapping.get(fid)
        if isinstance(idx, int) and 0 <= idx < n:
            entry["superset_index"] = idx
        else:
            entry["superset_index"] = None

    return ("ok", (
        f"{dataset}:{case_id}",
        {
            "dataset": dataset,
            "case_id": case_id,
            "superset": superset,
            "features": registry,
        },
    ))


def canonicalize_cases(max_workers=MAX_WORKERS):

    records = load_jsonl(INPUT)

    cases = group_cases(records)


    # Resume: keep cases already canonicalized in a prior run.
    results = load_json(OUTPUT, default={})


    tasks = [
        (dataset, case_id, case_records)
        for (dataset, case_id), case_records in cases.items()
        if f"{dataset}:{case_id}" not in results
    ]

    print(f"{len(tasks)} cases to canonicalize ({max_workers} concurrent)")


    # Fan out; the main loop is the sole writer (results dict + checkpoint).
    for status, payload in tqdm(
        parallel_yield(_canonicalize_one, tasks, max_workers),
        total=len(tasks),
    ):
        if status == "ok":
            key, case_dict = payload
            results[key] = case_dict
            save_json(OUTPUT, results)
        elif status == "fail":
            append_jsonl(FAILURES, payload)
            print(f"SKIP case {payload['dataset']}:{payload['case_id']}: "
                  f"{payload['error']}")



if __name__ == "__main__":
    canonicalize_cases()
