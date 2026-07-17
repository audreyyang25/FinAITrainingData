
import json
import re
import os
from tqdm import tqdm

from config import (
    ADAPTERS,
    load_dataset,
    output_path,
    MAX_WORKERS,
    CANONICALIZER_MODEL,
    EXTRACTORS
)

from config import EXTRACTOR_PROVIDER
from extract_gold_features import parse_json
from llm import call_llm
from schemas import ExtractionOutput
from prompts import EXTRACTION_SYSTEM
from canonicalize_case import canonicalize_cases
from canonicalize_global import canonicalize_global

from utils import (
    load_jsonl,
    append_jsonl,
    existing_keys,
    parallel_yield,
)



def _extract_one(task):
    """Worker: extract gold features for one case. No file I/O -- returns
    ("ok", record) or ("fail", failure_row) for the main thread to write."""

    extractor, dataset, case_id, model_family, model_key, model, truth = task

    extraction_prompt = f"""
Below is an answer to a legal/compliance analysis question.

Extract the reasoning features that drove this answer.

ANSWER:

{truth}
"""

    raw = None
    try:
        raw = call_llm(
            model=extractor,
            system=EXTRACTION_SYSTEM,
            user=extraction_prompt,
            # Extraction output is small (~800 tokens of feature JSON). 10k was
            # copied from generation and, on a 16k-context provider, pushes long
            # inputs over the limit -- 4k leaves ample room and stops 400s.
            max_tokens=4000,
            temperature=0,
            provider=EXTRACTOR_PROVIDER.get(extractor),
        )
        parsed = parse_json(raw)
        # Clamp per-feature importance into [0, 100] before validating: cheaper
        # extractors emit out-of-range scores (negatives, >100), and dropping the
        # whole feature list over one bad number loses the feature SELECTION,
        # which is the signal we actually use.
        for f in parsed.get("features", []):
            try:
                f["importance"] = max(0, min(100, int(f["importance"])))
            except (KeyError, TypeError, ValueError):
                pass   # leave genuinely malformed entries for validation to reject
        validated = ExtractionOutput(**parsed)

    except Exception as e:
        return ("fail", {
            "dataset": dataset,
            "case_id": case_id,
            "model": model,
            "error": str(e),
            "raw": (raw or "")[:2000],
        })

    return ("ok", {
        "dataset": dataset,
        "case_id": case_id,
        "model_family": model_family,
        "model_key": model_key,
        "model": model,
        "answer": truth,
        "features": [f.model_dump() for f in validated.features],
    })
        
def extract_features(extractor, output, failures, max_workers=MAX_WORKERS):
    generations = load_jsonl(output_path("generations.jsonl"))
    completed = existing_keys(
        output,
        ["dataset", "case_id", "model_family", "model_key", "model"],
    )

    tasks = []
    for g in generations:
            key = (g["dataset"], g["case_id"], g["model_family"], g["model_key"], g["model"])
            if key in completed:
                continue
            tasks.append((extractor, g["dataset"], g["case_id"], g["model_family"], g["model_key"], g["model"], g["answer"]))

    print(f"{len(tasks)} extractions to run ({max_workers} concurrent)")

    for status, payload in tqdm(
        parallel_yield(_extract_one, tasks, max_workers),
        total=len(tasks),
    ):
        if status == "ok":
            append_jsonl(output, payload)
        else:
            append_jsonl(failures, payload)
            print(f"SKIP {payload['model']} {payload['dataset']}/{payload['case_id']}: "
                  f"{payload['error']}")
    return output

if __name__ == "__main__":
    for e in EXTRACTORS:
        key, extractor = e["key"], e["extractor"]

        def model_output_path(name, _e=key):
            d = output_path(_e)
            os.makedirs(d, exist_ok=True)
            return os.path.join(d, name)

        extracted_path = extract_features(
            extractor,
            output=model_output_path("gen_feat.jsonl"),
            failures=model_output_path("gen_failures.jsonl"),
        )
        canonicalize_cases(input=extracted_path, output=model_output_path("case_feat.json"), failures=model_output_path("case_canon_failures.jsonl"))
        canonicalize_global(input=model_output_path("case_feat.json"), output=model_output_path("global_features.json"), failures=model_output_path("global_failures.jsonl"))
        




    