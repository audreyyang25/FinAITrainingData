import json
import re
from tqdm import tqdm

from config import (
    ADAPTERS,
    load_dataset,
    output_path,
    ANTHROPIC,
    OPENAI,
    GEMINI,
    QWEN,
    MAX_WORKERS,
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
FAILURES = output_path("generation_failures.jsonl")


ALL_MODELS = (
    ANTHROPIC
    + OPENAI
    + GEMINI
    + QWEN
)


MODEL_PARAMS = {
    "generation": {
        # The answer target is only 400-600 words; the headroom is for
        # reasoning tokens. Small reasoning models (qwen) were truncating at
        # 6000, and truncation isn't retried, so give them room.
        "max_tokens": 10000,
        "temperature": 0.0,
        "reasoning_effort": "medium",
    }
}


def parse_generation(raw):

    """
    Extract JSON from model response.
    """

    raw = raw.strip()

    # Handle accidental markdown fences
    if raw.startswith("```"):
        raw = (
            raw
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    # Model occasionally wraps the JSON in a prose preamble/epilogue; grab the
    # outermost {...} object rather than failing at char 0.
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object in response: {raw[:200]!r}")
        raw = m.group(0)

    return json.loads(raw)



def _generate_one(task):
    """Worker: one (case x model) generation. Pure network/CPU, no file I/O --
    returns ("ok", record) or ("fail", failure_row) for the main thread to
    write. One flaky response never sinks the run."""

    dataset_name, case_id, prompt, model_info = task

    raw = None
    try:
        raw = call_llm(
            model=model_info["model"],
            system=GENERATION_SYSTEM,
            user=prompt,
            **MODEL_PARAMS["generation"],
        )
        parsed = parse_generation(raw)
        validated = GenerationOutput(**parsed)

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
        "answer": validated.answer,
        "features": [f.model_dump() for f in validated.features],
    })


def generate_all(limit=None, max_workers=MAX_WORKERS):

    completed = existing_keys(
        OUTPUT,
        ["dataset", "case_id", "model"],
    )


    # Build the full task list first, skipping cells already done. Gold is
    # written by the `gold` stage, not here.
    tasks = []
    for dataset_name in ADAPTERS:
        for case_id, prompt, _truth in load_dataset(dataset_name, limit=limit):
            for model_info in ALL_MODELS:
                key = (dataset_name, case_id, model_info["model"])
                if key in completed:
                    continue
                tasks.append((dataset_name, case_id, prompt, model_info))

    print(f"{len(tasks)} generations to run ({max_workers} concurrent)")


    # Fan out the calls; the main loop is the sole writer (no append race).
    for status, payload in tqdm(
        parallel_yield(_generate_one, tasks, max_workers),
        total=len(tasks),
    ):
        if status == "ok":
            append_jsonl(OUTPUT, payload)
        else:
            append_jsonl(FAILURES, payload)
            print(f"SKIP {payload['dataset']}/{payload['case_id']}/"
                  f"{payload['model']}: {payload['error']}")



if __name__ == "__main__":
    generate_all()