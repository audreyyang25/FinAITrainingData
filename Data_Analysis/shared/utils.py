import json
import os
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def load_json(path, default=None):

    if not os.path.exists(path):
        return default

    with open(path) as f:
        return json.load(f)


def save_json(path, obj):

    ensure_dir(
        os.path.dirname(path)
    )

    with open(path, "w") as f:
        json.dump(
            obj,
            f,
            indent=2
        )


def append_jsonl(path, obj):

    ensure_dir(
        os.path.dirname(path)
    )

    with open(path, "a") as f:
        f.write(
            json.dumps(obj)
            + "\n"
        )


def load_jsonl(path):

    if not os.path.exists(path):
        return []

    with open(path) as f:
        return [
            json.loads(line)
            for line in f
        ]


def parse_json(raw):
    """Extract the first JSON object from a model response, tolerating markdown
    fences, a prose preamble, and trailing content after the object."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    # raw_decode parses the first complete JSON object and ignores anything after
    # it, so a valid object followed by trailing content (a second object, a stray
    # note, an extra code fence) no longer raises "Extra data".
    obj, _ = json.JSONDecoder().raw_decode(raw, start)
    return obj


def existing_keys(path, key_fields):

    records = load_jsonl(path)

    keys = set()

    for r in records:
        keys.add(
            tuple(
                r[k]
                for k in key_fields
            )
        )

    return keys


def parallel_yield(fn, items, max_workers):
    """Run fn(item) across a thread pool; yield results as they complete.

    For I/O-bound work (LLM HTTP calls): the GIL is released during the network
    wait, so threads give real concurrency. The CALLER does all file writes in
    its consuming loop, so there is a single writer and no append race.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for fut in as_completed(futures):
            yield fut.result()