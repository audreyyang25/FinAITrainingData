import json
import re

from tqdm import tqdm

from llm import call_llm
from schemas import GlobalBatchCanonicalization

from utils import (
    load_json,
    save_json,
    append_jsonl,
)

from config import (
    CANONICALIZER_MODEL,
    CANONICALIZATION_CONFIG,
    output_path,
)

INPUT = output_path("case_features.json")
OUTPUT = output_path("global_features.json")
FAILURES = output_path("global_canonicalization_failures.jsonl")

# Case features are folded into the global vocabulary a batch at a time, so
# neither prompt nor output grows unbounded. The vocabulary carries forward
# between batches and converges as later batches increasingly reuse it (a cheap
# integer index) rather than adding new entries (text). Kept modest so the
# model doesn't silently omit keys when enumerating a long mapping -- omitted
# features stay unmapped and are retried on the next run.
CHUNK_SIZE = 100


SYSTEM = """
You are building a GLOBAL reasoning-feature vocabulary for legal/compliance
analyses, incrementally, one batch at a time.

You will receive:
- EXISTING VOCABULARY: numbered global features already established (possibly
  empty).
- NEW FEATURES: numbered case-level reasoning features to fold in.


FOR EACH NEW FEATURE, decide its global feature:

1. If an EXISTING VOCABULARY entry represents the same underlying analytical
   factor, output that entry's NUMBER as a bare integer.

2. Otherwise, output a NEW global feature as a STRING -- a short, general,
   stance-neutral phrase. If several new features in this batch share the same
   new global feature, output the SAME string for each.

3. If the feature is not a genuine reasoning factor (a conclusion, a generic
   statement of law, or an unsupported tangent), output null.


MERGE RULE

Treat two features as the same global feature ONLY if they represent the same
underlying analytical factor.

MERGE:
"Client age affects suitability"
"Investor retirement horizon affects appropriateness"
(same demographic / time-horizon consideration)

DO NOT MERGE:
"Client age affects suitability"
with
"Client wealth affects ability to tolerate losses"
(different analytical dimensions)

Do not merge based on vocabulary similarity alone.


OUTPUT ONLY JSON, no markdown:

{
  "mapping": {
    "<new feature number>": <existing vocabulary number> | "new global feature" | null
  }
}

Every NEW feature number must appear exactly once as a key. Use a bare integer
for reuse, a quoted string for a new global feature, or null to discard.
"""

def collect_features(input=INPUT):

    cases = load_json(
        input,
        default={},
    )
    features = set()
    for case in cases.values():

        for f in case["superset"]:

            features.add(f)
    return sorted(features)

def build_batch_prompt(vocabulary, batch):

    if vocabulary:
        vocab_block = "\n".join(
            f"{i}. {v}"
            for i, v in enumerate(vocabulary)
        )
    else:
        vocab_block = "(empty)"

    feat_block = "\n".join(
        f"{i}. {f}"
        for i, f in enumerate(batch, start=1)
    )

    return (
        f"EXISTING VOCABULARY:\n{vocab_block}\n\n"
        f"NEW FEATURES:\n{feat_block}"
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

    start = raw.find("{")
    if start == -1:
        raise ValueError(
            f"no JSON object in response: {raw[:200]!r}"
        )

    # raw_decode parses the first complete JSON object and ignores anything
    # after it, so a valid object followed by trailing content (a second
    # object, a stray note, an extra code fence) no longer raises "Extra data".
    obj, _ = json.JSONDecoder().raw_decode(raw, start)

    return obj

def resolve(value, vocabulary):
    """Turn one mapping value into a global-feature string, or None to skip.

    value is an int index into `vocabulary`, a new global-feature string, or
    null. Mutates `vocabulary` (appends) when a genuinely new feature appears.
    """
    # Accept a quoted integer ("5") as an index too, defensively.
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)

    # bool is an int subclass -- never a valid index.
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        if 0 <= value < len(vocabulary):
            return vocabulary[value]
        return None   # out-of-range index -> leave unmapped, retried next run

    if isinstance(value, str) and value.strip():
        g = value.strip()
        if g not in vocabulary:
            vocabulary.append(g)
        return g

    return None

def canonicalize_global(input=INPUT, output=OUTPUT, failures=FAILURES):

    all_features = collect_features(input)

    # Resume from any prior partial run: a feature already in the mapping is
    # done. The vocabulary is preserved in code and never re-typed by the model.
    state = load_json(
        output,
        default={"vocabulary": [], "mapping": {}},
    )

    vocabulary = state["vocabulary"]
    mapping = state["mapping"]

    remaining = [
        f
        for f in all_features
        if f not in mapping
    ]

    print(
        f"{len(all_features)} unique case features; "
        f"{len(remaining)} left to map; "
        f"vocabulary so far: {len(vocabulary)}"
    )

    batches = [
        remaining[i:i + CHUNK_SIZE]
        for i in range(0, len(remaining), CHUNK_SIZE)
    ]

    for batch in tqdm(batches):

        prompt = build_batch_prompt(vocabulary, batch)

        try:
            raw = call_llm(
                model=CANONICALIZER_MODEL,
                system=SYSTEM,
                user=prompt,
                max_tokens=CANONICALIZATION_CONFIG["max_tokens"],
                temperature=CANONICALIZATION_CONFIG["temperature"],
                reasoning_effort=CANONICALIZATION_CONFIG["reasoning_effort"],
            )
            parsed = parse_json(raw)
            validated = GlobalBatchCanonicalization(**parsed)

        except Exception as e:

            append_jsonl(
                failures,
                {
                    "error": str(e),
                    "batch_size": len(batch),
                    "first_feature": batch[0] if batch else None,
                },
            )

            print(f"SKIP batch ({len(batch)} features): {e}")
            continue

        # Assign each batch feature (numbered from 1) its global feature.
        for i, feat in enumerate(batch, start=1):

            key = str(i)
            # Omitted by the model -> leave unmapped so a rerun retries it.
            if key not in validated.mapping:
                continue

            value = validated.mapping[key]
            # Explicit null -> a genuine discard (relevance filter). Record it
            # as None so it counts as done (not retried) and stays
            # distinguishable from a silent omission.
            if value is None:
                mapping[feat] = None
                continue

            global_feature = resolve(value, vocabulary)

            if global_feature is not None:
                mapping[feat] = global_feature
            # else: invalid/out-of-range index -> unmapped, retried next run

        # Checkpoint after every batch so a crash keeps prior work.
        save_json(
            output,
            {
                "vocabulary": vocabulary,
                "mapping": mapping,
            },
        )

    mapped = sum(1 for v in mapping.values() if v)
    discarded = sum(1 for v in mapping.values() if not v)
    unresolved = len(all_features) - len(mapping)

    print(
        f"Done. Vocabulary: {len(vocabulary)} global features; "
        f"{mapped} case features mapped, {discarded} discarded, "
        f"{unresolved} still unresolved (rerun to retry)."
    )

if __name__ == "__main__":
    canonicalize_global()
