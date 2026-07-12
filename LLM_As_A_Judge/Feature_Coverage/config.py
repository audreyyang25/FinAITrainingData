# config.py

import os
import json


# ============================================================
# Paths
# ============================================================

# Datasets live in the repo-root training-materials folder (two levels up),
# resolved from this file so cwd doesn't matter.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "AI Suitability Training Materials",
    "23_Folders_Suitability",
)

# Outputs live next to the code, resolved from this file so every stage writes
# to the same folder no matter which directory the pipeline is launched from.
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs",
)


def output_path(name):
    """Absolute path inside OUTPUT_DIR, independent of the current directory."""
    return os.path.join(OUTPUT_DIR, name)


def _format_convo(turns):
    """P14 conversations are a list of {speaker, text} dicts."""
    return "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in turns)

# ============================================================
# Dataset adapters
# ============================================================

ADAPTERS = ADAPTERS = {
    "standard": dict(  # P12, n=499
        file="suitability_only_P12.json",
        prompt=lambda r: f"{r['fact_pattern']}\n\nQuestion: {r['question']}",
        truth=lambda r: r["answer"],
        has_question=True,
    ),
    "borderline": dict(  # P13, n=250
        file="suitability_only_P13.json",
        prompt=lambda r: f"{r['fact_pattern']}\n\nQuestion: {r['question']} Respond with four clear components: arguments that the financial advisor violated their obligations, arguments that the financial advisor acted appropriately, list the key borderline factors, what is the likely outcome of a regulator's decisions.",
        truth=lambda r: f"Analysis: {r['analysis']}\n\nBorderline Factors: {r['borderline_factors']}\n\nLikely outcome: {r['likely_outcome']}",
        has_question=True,
    ),
    "conversations": dict(  # P14, n=50 (no explicit question)
        file="suitability_only_P14.json",
        prompt=lambda r: (
            "Review this client-advisor conversation and provide a legal/compliance "
            "assessment: identify any regulatory issues and their severity.\n\n"
            + _format_convo(r["conversation"])
        ),
        truth=lambda r: json.dumps(r["legal_assessment"], indent=2),
        has_question=False,
    ),
    "redflags": dict(  # P15, n=50 (no explicit question)
        file="suitability_only_P15.json",
        prompt=lambda r: (
            "Review this fact pattern. Identify the principal red flag, the specific "
            "regulatory concern, and the recommended compliance action.\n\n"
            + r["fact_pattern"]
        ),
        truth=lambda r: (
            f"Red flag: {r['red_flag']}\n"
            f"Regulatory concern: {r['regulatory_concern']}\n"
            f"Recommended action: {r['recommended_action']}"
        ),
        has_question=False,
    ),
    "adversarial": dict(  # P16, n=200
        file="suitability_only_P16.json",
        prompt=lambda r: f"{r['fact_pattern']}\n\nQuestion: {r['question']}",
        truth=lambda r: r["correct_answer"],
        has_question=True,
    ),
}


def load_dataset(name, limit=None, data_dir=DATA_DIR):

    a = ADAPTERS[name]

    with open(
        os.path.join(
            data_dir,
            a["file"]
        )
    ) as f:
        records = json.load(f)


    if limit:
        records = records[:limit]


    for r in records:

        yield (
            r["id"],
            a["prompt"](r),
            a["truth"](r),
        )


# ============================================================
# Models
# ============================================================

ANTHROPIC = [
    {
        "family": "Anthropic",
        "key": "frontier",
        "model": "anthropic/claude-opus-4.8",
    },
    {
        "family": "Anthropic",
        "key": "mid",
        "model": "anthropic/claude-sonnet-4.6",
    },
    {
        "family": "Anthropic",
        "key": "small",
        "model": "anthropic/claude-haiku-4.5",
    },
]


OPENAI = [
    {
        "family": "OpenAI",
        "key": "frontier",
        "model": "openai/gpt-5.5",
    },
    {
        "family": "OpenAI",
        "key": "mid",
        "model": "openai/gpt-5.4",
    },
    {
        "family": "OpenAI",
        "key": "small",
        "model": "openai/gpt-5.4-mini",
    },
]


GEMINI = [
    {
        "family": "Gemini",
        "key": "frontier",
        "model": "google/gemini-3.1-pro-preview",
    },
    {
        "family": "Gemini",
        "key": "mid",
        "model": "google/gemini-3.5-flash",
    },
    {
        "family": "Gemini",
        "key": "small",
        "model": "google/gemini-3.1-flash-lite",
    },
]


QWEN = [
    {
        "family": "Qwen",
        "key": "frontier",
        "model": "qwen/qwen3.5-397b-a17b",
    },
    {
        "family": "Qwen",
        "key": "mid",
        "model": "qwen/qwen3.6-35b-a3b",
    },
    {
        "family": "Qwen",
        "key": "small",
        "model": "qwen/qwen3.5-9b",
    },
]


EVALUATION_MODELS = (
    ANTHROPIC
    + OPENAI
    + GEMINI
    + QWEN
)


# ============================================================
# Canonicalization model
# ============================================================

# Canonicalizer + gold extractor. A neutral non-grid model is ideal
# (deepseek-r1 was tried, but its OpenRouter daily request cap throttled the
# run). Using opus-4.8 for reliability and no daily cap; note it is the
# Anthropic frontier generator, so results carry a mild Anthropic lean --
# document this as a known limitation.
CANONICALIZER_MODEL = (
    "anthropic/claude-opus-4.8"
)


# ============================================================
# Generation parameters
# ============================================================

GENERATION_CONFIG = {

    "temperature": 0,

    "max_tokens": 10000,

    "reasoning_effort": "medium",
}


CANONICALIZATION_CONFIG = {

    "temperature": 0,

    "max_tokens": 20000,

    # Dropped from "high": the original->canonical mapping for a feature-dense
    # case is a large output, and high reasoning was consuming the budget it
    # needed, truncating whole cases. Medium leaves room for the mapping.
    "reasoning_effort": "medium",
}


# ============================================================
# API / reliability
# ============================================================

MAX_RETRIES = 5

REQUEST_TIMEOUT = 300

# Concurrent in-flight LLM calls for the parallelized stages (generation, gold,
# case canonicalization). Raise for speed, lower if you hit OpenRouter rate
# limits; call_llm already retries transient 429s with backoff.
MAX_WORKERS = 8


# ============================================================
# Pipeline settings
# ============================================================

# Whether to include gold answer as feature source

EXTRACT_GOLD_FEATURES = True