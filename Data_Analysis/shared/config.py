# config.py

import csv
import os
import json


# Paths

# Path to original training data (gold standard)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "AI Suitability Training Materials",
    "23_Folders_Suitability",
)

# Outputs live at the Data_Analysis root (one level up from shared/).
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
)

# Materialized ADAPTERS table, also at the Data_Analysis root.
ADAPTERS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "adapters.csv",
)

# Shared SETUP output: one answer-only record per (case, model), at the root.
GENERATIONS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "generations.json",
)

def output_path(name):
    """Absolute path inside OUTPUT_DIR, independent of the current directory."""
    return os.path.join(OUTPUT_DIR, name)


def part_output(part, name=""):
    """Absolute path inside a per-part output subdir, created on demand, e.g.
    part_output('part2_coverage', 'coverage.csv') -> outputs/part2_coverage/coverage.csv.
    Groups every artifact under its pipeline part (part1_eval / part2_coverage /
    part3_distribution / setup)."""
    d = os.path.join(OUTPUT_DIR, part)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name) if name else d

def _format_convo(turns):
    """P14 conversations are a list of {speaker, text} dicts."""
    return "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in turns)

# Dataset adapters

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

    with open(os.path.join(data_dir, a["file"])) as f:
        records = json.load(f)

    if limit:
        records = records[:limit]

    for r in records:
        yield (
            r["id"],
            a["prompt"](r),
            a["truth"](r),
        )

# Models

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

# Canonicalization model

# Canonicalizer + gold extractor -- using opus-4.8 for reliability and no daily cap; note it is the
# Anthropic frontier generator, so results carry a mild Anthropic lean
CANONICALIZER_MODEL = (
    "anthropic/claude-opus-4.8"
)

EXTRACTORS = (
    # Dropped: deepseek-v3.2 emitted negative importance on ~15% of answers
    # (non-random, concentrated in OpenAI answers). Re-enable only with the
    # clamp/lenient-schema repair, and rely on feature SELECTION not importance.
    # {"key": "deepseek", "extractor": "deepseek/deepseek-v3.2"},
    {"key": "llama", "extractor": "meta-llama/llama-3.3-70b-instruct"},
)

# Per-extractor OpenRouter provider routing, keyed by slug. Llama 3.3-70B is
# served by many providers at different quantizations (fp8 vs bf16); left
# unpinned, the serving backend varies call-to-call and injects noise into an
# extraction comparison meant to isolate model identity. Restricting it to bf16
# makes every answer get extracted at the same precision. DeepSeek is
# effectively first-party on OpenRouter, so it needs no pin. For strict
# single-backend determinism, add "order": ["<provider>"] alongside the filter.
EXTRACTOR_PROVIDER = {
    "meta-llama/llama-3.3-70b-instruct": {
        "quantizations": ["bf16"],
        "allow_fallbacks": False,
    },
}

# Generation parameters

GENERATION_CONFIG = {
    "temperature": 0,
    "max_tokens": 10000,
    "reasoning_effort": "medium",
}


CANONICALIZATION_CONFIG = {
    "temperature": 0,
    "max_tokens": 20000,
    "reasoning_effort": "medium",
}

# API / reliability

MAX_RETRIES = 5

REQUEST_TIMEOUT = 300

MAX_WORKERS = 8

# Pipeline settings
EXTRACT_GOLD_FEATURES = True


# Materialized ADAPTERS table

def build_adapters_csv(out_path=ADAPTERS_CSV, data_dir=DATA_DIR, force=False):
    """Materialize every adapter over ALL its records into a flat CSV.

    Columns: FILE, ID, PROMPT, TRUTH, HAS_QUESTION. Every field is fully quoted
    so embedded commas/newlines/quotes in PROMPT and TRUTH round-trip safely
    (open in a spreadsheet, or load with pandas.read_csv). No row limit here;
    slice in the caller if you want fewer rows.

    Returns (rows_written, out_path). Skips the build when the file already
    exists unless force=True.
    """
    if os.path.exists(out_path) and not force:
        return None, out_path
    rows = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(["FILE", "ID", "PROMPT", "TRUTH", "HAS_QUESTION"])
        for a in ADAPTERS.values():
            with open(os.path.join(data_dir, a["file"])) as f:
                records = json.load(f)
            for r in records:
                writer.writerow([
                    a["file"], r["id"], a["prompt"](r), a["truth"](r), a["has_question"],
                ])
                rows += 1
    return rows, out_path


if __name__ == "__main__":
    n, path = build_adapters_csv()
    if n is None:
        print(f"adapters.csv already exists -> {path} (delete it or call "
              f"build_adapters_csv(force=True) to rebuild)")
    else:
        print(f"Wrote {n} rows -> {path}")