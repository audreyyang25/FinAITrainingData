import csv
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Repo-root-relative path to the five training files.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "AI Suitability Training Materials",
    "23_Folders_Suitability",
)

# Materialized ADAPTERS output: flat, human-readable form of the adapters below.
ADAPTERS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters.csv")

# Model suite: each model both generates and judges (self-bias study).
# All routed through OpenRouter (one OpenAI-compatible endpoint), so `model`
# is an OpenRouter slug. Verify exact slugs at https://openrouter.ai/models —
# they change over time; these are five distinct vendors for cross-model bias.
GEN_SUITE = [
    {"key": "claude",   "model": "anthropic/claude-opus-4.8"}, # opus-4.8
    {"key": "gpt",      "model": "openai/gpt-5.5"}, # gpt-5.5
    {"key": "gemini",   "model": "google/gemini-2.5-pro"},
    {"key": "qwen",    "model": "qwen/qwen3-235b-a22b"},
    {"key": "deepseek", "model": "deepseek/deepseek-r1"},
]

GPT_SUITE = [
    {"key": "gpt-3.5", "model": "openai/gpt-3.5-turbo"},
    {"key": "gpt-4", "model": "openai/gpt-4-turbo"},
    {"key": "gpt-4o", "model": "openai/gpt-4o"},
    {"key": "gpt-5.5", "model": "openai/gpt-5.5"},
]

JUDGE_SUITE = [
    {"key": "claude", "model": "anthropic/claude-sonnet-5"},
    {"key": "gpt", "model": "openai/gpt-4o-mini"},
    # {"key": "gemini-3.5", "model": "google/gemini-3.5-flash"},
]


def _format_convo(turns):
    """P14 conversations are a list of {speaker, text} dicts."""
    return "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in turns)


# Dataset adapters
# prompt(record)       -> the task shown to a GENERATOR
# truth(record)        -> the reference answer shown to a JUDGE
# has_question         -> True if the task contains an explicit question
ADAPTERS = {
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
        yield r["id"], a["prompt"](r), a["truth"](r)


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
