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

JUDGE_SUITE = [
    {"key": "claude", "model": "anthropic/claude-sonnet-5"},
    {"key": "gpt", "model": "openai/gpt-4o-mini"},

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
        prompt=lambda r: f"{r['fact_pattern']}\n\nQuestion: {r['question']}",
        truth=lambda r: f"{r['analysis']}\n\nLikely outcome: {r['likely_outcome']}",
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


if __name__ == "__main__":
    for name in ADAPTERS:
        rid, prompt, truth = next(load_dataset(name, limit=1))
        print(f"[{name}] id={rid}")
        print(f"  prompt[:100]: {prompt[:100]!r}")
        print(f"  truth[:100] : {truth[:100]!r}\n")
