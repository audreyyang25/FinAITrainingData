"""Target suite and knowledge cutoffs for the outcome probe.

Cutoffs drive the pre/post arm split, which is the control this whole design
rests on: a model cannot know the outcome of a case decided after its training
data ends, so its post-cutoff score is the empirical floor for that model. They
are recorded here rather than passed on the command line so a run cannot
silently use the wrong one.

Cutoffs for the three newer models were supplied by the author; the three
originals carry over from the memorization probe.

Llama 3.1 70B is the open-weight comparison, and it is not an arbitrary one:
Cooper et al. name it as substantially memorizing The Great Gatsby, which is
why Gatsby is the positive control in build_controls.py. Running it points the
harness back at the model the replicated result came from.

Two things follow from its 2023-12 cutoff, both worth remembering before
comparing it to the others:
  * Its post-cutoff arm is the largest in the suite (42 of 106 federal cases
    against 18-31 for the frontier models), so its floor is the best estimated.
  * Its pre-cutoff arm is therefore a DIFFERENT case mix, so a pre/post delta
    for Llama is not like-for-like with a pre/post delta for, say, Fable 5.
"""
from __future__ import annotations

TARGETS = [
    # (openrouter id, short label, knowledge cutoff YYYY-MM-DD)
    ("openai/gpt-5",                  "GPT-5",          "2024-09-30"),
    ("anthropic/claude-opus-4",       "Claude Opus 4",  "2025-01-31"),
    ("google/gemini-2.5-pro",         "Gemini 2.5 Pro", "2025-01-31"),
    ("openai/gpt-5.6-sol",            "GPT-5.6 Sol",    "2026-02-28"),
    ("anthropic/claude-fable-5",      "Claude Fable 5", "2026-01-31"),
    ("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro", "2025-01-31"),
    # Open-weight. Confirmed by the author as 2023-12.
    ("meta-llama/llama-3.1-70b-instruct", "Llama 3.1 70B", "2023-12-31"),
]

CUTOFF = {m: c for m, _, c in TARGETS}
LABEL = {m: l for m, l, _ in TARGETS}

# Judge. Claude is also a target family, so its verdicts carry a self-preference
# risk that is accepted rather than avoided -- every judge capable enough for
# nuanced legal comparison belongs to one of the three target families. The
# check is JUDGE_ALT on a subset: what matters is not absolute agreement but
# whether the Claude-vs-others gap moves when the judge changes family.
JUDGE = "anthropic/claude-opus-5"
JUDGE_ALT = "google/gemini-3.1-pro-preview"


def arm(date_filed: str, model: str) -> str:
    """pre_cutoff if the opinion predates the model's training data, else post."""
    c = CUTOFF.get(model, "")
    if not date_filed or not c:
        return "unknown"
    return "pre_cutoff" if date_filed[:10] <= c else "post_cutoff"
