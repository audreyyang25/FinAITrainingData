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


# --- the three-model common-arm design --------------------------------------
# A per-model arm split makes pre/post deltas incomparable ACROSS models,
# because each model's cutoff carves a different case set out of the corpus.
# Fable 5's pre-cutoff arm holds 104 cases and Gemini 3.1 Pro's holds 93, and
# they are not the same 93 -- so a Fable-vs-Gemini comparison of pre/post drop
# is partly a comparison of two different case mixes.
#
# This design fixes that by scoring every model on ONE case set: a case counts
# as pre only if it predates the EARLIEST cutoff in the suite, and post only if
# it postdates the LATEST. Between those two dates a case is post-cutoff for
# some of these models and pre-cutoff for others, which makes it uninterpretable
# in a pooled comparison -- so it is excluded rather than assigned.
#
# The band is wide because Gemini 3.1 Pro's cutoff is 2025-01-31, thirteen
# months earlier than GPT-5.6 Sol's 2026-02-28. It is NOT the one-month gap
# between Fable 5 and Sol; assuming that drops 0 cases and leaves 10 misassigned
# ones in, all filed 2025-07-01.
#
# Bounds are derived from CUTOFF rather than written down, so a corrected cutoff
# moves the boundaries instead of silently disagreeing with them.
NEWER = ["anthropic/claude-fable-5",
         "openai/gpt-5.6-sol",
         "google/gemini-3.1-pro-preview"]

COMMON_PRE_MAX = min(CUTOFF[m] for m in NEWER)    # 2025-01-31, Gemini 3.1 Pro
COMMON_POST_MIN = max(CUTOFF[m] for m in NEWER)   # 2026-02-28, GPT-5.6 Sol


def common_arm(date_filed: str) -> str | None:
    """pre_cutoff / post_cutoff for ALL of NEWER, or None if ambiguous.

    None means "exclude", and callers must honour it rather than defaulting it
    into an arm -- an ambiguous case scored as pre-cutoff recall is being graded
    on knowledge one of the three models could not have had.
    """
    d = (date_filed or "")[:10]
    if not d:
        return None
    if d <= COMMON_PRE_MAX:
        return "pre_cutoff"
    if d > COMMON_POST_MIN:
        return "post_cutoff"
    return None


# The corpus half of the same design: federal courts of appeals only. Court
# level shifts the disposition base rate (appellate courts affirm 75-80%; trial
# courts grant or deny), so mixing levels puts a second variable alongside the
# pre/post one. See Memorization/crawl/fetch_recent.py, which now enforces the
# same restriction at search time.
FED_APPEAL_DIR = "fed_appeal_court_opinions"
