"""General judging pipeline for the suitability evals (P12-P16).

A dataset-agnostic generalization of intrafamily_exp/family_model_judging.py.
The four judge dimensions live here; WHICH of them run for a given dataset, and
where each pulls its reference material, is declared per-dataset in
dataset_specs.py. The driver (`run_eval`) auto-detects which datasets are present
in a generations file and judges each against its own gold, so one call scores an
arbitrary model suite across whatever datasets the generations contain.

Dimensions
  1. factor_recall        checklist vs. the dataset's reference factors   (if any)
  2. legal_grounding      citation extraction + code-side fabrication flag (always)
  3. outcome              soft directional-consistency OR binary violation match
  4. content_similarity   holistic 0-1 substance match to the gold reference (always)

Outcome dimension (per the dataset spec's outcome.mode):
  - "soft":   direction_consistent + justified, vs a reference outcome string.
              -> summary keys outcome_direction, outcome_justified
  - "binary": the response's violation/no-violation verdict must EXACTLY match the
              gold verdict (read from a boolean field, or classified once from
              gold text). -> summary keys outcome_exact, outcome_justified

Judge: google/gemini-3.5-flash via OpenRouter, temperature 0, low reasoning effort.
Every binary judgment requires an evidence quote; no quote => NO. All judge output
is strict JSON, validated and retried.

Usage (programmatic):
    from part1_eval.general_judge import run_eval
    run_eval("generations.json",
             out_dir="outputs/part1_eval/gemini",
             exclude_models={"gold"})
"""

import csv
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import random

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tqdm import tqdm

from part1_eval.dataset_specs import SPECS, load_gold

# Provider-pinned judges (Llama bf16) route to ONE upstream shared pool that
# rate-limits under concurrency, so the default stays modest. Bump it only for an
# unpinned judge or when using your own provider key (BYOK).
DEFAULT_WORKERS = 4

# Key lives in the repo-root .env (three levels up: eval_pipeline -> Data_Analysis -> repo).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Judge registry: model slug -> per-model call config. `reasoning_effort` is None
# for non-reasoning models (e.g. Llama); `provider` is OpenRouter provider routing
# (None = unpinned). To add a judge, add an entry here.
JUDGES = {
    "google/gemini-3.5-flash": {"reasoning_effort": "low", "provider": None},
    # Same bf16 pin used as the feature-coverage extractor/judge, so precision does
    # not vary call-to-call (Llama 3.3-70B is served at mixed quantizations).
    "meta-llama/llama-3.3-70b-instruct": {
        "reasoning_effort": None,
        "provider": {"quantizations": ["bf16"], "allow_fallbacks": False},
        # The bf16 pin routes to the minority of providers serving this model at bf16,
        # with no fallback -- so concurrency collides with a small pool and 429s far
        # sooner than for unpinned models. Fewer workers is the right lever: relaxing
        # the pin would trade rate limits for a judge whose numeric precision drifts
        # mid-run, which is what a reliability study can least afford.
        "workers": 2,
    },
    # Cross-model judging panel. Both families sit OUTSIDE the 12-model generator
    # pool (anthropic / google / openai / qwen), so no judge ever scores its own
    # family's output and the panel carries no self-preference term.
    # DeepSeek is effectively first-party on OpenRouter, so it needs no provider pin.
    "deepseek/deepseek-v3.2": {"reasoning_effort": "low", "provider": None},
    "mistralai/mistral-large-2512": {"reasoning_effort": None, "provider": None},
}
DEFAULT_JUDGE = "google/gemini-3.5-flash"


def judge_workers(model: str, default: int) -> int:
    """Per-judge concurrency override from JUDGES, else `default`. Lets a
    provider-pinned judge run slower without throttling the rest of the panel."""
    return (JUDGES.get(model) or {}).get("workers") or default


JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 4000                    # headroom for JSON + evidence quotes
# 8 attempts with the rate-limit backoff below covers ~4 min of sustained 429s before
# giving up. Failures are logged, not written to the detail file, so a rerun retries
# only what failed -- but more headroom here means fewer reruns.
MAX_RETRIES = 8                            # total attempts per judge call
SCHEMA_RETRIES = 2                         # attempts for a DETERMINISTIC schema failure
RETRY_BASE_SLEEP = 2.0                     # backoff base for schema/transient errors
RATE_LIMIT_BASE_SLEEP = 8.0               # 429 from a shared provider pool: wait longer
RATE_LIMIT_MAX_SLEEP = 60.0

# Active judge config -- set_judge() mutates these so call_judge stays simple.
JUDGE_MODEL = DEFAULT_JUDGE
JUDGE_REASONING_EFFORT = JUDGES[DEFAULT_JUDGE]["reasoning_effort"]
JUDGE_PROVIDER = JUDGES[DEFAULT_JUDGE]["provider"]


def set_judge(model: str):
    """Select the judge model for the whole run. Known models pick up their pinned
    provider routing / reasoning setting from JUDGES; unknown models default to no
    reasoning effort and no provider pin."""
    global JUDGE_MODEL, JUDGE_REASONING_EFFORT, JUDGE_PROVIDER
    cfg = JUDGES.get(model, {"reasoning_effort": None, "provider": None})
    JUDGE_MODEL = model
    JUDGE_REASONING_EFFORT = cfg["reasoning_effort"]
    JUDGE_PROVIDER = cfg["provider"]


_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client


# Whitelist of real authorities for this question set. The judge only EXTRACTS
# citations; this list is how *code* flags likely fabrications for manual review.
AUTHORITY_WHITELIST_PATTERNS = [
    r"finra\s+rule\s+2111", r"rule\s+2111", r"suitability\s+rule",
    r"regulation\s+best\s+interest", r"reg\.?\s*bi", r"rule\s+15l-1",
    r"regulatory\s+notice\s+12-25", r"notice\s+12-25",
    r"securities\s+exchange\s+act(\s+of\s+1934)?",
    r"investment\s+advisers\s+act(\s+of\s+1940)?",
    r"finra\s+rule\s+2090", r"know\s+your\s+customer",
    r"release\s+no\.?\s*34-86031",
    r"finra\s+rule\s+3110",
    r"care\s+obligation", r"fiduciary\s+duty",
]

SHARED_SYSTEM_PROMPT = """You are an expert evaluator assessing AI responses to \
financial-advisory compliance case studies (FINRA suitability / Regulation Best Interest).

Rules that apply to every evaluation:
1. Judge substance, not style. Do not reward length, formatting, confident tone, or polish.
2. Semantic matching: a point counts if its SUBSTANCE is addressed, regardless of wording.
3. Evidence before verdict: for every judgment, first quote the exact passage(s) from the
   response that support your decision. If you cannot find a supporting quote, answer NO.
4. Do not use your own knowledge of the case to fill gaps. Evaluate only what is written.
5. You do not know which AI system produced the response. Do not guess or let suspected
   identity affect scores.
6. Output valid JSON only, exactly matching the schema provided. No preamble, no markdown
   fences, no commentary outside the JSON object."""


# ----------------------------------------------------------------------------
# Judge call plumbing
# ----------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def call_judge(user_prompt: str, validator) -> dict:
    """Call the judge with retries; `validator(parsed) -> None` raises on bad schema."""
    last_err = None
    last_raw = None
    extra_body = {}
    if JUDGE_REASONING_EFFORT:
        extra_body["reasoning"] = {"effort": JUDGE_REASONING_EFFORT}
    if JUDGE_PROVIDER:
        extra_body["provider"] = JUDGE_PROVIDER
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = dict(
                model=JUDGE_MODEL,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SHARED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = get_client().chat.completions.create(**kwargs)
            last_raw = resp.choices[0].message.content
            parsed = json.loads(_strip_fences(last_raw))
            validator(parsed)
            return parsed
        except Exception as e:  # JSON errors, schema errors, transient API errors
            last_err = e
            # Rate limits (429) from a shared upstream pool are transient but need a
            # longer, jittered wait than schema errors -- otherwise every worker
            # retries in lockstep and re-triggers the limit.
            msg = str(e)
            is_rate_limit = isinstance(e, RateLimitError) or "429" in msg or "rate-limit" in msg.lower()
            # A schema failure at temperature 0 is deterministic: the same prompt returns
            # the same unusable response every time, so retrying it just multiplies the
            # cost of a certain failure. Rate limits and transport errors DO clear, so
            # only those keep their full budget.
            if not is_rate_limit and isinstance(e, (AssertionError, json.JSONDecodeError)):
                if attempt >= SCHEMA_RETRIES - 1:
                    break
            base = RATE_LIMIT_BASE_SLEEP if is_rate_limit else RETRY_BASE_SLEEP
            cap = RATE_LIMIT_MAX_SLEEP if is_rate_limit else 30.0
            time.sleep(min(cap, base * (2 ** attempt)) + random.uniform(0, 2))
    # str() on a bare assert is empty, so name the exception type and show what the
    # judge actually returned -- otherwise the failure reports nothing usable.
    detail = str(last_err) or "(no message)"
    raise RuntimeError(
        f"Judge {JUDGE_MODEL} failed: {type(last_err).__name__}: {detail}"
        f"\n  raw response: {(last_raw or '')[:400]!r}")


# ----------------------------------------------------------------------------
# Dimension 1: factor recall (checklist vs. the dataset's reference factors)
# ----------------------------------------------------------------------------

def judge_factor_recall(fact_pattern: str, response: str, gold_factors: list) -> dict:
    factor_lines = "\n".join(f"{i+1}. {f}" for i, f in enumerate(gold_factors))
    prompt = f"""CASE FACT PATTERN:
{fact_pattern}

RESPONSE UNDER EVALUATION:
{response}

TASK: For each reference factor below, determine whether the response SUBSTANTIVELY
addresses it. The response does not need to use the same wording or reach the same
conclusion about it -- it only needs to engage with the underlying issue somewhere.

REFERENCE FACTORS:
{factor_lines}

Also list any substantive analytical points the response raises that do NOT appear in
the factor list (novel_points). Extract them verbatim or near-verbatim; do not score them.

OUTPUT SCHEMA (JSON only):
{{
  "factors": [
    {{"id": 1, "addressed": true, "evidence_quote": "..."}},
    {{"id": 2, "addressed": false, "evidence_quote": null}}
  ],
  "novel_points": ["...", "..."]
}}
The "factors" array must contain exactly {len(gold_factors)} entries, ids 1..{len(gold_factors)}."""

    def validate(p):
        assert isinstance(p.get("factors"), list) and len(p["factors"]) == len(gold_factors)
        for f in p["factors"]:
            assert isinstance(f.get("addressed"), bool)
            if f["addressed"]:
                assert f.get("evidence_quote"), "addressed=true requires evidence_quote"
        assert isinstance(p.get("novel_points"), list)

    out = call_judge(prompt, validate)
    out["recall"] = sum(f["addressed"] for f in out["factors"]) / len(gold_factors)
    return out


# ----------------------------------------------------------------------------
# Dimension 2: legal grounding (extraction by judge, verification in code)
# ----------------------------------------------------------------------------

def judge_legal_grounding(response: str) -> dict:
    prompt = f"""RESPONSE UNDER EVALUATION:
{response}

TASK: Extract EVERY specific rule, regulation, regulatory notice, statute, release
number, or court/arbitration case the response cites, verbatim as written.

OUTPUT SCHEMA (JSON only):
{{
  "cited_authorities": ["FINRA Rule 2111", "..."]
}}"""

    def validate(p):
        assert isinstance(p.get("cited_authorities"), list)

    out = call_judge(prompt, validate)
    flagged = []
    for cite in out["cited_authorities"]:
        low = cite.lower()
        if not any(re.search(pat, low) for pat in AUTHORITY_WHITELIST_PATTERNS):
            flagged.append(cite)
    out["unverified_citations"] = flagged
    return out


# ----------------------------------------------------------------------------
# Dimension 3a: outcome, SOFT (directional consistency vs a reference outcome)
# ----------------------------------------------------------------------------

def judge_outcome_soft(response: str, gold_outcome: str) -> dict:
    prompt = f"""RESPONSE UNDER EVALUATION:
{response}

REFERENCE OUTCOME ASSESSMENT:
{gold_outcome}

TASK: Evaluate the response's prediction of the likely regulatory/compliance outcome
with two independent binary judgments.

1. direction_consistent: Is the response's predicted outcome DIRECTIONALLY consistent
   with the reference? Directional consistency means agreeing on the general shape of
   the result (e.g., close call / partial liability / mitigated finding vs. full
   violation vs. full exoneration). Do NOT require matching specific damages,
   remedies, forums, or procedural details. If the response makes no outcome
   prediction at all, answer false.

2. justified: Is the predicted outcome JUSTIFIED by the response's own analysis -- does
   it connect the prediction to specific facts or arguments it raised, rather than
   asserting the outcome without support? Judge this independently of direction.

OUTPUT SCHEMA (JSON only):
{{
  "direction_consistent": {{"value": true, "evidence_quote": "..."}},
  "justified": {{"value": false, "evidence_quote": null}}
}}"""

    def validate(p):
        for k in ("direction_consistent", "justified"):
            assert isinstance(p.get(k, {}).get("value"), bool)
            if p[k]["value"]:
                assert p[k].get("evidence_quote"), f"{k}=true requires evidence_quote"

    return call_judge(prompt, validate)


# ----------------------------------------------------------------------------
# Dimension 3b: outcome, BINARY (violation verdict, exact match to gold)
# ----------------------------------------------------------------------------

_VERDICT_TO_BOOL = {"violation": True, "no_violation": False, "unclear": None}


def judge_verdict(text: str) -> dict:
    """Classify the bottom-line compliance conclusion of an analysis into
    violation / no_violation / unclear, plus whether it is justified by the text.
    Used for the model response AND (when gold has no boolean truth) for the gold
    reference itself, so the same classifier defines both sides of the match."""
    prompt = f"""COMPLIANCE ANALYSIS UNDER EVALUATION:
{text}

TASK: Classify the analysis's BOTTOM-LINE conclusion about whether the financial
advisor violated their suitability / best-interest obligations.

1. verdict: one of
   - "violation": concludes the advisor DID violate (or likely violated) their obligations.
   - "no_violation": concludes the advisor did NOT violate / acted appropriately.
   - "unclear": no clear bottom-line conclusion is stated either way.
2. justified: is the conclusion connected to specific facts or arguments in the text,
   rather than asserted without support? Judge independently of which verdict it is.

OUTPUT SCHEMA (JSON only):
{{
  "verdict": "violation",
  "justified": true,
  "evidence_quote": "..."
}}
evidence_quote is required unless verdict is "unclear"; use null only then."""

    def validate(p):
        # Messages are not optional here: a bare assert stringifies to "", so the
        # retry loop's final RuntimeError reported nothing at all about the cause.
        assert p.get("verdict") in _VERDICT_TO_BOOL, (
            f"verdict must be one of {sorted(_VERDICT_TO_BOOL)}, got {p.get('verdict')!r} "
            "(references that give different verdicts for different advisors do not fit "
            "this schema -- see the multi-party cases)")
        assert isinstance(p.get("justified"), bool), \
            f"justified must be a bool, got {type(p.get('justified')).__name__}"
        if p["verdict"] != "unclear":
            assert p.get("evidence_quote"), "a stated verdict requires evidence_quote"

    return call_judge(prompt, validate)


# ----------------------------------------------------------------------------
# Dimension 4: overall content similarity to gold (0-1, holistic)
# ----------------------------------------------------------------------------

def judge_content_similarity(response: str, gold_reference: str) -> dict:
    prompt = f"""REFERENCE (GOLD) ANALYSIS:
{gold_reference}

RESPONSE UNDER EVALUATION:
{response}

TASK: Give ONE overall content-similarity score from 0.0 to 1.0 for how well the
response matches the SUBSTANCE of the reference analysis -- the issues it raises,
the reasoning it applies, and the conclusions it reaches. Holistic, not a checklist:
  - 1.0 = substantively equivalent: same key issues, reasoning, and conclusions.
  - ~0.5 = captures some central points but misses or diverges on others.
  - 0.0 = unrelated to, or contradicts, the reference's substance.
Judge substance, not wording, length, tone, or formatting. Do not reward content
absent from the reference; do not penalize different phrasing of the same idea.

OUTPUT SCHEMA (JSON only):
{{
  "content_similarity": 0.0,
  "reasoning": "1-3 sentences: what substantively matched and what was missing or divergent."
}}"""

    def validate(p):
        v = p.get("content_similarity")
        assert isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0
        assert isinstance(p.get("reasoning"), str) and p["reasoning"].strip()

    out = call_judge(prompt, validate)
    out["content_similarity"] = max(0.0, min(1.0, float(out["content_similarity"])))
    return out


# Anchors for the cumulative quality score. Every level is described, not just the
# endpoints: the pilot panel (archive/judge.py) defined only 1.0 and 0.0 and its five
# judges diverged by up to 0.6 on answers they described identically, because nothing
# pinned the middle of the scale. The judge must also NAME the anchor it picked, which
# gives a coarse categorical agreement signal that survives numeric scale drift --
# two judges can land on 0.65 vs 0.75 and still agree the answer is "sound".
QUALITY_ANCHORS = """  1.0 -- Expert-level. Reaches the reference's conclusion, grounded in the applicable
        standards, covering the issues that matter. No legal errors.
  0.7 -- Sound. Right conclusion and broadly correct grounding, but misses a secondary
        issue or states a standard imprecisely.
  0.5 -- Mixed. Addresses some central issues but omits or errs on others, OR reaches
        the right conclusion on materially incomplete reasoning.
  0.3 -- Poor. Contains a major legal error, treats an inapplicable standard as
        governing, or reaches a conclusion its own reasoning does not support.
  0.0 -- Wrong. Reaches a bottom-line conclusion contrary to the reference, or
        fabricates authority."""


def judge_overall_quality(response: str, gold_reference: str, fact_pattern: str) -> dict:
    """Cumulative 0-1 quality score for the response as a whole, with an explanation.

    Distinct from content_similarity in BOTH what it asks and what it sees.
    content_similarity measures overlap with the reference and receives only the
    reference plus the response; this asks whether the response is a good answer ON
    THE FACTS, so it also receives the fact pattern.

    That difference is deliberate. Scoring "correctness of the conclusion" against the
    reference alone cannot distinguish a wrong answer from one that is right by a
    different route, and it makes the reference infallible by construction. Handing the
    judge the facts lets it assess the law, and lets a human rater doing the same task
    be compared to it on equal footing.
    """
    prompt = f"""OVERALL ANSWER QUALITY

CASE FACTS:
{fact_pattern}

REFERENCE ANALYSIS (the correct analysis of this case):
{gold_reference}

RESPONSE UNDER EVALUATION:
{response}

TASK: Give ONE cumulative quality score for the response as a whole -- how good an
answer it is to THIS CASE, taking together its legal grounding, the correctness of its
conclusion, and its coverage of the issues that matter.

Treat the reference analysis as CORRECT. A response that reaches a different bottom-line
conclusion is wrong, however well argued. The CASE FACTS are given so you can judge
whether the response's reasoning actually holds on this case -- not merely whether it
echoes the reference's wording.

The reference is authoritative but not exhaustive: if the response raises an ADDITIONAL
point that is correct on these facts, do not penalise it for going beyond the reference.
Do penalise a wrong conclusion, a misstated standard, and material issues left out.

Pick the closest anchor below, then optionally adjust by up to 0.1 to place the
response within that band:

{QUALITY_ANCHORS}

Judge substance, not wording, length, tone, or formatting. A short answer that is
correct and well grounded outranks a long one that is not. Credit only reasoning the
response actually states -- do not fill in steps it left out.

OUTPUT SCHEMA (JSON only):
{{
  "anchor": 0.7,
  "overall_quality": 0.65,
  "explanation": "2-4 sentences: what the response got right, what it got wrong, and why that places it at the anchor you chose."
}}"""

    def validate(p):
        assert p.get("anchor") in (0.0, 0.3, 0.5, 0.7, 1.0), "anchor must be one of the five"
        v = p.get("overall_quality")
        assert isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0
        assert isinstance(p.get("explanation"), str) and p["explanation"].strip()

    out = call_judge(prompt, validate)
    out["overall_quality"] = max(0.0, min(1.0, float(out["overall_quality"])))
    out["anchor"] = float(out["anchor"])
    # Kept raw rather than clamped to anchor +/- 0.1: a judge that names one anchor and
    # scores far from it is a calibration finding worth seeing, not one to paper over.
    out["anchor_gap"] = round(abs(out["overall_quality"] - out["anchor"]), 3)
    return out


# ----------------------------------------------------------------------------
# Per-response orchestration
# ----------------------------------------------------------------------------

def _gold_verdict(outcome_spec, gold, verdict_cache, cache_key) -> bool:
    """Resolve the gold violation verdict for a binary dataset: read a boolean
    field directly, or classify gold text once (cached per case)."""
    if "gold_violation" in outcome_spec:
        return bool(outcome_spec["gold_violation"](gold))
    # Derive from gold text (e.g. P16 correct_answer), classified once and cached.
    if cache_key not in verdict_cache:
        v = judge_verdict(outcome_spec["gold_text"](gold))
        verdict_cache[cache_key] = _VERDICT_TO_BOOL[v["verdict"]]
    return verdict_cache[cache_key]


# Dimensions judge_one can score. Each costs one judge call per response, so a run
# that only needs one of them should say so rather than pay for four. DEFAULT is the
# original full eval; the cross-model judging subset asks for overall_quality alone.
DEFAULT_DIMENSIONS = ("factor_recall", "legal_grounding", "outcome", "content_similarity")
ALL_DIMENSIONS = DEFAULT_DIMENSIONS + ("overall_quality",)


def judge_one(dataset, spec, gold, gen, verdict_cache,
              dimensions=DEFAULT_DIMENSIONS) -> dict:
    resp = gen["response"]
    rec = {
        "dataset": dataset,
        "case_id": gen["case_id"],
        "topic": gold.get("topic"),
        "judge": JUDGE_MODEL,
        "model": gen["model"],
        "model_family": gen.get("model_family"),
        "model_key": gen.get("model_key"),
        "run": gen.get("run", 0),
    }

    # 1. Factor recall (only if this dataset supplies a reference checklist).
    if "factor_recall" in dimensions and spec.get("factors") is not None:
        rec["factor_recall"] = judge_factor_recall(
            spec["fact_pattern"](gold), resp, spec["factors"](gold)
        )

    # 2. Legal grounding.
    if "legal_grounding" in dimensions:
        rec["legal_grounding"] = judge_legal_grounding(resp)

    # 3. Outcome (soft or binary, per the spec).
    oc = spec.get("outcome") if "outcome" in dimensions else None
    if oc and oc["mode"] == "soft":
        o = judge_outcome_soft(resp, oc["ref"](gold))
        rec["outcome"] = {"mode": "soft", **o}
    elif oc and oc["mode"] == "binary":
        pred = judge_verdict(resp)
        pred_viol = _VERDICT_TO_BOOL[pred["verdict"]]
        gold_viol = _gold_verdict(oc, gold, verdict_cache, (dataset, gen["case_id"]))
        rec["outcome"] = {
            "mode": "binary",
            "pred_verdict": pred["verdict"],
            "gold_violation": gold_viol,
            "exact": (pred_viol is not None) and (pred_viol == gold_viol),
            "justified": {"value": pred["justified"],
                          "evidence_quote": pred.get("evidence_quote")},
        }

    # 4. Content similarity.
    if "content_similarity" in dimensions:
        rec["content_similarity"] = judge_content_similarity(resp, spec["content_ref"](gold))

    # 5. Cumulative quality (anchored rubric + explanation). Unlike the other
    #    dimensions this also gets the fact pattern -- it judges the answer on the
    #    law, not just its overlap with the reference.
    if "overall_quality" in dimensions:
        rec["overall_quality"] = judge_overall_quality(
            resp, spec["content_ref"](gold), spec["fact_pattern"](gold))

    rec["summary"] = _summary(rec)
    return rec


def _summary(rec) -> dict:
    """Flat scalars for aggregation. Every key is None when its dimension was not
    scored, so a record from a single-dimension run has the same shape as a full one."""
    oc = rec.get("outcome", {})
    return {
        "content_similarity": (round(rec["content_similarity"]["content_similarity"], 3)
                               if "content_similarity" in rec else None),
        "factor_recall": (round(rec["factor_recall"]["recall"], 3)
                          if "factor_recall" in rec else None),
        "n_unverified_citations": (len(rec["legal_grounding"]["unverified_citations"])
                                   if "legal_grounding" in rec else None),
        # soft datasets populate outcome_direction; binary datasets populate outcome_exact
        "outcome_direction": (oc.get("direction_consistent", {}).get("value")
                              if oc.get("mode") == "soft" else None),
        "outcome_exact": (oc.get("exact") if oc.get("mode") == "binary" else None),
        "outcome_justified": (oc.get("justified", {}).get("value") if oc else None),
        # None unless the cumulative-quality dimension was enabled for this run.
        "overall_quality": (round(rec["overall_quality"]["overall_quality"], 3)
                            if "overall_quality" in rec else None),
        "quality_anchor": (rec["overall_quality"]["anchor"]
                           if "overall_quality" in rec else None),
    }


# ----------------------------------------------------------------------------
# Generations loading / normalization
# ----------------------------------------------------------------------------

# Default field map matches the SETUP generations.json (answer -> response).
DEFAULT_FIELD_MAP = {
    "dataset": "dataset",
    "case_id": "case_id",
    "model": "model",
    "response": "answer",
    "run": "run",
    "model_family": "model_family",
    "model_key": "model_key",
}


def load_generations(path, field_map=None):
    """Read a JSONL (or JSON list) of generations and normalize each row to the
    judge's shape: {dataset, case_id, model, run, response, model_family, model_key}.
    `field_map` overrides source field names (see DEFAULT_FIELD_MAP)."""
    fm = {**DEFAULT_FIELD_MAP, **(field_map or {})}
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    out = []
    for r in rows:
        out.append({
            "dataset": r[fm["dataset"]],
            "case_id": r[fm["case_id"]],
            "model": r[fm["model"]],
            "response": r[fm["response"]],
            "run": r.get(fm["run"], 0),
            "model_family": r.get(fm["model_family"]),
            "model_key": r.get(fm["model_key"]),
        })
    return out


# ----------------------------------------------------------------------------
# Concurrency helpers
# ----------------------------------------------------------------------------

def parallel_yield(fn, items, max_workers):
    """Run fn(item) across a thread pool; yield results as they complete. I/O-bound
    LLM calls release the GIL during the network wait, so threads give real
    concurrency; the CALLER does all writes (single writer, no append race).
    Mirrors shared/utils.py's parallel_yield."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for fut in as_completed(futures):
            yield fut.result()


def _append_jsonl(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")


def _precompute_gold_verdicts(tasks, gold_by_ds, workers):
    """For binary datasets that derive the gold verdict from TEXT (e.g. P16's
    correct_answer), classify each needed gold answer ONCE, in parallel, before the
    main judging loop. Worker threads then only read the result, so there is no
    shared-cache write race. Datasets with a direct boolean truth field (e.g. P12's
    `compliant`) need nothing here -- they are resolved inline in judge_one.
    Returns {(dataset, case_id): bool}."""
    need = {}  # (dataset, case_id) -> gold_text, deduped
    for g in tasks:
        oc = SPECS[g["dataset"]].get("outcome") or {}
        if oc.get("mode") == "binary" and "gold_text" in oc:
            k = (g["dataset"], g["case_id"])
            if k not in need:
                gold = gold_by_ds[g["dataset"]].get(g["case_id"])
                if gold is not None:
                    need[k] = oc["gold_text"](gold)
    if not need:
        return {}
    print(f"Pre-classifying {len(need)} gold verdicts (binary datasets)...")

    def classify(item):
        k, text = item
        return (k, _VERDICT_TO_BOOL[judge_verdict(text)["verdict"]])

    cache = {}
    for k, val in tqdm(parallel_yield(classify, list(need.items()), workers),
                       total=len(need)):
        cache[k] = val
    return cache


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run_eval(generations, out_dir="outputs/part1_eval/gemini", gold_dir=None,
             exclude_models=frozenset({"gold"}), datasets=None,
             limit=0, field_map=None, judge=DEFAULT_JUDGE, workers=DEFAULT_WORKERS,
             dimensions=DEFAULT_DIMENSIONS):
    """Judge a set of generations across whatever datasets they contain.

    generations:    path to a generations JSONL/JSON, or an already-normalized list
                    of {dataset, case_id, model, run, response, ...} dicts.
    out_dir:        output directory for detail/aggregate/review files.
    gold_dir:       override the gold directory (defaults to dataset_specs.GOLD_DIR).
    exclude_models: model labels to drop (default {"gold"} -- the canonicalizer answer).
    datasets:       restrict to these dataset names (default: all present that have a spec).
    limit:          judge only the first N generations after filtering (0 = all).
    field_map:      source->judge field-name overrides for loading (see load_generations).
    judge:          judge model slug (see JUDGES). Use a SEPARATE out_dir per judge --
                    resume keys don't include the judge, so mixing judges in one dir
                    would cross-contaminate.
    workers:        concurrent judge threads (default DEFAULT_WORKERS). I/O-bound, so
                    higher is usually faster; back off if you hit provider rate limits.
    dimensions:     which of ALL_DIMENSIONS to score. Each is one judge call per
                    response, so restrict this when a run only needs some of them.
                    Omitting "outcome" also skips the gold-verdict pre-classification.
    """
    set_judge(judge)
    print(f"Judge: {JUDGE_MODEL} (reasoning={JUDGE_REASONING_EFFORT}, "
          f"provider_pin={'yes' if JUDGE_PROVIDER else 'no'})")

    if isinstance(generations, (str, Path)):
        generations = load_generations(generations, field_map)
    gens = list(generations)

    # Filter: excluded models, unknown/unwanted datasets.
    kept, skipped_ds = [], set()
    want = set(datasets) if datasets else None
    for g in gens:
        if g["model"] in exclude_models:
            continue
        if g["dataset"] not in SPECS:
            skipped_ds.add(g["dataset"]);  continue
        if want and g["dataset"] not in want:
            continue
        kept.append(g)
    if skipped_ds:
        print(f"[warn] no spec for dataset(s) {sorted(skipped_ds)}; skipped")
    if limit:
        kept = kept[:limit]

    present = sorted({g["dataset"] for g in kept})
    print(f"Judging {len(kept)} generations across datasets: {present}")

    gold_kwargs = {"gold_dir": gold_dir} if gold_dir else {}
    gold_by_ds = {ds: load_gold(ds, **gold_kwargs) for ds in present}

    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    detail_path = outdir / "judgments_detail.jsonl"
    fails_path = outdir / "judgments_failures.jsonl"

    # Resume: skip any (dataset, case_id, model, run) already judged.
    done = set()
    if detail_path.exists():
        for line in detail_path.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                done.add((j["dataset"], j["case_id"], j["model"], j["run"]))
    tasks = [g for g in kept
             if (g["dataset"], g["case_id"], g["model"], g.get("run", 0)) not in done]
    print(f"{len(tasks)} to judge ({len(kept) - len(tasks)} already done), {workers} workers")

    # Resolve text-derived gold verdicts once, up front (see helper).
    # Only needed by the binary-outcome dimension; skipped entirely otherwise.
    verdict_cache = (_precompute_gold_verdicts(tasks, gold_by_ds, workers)
                     if "outcome" in dimensions else {})

    def work(gen):
        key = (gen["dataset"], gen["case_id"], gen["model"], gen.get("run", 0))
        gold = gold_by_ds[gen["dataset"]].get(gen["case_id"])
        if gold is None:
            return ("skip", {"key": list(key)})
        try:
            rec = judge_one(gen["dataset"], SPECS[gen["dataset"]], gold, gen, verdict_cache,
                            dimensions=dimensions)
            return ("ok", rec)
        except Exception as e:
            return ("fail", {"dataset": key[0], "case_id": key[1], "model": key[2],
                             "run": key[3], "error": str(e)})

    # Single writer (this consuming loop): workers only compute, so appends can't race.
    # Failures are logged but NOT written to the detail file, so a rerun retries them.
    n_ok = n_fail = n_skip = 0
    with detail_path.open("a") as fh:
        for status, payload in tqdm(parallel_yield(work, tasks, workers), total=len(tasks)):
            if status == "ok":
                fh.write(json.dumps(payload) + "\n"); fh.flush(); n_ok += 1
            elif status == "fail":
                _append_jsonl(fails_path, payload); n_fail += 1
            else:
                n_skip += 1
    msg = f"judged {n_ok}, failed {n_fail}"
    if n_fail:
        msg += f" (logged to {fails_path.name}; rerun to retry)"
    if n_skip:
        msg += f", skipped {n_skip} (no gold)"
    print(msg)

    agg_path = write_aggregate(out_dir)
    print(f"\nDone. Detail: {detail_path}")
    return agg_path


def write_aggregate(out_dir):
    """Recompute aggregate_by_dataset_model.csv + citations_for_review.txt from an
    existing judgments_detail.jsonl. Pure post-processing (no judging / no API), so
    it can be re-run any time the detail file changes."""
    outdir = Path(out_dir)
    detail_path = outdir / "judgments_detail.jsonl"
    recs = [json.loads(l) for l in detail_path.read_text().splitlines() if l.strip()]

    by_group = defaultdict(list)  # (dataset, family, key, model) -> [summary,...]
    meta = {}
    for r in recs:
        g = (r["dataset"], r.get("model_family"), r.get("model_key"), r["model"])
        by_group[g].append(r["summary"])
        meta[g] = g

    def mean(items, key):  # skips records where the metric is absent (None)
        vals = [x[key] for x in items if x.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else ""

    agg_path = outdir / "aggregate_by_dataset_model.csv"
    with agg_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model_family", "model_key", "model", "n",
                    "content_similarity", "factor_recall",
                    "unverified_citations_per_resp",
                    "outcome_direction_rate", "outcome_exact_rate",
                    "outcome_justified_rate",
                    "overall_quality", "quality_anchor"])
        for g in sorted(by_group, key=lambda t: tuple("" if x is None else x for x in t)):
            ds, fam, mkey, model = g
            s = by_group[g]
            w.writerow([
                ds, fam or "", mkey or "", model, len(s),
                mean(s, "content_similarity"),
                mean(s, "factor_recall"),
                mean(s, "n_unverified_citations"),
                mean(s, "outcome_direction"),
                mean(s, "outcome_exact"),
                mean(s, "outcome_justified"),
                mean(s, "overall_quality"),
                mean(s, "quality_anchor"),
            ])

    # Only meaningful when legal_grounding was scored; a run restricted to other
    # dimensions has no citations to review, so write nothing rather than KeyError.
    review_path = outdir / "citations_for_review.txt"
    with review_path.open("w") as fh:
        for r in recs:
            for c in (r.get("legal_grounding") or {}).get("unverified_citations", []):
                fh.write(f"{r['dataset']}\t{r['model']}\tcase={r['case_id']}\t"
                         f"run={r['run']}\t{c}\n")

    print(f"aggregate: {agg_path}\ncitation review: {review_path}")
    return agg_path
