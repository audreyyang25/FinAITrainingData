"""
Judging pipeline for the financial-advising suitability eval (P13 "borderline" set).

Design:
  - One judge call per dimension per response (3 dimensions), NOT one mega-call.
  - Every binary judgment requires an evidence quote; no quote => automatic NO.
  - Judge: google/gemini-3.5-flash via OpenRouter, temperature 0, low thinking effort.
  - All judge outputs are strict JSON, validated + retried on failure.
  - Citation-whitelist diff happens in code, not in the judge.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python judge_pipeline.py \
      --gold suitability_only_P13.json \
      --generations generations.json \
      --out results/

Expected input formats:
  gold: list of records with keys id, fact_pattern, question, analysis,
        borderline_factors, likely_outcome, applicable_standard
  generations: list of {"case_id": <id>, "model": "<slug>", "run": <int>,
                        "response": "<text>"}
"""

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI  # pip install openai

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

JUDGE_MODEL = "google/gemini-3.5-flash"   # pin this for the whole experiment
JUDGE_TEMPERATURE = 0.0
JUDGE_REASONING_EFFORT = "low"            # checklist verification != frontier reasoning
MAX_RETRIES = 4
RETRY_BASE_SLEEP = 2.0

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client

# Whitelist of real authorities for this question set. Extend as your bank grows.
# The judge only EXTRACTS citations; this list is how *code* flags fabrications.
AUTHORITY_WHITELIST_PATTERNS = [
    r"finra\s+rule\s+2111", r"rule\s+2111", r"suitability\s+rule",
    r"regulation\s+best\s+interest", r"reg\.?\s*bi", r"rule\s+15l-1",
    r"regulatory\s+notice\s+12-25", r"notice\s+12-25",
    r"securities\s+exchange\s+act(\s+of\s+1934)?",
    r"investment\s+advisers\s+act(\s+of\s+1940)?",
    r"finra\s+rule\s+2090", r"know\s+your\s+customer",   # commonly (validly) raised
    r"release\s+no\.?\s*34-86031",
    r"finra\s+rule\s+3110",                              # supervision, validly adjacent
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


RUBRIC_PROMPT = (
    "Describe the exact rubric you use to judge AI answers to these financial-advisory "
    "compliance case studies. Cover each dimension: factor recall (vs the reference "
    "borderline factors), outcome prediction (direction + justification), an OVERALL "
    "CONTENT-SIMILARITY score (0.0-1.0) for how closely the response's substance, reasoning, "
    "and conclusions match the gold-standard analysis, and citation extraction (identifying "
    "the legal authorities cited, for a fabrication check). For each, state what earns credit "
    "vs. not and any rules you apply (e.g. requiring an evidence quote; how you calibrate the "
    "0-1 similarity score and what distinguishes 0.3 from 0.7). Be concise. Plain text, no JSON."
)


def get_rubric() -> str:
    """Ask the judge model to articulate the rubric it applies (for before/after logging)."""
    resp = get_client().chat.completions.create(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        messages=[
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": RUBRIC_PROMPT},
        ],
        extra_body={"reasoning": {"effort": JUDGE_REASONING_EFFORT}},
    )
    return (resp.choices[0].message.content or "").strip()


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
    for attempt in range(MAX_RETRIES):
        try:
            resp = get_client().chat.completions.create(
                model=JUDGE_MODEL,
                temperature=JUDGE_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SHARED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"reasoning": {"effort": JUDGE_REASONING_EFFORT}},
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(_strip_fences(raw))
            validator(parsed)
            return parsed
        except Exception as e:  # JSON errors, schema errors, transient API errors
            last_err = e
            time.sleep(RETRY_BASE_SLEEP * (2 ** attempt))
    raise RuntimeError(f"Judge failed after {MAX_RETRIES} attempts: {last_err}")


# ----------------------------------------------------------------------------
# Dimension 1: factor recall (checklist vs. gold borderline_factors)
# ----------------------------------------------------------------------------

def judge_factor_recall(fact_pattern: str, response: str, gold_factors: list) -> dict:
    factor_lines = "\n".join(f"{i+1}. {f}" for i, f in enumerate(gold_factors))
    prompt = f"""CASE FACT PATTERN:
{fact_pattern}

RESPONSE UNDER EVALUATION:
{response}

TASK: For each reference factor below, determine whether the response SUBSTANTIVELY
addresses it. The response does not need to use the same wording, label it as a
"borderline factor", or reach the same conclusion about it — it only needs to engage
with the underlying issue somewhere in the response.

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
    n = len(gold_factors)
    out["recall"] = sum(f["addressed"] for f in out["factors"]) / n
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
    # Code-side fabrication check: anything cited that matches no whitelist pattern
    # is flagged for MANUAL REVIEW (not auto-penalized — it may be real but unlisted).
    flagged = []
    for cite in out["cited_authorities"]:
        low = cite.lower()
        if not any(re.search(pat, low) for pat in AUTHORITY_WHITELIST_PATTERNS):
            flagged.append(cite)
    out["unverified_citations"] = flagged
    return out


# ----------------------------------------------------------------------------
# Dimension 3: outcome prediction (two binary checks, evidence-backed)
# ----------------------------------------------------------------------------

def judge_outcome(response: str, gold_outcome: str) -> dict:
    prompt = f"""RESPONSE UNDER EVALUATION:
{response}

REFERENCE OUTCOME ASSESSMENT:
{gold_outcome}

TASK: Evaluate the response's prediction of the likely regulatory/arbitration outcome
with two independent binary judgments.

1. direction_consistent: Is the response's predicted outcome DIRECTIONALLY consistent
   with the reference? Directional consistency means agreeing on the general shape of
   the result (e.g., close call / partial liability / mitigated finding vs. full
   violation vs. full exoneration). Do NOT require matching specific damages
   percentages, remedies, forums, or procedural details. If the response makes no
   outcome prediction at all, answer false.

2. justified: Is the predicted outcome JUSTIFIED by the response's own analysis — i.e.,
   does the response connect its outcome prediction to specific facts or arguments it
   raised, rather than asserting the outcome without support? Judge this independently
   of whether the direction is correct.

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
# Dimension 4: overall content similarity to gold (0-1, holistic, with reasoning)
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


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def judge_one(gold: dict, gen: dict) -> dict:
    fp, resp = gold["fact_pattern"], gen["response"]
    rec = {
        "case_id": gold["id"],
        "topic": gold.get("topic"),
        "model": gen["model"],
        "run": gen.get("run", 0),
    }
    rec["factor_recall"] = judge_factor_recall(fp, resp, gold["borderline_factors"])
    rec["legal_grounding"] = judge_legal_grounding(resp)
    rec["outcome"] = judge_outcome(resp, gold["likely_outcome"])
    gold_ref = f"{gold.get('analysis', '')}\n\nLikely outcome: {gold.get('likely_outcome', '')}"
    rec["content_similarity"] = judge_content_similarity(resp, gold_ref)

    # Flat summary row for the CSV
    rec["summary"] = {
        "content_similarity": round(rec["content_similarity"]["content_similarity"], 3),
        "factor_recall": round(rec["factor_recall"]["recall"], 3),
        "n_unverified_citations": len(rec["legal_grounding"]["unverified_citations"]),
        "outcome_direction": rec["outcome"]["direction_consistent"]["value"],
        "outcome_justified": rec["outcome"]["justified"]["value"],
    }
    return rec


def run_family_judging(gold_path, generations, out_dir="results", limit=0):
    """Judge a set of generations against gold (the borderline P13 set). Callable
    directly (no CLI) so other scripts can orchestrate it.

    gold_path:   path to the gold JSON list (records with id, fact_pattern,
                 borderline_factors, likely_outcome, ...).
    generations: a list of {case_id, model, run, response} dicts, OR a path to a
                 JSON file containing such a list.
    out_dir:     output directory for the detail/aggregate/review files.
    limit:       judge only the first N generations (0 = all).
    """
    if isinstance(generations, (str, Path)):
        generations = json.loads(Path(generations).read_text())

    gold_by_id = {r["id"]: r for r in json.loads(Path(gold_path).read_text())}
    gens = list(generations)
    if limit:
        gens = gens[:limit]

    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    detail_path = outdir / "judgments_detail.jsonl"
    done = set()
    if detail_path.exists():  # resume support: skip already-judged (case, model, run)
        for line in detail_path.read_text().splitlines():
            j = json.loads(line)
            done.add((j["case_id"], j["model"], j["run"]))

    with detail_path.open("a") as fh:
        for i, gen in enumerate(gens):
            key = (gen["case_id"], gen["model"], gen.get("run", 0))
            if key in done:
                continue
            gold = gold_by_id.get(gen["case_id"])
            if gold is None:
                print(f"[skip] no gold record for case_id={gen['case_id']}")
                continue
            print(f"[{i+1}/{len(gens)}] case={key[0]} model={key[1]} run={key[2]}")
            try:
                rec = judge_one(gold, gen)
            except Exception as e:
                # A persistent failure on one generation shouldn't kill the run;
                # nothing is written for this key, so it's retried on the next run.
                print(f"[fail] {key}: {e}")
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    # Aggregate: mean per model across cases/runs
    import csv
    from collections import defaultdict
    all_recs = [json.loads(l) for l in detail_path.read_text().splitlines()]
    by_model = defaultdict(list)
    for r in all_recs:
        by_model[r["model"]].append(r["summary"])

    def mean(items, key):  # skips records missing the key (e.g. pre-metric records)
        vals = [x[key] for x in items if x.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else ""

    agg_path = outdir / "aggregate_by_model.csv"
    with agg_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "content_similarity", "factor_recall",
                    "unverified_citations_per_resp", "outcome_direction_rate",
                    "outcome_justified_rate"])
        for model, s in sorted(by_model.items()):
            w.writerow([
                model, len(s),
                mean(s, "content_similarity"),
                mean(s, "factor_recall"),
                mean(s, "n_unverified_citations"),
                mean(s, "outcome_direction"),
                mean(s, "outcome_justified"),
            ])

    # Dump every unverified citation for manual review
    review_path = outdir / "citations_for_review.txt"
    with review_path.open("w") as fh:
        for r in all_recs:
            for c in r["legal_grounding"]["unverified_citations"]:
                fh.write(f"{r['model']}\tcase={r['case_id']}\trun={r['run']}\t{c}\n")

    print(f"\nDone. Detail: {detail_path}\nAggregate: {agg_path}\nCitation review: {review_path}")
    return agg_path