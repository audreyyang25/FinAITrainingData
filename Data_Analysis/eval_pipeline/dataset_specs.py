"""Dataset judge-specs: how each suitability dataset (P12-P16) maps onto the
generic judge's four dimensions.

The generic judge (general_judge.py) knows four dimensions -- factor recall,
legal grounding, outcome, content similarity -- but nothing about any particular
dataset. Each spec below declares, for ONE dataset, how to pull the reference
material each dimension needs out of a gold record, or leaves a dimension out
(None) when the dataset has no clean reference for it.

Adding a new dataset = add one entry here; the judge and driver need no changes.

Spec shape
----------
gold_file      basename of the gold JSON in GOLD_DIR (records keyed by "id").
label          human-readable name for output/reporting.
fact_pattern   record -> str, the case context (used only by factor recall).
factors        record -> list[str] of reference factors, or None to SKIP factor
               recall for this dataset.
content_ref    record -> str, the gold reference text for content similarity.
outcome        the outcome dimension config, or None to skip it:
               {"mode": "soft", "ref": record -> str}
                   soft directional-consistency judge against a reference outcome.
               {"mode": "binary", "gold_violation": record -> bool}
                   exact violation/no-violation match; gold verdict read directly.
               {"mode": "binary", "gold_text": record -> str,
                "trap_text": record -> str|None}
                   exact match, but the gold verdict is derived by classifying
                   gold_text once (used when there is no boolean truth field).
"""

import json
from pathlib import Path

# Gold JSON lives with the training material -- the same directory both existing
# experiment configs already point at (repo_root/AI Suitability.../23_Folders_Suitability).
GOLD_DIR = (
    Path(__file__).resolve().parents[2]
    / "AI Suitability Training Materials"
    / "23_Folders_Suitability"
)


def _format_convo(turns):
    """P14 conversations are a list of {speaker, text} dicts."""
    return "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in turns)


def _p13_content(r):
    return (
        f"Analysis: {r['analysis']}\n\n"
        f"Borderline factors: {r['borderline_factors']}\n\n"
        f"Likely outcome: {r['likely_outcome']}"
    )


def _p14_content(r):
    return json.dumps(r["legal_assessment"], indent=2)


def _p14_outcome_ref(r):
    la = r["legal_assessment"]
    return (
        f"Issues identified: {la.get('issues_identified')}; "
        f"severity: {la.get('severity', 'n/a')}. "
        f"{la.get('analysis', '')}"
    )


def _p15_content(r):
    return (
        f"Red flag: {r['red_flag']}\n"
        f"Regulatory concern: {r['regulatory_concern']}\n"
        f"Recommended action: {r['recommended_action']}"
    )


SPECS = {
    "standard": {  # P12, n=499 -- clean boolean truth (`compliant`)
        "gold_file": "suitability_only_P12.json",
        "label": "P12 standard",
        "fact_pattern": lambda r: r["fact_pattern"],
        "factors": None,
        "content_ref": lambda r: r["answer"],
        # compliant == False  <=>  a violation occurred.
        "outcome": {"mode": "binary", "gold_violation": lambda r: r["compliant"] is False},
    },
    "borderline": {  # P13, n=250 -- soft "likely_outcome" (gray-area cases)
        "gold_file": "suitability_only_P13.json",
        "label": "P13 borderline",
        "fact_pattern": lambda r: r["fact_pattern"],
        "factors": lambda r: r["borderline_factors"],
        "content_ref": _p13_content,
        "outcome": {"mode": "soft", "ref": lambda r: r["likely_outcome"]},
    },
    "conversations": {  # P14, n=50 -- advisor/client transcript + legal_assessment
        "gold_file": "suitability_only_P14.json",
        "label": "P14 conversations",
        "fact_pattern": lambda r: _format_convo(r["conversation"]),
        "factors": None,
        "content_ref": _p14_content,
        "outcome": {"mode": "soft", "ref": _p14_outcome_ref},
    },
    "redflags": {  # P15, n=50 -- red_flag_indicators is a natural factor checklist
        "gold_file": "suitability_only_P15.json",
        "label": "P15 redflags",
        "fact_pattern": lambda r: r["fact_pattern"],
        "factors": lambda r: r["red_flag_indicators"],
        "content_ref": _p15_content,
        "outcome": {
            "mode": "soft",
            "ref": lambda r: (
                f"Principal red flag: {r['red_flag']}. "
                f"Recommended action: {r['recommended_action']}"
            ),
        },
    },
    "adversarial": {  # P16, n=200 -- correct_answer + a designed common_wrong_answer
        "gold_file": "suitability_only_P16.json",
        "label": "P16 adversarial",
        "fact_pattern": lambda r: r["fact_pattern"],
        "factors": None,
        "content_ref": lambda r: r["correct_answer"],
        # No boolean field; derive the gold verdict by classifying correct_answer once.
        "outcome": {
            "mode": "binary",
            "gold_text": lambda r: r["correct_answer"],
            "trap_text": lambda r: r.get("common_wrong_answer"),
        },
    },
}


def load_gold(dataset, gold_dir=GOLD_DIR):
    """Return {id: record} for one dataset's gold file."""
    spec = SPECS[dataset]
    records = json.loads((Path(gold_dir) / spec["gold_file"]).read_text())
    return {r["id"]: r for r in records}
