"""Family-model comparison: the full GPT family on the BORDERLINE (P13) cases.

Generates answers with the whole GPT family (config.GPT_SUITE), then judges them
with the 3-dimension borderline judge in family_model_judging.py. Outputs under
results/gpt_family/.

Anti-truncation is built in: gpt-5.5 is a reasoning model that otherwise runs long
and truncates at max_tokens, so it carries a brevity system prompt (reach the
conclusion early) applied ONLY to it -- every other family member uses the default
GEN_SYSTEM. A single full-family run therefore covers gpt-5.5 too; no separate rerun.

Run:  python family_model_comp.py
"""

import os
import sys
import json
import csv
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Data_Analysis/ on sys.path
from config import GPT_SUITE, DATA_DIR
from generations import run_generation
from family_model_judging import run_family_judging, get_rubric

HERE = os.path.dirname(os.path.abspath(__file__))         # Data_Analysis/intrafamily_exp
DA_ROOT = os.path.dirname(HERE)                           # Data_Analysis
OUT_DIR = os.path.join(DA_ROOT, "results", "gpt_family")
os.makedirs(OUT_DIR, exist_ok=True)
GEN_PATH = os.path.join(OUT_DIR, "generations.jsonl")
GOLD_PATH = os.path.join(DATA_DIR, "suitability_only_P13.json")  # borderline gold
RUBRIC_PATH = os.path.join(OUT_DIR, "rubric.csv")
LIMIT = 75  # borderline records to use; None for all of P13

# Anti-truncation: gpt-5.5 runs long and truncates at max_tokens, so it gets a
# brevity system prompt (reach the conclusion early). Applied ONLY to the models in
# NEEDS_BREVITY; the rest use the default GEN_SYSTEM. Attaching it to the suite spec
# (instead of running a separate pass) means one full-family run handles gpt-5.5.
BREVITY_SYSTEM = (
    "You are a securities-law and FINRA/SEC compliance expert. Answer directly and reach "
    "your conclusion quickly. Be brief, direct, and non-redundant, but thorough: address "
    "every part of the question and state your bottom-line conclusion explicitly and early. "
    "Do not pad, restate the fact pattern at length, or repeat points."
)
NEEDS_BREVITY = {"gpt-5.5"}
FAMILY_SUITE = [
    {**spec, "system": BREVITY_SYSTEM} if spec["key"] in NEEDS_BREVITY else spec
    for spec in GPT_SUITE
]


def generations_for_judge(path):
    """Map run_generation's JSONL rows -> the judge's {case_id, model, run, response}.
    (run_generation writes {dataset, id, generator, model, answer}; the judge keys
    on case_id/model/response, and we use the family key `generator` as the model
    label so the aggregate compares gpt-3.5 / gpt-4 / gpt-4o / gpt-5.5.)"""
    gens = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        gens.append({"case_id": r["id"], "model": r["generator"],
                     "run": 0, "response": r["answer"]})
    return gens


def _rubric_phases_present():
    present = set()
    if os.path.exists(RUBRIC_PATH):
        with open(RUBRIC_PATH, newline="") as fh:
            for row in csv.reader(fh):
                if row and row[0] != "phase":
                    present.add(row[0])
    return present


def capture_rubric(phase):
    """Ask the judge for its rubric and append it under `phase` (once). Idempotent
    -- skips if that phase is already recorded, so a resumed run keeps the original
    'before' and writes 'after' only at final completion."""
    if phase in _rubric_phases_present():
        return
    text = get_rubric()
    new_file = not os.path.exists(RUBRIC_PATH)
    with open(RUBRIC_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["phase", "timestamp", "rubric"])
        w.writerow([phase, datetime.datetime.now().isoformat(timespec="seconds"), text])
    print(f"rubric [{phase}] saved to {RUBRIC_PATH}")


if __name__ == "__main__":
    # 1. Generate: BORDERLINE cases, full GPT family, one pass. gpt-5.5 carries the
    #    brevity system prompt automatically (see FAMILY_SUITE). Resumable.
    run_generation(out_paths=[GEN_PATH], content_types=["borderline"],
                   gen_suites=[FAMILY_SUITE], limit=LIMIT)
    gens = generations_for_judge(GEN_PATH)

    # 2. Snapshot the judge's rubric BEFORE judging.
    capture_rubric("before")

    # 3. Judge with the 3-dimension borderline judge.
    run_family_judging(GOLD_PATH, gens, out_dir=OUT_DIR)

    # 4. Snapshot the judge's rubric AFTER the entire run.
    capture_rubric("after")
