"""Family-model comparison: GPT family on the BORDERLINE cases.

Generates answers to the borderline (P13) cases with the GPT family
(config.GPT_SUITE), then judges them with the 3-dimension borderline judge in
family_model_judging.py. Outputs under runs/gpt_family/.

Run:  python family_model_comp.py
"""

import os
import json
import csv
import datetime
from config import GPT_SUITE, DATA_DIR
from generations import run_generation
from family_model_judging import run_family_judging, get_rubric

HERE = os.path.dirname(os.path.abspath(__file__))
# OUT_DIR = os.path.join(HERE, "runs", "gpt_family")     # full GPT family run
OUT_DIR = os.path.join(HERE, "runs", "gpt55_instant")    # GPT-5.5-only rerun (brevity prompt)
os.makedirs(OUT_DIR, exist_ok=True)
GEN_PATH = os.path.join(OUT_DIR, "generations.jsonl")
GOLD_PATH = os.path.join(DATA_DIR, "suitability_only_P13.json")  # borderline gold
RUBRIC_PATH = os.path.join(OUT_DIR, "rubric.csv")
LIMIT = 75  # borderline records to use; None for all of P13

# GPT-5.5-only rerun: just gpt-5.5, with a brevity system prompt meant to emulate
# gpt-5.5-instant (get to the conclusion instead of running long and truncating).
GPT55_SUITE = [s for s in GPT_SUITE if s["key"] == "gpt-5.5"]
GPT55_SYSTEM = (
    "You are a securities-law and FINRA/SEC compliance expert. Answer directly and reach "
    "your conclusion quickly. Be brief, direct, and non-redundant, but thorough: address "
    "every part of the question and state your bottom-line conclusion explicitly and early. "
    "Do not pad, restate the fact pattern at length, or repeat points."
)


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
    # 1. Generate: BORDERLINE cases only, GPT-5.5 only, with the brevity system
    #    prompt (emulating gpt-5.5-instant). Resumable.
    run_generation(out_paths=[GEN_PATH], content_types=["borderline"],
                   gen_suites=[GPT55_SUITE], limit=LIMIT, gen_system=GPT55_SYSTEM)
    # --- full GPT family run (commented out for the GPT-5.5-only rerun) ---
    # run_generation(out_paths=[GEN_PATH], content_types=["borderline"],
    #                gen_suites=[GPT_SUITE], limit=LIMIT)
    gens = generations_for_judge(GEN_PATH)

    # 2. Snapshot the judge's rubric BEFORE judging.
    capture_rubric("before")

    # 3. Judge with the 3-dimension borderline judge.
    run_family_judging(GOLD_PATH, gens, out_dir=OUT_DIR)

    # 4. Snapshot the judge's rubric AFTER the entire run.
    capture_rubric("after")
