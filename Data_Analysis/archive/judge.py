import os
import re
import json
from openai import OpenAI
from config import ADAPTERS, JUDGE_SUITE, DATA_DIR, load_dataset

GEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generations.jsonl")
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judgments.jsonl")
CONDITIONS = ["no_q", "with_q"]

JUDGE_SYS = (
    "You are an expert evaluator of U.S. securities-law and FINRA/SEC compliance answers. "
    "Compare a CANDIDATE answer to a REFERENCE answer and rate how well the candidate "
    "captures the reference's substantive legal conclusions and applicable standards. "
    "Return ONLY a JSON object containing at least these fields: "
    '"score" (a number from 0.0 to 1.0, where 1.0 = fully captures the correct content and '
    "0.0 = wrong or contradictory; penalize incorrect statements even if other parts are right) "
    'and "rationale" (one or two sentences).'
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def judge_user(truth, candidate, task=None, guess=False):
    parts = []
    if task:
        parts.append(f"ORIGINAL TASK:\n{task}\n")
    parts.append(f"REFERENCE ANSWER:\n{truth}\n")
    parts.append(f"CANDIDATE ANSWER:\n{candidate}")
    if guess:
        parts.append(
            '\nThe original task/question was withheld from you. In addition to "score" '
            'and "rationale", include a "guessed_question" field with your best guess of '
            "what the original question asked."
        )
    return "\n".join(parts)


def parse_judgment(text):
    """Parse a judge response -> (score in [0,1] | None, rationale, guessed_question, valid).

    If we can't read a JSON object with a numeric score, the judgment is marked
    invalid (score=None) rather than fabricating a number from the raw text.
    """
    if not text:  # model returned no content (None/empty) -> invalid, don't crash
        return None, "", None, False
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)  # find an embedded JSON object
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                pass
    if obj is not None:
        try:
            score = max(0.0, min(1.0, float(obj["score"])))
            return score, obj.get("rationale", ""), obj.get("guessed_question"), True
        except Exception:
            pass
    # Unparseable / no numeric score -> invalid judgment, don't guess.
    return None, text[:300], None, False


def model_call(model, task_prompt_user):
    resp = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        # Cap hidden reasoning so the budget goes to the JSON, not thinking.
        # (no_q outputs are longer -- they also carry guessed_question.)
        # OpenRouter ignores this for non-reasoning models (gpt-4o, llama, ...).
        extra_body={"reasoning": {"effort": "low"}},
        messages=[
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": task_prompt_user},
        ],
    )
    msg = resp.choices[0].message
    if msg.content is None:  # diagnose which model/why returns no text
        print("NULL:", model,
              "| finish:", resp.choices[0].finish_reason,
              "| has_reasoning:", getattr(msg, "reasoning", None) is not None,
              "| err:", getattr(resp, "error", None))
    return msg.content


def build_truths(data_dir=DATA_DIR):
    """(dataset, id) -> (task_prompt, ground_truth), for every record."""
    truths = {}
    for name in ADAPTERS:
        for rid, prompt, truth in load_dataset(name, data_dir=data_dir):
            truths[(name, rid)] = (prompt, truth)
    return truths


def load_done(path, key_fields=("dataset", "id"), write_condition=True):
    """Done-keys with a VALID (non-null) score -> skipped on resume.

    A done-key is tuple(row[f] for f in key_fields) + (generator, judge) plus
    (condition,) when write_condition. Null/invalid judgments are intentionally
    left OUT of the done-set, so they can be re-run later (e.g. after raising
    max_tokens) without re-running the good entries. Also skips blank/partial lines.
    """
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("score") is None or not r.get("valid", False):
                continue  # leave null/invalid judgments re-runnable
            key = tuple(r[f] for f in key_fields) + (r["generator"], r["judge"])
            if write_condition:
                key += (r["condition"],)
            done.add(key)
    return done


def judge_answers(gen_path, out_path, lookup, key_fields=("dataset", "id"),
                  suite=JUDGE_SUITE, conditions=("with_q",)):
    """Generic, resumable judging loop shared by every experiment.

    lookup:     gen_row -> (task_prompt, ground_truth) for that generation.
    key_fields: identifying fields (besides generator/judge) copied onto each
                judgment row and used for resume.
    conditions: ("with_q",) judges once with the question and writes no
                `condition` field (the effect experiments). ("no_q", "with_q")
                loops both, adds a `condition` field, and on no_q asks the judge
                to guess the withheld question (`guessed_question`).
    """
    write_condition = len(conditions) > 1
    include_guess = "no_q" in conditions
    done = load_done(out_path, key_fields, write_condition)
    with open(out_path, "a") as out:
        for line in open(gen_path):
            g = json.loads(line)
            task, truth = lookup(g)
            base = tuple(g[f] for f in key_fields) + (g["generator"],)
            for spec in suite:                             # the judge
                for cond in conditions:
                    key = base + (spec["key"],) + ((cond,) if write_condition else ())
                    if key in done:
                        continue
                    user = judge_user(
                        truth, g["answer"],
                        task=task if cond == "with_q" else None,
                        guess=(cond == "no_q"),
                    )
                    try:
                        raw = model_call(spec["model"], user)
                    except Exception as e:
                        print(f"JUDGE FAIL {key}: {e}")
                        continue
                    score, rationale, guessed_question, valid = parse_judgment(raw)
                    row = {f: g[f] for f in key_fields}
                    row["generator"] = g["generator"]
                    row["judge"] = spec["key"]
                    row["judge_model"] = spec["model"]
                    if write_condition:
                        row["condition"] = cond
                    row["score"] = score
                    row["rationale"] = rationale
                    if include_guess:
                        row["guessed_question"] = guessed_question
                    row["valid"] = valid
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    done.add(key)
                    print(f"judged {key} -> {score}")


# separated into with question and no question
def run_judging(gen_path=GEN_PATH, out_path=PATH, data_dir=DATA_DIR):
    """Judge the built-in datasets under both no_q and with_q. Thin wrapper."""
    truths = build_truths(data_dir=data_dir)
    judge_answers(
        gen_path, out_path,
        lookup=lambda g: truths[(g["dataset"], g["id"])],
        key_fields=("dataset", "id"),
        conditions=CONDITIONS,
    )


if __name__ == "__main__":
    run_judging()
