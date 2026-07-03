"""#1: rewrite standard (P12) fact patterns into conversation format, preserving
all facts (auto-verified).

Output: standard_conversation.jsonl -- same schema as P12, but `fact_pattern`
is replaced with the dialogue rendered to text; `question` and `answer` are kept
byte-identical so the ground truth never moves. The only variable vs the original
is the prose format. Does NOT touch ADAPTERS. Resumable on record id.
"""

import os
import re
import json
from openai import OpenAI
from config import DATA_DIR

MODIFIED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modified_data")
os.makedirs(MODIFIED_DIR, exist_ok=True)
OUT = os.path.join(MODIFIED_DIR, "standard_conversation.jsonl")
SRC = os.path.join(DATA_DIR, "suitability_only_P12.json")
LIMIT = 45            # records to rewrite; None for all of P12
MAX_TRIES = 3         # rewrite->verify attempts before writing flagged-unverified
MODEL = "anthropic/claude-opus-4.8"  # adjust to a slug valid in your account

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

REWRITE_SYS = (
    "You reformat a third-person securities-compliance fact pattern into a realistic "
    "dialogue between a client and their financial advisor (add a compliance officer "
    "only if the original involves one). Preserve EVERY fact exactly: names, firms, "
    "dollar amounts, percentages, dates, account types, products, time horizons, and "
    "who did what. Do NOT add, omit, infer, editorialize, or resolve anything that "
    'the original left open. Output ONLY JSON: {"conversation": [{"speaker": str, '
    '"text": str}, ...]}.'
)

VERIFY_SYS = (
    "You check whether a rewritten dialogue preserves all facts from an original "
    "narrative, with no inventions. Return ONLY JSON: "
    '{"missing": [facts in ORIGINAL absent from REWRITE], '
    '"added": [facts or claims in REWRITE not supported by ORIGINAL], '
    '"ok": bool}. Set ok=true only if BOTH lists are empty. Ignore pure phrasing or '
    "formatting differences -- only substantive facts count."
)


def _json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def render(conversation):
    return "\n".join(
        f"{(t.get('speaker') or 'SPEAKER').upper()}: {t.get('text', '')}" for t in conversation
    )


def rewrite(fact_pattern, feedback=None):
    user = fact_pattern
    if feedback:
        user += (
            "\n\nYour previous attempt was rejected. "
            f"Missing facts to restore: {feedback.get('missing')}. "
            f"Invented content to remove: {feedback.get('added')}. "
            "Redo the dialogue, fixing these while keeping it natural."
        )
    resp = client.chat.completions.create(
        model=MODEL, max_tokens=4000,
        extra_body={"reasoning": {"effort": "low"}},
        messages=[{"role": "system", "content": REWRITE_SYS},
                  {"role": "user", "content": user}],
    )
    obj = _json(resp.choices[0].message.content)
    return (obj or {}).get("conversation")


def verify(original, rewrite_text):
    resp = client.chat.completions.create(
        model=MODEL, max_tokens=1500,
        extra_body={"reasoning": {"effort": "low"}},
        messages=[{"role": "system", "content": VERIFY_SYS},
                  {"role": "user", "content": f"ORIGINAL:\n{original}\n\nREWRITE:\n{rewrite_text}"}],
    )
    return _json(resp.choices[0].message.content) or {
        "missing": ["<verifier response unparseable>"], "added": [], "ok": False}


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


if __name__ == "__main__":
    records = json.load(open(SRC))
    if LIMIT:
        records = records[:LIMIT]
    done = load_done(OUT)

    with open(OUT, "a") as out:
        for rec in records:
            if rec["id"] in done:
                continue
            original = rec["fact_pattern"]

            conversation, check, feedback = None, None, None
            for _ in range(MAX_TRIES):
                conversation = rewrite(original, feedback)
                if not conversation:
                    feedback = None
                    continue
                check = verify(original, render(conversation))
                if check.get("ok"):
                    break
                feedback = check  # feed the gaps back into the next rewrite

            if not conversation:
                print(f"REWRITE FAIL id={rec['id']}: no usable dialogue")
                continue

            new = dict(rec)
            new["fact_pattern"] = render(conversation)  # question/answer untouched
            new["_verified"] = bool(check and check.get("ok"))
            new["_fact_check"] = check  # keep the audit so you can inspect failures
            out.write(json.dumps(new) + "\n")
            out.flush()
            done.add(rec["id"])
            print(f"ok id={rec['id']} verified={new['_verified']}")
