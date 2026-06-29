import os
import json
from openai import OpenAI
from config import ADAPTERS, SUITE, DATA_DIR, load_dataset

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generations.jsonl")
LIMIT = 15  # pilot: records per dataset. Set to None for the full run.

GEN_SYSTEM = (
    "You are a securities-law and FINRA/SEC compliance expert. "
    "Answer the question directly and completely."
)

# OpenRouter speaks the OpenAI API; just point the client at it.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def model_call(model, task_prompt):
    resp = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": task_prompt},
        ],
    )
    return resp.choices[0].message.content


def load_done(path):
    """Keys already present in the output file, so we can skip and resume.

    Skips blank/partial lines so an interrupted write can't break resume and
    force you to re-pay for the whole file.
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
            done.add((r["dataset"], r["id"], r["generator"]))
    return done


def run_generation(out_path=PATH, data_dir=DATA_DIR, limit=LIMIT):
    done = load_done(out_path)
    with open(out_path, "a") as out:
        for name in ADAPTERS:
            for rid, prompt, _truth in load_dataset(name, limit=limit, data_dir=data_dir):
                for spec in SUITE:
                    key = (name, rid, spec["key"])
                    if key in done:
                        continue
                    try:
                        answer = model_call(spec["model"], prompt)
                    except Exception as e:
                        # Don't write a line on failure -> key stays "not done"
                        # and the next run retries it.
                        print(f"FAIL {key}: {e}")
                        continue
                    out.write(json.dumps({
                        "dataset": name,
                        "id": rid,
                        "generator": spec["key"],
                        "model": spec["model"],
                        "answer": answer,
                    }) + "\n")
                    out.flush()
                    done.add(key)
                    print(f"ok {key}")


if __name__ == "__main__":
    run_generation()
