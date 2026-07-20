import os
import json
from openai import OpenAI
from config import ADAPTERS, GEN_SUITE, DATA_DIR, load_dataset

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


def model_call(model, task_prompt, system=GEN_SYSTEM):
    resp = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        # Cap reasoning so reasoning-model generators (gemini/deepseek/qwen) don't
        # spend the whole budget thinking and return empty/truncated answers.
        # OpenRouter ignores this for non-reasoning models.
        extra_body={"reasoning": {"effort": "low"}},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": task_prompt},
        ],
    )
    return resp.choices[0].message.content


def load_done(path, key_fields=("dataset", "id")):
    """Done-keys already present in the output file, so we can skip and resume.

    A done-key is tuple(row[f] for f in key_fields) + (generator,). Skips
    blank/partial lines so an interrupted write can't break resume and force you
    to re-pay for the whole file.
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
            done.add(tuple(r[f] for f in key_fields) + (r["generator"],))
    return done


def generate(items, out_path, key_fields=("dataset", "id"), suite=GEN_SUITE, gen_system=GEN_SYSTEM):
    """Generic, resumable generation loop shared by every experiment.

    items:      iterable of (key_dict, prompt). key_dict holds the identifying
                fields written on each row (e.g. {"dataset", "id"} or
                {"format", "id"} or {"gender", "type", "base_id", "demographic"}).
    key_fields: the subset of key_dict that (with generator) uniquely keys a row
                for resume. Must match what load_done reads back.

    Writes {**key_dict, generator, model, answer}. On failure or empty answer we
    write nothing, so the key stays "not done" and the next run retries it.
    """
    done = load_done(out_path, key_fields)
    with open(out_path, "a") as out:
        for key_dict, prompt in items:
            base = tuple(key_dict[f] for f in key_fields)
            for spec in suite:
                key = base + (spec["key"],)
                if key in done:
                    continue
                try:
                    # A suite spec may carry its own "system" (e.g. a brevity /
                    # anti-truncation prompt for one model); else use gen_system.
                    answer = model_call(spec["model"], prompt, system=spec.get("system", gen_system))
                except Exception as e:
                    print(f"GEN FAIL {key}: {e}")
                    continue
                if not answer:
                    print(f"GEN EMPTY {key}")
                    continue
                out.write(json.dumps({
                    **key_dict,
                    "generator": spec["key"],
                    "model": spec["model"],
                    "answer": answer,
                }) + "\n")
                out.flush()
                done.add(key)
                print(f"gen ok {key}")


def _dataset_items(content_types, limit, data_dir):
    """(key_dict, prompt) over the built-in datasets, for run_generation."""
    for name in ADAPTERS:
        if name in content_types:
            for rid, prompt, _truth in load_dataset(name, limit=limit, data_dir=data_dir):
                yield {"dataset": name, "id": rid}, prompt


def run_generation(out_paths=[PATH], data_dir=DATA_DIR, limit=LIMIT, content_types=["standard", "borderline", "conversations", "redflags", "adversarial"], gen_suites=[GEN_SUITE], gen_system=GEN_SYSTEM):
    """Generate over the built-in datasets. Thin wrapper over generate()."""
    for out_path, suite in zip(out_paths, gen_suites):
        generate(_dataset_items(content_types, limit, data_dir), out_path,
                 key_fields=("dataset", "id"), suite=suite, gen_system=gen_system)


if __name__ == "__main__":
    run_generation()
