# Feature Coverage

Measures how thoroughly different LLMs reason about legal/compliance
suitability cases, and profiles each model's reasoning "fingerprint" across the
dataset.

The distinctive idea: instead of an external judge extracting reasoning steps,
**each model self-reports the reasoning features that drove its own answer**,
along with an importance weight per feature (weights sum to 100). Those
self-reported features are then merged into a per-case canonical feature set
(the "superset"), and every model is scored on how much of that superset it
covered.

## What it produces

Given the five suitability datasets (`ADAPTERS` in `config.py`: standard P12,
borderline P13, conversations P14, redflags P15, adversarial P16), the pipeline
answers each case with a 4-family x 3-tier model grid and emits:

| File | Grain | Meaning |
|------|-------|---------|
| `outputs/generations.jsonl` | (case, model) | Each model's answer + self-reported features/importances. Gold (= reference `truth`) stored too. |
| `outputs/case_features.json` | case | Canonical feature superset per case + map from each raw feature to its canonical form. |
| `outputs/coverage.csv` | (case, model) | Fraction of the case superset that model covered. **Primary metric.** |
| `outputs/global_features.json` | dataset | One global reasoning vocabulary + map from case features to global features. |
| `outputs/reasoning_profiles.csv` | (model, global feature) | How often each model raises each reasoning dimension, and its mean importance. |
| `outputs/entropy.csv` | (case, model) | Shannon entropy of a model's importance weights — how concentrated vs. spread its reasoning is. |

## The model grid

4 families x 3 tiers = 12 contestant models (`config.py`), all routed through
OpenRouter:

- **Anthropic** — opus-4.8 / sonnet-4.6 / haiku-4.5
- **OpenAI** — gpt-5.5 / gpt-5.4 / gpt-5.4-mini
- **Gemini** — gemini-3.1-pro / gemini-3.5-flash / gemini-3.1-flash-lite
- **Qwen** — qwen3.5-397b / qwen3.6-35b / qwen3.5-9b

Canonicalization (case + global) and gold-feature extraction all use
`anthropic/claude-opus-4.8` as a fixed, high-reasoning arbiter.

## Self-evaluated Pipeline stages

Run via `run_pipeline.py --stage <name>` (or `--stage all`). Each stage reads
the previous stage's output file, so they must run in order. All stages are
checkpointed — rerunning skips work already written.

1. **`generation`** — 12 models answer every case and self-report their
   reasoning features. Output: `generations.jsonl`.
2. **`gold`** — opus extracts reasoning features from each reference (`truth`)
   answer, so gold can participate as a feature source.
3. **`case_canonicalization`** — per case, merge all models' (+gold's) raw
   features into one deduplicated canonical superset. Output:
   `case_features.json`.
4. **`coverage`** — per (case, model), count how many superset features that
   model hit / superset size. Output: `coverage.csv`.
5. **`global_canonicalization`** — fold every case superset into a single
   cross-case reasoning vocabulary, processed in batches of `CHUNK_SIZE` (the
   vocabulary carries forward and converges) and checkpointed per batch so the
   prompt/output never grow unbounded. Output: `global_features.json`.
6. **`profiles`** — per (model, global feature), appearance frequency + mean
   importance = each model's reasoning fingerprint. Output:
   `reasoning_profiles.csv`.
7. **`entropy`** — per (case, model), Shannon entropy of the importance
   distribution. Output: `entropy.csv`.

## Record schema (`generations.jsonl`)

```json
{
  "dataset": "standard",
  "case_id": "...",
  "model_family": "anthropic",
  "model_key": "frontier",
  "model": "anthropic/claude-opus-4.8",
  "answer": "...",
  "features": [
    {"feature": "Client age affects suitability", "importance": 40, "evidence": "..."}
  ]
}
```

Gold rows use `model: "gold"` with `answer` = the dataset's reference `truth`.

## Setup & run

Paths are resolved from the code, not the current directory: the OpenRouter
key is read from the repo-root `.env`, `DATA_DIR` points at the training
materials, and all outputs go to `Feature_Coverage/outputs/` — so you can
launch `run_pipeline.py` from anywhere and every stage reads/writes the same
place. No manual setup beyond installing deps:

```bash
pip install -r Feature_Coverage/requirements.txt   # needs OPENROUTER_API_KEY in repo-root .env

# Pilot on the first 100 cases of each dataset (~500 cases, 6000 generations):
python Feature_Coverage/run_pipeline.py --stage generation --limit 100
python Feature_Coverage/run_pipeline.py --stage gold --limit 100
python Feature_Coverage/run_pipeline.py --stage case_canonicalization
python Feature_Coverage/run_pipeline.py --stage coverage
python Feature_Coverage/run_pipeline.py --stage global_canonicalization
python Feature_Coverage/run_pipeline.py --stage profiles
python Feature_Coverage/run_pipeline.py --stage entropy
```

`--limit N` takes the first N records of each dataset. Omit it for the full run.
Every stage is checkpointed, so an interrupted run resumes where it left off.
Run `gold` before or after `generation` — they own separate output rows.

## Known limitations (not blocking)
- **Flaky responses are skipped, not fatal.** Reasoning models occasionally
  return empty/whitespace content or an importance set that doesn't sum to 100.
  `call_llm` retries empty responses; anything still failing (bad JSON, failed
  validation) is logged to `outputs/generation_failures.jsonl` (or
  `gold_failures.jsonl`) and skipped. The cell stays not-done, so a rerun
  retries just those cells. Check that file after a run to see what was
  dropped.
