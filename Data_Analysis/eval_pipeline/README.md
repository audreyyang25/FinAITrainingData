# General eval pipeline

A dataset-agnostic generalization of `intrafamily_exp/family_model_judging.py`. Give it
a generations file for **any** model suite and it judges each response against the right
gold, across **whatever suitability datasets (P12–P16) are present** in the file — one call,
no per-dataset wiring.

## What it scores

Four judge dimensions (same judge as before: `google/gemini-3.5-flash`, temp 0, evidence-quote
required for every binary call). Which dimensions run, and where each pulls its reference,
is declared per dataset in `dataset_specs.py`:

| dataset | factor recall | outcome dimension | content sim | citations |
|---|---|---|---|---|
| P12 `standard` | – | **binary** (exact match vs `compliant`) | ✓ | ✓ |
| P13 `borderline` | ✓ (`borderline_factors`) | **soft** (direction vs `likely_outcome`) | ✓ | ✓ |
| P14 `conversations` | – | **soft** (vs `legal_assessment`) | ✓ | ✓ |
| P15 `redflags` | ✓ (`red_flag_indicators`) | **soft** (vs red flag + action) | ✓ | ✓ |
| P16 `adversarial` | – | **binary** (exact match; gold verdict classified from `correct_answer`) | ✓ | ✓ |

**Outcome metric.** Binary datasets (P12, P16) have clean ground truth, so outcome is scored
as an **exact violation / no-violation match** → `outcome_exact_rate`. Soft datasets (P13/14/15)
use directional-consistency → `outcome_direction_rate`. Both also report `outcome_justified_rate`.
Each aggregate row is one dataset, so exactly one of the two outcome-rate columns is populated.

## Run

From `Data_Analysis/eval_pipeline/`:

```bash
# The 12 feature-importance models, all datasets, gold answer excluded (defaults):
python run_eval.py \
    --generations ../feature_importance_exp/outputs/generations.jsonl \
    --out ../results/feature_importance_eval

# Subset of datasets:
python run_eval.py --generations <path> --out <dir> --datasets borderline adversarial
```

Needs `OPENROUTER_API_KEY` in the repo-root `.env`. **Resumable** — reruns skip any
`(dataset, case_id, model, run)` already in `judgments_detail.jsonl`, so an interrupted
run just picks up where it left off.

**Concurrency.** Judging runs across a thread pool (`--workers`, default 4); LLM calls are
I/O-bound so this is a real speedup. A single consumer loop does all writes, so appends
never race. Failed responses (malformed JSON after retries, or exhausted rate-limit retries)
are logged to `judgments_failures.jsonl` and NOT written to detail, so **rerunning retries
them** — expect to run a couple of passes with a weaker judge.

**Rate limits.** The Llama bf16 pin (`allow_fallbacks: False`) routes every call to a single
upstream provider's shared pool, which returns 429s under concurrency. The judge already
retries 429s with a long jittered backoff (8→60s, up to 6 attempts), but the durable fixes are:
- keep `--workers` low (4 is the default for this reason; 2–3 if 429s persist), and/or
- add your own provider key via OpenRouter → Settings → Integrations (BYOK) for dedicated
  rate limits — then you can safely raise `--workers` and the run gets much faster.
An unpinned judge (e.g. `google/gemini-3.5-flash`) doesn't hit this and can run higher workers.

## Choosing the judge

Default judge is `google/gemini-3.5-flash`. To reuse the same judge as feature coverage:

```bash
python run_eval.py --generations <path> --out ../results/feature_importance_eval_llama \
    --judge meta-llama/llama-3.3-70b-instruct
```

Known judges (in `JUDGES`, `general_judge.py`) carry their own settings: Llama runs with
**no reasoning effort** and the **bf16 provider pin** (matching feature coverage, so
precision doesn't drift call-to-call); gemini runs with low reasoning effort. Add a judge
by adding one `JUDGES` entry.

Two notes:
- **Use a separate `--out` per judge.** The resume key is `(dataset, case_id, model, run)`
  and does *not* include the judge, so pointing two judges at one dir would cross-contaminate.
  Each detail record does record its `judge` for provenance.
- Empirically, the **outcome** metrics (`outcome_exact_rate`, `outcome_direction_rate`) are
  judge-robust; the calibrated dimensions (`content_similarity`, `factor_recall`, citation
  extraction) shift with the judge. Running two judges into two dirs and comparing is the
  cheap way to check agreement (cf. `feature_importance_exp/score_rubric.py --judge`).

## Outputs (in `--out`)

- `judgments_detail.jsonl` — one record per response, full per-dimension detail.
- `aggregate_by_dataset_model.csv` — one row per `(dataset, model)`: `content_similarity`,
  `factor_recall`, `unverified_citations_per_resp`, `outcome_direction_rate`,
  `outcome_exact_rate`, `outcome_justified_rate` (blank where a dimension doesn't apply).
  Also carries `model_family` / `model_key` for tier grouping.
- `citations_for_review.txt` — every cited authority not on the whitelist, for manual
  fabrication review.
- `judgments_failures.jsonl` — responses that errored out (e.g. malformed judge JSON after
  retries). Not in the detail file, so a rerun retries them.

`write_aggregate(out_dir)` recomputes the CSV + review file from the detail JSONL with no
API calls — handy after folding in more judged rows.

## Inputs it accepts

`load_generations` normalizes each row to `{dataset, case_id, model, run, response,
model_family, model_key}`. Defaults match `feature_importance_exp/outputs/generations.jsonl`
(`answer` → response). For a differently-shaped file, pass `field_map=` to `run_eval`
to remap source field names.

## Adding a dataset

Add one entry to `SPECS` in `dataset_specs.py`: its gold file, a `content_ref` extractor,
optionally a `factors` extractor (else `None` to skip factor recall), and an `outcome`
config (`soft` with a `ref`, or `binary` with `gold_violation` / `gold_text`). No judge or
driver changes needed.
