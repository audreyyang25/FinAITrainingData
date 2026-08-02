# Model reasoning on securities-suitability cases

How do 12 LLMs (4 families × 3 tiers) reason about legal/compliance suitability
cases, and how do their reasoning *fingerprints* differ? One shared **Setup**
collects an answer from every model on every case; three **Parts** analyze those
answers.

```
Setup ── generations.json ──┬── Part 1  Eval          (score answers vs gold)
                            ├── Part 2  Feature coverage (vs gold, indep. Llama)
                            └── Part 3  Feature distribution (20-criterion rubric)
```

All model calls go through **OpenRouter** (one OpenAI-compatible endpoint), so
every model is just a slug. Everything runs as a package from **this directory**
(`Data_Analysis/`) with `python -m <package>.<module>`.

---

## Layout

| Path | What |
|---|---|
| `shared/` | package imported by everything: `config` (models, adapters, paths), `llm`, `utils`, `schemas`, `prompts`, `jsd_stats` |
| `setup/` | `generation.py` — the 12 models answer every case (answer only) |
| `part1_eval/` | `run_eval`, `general_judge`, `dataset_specs` — judge answers against gold |
| `part2_coverage/` | independent-Llama feature extraction, canonicalization, coverage, gold review, nearest-neighbour geometry |
| `part3_distribution/` | 20-criterion `rubric`, `score_rubric`, `rubric_analysis`, tilt + separation figures |
| `tools/` | manual case/feature inspection (`read_case`, `examine_feature`, `build_concept_spotcheck`) |
| `adapters.csv` | the five datasets materialized flat (`FILE, ID, PROMPT, TRUTH, HAS_QUESTION`) |
| `generations.json` | **Setup output** — one answer per (case, model), the input to all three parts |
| `outputs/` | all pipeline artifacts, grouped by part: `part1_eval/<judge>/`, `part2_coverage/` (`<extractor>/`, CSVs, `figures/`), `part3_distribution/` (`rubric/`, CSVs, `figures/`), `setup/` |
| `archive/` | prior, unrelated experiments + their results, and the original self-report generations |

---

## Setup (once)

Create `.env` at the **repo root** (loaded automatically):

```
OPENROUTER_API_KEY=sk-or-...
```

Install deps: `pip install -r requirements.txt`

Model slugs live in `shared/config.py` (`EVALUATION_MODELS`). They're OpenRouter
slugs and drift over time — verify at https://openrouter.ai/models.

### Data

Source: `../AI Suitability Training Materials/23_Folders_Suitability/` (see
`config.DATA_DIR`). Five dataset "types", each mapped to a common
`(prompt, truth, has_question)` shape by an **adapter** in `config.ADAPTERS`:

| key | file | n | prompt from | truth from |
|---|---|---|---|---|
| `standard` | P12 | 499 | fact_pattern + question | answer |
| `borderline` | P13 | 250 | fact_pattern + question | analysis + likely_outcome |
| `conversations` | P14 | 50 | task + conversation | legal_assessment |
| `redflags` | P15 | 50 | task + fact_pattern | red_flag + concern + action |
| `adversarial` | P16 | 200 | fact_pattern + question | correct_answer |

- `python -m shared.config` → (re)builds `adapters.csv` if missing.
- `load_dataset(name, limit)` yields `(id, prompt, truth)` per record.

### Generations — `generations.json`

Every part reads one file: `generations.json` at the root — schema
`{dataset, case_id, model_family, model_key, model, answer}`, one row per
(case, model) plus a `gold` row per case (answer = reference truth).

```bash
# Replicate the paper's answers from the original run (strip self-reported
# features; no API calls):
python -m setup.generation --from-jsonl archive/selfreport_generations.jsonl

# OR a fresh answer-only run (resumable; --limit takes first N cases/dataset):
python -m setup.generation --limit 100
```

---

## Part 1 — Eval

Judges each answer against its dataset's gold across the applicable dimensions
(factor recall, outcome match/direction, content similarity, citations),
declared per dataset in `dataset_specs.py`. Resumable.

```bash
python -m part1_eval.run_eval                       # -> outputs/part1_eval/gemini  (gold excluded)
python -m part1_eval.run_eval --judge meta-llama/llama-3.3-70b-instruct \
    --out outputs/part1_eval/llama                   # separate --out per judge
```

Outputs (in `--out`): `judgments_detail.jsonl`, `aggregate_by_dataset_model.csv`,
`citations_for_review.txt`. See `part1_eval/README.md` for judge/rate-limit notes.

---

## Part 2 — Feature coverage (vs gold)

An **independent** model (Llama 3.3-70B) extracts reasoning features from every
answer — including gold — so coverage is measured against a neutral extractor,
not model self-report. Features are canonicalized per case into a superset; each
model is scored on the fraction of that superset it covered.

```bash
python -m part2_coverage.indep_extractor_pipeline   # -> outputs/part2_coverage/<extractor>/{gen_feat,case_feat,global_features}
python -m part2_coverage.coverage                   # -> outputs/part2_coverage/coverage.csv  (primary metric)
python -m part2_coverage.extract_gold_review --n 100 # -> outputs/part2_coverage/gold_review.csv (audit list)
python -m part2_coverage.nearest_neighbor --suffix _freq  # geometry: MDS embedding, gold ranking, heatmap
```

`nearest_neighbor` uses **selection frequency** (not importance): who reasons like
whom, and does the family geometry replicate. Writes `selection_matrix_*.csv`,
`js_divergence_*.csv`, figures, and **`stats_*.json`** (replication r, family-vs-verbosity
and NN-recovery permutation p, gold's family-win fractions) under `outputs/part2_coverage/`.

---

## Part 3 — Feature distribution (20-criterion rubric)

Every answer is scored against a fixed 20-criterion codebook (`rubric.py`) by one
labeler, so the axes are model-independent. Each model becomes a distribution
over the 20 criteria; families are compared by JS divergence, nearest-neighbour
purity, and permutation tests.

```bash
python -m part3_distribution.score_rubric           # -> outputs/part3_distribution/rubric/<judge>.jsonl
python -m part3_distribution.rubric_analysis        # JSD clustering + permutation test
python -m part3_distribution.plot_family_tilts      # per-family over/under-emphasis
python -m part3_distribution.plot_model_tilts       # per-model version
python -m part3_distribution.plot_three_ratios      # across/within ratio: open-vocab vs rubric
python -m part3_distribution.plot_substance_vs_style # within-vs-across gap, both spaces
```

The headline: family clustering survives (and sharpens) on the clean fixed
codebook — it's substance, not vocabulary.

**Recorded statistics.** The permutation p-values are no longer print-only.
`rubric_analysis` computes both granularities (rubric, plus open-vocab read from Part 2's
`js_divergence_freq.csv`) and writes one `outputs/part3_distribution/stats_<judge>.json`
(within/across JSD, ratio, permutation p per granularity, NN purity, per-criterion base rates).
**Both ratio plots (`plot_three_ratios`, `plot_substance_vs_style`) are pure readers of that
file**, so they can't drift apart. Part 2's `stats_*.json` records its four geometry tests.
Cite the paper from these files, not console scrollback. Run order: `nearest_neighbor` →
`rubric_analysis` → the plots.

---

## Conventions

- **OpenRouter client**: the `openai` SDK pointed at `https://openrouter.ai/api/v1`.
  Llama extraction/judging is pinned to the **bf16** provider (`config.EXTRACTOR_PROVIDER`)
  so precision doesn't drift call-to-call.
- **Resumable**: generation, extraction, rubric scoring, and eval all skip work
  already written, so an interrupted run resumes where it left off.
- **Concurrency**: LLM calls fan out over a thread pool; a single consumer loop
  does all writes, so appends never race.
