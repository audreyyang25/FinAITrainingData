# LLM-as-a-Judge Pipeline

An LLM-as-a-judge evaluation harness for the AI-suitability (securities compliance)
training data, plus two controlled-perturbation studies built on top of it:

1. **Main eval** — each model generates an answer to a case; a suite of judge
   models scores each answer against the ground-truth answer.
2. **Formatting effect** — does presenting the *same* case as a narrative vs a
   conversation change the scores?
3. **Gender effect** — does swapping the *client's* name (male- vs female-coded)
   change the scores?

All model calls go through **OpenRouter** (one OpenAI-compatible endpoint), so
every model is just a slug.

---

## Setup

```bash
pip install openai pandas matplotlib scipy python-dotenv spacy
python -m spacy download en_core_web_sm      # only needed by the deterministic name_swap
```

Create `.env` in the repo (loaded automatically via `config.py`):

```
OPENROUTER_API_KEY=sk-or-...
```

> Model slugs live in `config.py` (`GEN_SUITE`, `JUDGE_SUITE`). They're OpenRouter
> slugs and drift over time — verify them at https://openrouter.ai/models. A bad
> slug shows up as `GEN FAIL` / `JUDGE FAIL` for that model.

---

## Data

Source: `../AI Suitability Training Materials/23_Folders_Suitability/` (see
`config.DATA_DIR`). Five dataset "types", each with its own schema, mapped to a
common `(prompt, ground_truth)` shape by an **adapter** in `config.ADAPTERS`:

| key | file | prompt from | truth from |
|---|---|---|---|
| `standard` | P12 (n=499) | fact_pattern + question | answer |
| `borderline` | P13 (n=250) | fact_pattern + question | analysis + likely_outcome |
| `conversations` | P14 (n=50) | synthesized task + conversation | legal_assessment |
| `redflags` | P15 (n=50) | synthesized task + fact_pattern | red_flag + concern + action |
| `adversarial` | P16 (n=200) | fact_pattern + question | correct_answer |

`load_dataset(name, limit, data_dir)` yields `(id, prompt, truth)` per record.

---

## Core conventions

- **OpenRouter client**: the `openai` SDK pointed at `https://openrouter.ai/api/v1`
  with `OPENROUTER_API_KEY`. Generation and judging both pass
  `extra_body={"reasoning": {"effort": "low"}}` so reasoning models
  (gemini/deepseek/qwen) don't spend the token budget thinking and return
  empty/truncated output.
- **Resumability**: every stage appends one JSON line per unit of work, keyed by a
  tuple (e.g. `(dataset, id, generator)`). On rerun, already-present keys are
  skipped — Ctrl-C and rerun is always safe, and **you never re-pay for completed
  work**. The judge marks unparseable responses `valid: false`; those are *not*
  counted as done, so they can be re-run later.
- **Two entry points don't mix**: outputs are namespaced per run
  (`runs/v1`, `runs/v2`, `runs/v3`). Don't point two scripts at the same file.

---

## Files

### Core eval
- **`config.py`** — `DATA_DIR`, `GEN_SUITE` (generators), `JUDGE_SUITE` (judges),
  `ADAPTERS` (the five datasets), `load_dataset`. Loads `.env`.
- **`generations.py`** — `model_call` (generator, `reasoning: low`) + resumable
  `run_generation`. Also imported by the effect pipelines.
- **`judge.py`** — `JUDGE_SYS`, `judge_user` (optional question / `guessed_question`),
  `parse_judgment` (tolerant JSON parse; returns `valid=False` rather than
  fabricating a score), resumable `run_judging`.
- **`aggregate.py`** — main-eval aggregation (generator × judge matrix, marginals).
- **`run.py`** — orchestrates the main eval per run (`runs/v1`), with a cross-run
  variance block for comparing a second dataset later.
- **`figures.py`** — shared `save_table_fig` / `save_bar` / `save_heatmap`
  (headless PNG rendering; used by v2 and v3).

### Data rewriting (perturbations)
- **`rewrite_format.py`** — narrative → client/advisor **conversation**, with an
  LLM **fact-verifier loop** (rewrite → check `missing`/`added` → retry). Writes
  `modified_data/standard_conversation.jsonl`, one record per id, tagged
  `_verified` + `_fact_check`. Question/answer kept byte-identical.
- **`rewrite_names.py`** — LLM-based **client-only name swap** across a
  gender × demographic grid. Identifies the client by **role** (not position),
  **skips cases with ≥2 named clients** (couples confound the gender signal), and
  gates every variant through a **token-diff verifier** (rejects any change beyond
  the client's name / pronouns / spouse-relation terms; also catches the advisor
  being renamed). Currently **gender-only** (`DEMOS = ("white",)`). Writes
  `modified_data/name_variants.jsonl`; caches roles in `person_roles.jsonl`.
- **`name_swap.py`** — deterministic name-swap toolkit used by `rewrite_names`
  (name bank `NAMES`, `assign_names` w/ advisor-avoidance, and the verifier
  primitives `changed_tokens` / `find_unexpected_changes` / `find_leaks`; plus a
  fully-deterministic `make_variant` with spaCy-based "her" → his/him resolution,
  retained from the earlier deterministic approach).

### Studies
- **`formatting_effect.py`** — run **v2**. Narrative vs conversation, paired by id.
- **`gender_effect.py`** — run **v3**. Male vs female variant, paired by case.
- **`extract_identical.py`** — inspection helper: pulls the judgment records for
  `delta == 0` gender pairs into `runs/v3/identical_score_judgments.jsonl`.

---

## The three studies

### 1. Main eval — `runs/v1`
Every `GEN_SUITE` model answers every record in the five datasets; every
`JUDGE_SUITE` model scores each answer (two conditions: `no_q` = reference +
candidate; `with_q` = also shows the original question). Aggregation builds the
generator × judge score matrix and marginals.

```bash
python run.py          # generate -> judge -> aggregate, into runs/v1
```

> Originally a **self-bias** study (generators == judges, diagonal = self). The
> suites are now split (`GEN_SUITE` ≠ `JUDGE_SUITE`), so the "diagonal = self"
> reading no longer applies — the matrix is still valid, but keys `claude`/`gpt`
> mean *different* models on the generator vs judge side.

### 2. Formatting effect — `runs/v2`
Same standard cases in two formats (only `fact_pattern` differs; question/answer
identical). **Judged with-question only.** Paired signed effect:
`delta = score(conversation) − score(narrative)`.

```bash
python rewrite_format.py       # produce modified_data/standard_conversation.jsonl
python formatting_effect.py    # generate -> judge -> aggregate + figures
python formatting_effect.py agg   # <- stats + figures ONLY (no API calls)
```

- Sample size = verified conversation records (`_verified: true`) ∩ P12.
  **To grow it:** raise `LIMIT` in `rewrite_format.py`, rerun it (resume adds only
  the new ids), then rerun `formatting_effect.py`. `formatting_effect` has no
  `LIMIT` of its own.
- Unverified rewrites are excluded by a `_verified` filter in `load_formats`. To
  retry an unverified id: strip its line from `standard_conversation.jsonl`, then
  rerun `rewrite_format.py` (it's keyed by id).

### 3. Gender effect — `runs/v3`
Male vs female variant of each case (from `name_variants.jsonl`), demographic held
constant. Judged with-question. Signed effect: `delta = score(male) − score(female)`
(+ ⇒ male scored higher).

```bash
python rewrite_names.py     # produce modified_data/name_variants.jsonl
python gender_effect.py     # generate -> judge -> aggregate + figures
```

---

## Outputs

Each study writes to `runs/<v>/`:
- `generations.jsonl`, `judgments.jsonl` — raw, resumable.
- `*_deltas.csv` — one row per pair with both arms + `delta`.
- `*_{overall,by_judge,by_generator,gen_x_judge}.csv` (v3 also `by_type`).
- `figures/*.png` — table images + a by-generator bar + a gen×judge heatmap.

Terminal output is intentionally minimal (progress + "saved …"); read the figures/CSVs.

---

## Reading the results

- **`delta`** = arm A − arm B (v2: conversation−narrative; v3: male−female). At the
  raw level there's one delta per `(case, generator, judge)`, and it exists only
  when **both** arms got a `valid` score.
- **Significance and descriptive tables use different units.** The two judges score
  the *same* answers, so they aren't independent — counting both inflates `n` and
  the p-value. The significance tables therefore **collapse judges** (average them):
  - `overall`, `by_generator`, `by_type` → one delta per **`case × generator`**
    (judges averaged). Their `n` ≈ cases × generators.
  - `by_judge`, `gen×judge` → left on the **raw per-judge pairs** (descriptive).
  So `n` in `overall`/`by_generator` is *not* comparable to `n` in `by_judge`.
- **`t` and `p`** (in every summary table) test whether the mean delta = 0 (no effect):
  `t = mean / (std/√n)`, `p =` two-sided Student's t with `df = n−1`. A Wilcoxon
  signed-rank p (non-parametric, robust to outliers, drops ties) is also printed for
  the overall. Rule of thumb: `|t| ≳ 2` / `p < 0.05` ≈ significant.
- **`std` is spread, not precision.** It's the case-to-case scatter of deltas; the
  uncertainty of the *mean* is `std/√n` (shrinks with n). A tiny mean can still be
  significant if `std` is small and n large (e.g. gender `redflags`).
- **Uneven `n` across models** = uneven success: a model with more `GEN EMPTY`/
  `GEN FAIL`, or a judge with more `valid:false`, loses pairs (reasoning models and
  `claude-sonnet-5` as a judge drop the most). The delta is within-judge, so this
  doesn't bias the effect — but check `by_judge` to confirm the two judges agree in
  direction before trusting the pooled number.
- **Watch multiple comparisons** in `by_type` (5 tests): expect ~1 to cross p<0.05
  by chance, so treat a lone significant subgroup skeptically.

### Diagnostics (run from `LLM_As_A_Judge/`)
```bash
# paired cases per type + how many are missing/skipped (name variants)
python3 -c "import json,collections; g=collections.defaultdict(set)
[g[(json.loads(l)['type'],json.loads(l)['base_id'])].add(json.loads(l)['gender']) for l in open('../modified_data/name_variants.jsonl')]
print('paired cases:', sum(len(v)==2 for v in g.values()))"

# per-judge valid vs invalid judgments
python3 -c "import json,collections; v,i=collections.Counter(),collections.Counter()
[(v if json.loads(l).get('valid') else i).update([json.loads(l)['judge']]) for l in open('runs/v3/judgments.jsonl')]
print('valid',dict(v));print('invalid',dict(i))"
```

---

## Key design decisions

- **`reasoning: low`** on generation and judging — prevents reasoning-model
  truncation/empty output. Trade-off: reasoning models are evaluated at reduced
  depth (fine for a pilot; raise `max_tokens` instead if you want full depth).
- **Fact-verifier gate (formatting)** — LLM rewrites drift; the verifier catches
  dropped/invented facts and flags `_verified: false` so bad rewrites don't
  contaminate the comparison.
- **Client-only, role-based name swap (gender)** — only the client's name/pronouns
  change; the advisor is fixed (and protected by the token-diff gate). Couples
  (≥2 named clients) are skipped because flipping one client's gender would drag
  the spouse's gender along, confounding the signal.
- **Paired / within-subject design** — the same case appears in both arms, so the
  signed difference isolates the perturbation and cancels case difficulty and
  judge scale.
