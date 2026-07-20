# Gender effect

Does swapping the **client's** name (male- vs female-coded) change the scores, holding
the case and demographic constant? Generates + judges both arms and runs a paired-delta
test (`delta = male − female`). Outputs to `../results/gender_effect/`.

## Run

From the `Data_Analysis/` directory, in order:

```bash
# 1. build the name-swapped variants -> ../../modified_data/name_variants.jsonl
python gender_effect_exp/rewrite_names.py

# 2. generate answers + judge (with question) + aggregate
python gender_effect_exp/gender_effect.py

# 3. (optional) extract the pairs the judges scored identically
python gender_effect_exp/extract_identical.py
```

Stats + figures only, from the existing judgments (no API calls):

```bash
python gender_effect_exp/gender_effect.py agg
```

`rewrite_names.py` uses spaCy's `en_core_web_sm` (via `name_swap.py`) for deterministic
name detection. Requires `OPENROUTER_API_KEY` in the repo-root `.env` (see the
`Data_Analysis/` README for setup + dependencies).
