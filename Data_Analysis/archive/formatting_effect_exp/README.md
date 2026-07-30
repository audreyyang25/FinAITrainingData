# Formatting effect

Does presenting the **same** case as a conversation vs a narrative change the scores?
Pairs each P12 narrative with its verified conversation rewrite, generates + judges
both arms, and runs a paired-delta test (`delta = conversation − narrative`). Outputs
to `../results/convo_formatting_effect/`.

## Run

From the `Data_Analysis/` directory, in order:

```bash
# 1. build the conversation rewrites -> ../../modified_data/standard_conversation.jsonl
python formatting_effect_exp/rewrite_format.py

# 2. generate answers + judge (with question) + aggregate
python formatting_effect_exp/formatting_effect.py
```

Stats + figures only, from the existing judgments (no API calls):

```bash
python formatting_effect_exp/formatting_effect.py agg
```

Both steps are resumable. Requires `OPENROUTER_API_KEY` in the repo-root `.env`
(see the `Data_Analysis/` README for setup + dependencies).
