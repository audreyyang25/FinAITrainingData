# Intra-family comparison

Compares the GPT family (gpt-3.5 / gpt-4 / gpt-4o / gpt-5.5) on the **borderline (P13)**
cases, scored by a 4-dimension borderline judge (factor recall, citation-fabrication
check, outcome direction/justification, content similarity). Outputs to
`../results/gpt_family/`.

## Run

From the `Data_Analysis/` directory:

```bash
python intrafamily_exp/family_model_comp.py
```

One pass covers the whole family: gpt-5.5 automatically gets a brevity system prompt
(`BREVITY_SYSTEM`) so it reaches its conclusion instead of running long and truncating —
no separate rerun. Generation and judging are resumable. Produces
`generations.jsonl`, `judgments_detail.jsonl`, `aggregate_by_model.csv`,
`citations_for_review.txt`, and `rubric.csv` (judge rubric snapshot, before + after).

Requires `OPENROUTER_API_KEY` in the repo-root `.env` (see the `Data_Analysis/` README
for setup + dependencies).
