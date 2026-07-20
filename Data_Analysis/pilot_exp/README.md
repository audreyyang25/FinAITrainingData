# Pilot

The base LLM-as-a-judge run: every generator answers each case, a judge suite scores
each answer **both with and without the question shown**, then self-bias /
question-effect / leniency aggregation. Outputs to `../results/pilot_run/`.

## Run

From the `Data_Analysis/` directory:

```bash
python pilot_exp/pilot.py
```

Generation and judging are resumable — rerunning skips anything already in
`results/pilot_run/{generations,judgments}.jsonl`, so an interrupted run continues
where it left off. Requires `OPENROUTER_API_KEY` in the repo-root `.env` (see the
`Data_Analysis/` README for setup + dependencies).
