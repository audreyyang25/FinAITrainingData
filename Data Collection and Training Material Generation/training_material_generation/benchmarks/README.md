# Benchmark result files

Model-evaluation outputs (not training data). See `../logs/benchmark/` for the run logs.

| File | What it is | Status |
|------|------------|--------|
| `March_23_Suitability_Benchmark.json` | Suitability bench, GPT-4o vs GPT-5.4, keyed `model::phase::id` | ⚠️ GPT-5.4 incomplete (P12 391/499); GPT-4o complete |
| `New_Folders_Benchmark_4o_vs_54.json` | New product folders, GPT-4o vs GPT-5.4 | Final |
| `New_Folders_Benchmark.json` | New product folders, GPT-4o vs GPT-5.2 | Older model pairing |
| `March_23_P13_binary_cache.json` | Cached P13 yes/no verdicts (speeds up reruns) | Working cache |
| `March_17_GPT_Benchmark.json` | Early adversarial bench (4o vs 5.4), dated 2026-03-16 | ⚠️ Superseded by later runs / Apr-4 exported PDF |
