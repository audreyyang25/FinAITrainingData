# FinAdvisor Hypotheticals

983 synthetic Q&A hypotheticals for training a financial adviser conduct AI.

## Contents

| File | Description |
|------|-------------|
| `0001_compliant.txt` … `0983_*.txt` | One file per hypothetical, full fact pattern + question + answer + reasoning |
| `hypotheticals.json` | Full dataset in JSON (all fields) |
| `all_hypotheticals.txt` | All 983 concatenated into one file |

## Structure of Each Entry

- **FACTS** — Detailed scenario describing adviser, client, recommendation, and context
- **QUESTION** — Compliance question (yes/no framing)
- **ANSWER** — Verdict + full legal reasoning, citing specific rules and standards
- **APPLICABLE STANDARD** — Reg BI, FINRA Rules, IA Act, ERISA, etc.
- **KEY FACTOR** — Single most important compliance/violation factor

## Coverage

| Batch | Count | Notes |
|-------|-------|-------|
| Legacy (source_folder: unknown) | 485 | Generated Feb 25 – Mar 2; many have detailed answers baked in |
| Phase 10 tagged (folders 01–23) | 498 | Generated Mar 3–7; answer + reasoning in separate fields (merged here) |

## Topic Buckets

Reg BI, IA fiduciary duty, FINRA suitability (Rule 2111), variable annuities, rollovers, 
DOL/ERISA, senior investors, ESG, crypto, robo-advisers, state law, FINRA arbitration, 
SEC enforcement, OHO/ALJ decisions, PIABA, international standards (FCA, MiFID II, ASIC), and more.

## Verdict Split

Approximately 50% compliant / 50% non-compliant by design.

## Generated

March 7, 2026 — FinAdvisor Training Project, Phase 10
