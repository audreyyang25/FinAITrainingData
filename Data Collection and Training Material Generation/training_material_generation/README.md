# AI Training Materials — folder index & dataset catalog

Reorganized 2026-06-22. (Pre-reorg snapshot saved at `~/AI_training_materials_before/`.)
Note: the generator/benchmark scripts referenced in the logs are **not** in this repo — they
live on the original author's machine. Nothing here reads these files by path, so the
reorganization breaks nothing locally.

## Structure

| Folder | Contents |
|--------|----------|
| `datasets/` | The 5 phase datasets (`.json` + `.txt`) — the deliverable training data |
| `benchmarks/` | Model-evaluation result files (JSON) + the P13 binary cache |
| `logs/generation/` | Logs from generating the datasets (phase10, phase12–16, fill-up, hypotheticals_v2) |
| `logs/benchmark/` | Current/final benchmark run logs |
| `logs/_archived/` | Superseded earlier log iterations (see that folder's README) |
| `hypotheticals/` | 983 flat per-scenario `.txt` files + `all_hypotheticals.txt` + `hypotheticals.json` + the Phase-10 source (`hypotheticals_phase10_498.json`) + `_older_versions/` |
| `backups_mar21/`, `_v2/`, `_v3/` | Progressive Mar-21 filtering snapshots (not byte-identical) |
| `Suitability Only/` | Suitability-focused variant of the datasets (P12–P16) |
| `New_Folders_Suitability/` | Six new-product-folder datasets (P12–P13) |

## Phase map

| Phase | Dataset file (`datasets/`) | Records | Type |
|-------|--------------|---------|------|
| P12 | `March_6_AI_Training` | 499 | Standard |
| P13 | `March_6_AI_Training_Borderline` | 250 | Borderline |
| P14 | `March_7_AI_Training_Materials_Conversations` | 50 | Advisor–client conversations |
| P15 | `AI_training_materials_Red_Flags` | 50 | Red-flag scenarios |
| P16 | `March_14_Adversarial` | 200 | Adversarial |

## Log → output map

- `logs/generation/phase12–16_log.txt` → the corresponding P12–P16 datasets
- `logs/generation/phase10_log.txt` → original hypotheticals generation (`hypotheticals/`)
- `logs/generation/phase_fill_up_log.txt` → top-up pass after filtering (2026-03-21)
- `logs/generation/hypotheticals_v2_log.txt` → `hypotheticals/hypotheticals_phase10_498.json`
- `logs/benchmark/bench_gpt54_v3_log.txt` → final GPT-5.4 suitability bench (P13–P16)
- `logs/benchmark/March_23_Suitability_Benchmark_log.txt` → `benchmarks/March_23_Suitability_Benchmark.json`
- `logs/benchmark/new_folders_bench_4o_vs_54_log.txt` → `benchmarks/New_Folders_Benchmark_4o_vs_54.json`
- `logs/benchmark/run_gen_bench_log.txt` → master orchestrator driving generation + benchmark

## Known caveats

- `benchmarks/March_23_Suitability_Benchmark.json` is **incomplete for GPT-5.4** (P12 391/499); GPT-4o is complete.
- `benchmarks/March_17_GPT_Benchmark.json` is a **superseded** early adversarial run (see `benchmarks/README.md`).

---

# Dataset Catalog: Suitability & Best-Interest Items

> **Originally:** "AI Training Materials Catalog" report, dated **April 5, 2026**, integrated
> into this README on 2026-06-22 (original preserved at
> `FinAdvisor_Training/_merged_sources/review_training_materials_catalog.md`).
>
> **Verification (2026-06-22):** All per-file counts below were re-checked against the live
> files. Everything matched **except New_Folders_Suitability**, which has grown since the
> report (P12 420→600, P13 301→600; total **721 → 1,200**). The tables and totals below have
> been updated to reflect this, and file paths/names were updated to the reorganized structure.
> Derived/analytical figures that required the original deep content analysis — the
> compliant/non-compliant estimates (§4), difficulty split (§5), and topic clusters (§6) — are
> kept **as-of April 5, 2026** and were **not** recomputed; treat them as approximate.

## 1. Executive Summary

The collection is a substantial corpus of suitability- and best-interest-focused training data
across multiple formats and difficulty levels. As of this verification it catalogs roughly
**4,668 suitability/best-interest training items** (excluding benchmark/evaluation data), plus
**1,981 benchmark evaluation items**. The materials cover an impressive breadth of topics —
over **1,300 unique topic labels** (as-of Apr 5) — spanning Reg BI care obligations, FINRA
suitability rules, fiduciary duties, DOL/ERISA requirements, state-level standards, and
international comparative frameworks.

## 2. Total Counts by Source

### Primary Training Data (~4,668 suitability items)

| Source | Total Items | Suitability Items | % Suitability |
|--------|-------------|-------------------|---------------|
| `hypotheticals/` (984 .txt files) | 984 | 879 | 89.3% |
| `hypotheticals/hypotheticals_phase10_498.json` (was hypotheticals_v2.json) | 498 | 493 | 99.0% |
| `datasets/March_6_AI_Training.json` | 499 | 499 | 100.0% |
| `datasets/March_6_AI_Training_Borderline.json` | 250 | 248 | 99.2% |
| `datasets/March_14_Adversarial.json` | 200 | 200 | 100.0% |
| `datasets/AI_training_materials_Red_Flags.json` | 50 | 50 | 100.0% |
| `datasets/March_7_..._Conversations.json` | 50 | 50 | 100.0% |
| `Suitability Only/` (5 JSON files) | 1,049 | 1,049 | 100.0% |
| `New_Folders_Suitability/` (2 JSON files) ⬆ | 1,200 | 1,200 | 100.0% |
| **TOTAL** | **4,780** | **4,668** | **97.7%** |

> ⬆ New_Folders grew from 721 → 1,200 since the Apr-5 report. Its suitability count is assumed
> ~100% (consistent with the original methodology); the per-item suitability split was not re-analyzed.

### Benchmark/Evaluation Data (not training, used for model evaluation)

| Source | Total Items | Suitability Items |
|--------|-------------|-------------------|
| `benchmarks/March_23_Suitability_Benchmark.json` | 1,990 | 1,981 |
| `benchmarks/March_17_GPT_Benchmark.json` | 400 | 221 |

### Backup Folders (older versions of current files, NOT additional items)

- **backups_mar21/** — 5 JSON files, pre-topic-filter versions of the main training files
- **backups_mar21_v2/** — 7 files including pre-filter intermediate versions
- **backups_mar21_v3/** — 9 files including "pretopicfilter" labeled originals

These contain earlier iterations of the same datasets — NOT additional unique items; do not double-count.

## 3. Breakdown by Data Format/Phase

### Suitability Only Folder (1,049 items)
| File | Count | Format |
|------|-------|--------|
| suitability_only_P12.json | 499 | Standard Q&A (fact_pattern, question, answer) |
| suitability_only_P13.json | 250 | Borderline cases (analysis, why_borderline, borderline_factors) |
| suitability_only_P14.json | 50 | Conversation format (multi-turn dialogues) |
| suitability_only_P15.json | 50 | Red flag identification (red_flag, indicators, recommended_action) |
| suitability_only_P16.json | 200 | Adversarial (adversarial_type, common_wrong_answer) |

### New_Folders_Suitability (1,200 items — updated 2026-06-22)
| File | Count | Format |
|------|-------|--------|
| new_folders_P12.json | 600 | Standard Q&A with source_folder attribution |
| new_folders_P13.json | 600 | Borderline analysis format |

### Hypotheticals (879 suitability items out of 984)
- Format: Structured .txt files with ID, Source Folder, Topic, Difficulty, Verdict, FACTS, QUESTION, ANSWER, APPLICABLE STANDARD, KEY FACTOR sections
- Non-suitability items (105) cover best execution, state registration violations, ERISA case law, trust investment management, etc.

## 4. Compliant vs. Non-Compliant Breakdown *(as-of April 5, 2026 — not recomputed)*

| Source | Non-Compliant | Compliant | Borderline/Unknown |
|--------|--------------|-----------|-------------------|
| hypotheticals/ (.txt) | 544 | 334 | 1 |
| Suitability Only/ | 424 | 75 | 550 |
| New_Folders_Suitability/ | 419 | 1 | 301 |
| March_6_AI_Training.json | 17 | 12 | 470 |
| March_14_Adversarial.json | 51 | 8 | 141 |
| **Estimated Totals** | **~1,455** | **~430** | **~1,463+** |

**Key Observation:** Significant imbalance toward non-compliant scenarios. Compliant examples
are underrepresented, particularly in Suitability Only and New_Folders. The "Unknown/Borderline"
category is large because many JSON items lack an explicit compliant/non-compliant label — the
determination is embedded in the answer text. P13 items are intentionally ambiguous by design.
*(Note: New_Folders has since grown to 1,200 items, so its row understates current totals.)*

## 5. Difficulty Distribution (Hypotheticals .txt files)

| Difficulty | Count |
|------------|-------|
| Easy | 164 |
| Medium | 365 |
| Hard | 349 |

Reasonably balanced with a slight emphasis on medium/hard — appropriate for professional training.
*(Hypotheticals are unchanged since Apr 5, so these remain accurate.)*

## 6. Unique Topics Covered *(as-of April 5, 2026)*

**Total unique topic labels: 1,316** — though many are near-duplicates with minor naming
variations. Grouping by theme, the major topic clusters are:

### Top 30 Topic Clusters (by frequency)

1. Inappropriate Risk Profile / Risk Mismatch — 159+
2. Senior Investor Exploitation / Protection — 140+
3. Failure to Know Customer / KYC — 128+
4. Complex Products for Unsophisticated Investors — 120+
5. High Fees / Unsuitable Cost Structure — 108+
6. Leveraged ETF Suitability — 79+
7. Illiquid Investments / Products — 73+
8. Illiquid Real Estate (Non-Traded REITs) — 63+
9. Variable Annuity Suitability — 62+
10. Excessive Trading / Churning — 56+
11. Tax-Deferred Unnecessary Annuity — 54+
12. Alternative Investments Unsuitable — 53+
13. Annuity Switching / Replacement — 46+
14. Reverse Churning — 46+
15. Private Placements Unsuitable — 46+
16. Options Suitability — 44+
17. Unsuitable for Income Investor — 44+
18. Structured Products Unsuitable — 42+
19. Mutual Fund Switching / Churning — 42+
20. Crypto/Digital Assets Unsuitable — 37+
21. Excessive Load Funds — 37+
22. Overconcentration / Single Stock — 36+
23. Penny Stocks Unsuitable — 35+
24. Margin Trading Unsuitable — 34+
25. Speculative for Conservative Investor — 34+
26. Bond Duration Mismatch — 34+
27. Market Timing Unsuitable — 32+
28. Junk Bonds for Conservative Investor — 31+
29. Lack of Diversification — 29+
30. Reg BI Care Obligation — 25+

### Regulatory Framework Topics
Reg BI (Care/Disclosure/Conflict) 100+ · FINRA Suitability (Rule 2111) 25+ explicit ·
FINRA AWC Patterns 18+ · FINRA Quantitative Suitability 15+ · IA Fiduciary Duty 25+ ·
DOL/ERISA Fiduciary 30+ · DOL Rollover Analysis 18+ · PTE 2020-02 9+ · State Fiduciary Rules
15+ (NY Reg 187, Nevada, Maryland, Colorado) · NAIC Model 275 5+ · CTA/CPO Suitability 14+

### Specialized / Emerging Topics
AI/Robo-Adviser 8+ · ESG/Sustainable 9+ · Digital Assets/Crypto 9+ · MiFID II 4+ ·
Singapore FAA 3+ · IOSCO Cross-Border 4+ · ASIC Best Interest 3+ · FCA Consumer Duty (in
non-suitability items) · New Zealand Financial Advice Regime 1

## 7. Assessment of Topic Diversity and Balance

### Strengths
1. **Exceptional breadth** — covers virtually every major area of US suitability law plus international comparisons (MiFID II, ASIC, Singapore FAA, IOSCO).
2. **Multiple format types** — standard Q&A, borderline analysis, adversarial trick questions, conversational dialogues, red-flag identification.
3. **Difficulty stratification** — easy to expert.
4. **Real-world relevance** — leveraged ETFs for seniors, VA switching, non-traded REIT concentration are among the most common actual violations.
5. **Emerging issues** — AI/robo-adviser suitability, digital assets, ESG fiduciary duties, predictive data analytics.

### Weaknesses / Gaps
1. **Compliant example deficit** — non-compliant outnumber compliant ~3.4:1; risk of a violation-finding bias. Add more "close but compliant" scenarios.
2. **Topic label fragmentation** — 1,316 labels, many near-duplicates; consider consolidating to ~50–100 canonical categories.
3. **Senior-investor overrepresentation** — "Margaret" (retired schoolteacher, 70s, Tucson AZ) recurs heavily; add diverse client profiles.
4. **Thin international coverage** relative to US frameworks.
5. **Insurance-only suitability** (NAIC Model 275, state insurance rules) could use more coverage.

## 8. Sample Quality Assessment

**Overall Quality: STRONG (8/10)** *(based on review of 25+ items across all major sources)*

**Strengths:** detailed/realistic fact patterns (specific dollar amounts, ages, expense ratios,
timelines); generally accurate legal analysis (Reg BI §240.15l-1(a)(2)(i), FINRA Rule 2111, IA
Act fiduciary standards correctly cited); well-crafted adversarial traps; genuinely ambiguous
borderline items; red-flag items capturing compounding violations.

**Issues:** character-name recycling ("Margaret"/"Derek"/"Kevin") risks spurious pattern
associations; some P12 answers one-sided (light on mitigating factors/defenses); occasional
temporal-consistency looseness on which rule version applies; benchmark data uses composite keys
(`gpt-4o::P12::2`) and is evaluation data — keep separate from training.

## 9. Duplicate Analysis

**~90 duplicates** found between `datasets/March_6_AI_Training.json` and
`hypotheticals/hypotheticals_phase10_498.json` (formerly hypotheticals_v2.json), based on
fact-pattern matching — likely because the latter is a filtered/reorganized cut of the former.

**No duplicates** between: hypotheticals/ .txt files ↔ the phase-10 JSON; Suitability Only/ ↔
New_Folders_Suitability/.

**Recommendation:** deduplicate those two files before training, or use only one.

## 10. File Inventory

### Active Training Files (Suitability/Best Interest)

| File/Folder | Suit. Items | Format | Phase |
|-------------|-------------|--------|-------|
| `hypotheticals/*.txt` | 879 | Structured text | P12 (original) |
| `hypotheticals/hypotheticals_phase10_498.json` | 493 | JSON | P12 (v2) |
| `datasets/March_6_AI_Training.json` | 499 | JSON | P12 |
| `datasets/March_6_AI_Training_Borderline.json` | 248 | JSON | P13 |
| `datasets/March_14_Adversarial.json` | 200 | JSON | P16 |
| `datasets/AI_training_materials_Red_Flags.json` | 50 | JSON | P15 |
| `datasets/March_7_..._Conversations.json` | 50 | JSON | P14 |
| `Suitability Only/suitability_only_P12.json` | 499 | JSON | P12 |
| `Suitability Only/suitability_only_P13.json` | 250 | JSON | P13 |
| `Suitability Only/suitability_only_P14.json` | 50 | JSON | P14 |
| `Suitability Only/suitability_only_P15.json` | 50 | JSON | P15 |
| `Suitability Only/suitability_only_P16.json` | 200 | JSON | P16 |
| `New_Folders_Suitability/new_folders_P12.json` | 600 | JSON | P12 |
| `New_Folders_Suitability/new_folders_P13.json` | 600 | JSON | P13 |

### Benchmark/Evaluation Files (`benchmarks/`)

| File | Items | Notes |
|------|-------|-------|
| March_23_Suitability_Benchmark.json | 1,990 | GPT-4o (1,049) and GPT-5.4 (941) model answers |
| March_17_GPT_Benchmark.json | 400 | GPT-4o (200) and GPT-5.4 (200) adversarial answers |
| New_Folders_Benchmark.json | — | Benchmark for new-folders data (4o vs 5.2) |
| New_Folders_Benchmark_4o_vs_54.json | — | Model comparison benchmark (4o vs 5.4) |

### Backup Folders (historical, not for training)
- backups_mar21/ (5 files) · backups_mar21_v2/ (7 files) · backups_mar21_v3/ (9 files)

### Log Files (process documentation, not training data)
- See `logs/generation/`, `logs/benchmark/`, `logs/_archived/`, and each folder's `generation_log.txt`.

## 11. Phase Structure Explanation

- **P12 (Standard Q&A)** — fact pattern → question → answer with verdict. Core format. ~2,470 suitability items.
- **P13 (Borderline Analysis)** — ambiguous scenarios requiring nuanced analysis. ~1,100 items.
- **P14 (Conversations)** — multi-turn advisor-client dialogues with legal assessment. ~100 items.
- **P15 (Red Flags)** — identify red flags, regulatory concerns, recommended actions. ~100 items.
- **P16 (Adversarial)** — trick questions testing common reasoning errors. ~400 items.

*(P12/P13 totals updated for the New_Folders expansion; others as-of Apr 5.)*

## 12. Recommendations

1. **Deduplicate** `datasets/March_6_AI_Training.json` and `hypotheticals/hypotheticals_phase10_498.json` (~90 overlapping items)
2. **Increase compliant examples** to at least 2:1 non-compliant:compliant (currently ~3.4:1)
3. **Consolidate topic labels** from 1,316 to ~50–100 canonical categories
4. **Diversify client demographics** beyond the "Margaret the retired schoolteacher" archetype
5. **Expand P14 (conversation) and P15 (red flag)** formats — currently ~100 each vs. 2,400+ for P12
6. **Add more international items** if cross-border suitability is in scope
7. **Keep training and benchmark data clearly separated** (now done: `datasets/` vs `benchmarks/`)
