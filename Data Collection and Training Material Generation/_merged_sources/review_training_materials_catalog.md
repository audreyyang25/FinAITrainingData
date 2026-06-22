# AI Training Materials Catalog: Suitability & Best Interest Items

**Report Date:** April 5, 2026  
**Directory:** ~/Desktop/FinAdvisor_Training/AI training materials/  
**Scope:** All training items where suitability or best interest of the customer is a question or focal point

---

## 1. Executive Summary

The AI training materials collection contains a substantial corpus of suitability and best-interest focused training data across multiple formats and difficulty levels. After systematic analysis of all files and folders, this report catalogs **4,189 unique suitability/best-interest training items** (excluding benchmark/evaluation data), plus **1,981 benchmark evaluation items**. The materials cover an impressive breadth of topics — over **1,300 unique topic labels** — spanning Reg BI care obligations, FINRA suitability rules, fiduciary duties, DOL/ERISA requirements, state-level standards, and international comparative frameworks.

---

## 2. Total Counts by Source

### Primary Training Data (4,189 suitability items)

| Source | Total Items | Suitability Items | % Suitability |
|--------|-------------|-------------------|---------------|
| hypotheticals/ (984 .txt files) | 984 | 879 | 89.3% |
| hypotheticals_v2.json | 498 | 493 | 99.0% |
| March_6_AI_Training.json | 499 | 499 | 100.0% |
| March_6_AI_Training_Borderline.json | 250 | 248 | 99.2% |
| March_14_Adversarial.json | 200 | 200 | 100.0% |
| AI_training_materials_Red_Flags.json | 50 | 50 | 100.0% |
| March_7_Conversations.json | 50 | 50 | 100.0% |
| Suitability Only/ (5 JSON files) | 1,049 | 1,049 | 100.0% |
| New_Folders_Suitability/ (2 JSON files) | 721 | 721 | 100.0% |
| **TOTAL** | **4,301** | **4,189** | **97.4%** |

### Benchmark/Evaluation Data (not training, used for model evaluation)

| Source | Total Items | Suitability Items |
|--------|-------------|-------------------|
| March_23_Suitability_Benchmark.json | 1,990 | 1,981 |
| March_17_GPT_Benchmark.json | 400 | 221 |

### Backup Folders (older versions of current files, NOT additional items)

- **backups_mar21/** — 5 JSON files, pre-topic-filter versions of the main training files
- **backups_mar21_v2/** — 7 files including pre-filter intermediate versions
- **backups_mar21_v3/** — 9 files including "pretopicfilter" labeled originals

These backup folders contain earlier iterations of the same datasets. They are NOT additional unique training items and should not be double-counted.

---

## 3. Breakdown by Data Format/Phase

### Suitability Only Folder (1,049 items)
| File | Count | Format |
|------|-------|--------|
| suitability_only_P12.json | 499 | Standard Q&A (fact_pattern, question, answer) |
| suitability_only_P13.json | 250 | Borderline cases (analysis, why_borderline, borderline_factors) |
| suitability_only_P14.json | 50 | Conversation format (multi-turn dialogues) |
| suitability_only_P15.json | 50 | Red flag identification (red_flag, indicators, recommended_action) |
| suitability_only_P16.json | 200 | Adversarial (adversarial_type, common_wrong_answer) |

### New_Folders_Suitability (721 items)
| File | Count | Format |
|------|-------|--------|
| new_folders_P12.json | 420 | Standard Q&A with source_folder attribution |
| new_folders_P13.json | 301 | Borderline analysis format |

### Hypotheticals (879 suitability items out of 984)
- Format: Structured .txt files with ID, Source Folder, Topic, Difficulty, Verdict, FACTS, QUESTION, ANSWER, APPLICABLE STANDARD, KEY FACTOR sections
- Non-suitability items (105) cover topics like best execution, state registration violations, ERISA case law, trust investment management, etc.

---

## 4. Compliant vs. Non-Compliant Breakdown

| Source | Non-Compliant | Compliant | Borderline/Unknown |
|--------|--------------|-----------|-------------------|
| hypotheticals/ (.txt) | 544 | 334 | 1 |
| Suitability Only/ | 424 | 75 | 550 |
| New_Folders_Suitability/ | 419 | 1 | 301 |
| March_6_AI_Training.json | 17 | 12 | 470 |
| March_14_Adversarial.json | 51 | 8 | 141 |
| **Estimated Totals** | **~1,455** | **~430** | **~1,463+** |

**Key Observation:** There is a significant imbalance toward non-compliant scenarios. Compliant examples are underrepresented, particularly in the Suitability Only and New_Folders_Suitability datasets. The "Unknown/Borderline" category is large because many JSON items do not have an explicit compliant/non-compliant label — instead, the compliance determination is embedded in the answer text. The borderline items (P13 format) are intentionally ambiguous by design.

---

## 5. Difficulty Distribution (Hypotheticals .txt files)

| Difficulty | Count |
|------------|-------|
| Easy | 164 |
| Medium | 365 |
| Hard | 349 |

The difficulty distribution is reasonably balanced with a slight emphasis on medium and hard items, which is appropriate for professional training.

---

## 6. Unique Topics Covered

**Total unique topic labels: 1,316**

However, many of these are near-duplicates with minor naming variations (e.g., "leveraged_etf_unsuitable" vs "leveraged_etf_unsuitable_recommendation" vs "unsuitable_leveraged_etf_recommendation_elderly_investor"). Grouping by theme, the major topic clusters are:

### Top 30 Topic Clusters (by frequency)

1. **Inappropriate Risk Profile / Risk Mismatch** — 159+ items
2. **Senior Investor Exploitation / Protection** — 140+ items
3. **Failure to Know Customer / KYC** — 128+ items
4. **Complex Products for Unsophisticated Investors** — 120+ items
5. **High Fees / Unsuitable Cost Structure** — 108+ items
6. **Leveraged ETF Suitability** — 79+ items
7. **Illiquid Investments / Products** — 73+ items
8. **Illiquid Real Estate (Non-Traded REITs)** — 63+ items
9. **Variable Annuity Suitability** — 62+ items
10. **Excessive Trading / Churning** — 56+ items
11. **Tax-Deferred Unnecessary Annuity** — 54+ items
12. **Alternative Investments Unsuitable** — 53+ items
13. **Annuity Switching / Replacement** — 46+ items
14. **Reverse Churning** — 46+ items
15. **Private Placements Unsuitable** — 46+ items
16. **Options Suitability** — 44+ items
17. **Unsuitable for Income Investor** — 44+ items
18. **Structured Products Unsuitable** — 42+ items
19. **Mutual Fund Switching / Churning** — 42+ items
20. **Crypto/Digital Assets Unsuitable** — 37+ items
21. **Excessive Load Funds** — 37+ items
22. **Overconcentration / Single Stock** — 36+ items
23. **Penny Stocks Unsuitable** — 35+ items
24. **Margin Trading Unsuitable** — 34+ items
25. **Speculative for Conservative Investor** — 34+ items
26. **Bond Duration Mismatch** — 34+ items
27. **Market Timing Unsuitable** — 32+ items
28. **Junk Bonds for Conservative Investor** — 31+ items
29. **Lack of Diversification** — 29+ items
30. **Reg BI Care Obligation** — 25+ items

### Regulatory Framework Topics

- **Reg BI (Care, Disclosure, Conflict)** — 100+ items across subtopics
- **FINRA Suitability (Rule 2111)** — 25+ items explicitly labeled, many more embedded
- **FINRA AWC Patterns** — 18+ items
- **FINRA Quantitative Suitability** — 15+ items
- **IA Fiduciary Duty** — 25+ items
- **DOL/ERISA Fiduciary** — 30+ items
- **DOL Rollover Analysis** — 18+ items
- **PTE 2020-02** — 9+ items
- **State Fiduciary Rules** — 15+ items (including NY Reg 187, Nevada, Maryland, Colorado)
- **NAIC Model 275 (Annuity Best Interest)** — 5+ items
- **CTA/CPO Suitability** — 14+ items

### Specialized / Emerging Topics

- **AI/Robo-Adviser Suitability** — 8+ items
- **ESG/Sustainable Investing** — 9+ items
- **Digital Assets/Crypto** — 9+ items
- **MiFID II Suitability** — 4+ items
- **Singapore FAA Know Your Client** — 3+ items
- **IOSCO Cross-Border Suitability** — 4+ items
- **ASIC Best Interest Duty** — 3+ items
- **FCA Consumer Duty** — present in non-suitability items
- **New Zealand Financial Advice Regime** — 1 item

---

## 7. Assessment of Topic Diversity and Balance

### Strengths
1. **Exceptional breadth**: The collection covers virtually every major area of suitability law and regulation in the US, plus international comparisons (MiFID II, ASIC, Singapore FAA, IOSCO).
2. **Multiple format types**: Standard Q&A, borderline analysis, adversarial trick questions, conversational dialogues, and red flag identification provide varied training signals.
3. **Difficulty stratification**: Items span easy to expert difficulty levels.
4. **Real-world relevance**: Topics like leveraged ETFs for seniors, variable annuity switching, and non-traded REIT concentration are among the most common actual regulatory violations.
5. **Emerging issues covered**: AI/robo-adviser suitability, digital assets, ESG fiduciary duties, and predictive data analytics are forward-looking.

### Weaknesses / Gaps
1. **Compliant example deficit**: Non-compliant scenarios heavily outnumber compliant ones (~3.4:1 ratio). A model trained primarily on violations may develop a bias toward finding violations. Recommendation: increase compliant examples, especially "close but compliant" scenarios.
2. **Topic label fragmentation**: With 1,316 unique topic labels, many are near-duplicates. This could cause issues if topic labels are used as training features. Consider consolidating to ~50-100 canonical topic categories.
3. **Senior investor overrepresentation**: "Margaret" (a retired schoolteacher in her 70s from Tucson, Arizona) appears as the prototypical client across many items. This could lead to demographic overfitting. More diverse client profiles (young professionals, middle-aged business owners, institutional investors, non-English-speaking clients) would improve generalization.
4. **International coverage is thin**: While there are a few MiFID II, ASIC, and Singapore items, international suitability regimes are minimally represented compared to US frameworks.
5. **Insurance-only suitability**: NAIC Model 275 and state insurance suitability requirements could use more coverage given the importance of annuity sales.

---

## 8. Sample Quality Assessment

I reviewed 25+ items across all major sources. Assessment:

### Overall Quality: STRONG (8/10)

**Strengths observed:**
1. **Fact patterns are detailed and realistic**: Items include specific dollar amounts, percentages, ages, income figures, expense ratios, commission structures, and timelines. They read like actual FINRA enforcement cases or arbitration claims.
2. **Legal analysis is generally accurate**: References to Reg BI §240.15l-1(a)(2)(i), FINRA Rule 2111, and IA Act fiduciary standards are correctly cited. The analysis correctly applies reasonable-basis, customer-specific, and quantitative suitability tests.
3. **Adversarial items are well-crafted**: The P16/adversarial items include genuine traps — e.g., a broker who conducted thorough due diligence on a reverse convertible but used pre-COVID volatility data, creating a reasonable-basis suitability failure despite apparent diligence.
4. **Borderline items capture genuine ambiguity**: P13 items present scenarios where reasonable regulators might disagree, such as a 70% equity allocation for a 74-year-old with adequate pension income who stated moderate risk tolerance.
5. **Red flag items identify multiple simultaneous concerns**: The red flag format correctly identifies compounding violations (e.g., churning + undisclosed lending from client + inadequate supervision response).

**Issues identified:**
1. **Character name recycling**: "Margaret" (various last names) appears as the client in a very high proportion of items, often as "a retired schoolteacher in Tucson, Arizona." "Derek" and "Kevin" are frequently used as advisor names. This could create spurious pattern associations.
2. **Some answers could be more nuanced**: A few P12 items present complex situations but give relatively one-sided answers. In real regulatory practice, there would be more discussion of mitigating factors and defenses.
3. **Temporal consistency**: Some items reference March 2023 events while applying rules that evolved over 2020-2024. The temporal treatment is generally correct but could be more explicit about which version of rules applies.
4. **Benchmark data structure note**: March_23_Suitability_Benchmark.json uses composite keys (e.g., "gpt-4o::P12::2") and contains model-generated answers from GPT-4o and GPT-5.4. This is evaluation data, not training data, and should be kept separate.

---

## 9. Duplicate Analysis

**90 duplicates found** between March_6_AI_Training.json and hypotheticals_v2.json based on fact pattern matching. These appear to be the same items appearing in both files, likely because hypotheticals_v2.json is a filtered/reorganized version of March_6_AI_Training.json.

**No duplicates found** between:
- hypotheticals/ .txt files and hypotheticals_v2.json (different fact patterns despite similar structure)
- Suitability Only/ and New_Folders_Suitability/ (distinct datasets)

**Recommendation**: Deduplicate March_6_AI_Training.json against hypotheticals_v2.json before training, or use only one of the two.

---

## 10. File Inventory

### Active Training Files (Suitability/Best Interest)

| File/Folder | Suit. Items | Format | Phase |
|-------------|-------------|--------|-------|
| hypotheticals/*.txt | 879 | Structured text | P12 (original) |
| hypotheticals_v2.json | 493 | JSON | P12 (v2) |
| March_6_AI_Training.json | 499 | JSON | P12 |
| March_6_AI_Training_Borderline.json | 248 | JSON | P13 |
| March_14_Adversarial.json | 200 | JSON | P16 |
| AI_training_materials_Red_Flags.json | 50 | JSON | P15 |
| March_7_Conversations.json | 50 | JSON | P14 |
| Suitability Only/suitability_only_P12.json | 499 | JSON | P12 |
| Suitability Only/suitability_only_P13.json | 250 | JSON | P13 |
| Suitability Only/suitability_only_P14.json | 50 | JSON | P14 |
| Suitability Only/suitability_only_P15.json | 50 | JSON | P15 |
| Suitability Only/suitability_only_P16.json | 200 | JSON | P16 |
| New_Folders_Suitability/new_folders_P12.json | 420 | JSON | P12 |
| New_Folders_Suitability/new_folders_P13.json | 301 | JSON | P13 |

### Benchmark/Evaluation Files

| File | Items | Notes |
|------|-------|-------|
| March_23_Suitability_Benchmark.json | 1,990 | GPT-4o (1,049) and GPT-5.4 (941) model answers |
| March_17_GPT_Benchmark.json | 400 | GPT-4o (200) and GPT-5.4 (200) adversarial answers |
| New_Folders_Benchmark.json | — | Benchmark for new folders data |
| New_Folders_Benchmark_4o_vs_54.json | — | Model comparison benchmark |

### Backup Folders (historical, not for training)

- backups_mar21/ (5 files) — Pre-topic-filter originals from March 21
- backups_mar21_v2/ (7 files) — Intermediate filtered versions
- backups_mar21_v3/ (9 files) — Pre-topic-filter with explicit labeling

### Log Files (process documentation, not training data)

- generation_log.txt (in Suitability Only/ and New_Folders_Suitability/)
- phase12_log.txt through phase16_log.txt
- Various benchmark log files

---

## 11. Phase Structure Explanation

The training data is organized into 5 phases:

- **P12 (Standard Q&A)**: Fact pattern → question → answer with verdict. The core format. ~2,290 suitability items.
- **P13 (Borderline Analysis)**: Ambiguous scenarios requiring nuanced analysis of why the case is borderline. ~799 items.
- **P14 (Conversations)**: Multi-turn advisor-client dialogues with legal assessment. ~100 items.
- **P15 (Red Flags)**: Fact patterns requiring identification of red flags, regulatory concerns, and recommended actions. ~100 items.
- **P16 (Adversarial)**: Trick questions designed to test whether models fall for common reasoning errors. ~400 items.

---

## 12. Recommendations

1. **Deduplicate** March_6_AI_Training.json and hypotheticals_v2.json (~90 overlapping items)
2. **Increase compliant examples** to at least a 2:1 non-compliant:compliant ratio (currently ~3.4:1)
3. **Consolidate topic labels** from 1,316 to ~50-100 canonical categories for cleaner topic-based analysis
4. **Diversify client demographics** beyond the "Margaret the retired schoolteacher" archetype
5. **Expand P14 (conversation) and P15 (red flag) formats** — currently only 100 items each, vs. 2,290+ for P12
6. **Add more international items** if cross-border suitability is in scope
7. **Clearly separate training from benchmark data** in the directory structure

---

*Report generated by automated analysis of all files in ~/Desktop/FinAdvisor_Training/AI training materials/*
