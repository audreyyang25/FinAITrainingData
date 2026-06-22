# Audits & Reviews (combined)

> Merged 2026-06-22 from 5 point-in-time reports. Originals preserved in `_merged_sources/`.
> Each section keeps its original title and date.

---



<!-- ===== source: PHASE11_REPORT_AND_RECOMMENDATIONS.md ===== -->

# FinAdvisor Training Dataset — Phase 11 Final Report & Recommendations

Date: 2026-03-05 13:15 HK

---

## Final Dataset State

**Total PDFs:** 5,399 across 23 folders
**Index entries:** 12,305

| Folder | PDFs |
|--------|------|
| 01_SEC_Guidance_NoAction_Interpretive | 87 |
| 02_SEC_Enforcement_Actions | 220 |
| 03_FINRA_AWC | 497 |
| 04_RegBI_Compliance_Guidance | 155 |
| 05_SEC_Exam_Findings | 54 |
| 06_DOL_Fiduciary | 85 |
| 07_Practitioner_Commentary | 124 |
| 08_State_Fiduciary_Rules | 104 |
| 09_Law_Firm_Memos | 143 |
| 10_Academic_Articles | 206 |
| 11_Case_Law | 156 |
| 12_FINRA_Arbitration_Awards | 2,668 |
| 13_FINRA_Rulebook | 55 |
| 14_NASAA_State_Enforcement | 85 |
| 15_FINRA_Regulatory_Notices | 288 |
| 16_Congressional_Materials | 80 |
| 17_SEC_Staff_Bulletins | 13 |
| 18_OCC_Bank_Trust_Guidance | 40 |
| 19_CFTC_Guidance | 37 |
| 20_SEC_ALJ_OALJ_Decisions | 24 |
| 21_FINRA_OHO_Decisions | 160 |
| 22_PIABA_Materials | 27 |
| 23_International_Comparative | 91 |
| **TOTAL** | **5,399** |

---

## Phase 11 Gap Fill Results

| Folder | Before | After | Added |
|--------|--------|-------|-------|
| 09 Law Firm Memos | 87 | 143 | +56 |
| 10 Academic Articles | 186 | 206 | +20 |
| 12 FINRA Awards | 2,631 | 2,668 | +37 (2025 stride=1 fill) |
| 13 FINRA Rulebook | 30 | 55 | +25 |
| 16 Congressional | 37 | 80 | +43 |
| 21 FINRA OHO Decisions | 45 | 160 | +115 |
| **Net new** | **5,105** | **5,399** | **+294** |

Note: Folder 20 (SEC ALJ) held at 24 — SEC rate-limited direct scraping; Brave search returned limited unique results. See recommendations below.

---

## Recommendations: Next Steps for a Fantastic AI Training Set

### 1. 📋 Complete the Synthetic Q&A Layer (Phase 10 — PAUSED)
**Status:** 655 hypotheticals done; ~330 more needed to cover folders 09–23
**Action:** Resume `phase10_hypotheticals_opus.py` — resume-capable, auto-skips completed folders
**Why it matters:** Raw PDFs teach the model facts. Grounded Q&A pairs teach it to *reason*. This is the highest-leverage thing left.
**Cost:** ~$3–4 more at Opus pricing

---

### 2. 🏷️ Add Metadata / Classification Labels
**Current state:** Raw PDFs, no labels
**Recommendation:** Create a `metadata.json` with per-document fields:
- `document_type`: enforcement / guidance / case_law / academic / rulebook
- `violation_type`: suitability / churning / disclosure / supervision / variable_annuity / concentration / etc.
- `outcome`: violation_found / no_violation / settled / guidance_only
- `year`: document year
- `regulator`: SEC / FINRA / DOL / State / Court
**Why:** Labels enable supervised fine-tuning and structured evaluation. Without them, the model can learn patterns but not categories.

---

### 3. 🔄 Add Negative Examples — Vindicated Advisors
**What's missing:** Cases where the advisor was *cleared*
**Sources:**
- FINRA arbitration awards where the customer **lost** (no liability found) — in folder 12 but unlabeled
- OHO decisions where respondent was found **not liable**
- SEC no-action letters where conduct was **approved**
**Why:** A model trained only on violations will over-flag compliant behavior. It needs to see the boundary from both sides.

---

### 4. 📊 Add Customer Complaint Narratives
**What's missing:** The investor's voice — how victims describe harm
**Sources:**
- FINRA BrokerCheck complaint disclosures (public per-broker records)
- CFPB consumer complaint database (financial advisor complaints subset)
- PIABA member case summaries (have some in folder 22, can expand)
**Why:** Regulatory documents describe violations in legal terms. Customer complaints describe them in plain English — exactly the language a retail investor would use when seeking help from the AI.

---

### 5. 🔍 Expand State-Level Enforcement Actions
**Current state:** Folder 14 has NASAA aggregate reports but limited individual state orders
**Sources:**
- CA DFPI, NY DFS, MA Securities Division, TX SSB, FL OFR — all publish enforcement orders online
- Individual consent orders, cease-and-desist orders, revocation orders
**Why:** State enforcement catches advisors who fall below the federal radar. Patterns differ significantly by state — important for geographic coverage.

---

### 6. 🤖 Synthetic Edge-Case Scenarios
**What's missing:** Complex multi-violation scenarios
**Recommendation:** ~200 additional hypotheticals involving compound violations (e.g., elderly client + variable annuity + concentration + inadequate supervision all at once), grounded in real arbitration awards from folder 12
**Why:** Real cases are rarely clean single-issue violations. The model needs to handle messy, layered situations and prioritize what matters most.

---

### 7. 🧪 Build an Evaluation / Test Set
**What's missing:** A held-out test set to measure model performance
**Recommendation:** Set aside ~100 cases (mix of real awards + hypotheticals) *not used in training*
**Split:** 80% train / 10% validation / 10% test
**Why:** Without a test set you can't measure whether the model is improving or just memorizing. Essential before any fine-tuning run.

---

### 8. 📝 Document Chunking Strategy
**Current state:** Full PDFs as training data
**Consideration:** Many PDFs are 20–100+ pages. For fine-tuning you want:
- Chunked excerpts (~1,000–2,000 tokens) with source metadata
- Key passage extraction (violation findings, regulatory analysis sections)
- Summary + full-text pairs for longer documents
**Why:** LLMs have context windows. A 50-page PDF isn't a training example — it's 20–30 training examples if chunked correctly. This decision affects everything downstream.

---

### 9. 🔁 Ongoing Maintenance Schedule
**Recommendation:** Quarterly refresh:
- Probe new FINRA awards (re-run stride=5 for current year)
- Scrape new FINRA AWC monthly disciplinary pages
- Check SEC enforcement actions (new IA-XXXX releases)
- Pull new OHO/ALJ decisions
- Add new academic papers from SSRN
**Why:** Reg BI enforcement is still evolving. A stale training set becomes a liability — especially for a model advising on current standards.

---

### 10. 🌐 Expand International Coverage
**Current state:** Folder 23 has 91 docs (UK, EU, AU, NZ, SG)
**Consider adding:**
- Canada (CIRO suitability rules, OSC enforcement)
- Hong Kong (SFC suitability requirements)
- India (SEBI investment adviser regulations)
**Why:** Useful if the model will be deployed internationally or used for comparative analysis.

---

## Priority Order

If bandwidth is limited, do these first:

1. **Resume Phase 10** — finish the Q&A layer (highest training signal, nearly done)
2. **Add metadata labels** — unlocks structured fine-tuning
3. **Add negative examples** — prevents over-flagging compliant behavior
4. **Define chunking strategy** — determines how PDFs become actual training examples
5. **Build evaluation set** — can't improve what you can't measure
6. Everything else improves breadth and diversity


---


<!-- ===== source: COMPLETENESS_AUDIT.md ===== -->

# FinAdvisor Completeness Audit
Date: 2026-03-05 10:16 HK

---


---

## Summary

**Total PDFs:** 5105 across 23 folders

### Overall Assessment
- **Strongest folders:** 03 (FINRA AWC), 04 (Reg BI), 12 (Arbitration Awards), 15 (Reg Notices) — substantial volume and breadth
- **Thinnest folders:** 17 (SEC Bulletins), 20 (ALJ Decisions), 22 (PIABA) — relatively small for their topic scope
- **Coverage period:** Solid 2015-2025 for enforcement-heavy folders; academic and practitioner commentary may lag 2024-2025

### Top Gaps to Fill
1. **Folder 09:** 2024-2025 law firm memos likely missing (years found: ['2015', '2016', '2018', '2019', '2020', '2021', '2022', '2023', '2024'])
2. **Folder 10:** Missing recent 2024-2025 academic work
3. **Folder 12:** ~20% gap due to stride=5 method; 2025 ongoing
4. **Folder 13:** Missing rules: ['4512', '4511']
5. **Folder 16:** Missing recent GAO reports and hearing materials 2024-2025
6. **Folder 20:** Only 24 ALJ decisions — significantly under-represented
7. **Folder 21:** Only 45 OHO decisions — could expand

---

## Folder-by-Folder Results


### 01_SEC_Guidance_NoAction_Interpretive (87 PDFs)
- **Coverage:** Years detected: ['2015', '2016', '2017', '2019', '2020', '2022', '2023', '2024', '2025']
- **Source check:** SEC IM Guidance page returned 0 IM-series references visible in HTML
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 02_SEC_Enforcement_Actions (220 PDFs)
- **Coverage:** IA range 4124–6941
- **Source check:** Our IA range: 4124–6941 (220 files). Recent page shows up to IA-unknown
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 03_FINRA_AWC (497 PDFs)
- **Coverage:** Years: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Source check:** FINRA site indicates ~unknown total AWC results. We have 497 PDFs, years: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 04_RegBI_Compliance_Guidance (155 PDFs)
- **Coverage:** 2019-2025 (Reg BI adoption + compliance period)
- **Source check:** SEC Reg BI page has ~0 PDF links visible. We have 155 docs.
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 05_SEC_Exam_Findings (54 PDFs)
- **Coverage:** Years: ['2015', '2016', '2020', '2021', '2022', '2023', '2024', '2025']
- **Source check:** SEC EXAMS page has 0 'risk alert' mentions, 0 'examination priorities' mentions
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 06_DOL_Fiduciary (85 PDFs)
- **Coverage:** 2006-2025 (FABs, PTEs, fiduciary rule, Retirement Security Rule)
- **Source check:** DOL EBSA fiduciary page loaded. 1 PTE/prohibited-transaction references found. We have 85 docs.
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 07_Practitioner_Commentary (124 PDFs)
- **Coverage:** Years: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Source check:** FINRA notices page: 0 notice references. We have 124 docs, years: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 08_State_Fiduciary_Rules (104 PDFs)
- **Coverage:** Key states + NASAA model rule
- **Source check:** NASAA page fetched. States we appear to cover: ['NY', 'MA', 'NJ', 'NV', 'CT', 'CO', 'NASAA']
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 09_Law_Firm_Memos (87 PDFs)
- **Coverage:** Years: ['2015', '2016', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
- **Source check:** Dechert 2024 publications: 0 Reg BI/fiduciary-related references. We have 87 docs.
- **Gaps identified:** 2024-2025 law firm memos likely missing (years found: ['2015', '2016', '2018', '2019', '2020', '2021', '2022', '2023', '2024'])
- **Verdict:** ⚠️ Partial

### 10_Academic_Articles (187 PDFs)
- **Coverage:** Years: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2025']
- **Source check:** SSRN search returned ~unknown papers on Reg BI/fiduciary. We have 187.
- **Gaps identified:** Missing recent 2024-2025 academic work
- **Verdict:** ⚠️ Partial

### 11_Case_Law (156 PDFs)
- **Coverage:** Years: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Source check:** CourtListener: ~unknown precedential opinions on 'Regulation Best Interest'. We have 156 docs.
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 12_FINRA_Arbitration_Awards (2631 PDFs)
- **Coverage:** Years: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Source check:** Probed 2025 awards: highest accessible ≈ 25-02100. We have 2631 total awards across ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025'].
- **Gaps identified:** 2025 awards: we likely have up to ~2100 but actual current max appears higher
- **Verdict:** ⚠️ Partial — stride=5 probe means ~80% coverage by design

### 13_FINRA_Rulebook (30 PDFs)
- **Coverage:** Core suitability and supervision rules
- **Source check:** FINRA rulebook page fetched. Key rules in filenames: ['2111', '2330', '3110', '2010']. Missing from filenames: ['4512', '4511']
- **Gaps identified:** Key rules possibly missing: ['4512', '4511']
- **Verdict:** ⚠️ Partial

### 14_NASAA_State_Enforcement (85 PDFs)
- **Coverage:** Years: ['2015', '2017', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Source check:** NASAA enforcement page: 0 report references. Years in our files: ['2015', '2017', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 15_FINRA_Regulatory_Notices (289 PDFs)
- **Coverage:** Years: ['2015', '2016', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Source check:** We have 289 notices, 256 with notice numbers in filename, years: ['2015', '2016', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 16_Congressional_Materials (37 PDFs)
- **Coverage:** Years: ['2017', '2019', '2020', '2021', '2023', '2024']
- **Source check:** GAO search returned 0 product links. We have 37 congressional docs.
- **Gaps identified:** Missing recent GAO reports and hearing materials 2024-2025
- **Verdict:** ⚠️ Partial

### 17_SEC_Staff_Bulletins (13 PDFs)
- **Coverage:** 2020-2023 Reg BI obligations bulletins
- **Source check:** SEC staff bulletins page: 0 bulletin references visible. We have 13.
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 18_OCC_Bank_Trust_Guidance (40 PDFs)
- **Coverage:** OCC Comptroller's Handbook fiduciary sections
- **Source check:** OCC Handbook index: 0 fiduciary/trust/investment-management references. We have 40 docs.
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 19_CFTC_Guidance (37 PDFs)
- **Coverage:** Years: ['2016', '2018', '2019', '2020', '2021', '2022', '2024', '2025']
- **Source check:** CFTC staff letters index: 0 letter references. We have 37 docs, years: ['2016', '2018', '2019', '2020', '2021', '2022', '2024', '2025']
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 20_SEC_ALJ_OALJ_Decisions (24 PDFs)
- **Coverage:** Years: []
- **Source check:** SEC ALJ decisions page: 0 IA/broker-dealer references. We have 24 decisions.
- **Gaps identified:** Only 24 ALJ decisions — SEC ALJ page has substantial IA/BD decisions 2015-2025
- **Verdict:** ⚠️ Partial

### 21_FINRA_OHO_Decisions (45 PDFs)
- **Coverage:** Years: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2024', '2025']
- **Source check:** FINRA OHO page: 0 OHO references. We have 45 decisions.
- **Gaps identified:** Only 45 OHO decisions — FINRA issues ~20-30 formal decisions/year
- **Verdict:** ⚠️ Partial

### 22_PIABA_Materials (27 PDFs)
- **Coverage:** Years: ['2019', '2020', '2021', '2022', '2024']
- **Source check:** PIABA resources: ~0 PDFs visible. We have 27.
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

### 23_International_Comparative (91 PDFs)
- **Coverage:** UK, EU, AU, NZ, SG. Years: ['2015', '2016', '2020', '2022', '2023', '2024', '2025']
- **Source check:** FCA Consumer Duty page: 0 PDF references. We have 91 international docs.
- **Gaps identified:** None identified
- **Verdict:** ✅ Solid

---


<!-- ===== source: FINAL_REVIEW_REPORT.md ===== -->

# COMPREHENSIVE REVIEW: AI Training Materials vs. Raw Legal Materials
# Focus: Suitability and Best Interest of the Customer
# Prepared by Captain Trips
# Date: April 5, 2026

================================================================================
## EXECUTIVE SUMMARY
================================================================================

I reviewed ~6,000+ raw legal source files across 29 categories and ~4,189 AI 
training items (984 hypotheticals + ~3,200 JSON-based scenarios) generated by 
openclaw. My focus was exclusively on materials where SUITABILITY or BEST INTEREST 
OF THE CUSTOMER is a question or focal point.

OVERALL ASSESSMENT: The training materials are STRONG (8/10) but have significant 
gaps that should be addressed before the AI model is deployed.

KEY FINDINGS:
1. GOOD: Comprehensive coverage of Reg BI care obligation and FINRA suitability
2. GOOD: Realistic, detailed fact patterns with specific dollar amounts and ages
3. GOOD: Both compliant and non-compliant examples included
4. GAP: Heavy non-compliant skew (3.4:1 ratio) — model may learn to see violations everywhere
5. GAP: Several raw material categories have suitability content NOT reflected in training
6. GAP: Character recycling ("Margaret the retired schoolteacher" appears repeatedly)
7. GAP: Underrepresentation of gray-area/borderline cases
8. GAP: Missing product-specific suitability scenarios for several categories

================================================================================
## SECTION 1: WHAT THE TRAINING MATERIALS COVER WELL
================================================================================

The training materials do an excellent job on these suitability/best-interest topics:

CORE STANDARDS (well-covered):
  - Reg BI Care Obligation (50+ scenarios) — §240.15l-1(a)(2)(i)
  - FINRA Rule 2111 Suitability (25+ scenarios) — all 3 components
  - IA Fiduciary Duty (25+ scenarios) — Advisers Act §206
  - DOL Rollover Analysis (25+ scenarios) — PTE 2020-02
  - FINRA Quantitative Suitability (15 scenarios) — turnover rates, cost-equity
  - Churning/Excessive Trading (18+ scenarios)

PRODUCT-SPECIFIC SUITABILITY (well-covered):
  - Variable Annuities (28+ scenarios) — FINRA Rule 2330, surrender charges, riders
  - Senior Investor scenarios (26+ across multiple topics)
  - Concentration Risk (10+ scenarios)
  - Leveraged/Inverse ETFs (4-12 scenarios including JSON)

JSON DATASETS - STRONG ON:
  - inappropriate_risk_profile (159 items)
  - senior_investor_exploitation (142 items)
  - failure_to_know_customer (133 items)
  - complex_products_unsophisticated_investor (121 items)
  - high_fees_unsuitable (108 items)
  - leveraged_etf_unsuitable (79 items)
  - illiquid_investments (73 items)
  - variable_annuity_unsuitable (62 items)
  - excessive_churning (59 items)

QUALITY STRENGTHS:
  - Fact patterns are realistic with named characters, specific financial details
  - Legal citations are accurate (correct rule numbers, case citations)
  - Answers explain reasoning, not just conclusions
  - Difficulty levels vary (easy/medium/hard)
  - Format is consistent and machine-parseable

================================================================================
## SECTION 2: IDENTIFIED GAPS — SUITABILITY/BEST INTEREST TOPICS
================================================================================

### GAP 1: COMPLIANT/NON-COMPLIANT IMBALANCE [HIGH PRIORITY]

Current ratio: ~3.4 non-compliant : 1 compliant
Problem: An AI trained on this data will be biased toward finding violations.
A financial advisor AI needs to recognize COMPLIANT conduct equally well.

RECOMMENDATION: Generate 500-800 additional COMPLIANT suitability scenarios
covering the same topics. Focus on:
  - Compliant variable annuity recommendations with proper documentation
  - Compliant high-fee product recommendations with adequate justification
  - Compliant concentration positions with documented rationale
  - Compliant recommendations to seniors with proper alternatives analysis
  - Compliant rollover recommendations with PTE 2020-02 compliance

### GAP 2: OPTIONS SUITABILITY [HIGH PRIORITY]

Raw materials: Folder 26_Options has 58 files including FINRA Rule 2360 (options),
OCC risk disclosures, CBOE suitability guidance, and options-specific enforcement.
Training materials: Only 5 "options_complex" hypotheticals + ~45 JSON items.

MISSING SCENARIOS:
  - Options account approval process and suitability determination
  - Covered call writing for income investors (borderline suitable)
  - Naked option writing suitability standards
  - Options spread strategies — when suitable vs. when too complex
  - FINRA Rule 2360 specific requirements (approval levels, margin)
  - Options in retirement accounts — enhanced suitability scrutiny

### GAP 3: STRUCTURED PRODUCTS SUITABILITY [HIGH PRIORITY]

Raw materials: Folder 28_Structured_Products has 83 files including SEC staff
guidance on structured notes, FINRA regulatory notices, and enforcement actions.
Training materials: ~42 JSON items on "structured_products_unsuitable" but mostly 
non-compliant. Very few compliant structured product scenarios.

MISSING SCENARIOS:
  - When structured products ARE suitable (accredited investors, specific hedging)
  - Principal-protected notes vs. non-protected — different suitability analysis
  - Autocallable notes — early redemption features and suitability implications
  - Structured CDs vs. structured notes — different risk profiles
  - Reverse convertibles — high coupon but downside equity risk
  - Issuer credit risk as a suitability factor

### GAP 4: PRIVATE PLACEMENTS / DPPs SUITABILITY [HIGH PRIORITY]

Raw materials: Folder 27_Private_Placements_DPPs has 95 files covering Reg D
offerings, non-traded REITs, oil & gas programs, equipment leasing DPPs.
Training materials: ~46 JSON items on "private_placements_unsuitable" + some 
in illiquid category, but limited scope.

MISSING SCENARIOS:
  - Non-traded REIT suitability (liquidity constraints, fee structures)
  - Accredited investor status verification as suitability component
  - DPP tax benefits vs. investment merit — suitability analysis
  - Oil & gas DPPs for retirement accounts
  - Concentration limits for illiquid alternatives
  - Due diligence obligations on private offerings

### GAP 5: CRYPTO/DIGITAL ASSETS SUITABILITY [MEDIUM-HIGH PRIORITY]

Raw materials: Folder 29_Crypto_Digital_Assets has 236 files — largest specialty 
folder. Covers SEC guidance, FINRA notices, state enforcement, academic research.
Training materials: Only 10 hypotheticals + 37 JSON items.

MISSING SCENARIOS:
  - Bitcoin/Ethereum suitability for different investor profiles
  - Crypto exchange-traded products vs. direct holdings
  - Staking and DeFi yield products — suitability for retail
  - Crypto in retirement accounts (Bitcoin IRAs)
  - NFT-related investment suitability
  - Regulatory uncertainty as a suitability disclosure factor
  - Spot Bitcoin ETF suitability analysis (post-2024 approval)

### GAP 6: STATE-LEVEL FIDUCIARY STANDARDS [MEDIUM PRIORITY]

Raw materials: Folder 08_State_Fiduciary_Rules has 145 files covering state-level
best interest standards (MA fiduciary rule, NJ best interest, NV fiduciary duty).
Training materials: 15 "state_fiduciary_rules" + 9 each for state_fiduciary_bd 
and state_fiduciary_annuity = 33 total.

MISSING SCENARIOS:
  - Massachusetts fiduciary standard specific requirements (stricter than Reg BI)
  - New Jersey best interest regulation for annuity sales
  - Nevada fiduciary duty for broker-dealers
  - State-vs-federal conflicts — when state standard is higher
  - Insurance product suitability under state-specific annuity models
  - NAIC Suitability in Annuity Transactions Model Regulation

### GAP 7: ERISA/RETIREMENT PLAN SUITABILITY [MEDIUM PRIORITY]

Raw materials: Folders 06_DOL_Fiduciary (112 files) + DOL-related enforcement.
Training materials: 15 "dol_erisa" + 10 "dol_plan_adviser" + 11 "erisa_case_law" 
+ 9 "pte_2020_02" + 20 "dol_rollover_analysis" = 65 total. Decent but could expand.

MISSING SCENARIOS:
  - 401(k) plan investment menu selection — fiduciary analysis
  - Target date fund suitability in retirement plans
  - Revenue sharing arrangements and their impact on suitability
  - ERISA §404(a)(1) prudent expert standard applied to plan investments
  - DOL 2024 Retirement Security Rule implications (if not vacated)
  - Plan-to-plan rollovers vs. plan-to-IRA — comparative analysis

### GAP 8: BANK TRUST/FIDUCIARY PRODUCTS [MEDIUM PRIORITY]

Raw materials: Folder 18_OCC_Bank_Trust_Guidance has 41 files on bank fiduciary 
standards, OCC bulletins, and trust investment management.
Training materials: 14 "bank_fiduciary" + 8 "trust_investment_management" = 22.

MISSING SCENARIOS:
  - Prudent investor rule for trust portfolios
  - Bank-sold annuity suitability (Regulation R implications)
  - Trust beneficiary conflicts — income vs. remainder beneficiaries
  - Uniform Prudent Investor Act specific requirements
  - Bank sweep account suitability
  - FDIC-insured vs. non-insured product suitability disclosures

### GAP 9: ESG/SUSTAINABLE INVESTING SUITABILITY [MEDIUM PRIORITY]

Raw materials: Scattered across practitioner commentary and academic articles.
Training materials: Only 10 hypotheticals on "esg_sustainable_investing."

MISSING SCENARIOS:
  - ESG fund suitability when client has no ESG preference
  - Greenwashing and suitability — when ESG claims affect product selection
  - DOL ESG rule for retirement plan investments
  - Performance trade-offs in ESG investing — disclosure obligations
  - ESG scoring methodology differences and suitability implications

### GAP 10: ROBO-ADVISER SUITABILITY [LOW-MEDIUM PRIORITY]

Raw materials: Found in SEC exam findings, practitioner commentary.
Training materials: 8 "ai_robo_adviser" hypotheticals + 3 JSON items.

MISSING SCENARIOS:
  - Robo-adviser questionnaire adequacy for suitability determination
  - Algorithm-driven rebalancing — ongoing suitability obligations
  - Hybrid human-robo models — who bears suitability responsibility
  - Robo-adviser for complex client situations (trusts, estate planning)
  - SEC/FINRA guidance on digital investment advice suitability

================================================================================
## SECTION 3: QUALITY ISSUES IN EXISTING TRAINING MATERIALS
================================================================================

### ISSUE 1: Character Name Recycling
"Margaret" appears as a retired schoolteacher in multiple unrelated scenarios.
Similar character archetypes repeat. This could cause overfitting — the model
may associate "retired schoolteacher" with specific outcomes.
FIX: Diversify character names, professions, and demographics.

### ISSUE 2: Topic Label Fragmentation
~130+ unique topic labels, many overlapping:
  - "churning_excessive" vs "excessive_trading_churning" vs "finra_awc_churning"
  - "senior_investor" vs "senior_investor_expanded" vs "senior_investor_protection"
This makes analysis harder and could confuse training.
FIX: Consolidate to ~40-50 canonical topic labels with a mapping file.

### ISSUE 3: ~90 Duplicates Found
Between March_6_AI_Training.json and hypotheticals_v2.json.
FIX: Deduplicate before training.

### ISSUE 4: Missing "Source Folder" Attribution
Many hypotheticals have "Source Folder: unknown" — can't trace back to raw materials.
FIX: Add provenance tracking for all training items.

### ISSUE 5: Limited Conversation/Dialogue Format
All items are structured as FACTS/QUESTION/ANSWER. No client-advisor dialogue 
format, no multi-turn conversations, no "red flag" recognition scenarios.
FIX: Create conversation-format training items where the AI must identify 
suitability issues in natural dialogue.

### ISSUE 6: Insufficient "Gray Area" Scenarios  
Most scenarios are clearly compliant or clearly non-compliant. Real-world 
suitability analysis often involves genuine judgment calls.
FIX: Create 100+ "borderline" scenarios where reasonable professionals could 
disagree, with analysis of factors supporting each position.

================================================================================
## SECTION 4: RAW MATERIAL CATEGORIES — SUITABILITY RELEVANCE RANKING
================================================================================

HIGH RELEVANCE (>70% of files touch suitability/best interest):
  04  RegBI Compliance Guidance (311 files) — ~95% relevant
  17  SEC Staff Bulletins (14 files) — ~100% relevant
  13  FINRA Rulebook (56 files) — ~90% relevant
  03  FINRA AWC (645 files) — ~70% relevant (many are suitability violations)
  24  Variable Annuities (118 files) — ~85% relevant
  06  DOL Fiduciary (112 files) — ~80% relevant
  21  FINRA OHO Decisions (161 files) — ~75% relevant

MEDIUM RELEVANCE (30-70%):
  01  SEC Guidance/No-Action (135 files) — ~50% relevant
  02  SEC Enforcement Actions (378 files) — ~60% relevant
  05  SEC Exam Findings (90 files) — ~65% relevant
  08  State Fiduciary Rules (145 files) — ~55% relevant
  11  Case Law (179 files) — ~50% relevant
  12  FINRA Arbitration Awards (2670 files) — ~45% relevant
  14  NASAA State Enforcement (88 files) — ~50% relevant
  15  FINRA Regulatory Notices (305 files) — ~40% relevant
  20  SEC ALJ/OALJ Decisions (25 files) — ~60% relevant
  25  Leveraged/Inverse ETFs (67 files) — ~65% relevant
  26  Options (58 files) — ~60% relevant
  27  Private Placements/DPPs (95 files) — ~55% relevant
  28  Structured Products (83 files) — ~50% relevant

LOWER RELEVANCE (but still contains suitability content):
  07  Practitioner Commentary (207 files) — ~35% relevant
  09  Law Firm Memos (198 files) — ~30% relevant
  10  Academic Articles (245 files) — ~25% relevant
  16  Congressional Materials (82 files) — ~25% relevant
  18  OCC Bank Trust Guidance (41 files) — ~40% relevant
  19  CFTC Guidance (39 files) — ~30% relevant
  22  PIABA Materials (28 files) — ~40% relevant
  23  International Comparative (92 files) — ~20% relevant
  29  Crypto/Digital Assets (236 files) — ~30% relevant

================================================================================
## SECTION 5: RECOMMENDATIONS — PRIORITY ORDER
================================================================================

IMMEDIATE (before any training run):
  1. Deduplicate the 90 known duplicates
  2. Fix the compliant/non-compliant imbalance — add 500+ compliant scenarios
  3. Consolidate topic labels to canonical set

HIGH PRIORITY (significant gaps in suitability coverage):
  4. Add 30-50 options suitability scenarios (from folder 26 raw materials)
  5. Add 30-50 structured products suitability scenarios (from folder 28)
  6. Add 30-50 private placement/DPP suitability scenarios (from folder 27)
  7. Add 30-50 crypto/digital asset suitability scenarios (from folder 29)
  8. Create 100+ gray-area/borderline scenarios

MEDIUM PRIORITY (enhance coverage):
  9. Add 20-30 state-specific fiduciary standard scenarios
  10. Expand ERISA/retirement plan suitability by 20-30 scenarios
  11. Add 15-20 bank trust fiduciary scenarios
  12. Add 15-20 ESG suitability scenarios
  13. Create conversation/dialogue format training items (50-100)

QUALITY IMPROVEMENTS:
  14. Diversify character names and demographics
  15. Add source provenance tracking to all items
  16. Create a topic taxonomy/mapping document

================================================================================
## SECTION 6: SUMMARY STATISTICS
================================================================================

Raw Materials:
  Total files:           ~6,900
  Estimated suitability-relevant: ~3,000 (43%)
  Categories:            29

AI Training Materials:
  Total items:           ~4,189 (excl. benchmarks)
  Hypotheticals:         984
  JSON scenarios:        ~3,205
  Suitability-focused:   ~3,500 (84%)
  Compliant:             ~1,030 (25%)
  Non-compliant:         ~3,159 (75%)
  Unique topics:         130+

Gaps Identified:         10 major gaps
Quality Issues:          6 issues
Priority Recommendations: 16

================================================================================
END OF REPORT
================================================================================


---


<!-- ===== source: CORPUS_COVERAGE_AUDIT_2026-06-19.md ===== -->

# Corpus Coverage Audit — Financial-Advisor Standard-of-Conduct Materials
**Date:** 2026-06-19 · **Auditor role:** coverage/gap, not legal substance · **Search engine:** Brave Search API
**Folder audited:** `~/Desktop/FinAdvisor_Training/` (29 subfolders, ~6,049 PDFs)
**Scope of this audit:** the four core regimes — RIA fiduciary duty, Reg BI (broker-dealer best interest), FINRA suitability, best execution — plus the high-churn ERISA/DOL retirement overlay.

---

## 1. Coverage Summary

The folder is **strong on enforcement/applied material and broad secondary coverage, but has a recurring hole in the foundational *rule-text and statutory* layer** of every regime. It holds the two anchor SEC adopting releases (Reg BI 34-86031, Form CRS 34-86032) and the SEC IA fiduciary interpretation (IA-5248, present in 23 copies), plus deep AWC/enforcement, arbitration, and commentary collections. What's missing is the **primary black-letter text**: the FINRA rules themselves (2111 suitability, 2010 commercial honor, **5310 best execution**, 2330 variable annuities) are not present as standalone rule documents; the foundational suitability interpretive notices (**11-25, 12-25, 12-55**) and the **best-execution guidance Notice 15-46** are absent; the Advisers Act **§206** statutory text and the cornerstone case **SEC v. Capital Gains Research Bureau (375 U.S. 180 (1963))** are not in the corpus. The single most urgent **currency** problem is ERISA/DOL: the folder's DOL holdings are built around the 2016 rule, PTE 2020-02, and the 2024 Retirement Security Rule as if live — but as of **March 2026 the 2024 Rule was vacated and DOL formally restored the prior rule** (Federal Register notice of court vacatur, 2026-03-20). Any DOL material in the folder must now be read as historical, not operative.

**Limitation (stated per instructions):** This audit reduces gap risk; it does not guarantee completeness. Keyword search misses items, and enforcement/settled orders are effectively endless — discovery here was scoped to foundational authorities plus recent significant developments, not every order. Items I could not confirm via Brave are flagged "could not confirm," not asserted.

---

## 2. Inventory Table (foundational-authority layer)

Primary-law and key-guidance documents actually present (enforcement/arbitration bulk folders summarized, not enumerated):

| # | Document (in folder) | Type | Issuing body | Citation | Date/Ver | Folder | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | Reg BI Final Rule | Rule/adopting release | SEC | Rel. 34-86031 | 2019 | 01, 04 | confirmed |
| 2 | Form CRS Final Rule | Rule/adopting release | SEC | Rel. 34-86032 | 2019 | 01, 04 | confirmed |
| 3 | IA Fiduciary Interpretation | Interpretive release | SEC | IA-5248 | 2019 | 01, 11, 17 (×23) | confirmed |
| 4 | Form CRS Interpretation | Interpretive release | SEC | IA-5249 | 2019 | 01 | confirmed |
| 5 | IA-4509 Form ADV amendments | Rule | SEC | IA-4509 | 2016 | 01 | confirmed |
| 6 | IA-5653 Marketing Rule | Rule | SEC | IA-5653 | 2020 | 01 | confirmed |
| 7 | IA-3043 Pay-to-Play | Rule | SEC | IA-3043 | 2010 | 01 | confirmed |
| 8 | Proposed Commission Interpretation (pre-5248) | Proposed release | SEC | IA-4889 | 2018 | 07, 20 | confirmed |
| 9 | Conflicts-of-Interest Staff Bulletin | Staff bulletin | SEC | Aug 2022 | 2022-08-03 | 17 | confirmed |
| 10 | 2019 Reg BI obligations bulletin | Staff guidance | SEC | — | 2019 | 17 | needs confirmation (which release) |
| 11 | DOL 2016 Fiduciary/Conflict-of-Interest Rule | Rule (Fed Reg) | DOL | 2016 | 2016 | 06 | confirmed |
| 12 | DOL 2016 Best Interest Contract Exemption | PTE | DOL | BIC | 2016 | 06 | confirmed |
| 13 | PTE 2020-02 | PTE | DOL | 2020-02 | 2020 | 06 | confirmed |
| 14 | PTE 84-24 | PTE | DOL | 84-24 | (amended) | 06 | confirmed |
| 15 | PTEs 75-1/77-4/80-83/83-1 | PTE bundle | DOL | various | various | 06 | confirmed |
| 16 | Retirement Security Rule + class-PTE amendments | Rule (final + proposed) | DOL | 2024 | 2024 | 06 | **confirmed but STALE — see §4** |
| 17 | DOL Field Assistance Bulletins | Guidance | DOL | FAB 2017-01 → 2025-01 | 2017–2025 | 06 | confirmed |
| 18 | *Chamber of Commerce v. DOL* (vacatur of 2016 rule) | Case | 5th Cir. | 885 F.3d 360 | 2018 | 06, 11 | confirmed |
| 19 | *XY Planning Network v. SEC* | Case | 2d Cir. | — | 2020 | 11 | confirmed |
| 20 | *Robare Group v. SEC* | Case | D.C. Cir. | — | 2019 | 11 | confirmed |
| 21 | FINRA Rule 2111 Suitability **FAQ** | Guidance (FAQ only) | FINRA | — | — | 13 | confirmed (FAQ, **not rule text**) |
| 22 | Combined Suitability FAQs 12-10-12 | Guidance | FINRA | — | 2012 | 13 | confirmed |
| 23 | FINRA Rule 2330 industry material (IRI) | Secondary on rule | IRI/FINRA | re: Rule 2330 | 2020 | 13 | confirmed (secondary, not rule text) |
| 24 | FINRA Rule 3110 supervision | Guidance page | FINRA | 3110 | — | 13 | confirmed |
| 25 | FINRA Rule 4511/4512 books & records | Rule filings | FINRA | 4511, 4512 | — | 13 | confirmed |
| 26 | FINRA Reg Notices (large set) | Reg notices | FINRA | 99-24 … 25-19 | 1999–2025 | 13, 15 | confirmed (set) |
| 27 | SEC Care Obligations Staff Bulletin | Staff bulletin | SEC | 2023 | 2023 | — | **MISSING — see §3** |
| 28 | FINRA Rule 5310 best execution | Rule | FINRA | 5310 | — | — | **MISSING — see §3** |
| 29 | Reg BI FINRA AWC enforcement set | Enforcement | FINRA | many | 2021–2024 | 04 | confirmed (bulk) |
| 30 | SEC Enforcement Actions (IA-series) | Enforcement | SEC | IA-4135→7100+ | 2015–2025 | 02 | confirmed (bulk, 220) |
| 31 | FINRA Arbitration Awards | Awards | FINRA | case-numbered | 2016–2025 | 12 | confirmed (bulk, ~2,630) |

**Normalization note:** IA-5248 appears under many filenames (`ia5248_*`, `IA_FiduciaryInterpretation_IA-5248_2019`, `interp_ia-5248`, `noaction_ia-5248`, `case_ia-5248`) across folders 01/04/11/17 — counted once. Reg BI 34-86031 and Form CRS 34-86032 likewise appear in both 01 and 04 — counted once each.

---

## 3. Gap Report (Bucket C — missing from folder, present online)

Sorted by regime, then binding law first. All URLs verified live via Brave on 2026-06-19.

### Fiduciary duty (RIA)
| Priority | Authority | Type | Citation | Date | Source URL | Tier | Why it belongs |
|---|---|---|---|---|---|---|---|
| 1 | **SEC v. Capital Gains Research Bureau** | Case (binding) | 375 U.S. 180 | 1963 | sec.gov/divisions/investment/capitalgains1963.pdf | 1 | The Supreme Court foundation of the IA fiduciary standard; corpus has the modern interpretation (IA-5248) but not its root. |
| 2 | **Investment Advisers Act §206** (statutory text) | Statute | 15 U.S.C. §80b-6 | 1940 (current) | uscode.house.gov (subchapter II); govinfo COMPS-1878 | 1 | The anti-fraud/fiduciary statutory basis itself. Folder has releases interpreting it but not the section. |

### Best interest — broker-dealers (Reg BI)
| Priority | Authority | Type | Citation | Date | Source URL | Tier | Why it belongs |
|---|---|---|---|---|---|---|---|
| 1 | **Exchange Act Rule 15l-1** (Reg BI rule text) | Rule text | 17 CFR 240.15l-1 | 2019 | sec.gov/regulation-best-interest | 1 | Folder has the *adopting release* (34-86031) but not the codified rule text as a standalone. |
| 2 | **SEC Staff Bulletin: Standards of Conduct — Care Obligations** | Staff bulletin | 2023 | 2023 | sec.gov/.../staff-bulletin-standards-conduct-broker-dealers-investment-advisers-care-obligations | 1 | Folder has the *Conflicts* (2022) and *Account Recommendations* topics but is missing the **Care Obligations** bulletin — a core part of the 3-bulletin staff series. |

### Suitability (FINRA)
| Priority | Authority | Type | Citation | Date | Source URL | Tier | Why it belongs |
|---|---|---|---|---|---|---|---|
| 1 | **FINRA Rule 2111** (rule text) | Rule text | FINRA 2111 | current | finra.org/rules-guidance/rulebooks/finra-rules/2111 | 1 | Folder has the FAQ but not the black-letter rule. |
| 2 | **FINRA Rule 2010** (rule text) | Rule text | FINRA 2010 | current | finra.org/rules-guidance/rulebooks/finra-rules/2010 | 1 | Commercial-honor catch-all; cited throughout enforcement but rule text absent. |
| 3 | **FINRA Rule 2330** (rule text) | Rule text | FINRA 2330 | current | finra.org/rules-guidance/rulebooks/finra-rules/2330 | 1 | Deferred variable annuity suitability; folder has only IRI secondary material. |
| 4 | **Regulatory Notice 11-25** | Reg notice | 11-25 | 2011 | finra.org/rules-guidance/notices/11-25 | 1 | Original 2111 implementation guidance. |
| 5 | **Regulatory Notice 12-25** | Reg notice | 12-25 | 2012 | finra.org/rules-guidance/notices/12-25 | 1 | Additional 2111 interpretive guidance (the "know-your-customer"/reasonable-basis Q&A). |
| 6 | **Regulatory Notice 12-55** | Reg notice | 12-55 | 2012 | finra.org/rules-guidance/notices/12-55 | 1 | Clarifies "customer" and "investment strategy" under 2111. Surfaced during discovery. |

### Best execution
| Priority | Authority | Type | Citation | Date | Source URL | Tier | Why it belongs |
|---|---|---|---|---|---|---|---|
| 1 | **FINRA Rule 5310** (rule text) | Rule text | FINRA 5310 | current | finra.org/rules-guidance/rulebooks/finra-rules/5310 | 1 | The best-execution & interpositioning rule. **Entirely absent** — only incidental AWC mentions exist. |
| 2 | **Regulatory Notice 15-46** | Reg notice | 15-46 | Nov 2015 | finra.org/sites/default/files/notice_doc_file_ref/Notice_Regulatory_15-46.pdf | 1 | The headline best-execution guidance (incl. PFOF). Absent. |
| 3 | **Regulatory Notice 12-13** | Reg notice | 12-13 | 2012 | finra.org/rules-guidance/notices/12-13 | 1 | Earlier best-execution guidance; surfaced during discovery alongside 5310. |
| 4 | **SEC OCIE Risk Alert — IA Best Execution** | Exam risk alert | — | (OCIE) | sec.gov/files/OCIE%20Risk%20Alert%20-%20IA%20Best%20Execution.pdf | 1 | Adviser-side best-execution expectations; complements the BD/FINRA side. Not in folder 05. |

### Retirement / ERISA (highest churn)
| Priority | Authority | Type | Citation | Date | Source URL | Tier | Why it belongs |
|---|---|---|---|---|---|---|---|
| 1 | **Federal Register — Notice of Court Vacatur, Retirement Security Rule** | Rule status notice | 2026-05492 | 2026-03-20 | federalregister.gov/documents/2026/03/20/2026-05492/ | 1 | The dispositive currency document: the 2024 Rule was vacated; folder lacks it. **Add immediately.** |
| 2 | **DOL/EBSA news release restoring prior rule** | Agency release | EBSA 2026-03-18 | 2026-03-18 | dol.gov/newsroom/releases/ebsa/ebsa20260318 | 1 | Confirms DOL restored the long-standing prior investment-advice rule. |
| 3 | **ERISA §§3(21), 404, 406** (statutory text) | Statute | 29 U.S.C. §1002(21), §1104, §1106 | current | govinfo.gov / uscode.house.gov | 1 | Statutory fiduciary definitions/duties/prohibited transactions. Folder has FABs and PTEs but not the sections. |

---

## 4. Stale / Superseded Report (Bucket B)

| Folder item | Status now | Superseded/changed by | Date of change | Source |
|---|---|---|---|---|
| `6_D_retirement-security-rule-and-amendments-to-class-pte-fo` and all `*retirement-security-rule*` files (folder 06) | **VACATED / not operative** | 2024 Rule vacated by district court; DOL did not defend; prior (pre-2024) rule restored | Vacatur Fed Reg notice **2026-03-20**; DOL restoration **2026-03-18** | federalregister.gov/.../2026-05492; dol.gov/.../ebsa20260318 |
| DOL 2016 Fiduciary Rule + 2016 BIC Exemption (folder 06) | **Historical** (already vacated in 2018) | *Chamber of Commerce v. DOL*, 5th Cir. — folder HAS this case (good); just confirm the 2016 rule files are labeled historical | 2018 | folder 06/11 |
| PTE 2020-02 (folder 06) | **Operative but context-shifted** | With 2024 Rule vacated, the regulatory baseline reverts to the pre-2024 framework under which 2020-02 operates; verify current amended text | post-2026-03 | dol.gov (confirm current PTE 2020-02 text) |
| `2019_RBI_Reg_BI_Obligations` / 2019-vintage Reg BI staff guidance (folder 17) | **Supplemented** | SEC Staff Bulletins series (Account Recommendations 2022, Conflicts 2022, **Care Obligations 2023**) refine the 2019 guidance | 2022–2023 | sec.gov staff bulletins (Care Obligations missing — see §3) |
| FINRA Reg Notice set ends ~25-19 (folder 15) | **Possibly trailing** | Confirm no 2025–2026 suitability/Reg BI/best-ex notices issued after the last captured notice | ongoing | finra.org/rules-guidance/notices |

---

## 5. Currency Flags (Step 5)

1. **ERISA/DOL — CRITICAL, time-pinned.** The entire 2024 Retirement Security Rule line in folder 06 is **vacated as of 2026-03-20** (Fed Reg notice of court vacatur, doc 2026-05492) and DOL restored the prior rule on **2026-03-18** (EBSA release). Treat every `*retirement-security-rule*` and 2024-amendment file as historical. This is the highest-churn bucket and it has now turned over since the folder was last built (folder 06 last modified Mar 2026, but the build predates the vacatur paperwork).
2. **PTE 2020-02 operative status** depends on the post-vacatur baseline — flag for re-pull of the current amended text. (Source: dol.gov; confirm release date on re-pull.)
3. **Reg BI staff-guidance vintage.** Folder leans on 2019 Reg BI obligations material; the 2022–2023 SEC Staff Bulletins (esp. **Care Obligations 2023**) are the current operative staff interpretation and one is missing.
4. **FINRA rule text predates nothing but is simply absent** — not a staleness issue but a coverage hole (rules 2010/2111/2330/5310). Pull current rulebook versions directly.
5. **Litigation/pending status to monitor:** the Fifth Circuit dismissed the fiduciary-rule appeals (Dec 2025) and DOL withdrew its appeal of the stay (Nov 2025) — the vacatur is now effectively final, not merely stayed. Pin to 401kspecialist/PLANADVISER (Tier 2 leads) → confirm via Fifth Circuit docket (Tier 1) on re-pull.

---

## 6. Unmatched Folder Items (Bucket D — for human review)

These are in the folder but did **not** surface as foundational checklist authorities; not invalid — flagged for human judgment:

- `01/guidance2_*` cybersecurity, business-continuity, AML, and "Industry Snapshot"/"Annual Regulatory Oversight Report" files — **off-core-topic** for the four standard-of-conduct regimes; legitimately useful context but not standard-of-conduct primary law. (Consistent with the project's own note that folders accumulated adjacent-topic material.)
- `13_FINRA_Rulebook/4511`, `4512` books-&-records rule filings — valid FINRA rules but **recordkeeping**, not standard-of-conduct.
- `11_Case_Law/cl_2014…cl_2026…` — a large general securities-litigation set (ERISA fee cases, insider-trading, fraudulent-conveyance). Many are tangential to advisor standard-of-conduct; *Tibble*, *Hughes*, *XY Planning*, *Robare* are the on-point ones — the rest are context. Human should confirm which to keep in a standard-of-conduct training set.
- `gap1_*`, `gap4_*`, `gap6_*`, `brave_*` prefixed files — provenance-tagged duplicates/leads from earlier gap-fill passes; check for de-duplication against canonical copies.
- `IAA-Conflicts-CLE-Outline`, `CFP-Board-Code…`, EY/Chapman secondary memos (folder 17) — **Tier 2 secondary**; fine as commentary, not citations of record.

---

## 7. Search Log (Brave queries run 2026-06-19)

| # | Query | Purpose |
|---|---|---|
| Q1 | `FINRA Rule 5310 best execution current site:finra.org` | Confirm 5310 exists/current |
| Q2 | `FINRA Regulatory Notice 15-46 best execution guidance` | Confirm 15-46 |
| Q3 | `FINRA Rule 2111 suitability rule text site:finra.org` | Confirm 2111 rule text |
| Q4 | `FINRA Regulatory Notice 11-25 12-25 suitability rule 2111` | Confirm 11-25/12-25 |
| Q5 | `SEC v. Capital Gains Research Bureau 375 U.S. 180 1963 supreme court` | Confirm anchor case |
| Q6 | `Investment Advisers Act 1940 section 206 anti-fraud full text govinfo` | Confirm §206 statute |
| Q7 | `Regulation Best Interest adopting release 34-86031 SEC site:sec.gov` | Confirm Reg BI release (folder match) |
| Q8 | `DOL Retirement Security Rule 2024 status litigation stay vacated` | Currency — DOL |
| Q9 | `2024 Retirement Security Rule vacated fifth circuit Texas 2025 2026 fiduciary` | Pin vacatur date/source |
| Q10 | `SEC Regulation Best Interest enforcement action 2024 2025` | Discovery — recent Reg BI enforcement |
| Q11 | `FINRA suitability regulatory notice 2024 2025 rule 2111` | Discovery — recent suitability |
| Q12 | `SEC investment adviser fiduciary duty guidance bulletin 2024 2025` | Discovery — IA guidance updates |
| Q13 | `SEC Staff Bulletin standards of conduct care obligations 2023 site:sec.gov` | Confirm Care Obligations bulletin |
| Q14 | `SEC investment adviser best execution obligations guidance` | Discovery — adviser best-ex |
| Q15 | `FINRA Rule 2330 variable annuities suitability site:finra.org` | Confirm 2330 |
| Q16 | `FINRA Rule 2010 standards of commercial honor site:finra.org` | Confirm 2010 |

**Re-run note:** queries are reproducible via the Brave Web Search API with the project key. To extend coverage, iterate Q10–Q12 with year filters and add docket-level Tier-1 confirmation for the Fifth Circuit fiduciary-rule dismissal.

---

### Discipline footer
Primary law is distinguished from secondary commentary in every line above. No citation, release number, case name, or rule number was asserted without a Brave-verified source URL and tier. Forum is flagged where it matters (SEC vs. FINRA vs. 5th Cir. vs. district court). This audits coverage only — it does not opine on whether any authority is good law.


---


<!-- ===== source: GAPFILL_ADDED_2026-06-19.md ===== -->

# Gap-Fill — Documents Added 2026-06-19

Added the 15 missing Tier-1 authorities identified in `CORPUS_COVERAGE_AUDIT_2026-06-19.md`.
All files verified to contain the correct primary-source text. Items pulled via the Internet
Archive (Wayback) carry a one-line archive banner at the top of page 1, followed by the full
original document — this was necessary because FINRA (Cloudflare), SEC.gov, and
FederalRegister.gov block automated direct download.

| # | File | Folder | Authority | Source route |
|---|------|--------|-----------|--------------|
| 1 | SCOTUS_SEC_v_Capital_Gains_Research_Bureau_375US180_1963.pdf | 11_Case_Law | SEC v. Capital Gains Research Bureau, 375 U.S. 180 (1963) | sec.gov (direct, w/ referer) |
| 2 | Advisers_Act_Section_206_15USC80b-6.pdf | 01_SEC_Guidance… | Investment Advisers Act §206 / 15 U.S.C. §80b-6 | uscode.house.gov |
| 3 | SEC_Staff_Bulletin_Care_Obligations_2023.pdf | 17_SEC_Staff_Bulletins | SEC Staff Bulletin — Care Obligations (2023) | Wayback (sec.gov) |
| 4 | FINRA_Rule_2111_Suitability_ruletext.pdf | 13_FINRA_Rulebook | FINRA Rule 2111 (rule text) | Wayback (finra.org) |
| 5 | FINRA_Rule_2010_Commercial_Honor_ruletext.pdf | 13_FINRA_Rulebook | FINRA Rule 2010 (rule text) | Wayback (finra.org) |
| 6 | FINRA_Rule_2330_Variable_Annuities_ruletext.pdf | 13_FINRA_Rulebook | FINRA Rule 2330 (rule text) | Wayback (finra.org) |
| 7 | FINRA_Rule_5310_Best_Execution_ruletext.pdf | 13_FINRA_Rulebook | FINRA Rule 5310 (rule text) | Wayback (finra.org) |
| 8 | Regulatory_Notice_11-25_Suitability_KYC.pdf | 15_FINRA_Reg_Notices | FINRA Reg Notice 11-25 | renamed from existing p123701.pdf (already in folder 13) |
| 9 | Regulatory_Notice_12-25_Suitability.pdf | 15_FINRA_Reg_Notices | FINRA Reg Notice 12-25 | renamed from existing p126431.pdf (already in folder 13) |
| 10 | Regulatory_Notice_12-55_Suitability.pdf | 15_FINRA_Reg_Notices | FINRA Reg Notice 12-55 | Wayback (finra.org, p197435) |
| 11 | Regulatory_Notice_15-46_Best_Execution.pdf | 15_FINRA_Reg_Notices | FINRA Reg Notice 15-46 | Wayback (finra.org) |
| 12 | Regulatory_Notice_12-13_Best_Execution.pdf | 15_FINRA_Reg_Notices | FINRA Reg Notice 12-13 | Wayback (finra.org, p125747) |
| 13 | OCIE_Risk_Alert_IA_Best_Execution.pdf | 05_SEC_Exam_Findings | OCIE Risk Alert — IA Best Execution (Jul 11 2018) | sec.gov (direct, w/ referer) |
| 14 | FedReg_2026-05492_…Notice_of_Court_Vacatur_2026-03-20.pdf | 06_DOL_Fiduciary | Fed Reg notice of court vacatur, 2024 Retirement Security Rule | Wayback (federalregister.gov) |
| 15 | DOL_EBSA_Release_2026-03-18_Restores_Prior_Investment_Advice_Rule.pdf | 06_DOL_Fiduciary | DOL/EBSA release restoring prior rule | dol.gov (direct) |

## Update (same day) — FINRA rule texts re-pulled clean

The OpenClaw browser tool was enabled and started (profile `openclaw`, its own persistent
Chrome). It passes FINRA's Cloudflare challenge where headless `curl`/Chrome could not.
Items 4–7 (Rules 2010/2111/2330/5310) were **re-pulled directly from finra.org** and now
contain pristine rule text with **no Wayback banner** (verified `banner=0`, real rule text on
line 1). The FINRA notices (items 10–12) remain Wayback copies — content is complete and the
one-line banner is cosmetic, so they were left as-is.

## Notes
- **Items 8 & 9 were already in the corpus** under opaque FINRA document IDs (`p123701.pdf`,
  `p126431.pdf` in folder 13) — the original audit flagged them as missing because the
  filenames gave no hint of the notice number. They are now also present under clear names in
  folder 15; the originals in folder 13 were left untouched.
- The DOL vacatur pair (items 14–15) resolves the audit's top **currency** flag: the 2024
  Retirement Security Rule line in folder 06 should be treated as historical.
- §206 (item 2) is the US Code current-version HTML rendered to PDF; for a citation-of-record
  consider also the govinfo COMPS-1878 compiled PDF.


---
