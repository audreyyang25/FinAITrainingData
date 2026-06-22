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
