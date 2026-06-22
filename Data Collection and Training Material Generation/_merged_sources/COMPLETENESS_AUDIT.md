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