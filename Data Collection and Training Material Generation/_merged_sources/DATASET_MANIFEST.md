# FinAdvisor Training Dataset Manifest

> **Historical snapshot only (Phase 4 build).** Generated 2026-02-28. Superseded by
> [`../RAW_MATERIALS.md`](../RAW_MATERIALS.md) (corpus inventory) and
> [`../AUDITS_AND_REVIEWS.md`](../AUDITS_AND_REVIEWS.md) (audits and current counts).
> Kept here for provenance — do not use these numbers for the live dataset.

---

Generated: 2026-02-28 01:52 

## Summary
- **Total PDFs**: 1171
- **Total Hypotheticals**: 320
- **Folders**: 11

## Folder Breakdown

| Folder | PDFs | Notes |
|--------|------|-------|
| 10_Academic_Articles | 51 | Stanford L. Rev., law reviews, Sitkoff, Schanzenbach, academic analysis |
| 11_Case_Law | 155 | SCOTUS, 2nd Cir, 9th Cir, DC Cir, ERISA, IA enforcement; XY Planning v. SEC; Robare Group v. SEC |
| 1_SEC_Guidance_NoAction_Interpretive | 78 | IM Guidance 2013-2017, Reg BI rules, interpretive releases, no-action letters |
| 2_SEC_Enforcement_Actions | 158 | Admin proceedings IA-XXXX series 2015-2024 |
| 3_FINRA_AWC | 147 | Month-by-month AWC 2015-2025 |
| 4_RegBI_Compliance_Guidance | 154 | Reg BI final rule, FINRA AWC Reg BI cases, FAQs |
| 5_SEC_Exam_Findings | 47 | OCIE/EXAMS risk alerts, exam priorities 2015-2025 |
| 6_DOL_Fiduciary | 74 | DOL fiduciary rule, PTE 2020-02, Retirement Security Rule 2024, PTEs 75-1/77-4/80-83/84-24 |
| 7_Practitioner_Commentary | 120 | SIFMA, IAA, CFP Board, CFA Institute, FINRA guidance |
| 8_State_Fiduciary_Rules | 100 | NY Reg 187, MA, NJ, NV, NAIC Model 275, state adoptions |
| 9_Law_Firm_Memos | 87 | Dechert, Sidley, K&L Gates, Willkie, Groom Law memos |

## Hypotheticals

**Total**: 320 items

Compliant: 121 | Non-compliant: 199
Difficulty: easy=63 medium=144 hard=113

| Topic | Count |
|-------|-------|
| ia_fiduciary_duty | 25 |
| finra_suitability | 25 |
| reg_bi_care_expanded | 25 |
| reg_bi_care | 20 |
| variable_annuity | 20 |
| reg_bi_conflict_expanded | 20 |

| reg_bi_conflict | 15 |
| ia_duty_of_care | 15 |
| dol_erisa | 15 |
| churning_excessive | 15 |
| undisclosed_conflict | 15 |
| form_crs_disclosure | 15 |
| ia_cherry_picking | 15 |
| ia_best_execution | 15 |
| ia_fee_arrangements | 15 |
| reg_bi_disclosure | 10 |
| concentration | 10 |
| compliant_edge | 10 |
| ia_dual_registration | 10 |
| senior_investor | 5 |
| options_complex | 5 |

## Key Landmark Docs

### Case Law Highlights
- cl_2020_ca2_xy_planning_network_llc_v._sec — XY Planning Network v. SEC (2nd Cir 2020) — Reg BI legal challenge
- cl_2019_cadc_the_robare_group_ltd._v._sec — Robare Group v. SEC (DC Cir 2019) — IA fiduciary + conflicts
- cl_2021_ca2_sacerdote_v._new_york_university — Sacerdote v. NYU (2nd Cir 2021) — ERISA fees
- cl_2022_ca2_haley_v._tiaa — Haley v. TIAA (2nd Cir 2022) — ERISA fiduciary
- cl_2019_ca3_jennifer_sweda_v._university_of_pen — Sweda v. U. Penn (3rd Cir 2019) — ERISA monitoring
- scotus_2015_tibble_v_edison_erisa_fiduciary — Tibble v. Edison (SCOTUS 2015) — duty to monitor
- scotus_2022_northwestern_hughes_erisa_fees — Hughes v. Northwestern (SCOTUS 2022) — ERISA fees

### SEC Guidance Highlights
- ia5248_fiduciary_interpretation_2019 — 2019 IA Fiduciary Duty Interpretation Release
- reg_bi_34-86031_final_rule_2019 — Reg BI Final Rule
- reg_bi_34-86032_form_crs_2019 — Form CRS Final Rule
- im-guidance-2017-02 — Robo-Adviser Guidance
- im-guidance-2015-07 — Best Execution Guidance
