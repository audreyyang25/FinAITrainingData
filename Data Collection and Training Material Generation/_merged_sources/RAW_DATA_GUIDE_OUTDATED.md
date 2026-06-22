# FinAdvisor Training Dataset

**Topic:** Financial advisor conduct — suitability, best interest, and fiduciary duty  
**Coverage:** 2015–2025  
**Jurisdiction:** Primarily United States, with international comparative materials  
**Purpose:** Training data for a financial advisor compliance AI

---

## Dataset Structure

| # | Folder | Contents |
|---|--------|----------|
| 01 | `01_SEC_Guidance_NoAction_Interpretive` | SEC staff no-action letters, interpretive releases, IM Guidance series on investment adviser conduct and Reg BI |
| 02 | `02_SEC_Enforcement_Actions` | SEC administrative proceedings against investment advisers and broker-dealers (IA-numbered releases, 2015–2025) |
| 03 | `03_FINRA_AWC` | FINRA Acceptance, Waiver & Consent disciplinary settlements — suitability violations, churning, supervision failures, 2015–2025 |
| 04 | `04_RegBI_Compliance_Guidance` | Reg BI adopting release, Form CRS rule, fiduciary interpretation, FINRA Reg BI AWC cases, compliance FAQs |
| 05 | `05_SEC_Exam_Findings` | OCIE/EXAMS annual priorities and risk alerts (share class, wrap fee, best execution, senior investors, Reg BI) |
| 06 | `06_DOL_Fiduciary` | DOL fiduciary rule, PTE 2020-02, Retirement Security Rule 2024, FABs 2006–2025, 5th Circuit vacature materials |
| 07 | `07_Practitioner_Commentary` | SIFMA, IAA, CFP Board, FINRA regulatory notices, industry association Reg BI reports and compliance guides |
| 08 | `08_State_Fiduciary_Rules` | NY Reg 187, MA, NJ, NV, CT, CO, NAIC Model 275, NASAA model rule — state-level best interest standards |
| 09 | `09_Law_Firm_Memos` | Client alerts from Dechert, Sidley, K&L Gates, Willkie, Groom Law on Reg BI and fiduciary developments |
| 10 | `10_Academic_Articles` | Law review articles, economics papers, GAO reports, NBER working papers on fiduciary duty, suitability, Reg BI |
| 11 | `11_Case_Law` | Federal circuit and SCOTUS decisions: XY Planning v. SEC, Robare Group, Tibble v. Edison, Hughes v. Northwestern, 5th Circuit DOL cases |
| 12 | `12_FINRA_Arbitration_Awards` | FINRA customer arbitration award PDFs, 2015–2025 — suitability, churning, variable annuities, concentration, senior investors |
| 13 | `13_FINRA_Rulebook` | FINRA Rule 2111 (suitability), Rule 2330 (variable annuities), Rule 4512 (KYC), Rule 3110 (supervision), related guides and FAQs |
| 14 | `14_NASAA_State_Enforcement` | State securities regulator enforcement actions and NASAA annual enforcement reports |
| 15 | `15_FINRA_Regulatory_Notices` | FINRA regulatory notices and comment letters, 2015–2025 — suitability, complex products, senior investors, Reg BI |
| 16 | `16_Congressional_Materials` | Dodd-Frank §913 study, GAO reports on Reg BI, CRS reports, congressional hearing transcripts |
| 17 | `17_SEC_Staff_Bulletins` | SEC staff bulletins on Reg BI obligations (care, conflict, disclosure), 2020–2023 |
| 18 | `18_OCC_Bank_Trust_Guidance` | OCC Comptroller's Handbook on fiduciary activities, investment management, conflicts of interest for bank trust departments |
| 19 | `19_CFTC_Guidance` | CFTC no-action letters and guidance on commodity trading advisers (CTAs) and suitability-adjacent obligations |
| 20 | `20_SEC_ALJ_OALJ_Decisions` | SEC Administrative Law Judge initial decisions and Commission opinions on investment adviser and broker-dealer misconduct |
| 21 | `21_FINRA_OHO_Decisions` | FINRA Office of Hearing Officers formal hearing panel decisions and NAC appeal decisions (contested cases, not AWC settlements) |
| 22 | `22_PIABA_Materials` | PIABA amicus briefs, annual arbitration studies, comment letters, and research reports on investor protection |
| 23 | `23_International_Comparative` | FCA Consumer Duty (UK), IOSCO suitability standards, MiFID II/ESMA guidelines, ASIC (Australia), New Zealand, Singapore |

---

## Key Files

- `index.json` — tracks all downloaded URLs; prevents duplicates across sessions
- `hypotheticals.json` — 485 synthetic Q&A pairs across 33 violation topic buckets
- `hypotheticals.txt` — plain-text version of hypotheticals
- `_merged_sources/DATASET_MANIFEST.md` — Phase 4 snapshot (Feb 2026); superseded by `RAW_MATERIALS.md`
- `CRITIQUE.txt` — dataset quality analysis

---

## Coverage Notes

- **Primary scope:** Suitability (FINRA Rule 2111), Regulation Best Interest (Reg BI), investment adviser fiduciary duty under the Advisers Act, DOL fiduciary standards for retirement accounts
- **Excluded:** Fund custody, AML/BSA, affiliated fund distribution, pure securities fraud (unrelated to investment advice)
- **Enforcement emphasis:** Real named cases across SEC, FINRA AWC, FINRA arbitration awards, state enforcement, and ALJ/OHO formal proceedings
- **Date filter:** Documents dated 2015–2025 (a small number of foundational pre-2015 documents are retained where no current equivalent exists)

---

## Build History

| Phase | Date | Result |
|-------|------|--------|
| Phase 1–5 | Feb 25–Mar 1, 2026 | Folders 01–11, 1,271 PDFs + 485 hypotheticals |
| Phase 6 | Mar 1, 2026 | Folders 12–19, +902 PDFs |
| Phase 7 | Mar 1, 2026 | Awards expanded, folders 20–22 added |
| Phase 8 | Mar 2, 2026 | Quality pass, folder 23 added, grand total 2,788 |
| Phase 9 | Mar 2, 2026 | FINRA AWC 2020–2025 sweep (+349 AWC), Awards 2016–2025 full probe (+2,022), grand total **5,102 PDFs** |
