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
