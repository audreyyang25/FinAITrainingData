# FinAdvisor Training Data

Training corpus and synthetic materials for a financial-adviser conduct AI focused on **suitability**, **Regulation Best Interest**, and **fiduciary duty** (primarily U.S., 2015–2026).

**Topic:** Investment adviser and broker-dealer standard-of-conduct — suitability (FINRA Rule 2111), Reg BI, IA fiduciary duty (Advisers Act §206), DOL/ERISA retirement advice.

**Current scale (approx.):** ~6,064 source PDFs across 29 folders · ~4,700+ suitability training items · ~2,000 benchmark evaluation rows.

---

## Top-level layout

```
FinAITrainingData/
├── README.md                          ← you are here
│
├── Data Collection and Training Material Generation/   ← canonical working tree
│   ├── RAW_MATERIALS.md               ← folder-by-folder corpus guide (29 folders)
│   ├── AUDITS_AND_REVIEWS.md          ← audits, gap reports, recommendations
│   │
│   ├── raw_data/                      ← source PDF corpus (folders 01–29)
│   │   ├── 01_SEC_Guidance_NoAction_Interpretive/
│   │   ├── …                          (29 numbered folders — see table below)
│   │   ├── 29_Crypto_Digital_Assets/
│   │   └── logs/
│   │       ├── index.json             ← machine-readable scrape index (per-PDF metadata)
│   │       ├── README.md
│   │       └── phase*_log.txt         ← corpus-build run logs
│   │
│   ├── training_material_generation/  ← synthetic training data + benchmarks
│   │   ├── README.md                  ← dataset catalog, phase map, caveats
│   │   ├── datasets/                  ← P12–P16 deliverable JSON + TXT
│   │   ├── hypotheticals/             ← 983 per-scenario .txt + hypotheticals.json
│   │   ├── Suitability Only/          ← suitability-filtered P12–P16 copies
│   │   ├── New_Folders_Suitability/   ← P12/P13 for product folders 24–29
│   │   ├── benchmarks/                ← model evaluation outputs (not training)
│   │   ├── logs/
│   │   │   ├── generation/            ← phase10, phase12–16, fill-up logs
│   │   │   ├── benchmark/
│   │   │   └── _archived/
│   │   ├── backups_mar21/             ← pre-filter snapshots (do not double-count)
│   │   ├── backups_mar21_v2/
│   │   └── backups_mar21_v3/
│   │
│   ├── training_material_FINAL/       ← human-readable PDF bundle (Apr 4, 2026)
│   │   ├── Phase_12_…Phase_16_*.pdf   ← from datasets/ (Mar 21 generation)
│   │   ├── SuitabilityOnly_P12_…P16_*.pdf  ← from Suitability Only/ (Mar 22)
│   │   ├── New_Folders_P12/P13.pdf
│   │   └── Benchmark_*.pdf
│   │
│   └── _merged_sources/               ← verbatim originals of merged docs
│       ├── README.md
│       ├── DATASET_MANIFEST.md        ← Phase 4 snapshot (Feb 2026) — historical only
│       └── …
│
└── AI Suitability Training Materials/  ← curated export / training deliverable subset
    ├── 23_Folders_Suitability/         ← suitability_only_P12–P16.json (folders 01–23)
    └── New_6_Folders_Suitability/      ← per-product Standard + Borderline JSON (folders 24–29)
        ├── Variable Annuities/
        ├── Leveraged Inverse ETFs/
        ├── Options/
        ├── Private Placements/
        ├── Structured Products/
        └── Crypto Digital Assets/
```
### Project Setup & Data
The source code is tracked in this repository, but the raw data is stored externally due to size constraints.

1. Clone this repository.
2. Download the raw materials from [this link](https://drive.google.com/drive/folders/1h8UWGut_EyAJ93dnVQQGSiyUiQPo5Usa?usp=drive_link).
3. Place the files inside a folder named `/raw_data/` inside the Data Collection and Training Material Generation folder.

### Which tree to use

| Need | Go to |
|------|--------|
| Full corpus + all generation artifacts + logs | `Data Collection and Training Material Generation/` |
| Training JSON ready for fine-tuning (suitability-only) | `AI Suitability Training Materials/` |
| Human-readable PDF review of all training phases | `training_material_FINAL/` |
| What each raw folder contains | [`RAW_MATERIALS.md`](Data%20Collection%20and%20Training%20Material%20Generation/RAW_MATERIALS.md) |
| Dataset counts, formats, known duplicates | [`training_material_generation/README.md`](Data%20Collection%20and%20Training%20Material%20Generation/training_material_generation/README.md) |
| Coverage gaps, audits, reviews | [`AUDITS_AND_REVIEWS.md`](Data%20Collection%20and%20Training%20Material%20Generation/AUDITS_AND_REVIEWS.md) |
| Per-PDF scrape provenance | `raw_data/logs/index.json` |

---

## `training_material_FINAL/` vs `AI Suitability Training Materials/`

These are **not the same thing**. They were built on different dates for different purposes.

### What `training_material_FINAL/` contains

PDF-only bundle exported **Apr 4, 2026** for human review. No JSON. Fifteen files:

| PDF | Source JSON | Generation | Records |
|-----|-------------|------------|---------|
| `Phase_12_Standard_Hypotheticals.pdf` | `datasets/March_6_AI_Training.json` | Mar 21 (topic-filter + fill-up) | 499 |
| `Phase_13_Borderline_Hypotheticals.pdf` | `datasets/March_6_AI_Training_Borderline.json` | Mar 21 | 250 |
| `Phase_14_Advisor_Client_Conversations.pdf` | `datasets/March_7_…_Conversations.json` | Mar 6–7 | 50 |
| `Phase_15_Red_Flag_Scenarios.pdf` | `datasets/AI_training_materials_Red_Flags.json` | Mar 6–7 | 50 |
| `Phase_16_Adversarial_Hypotheticals.pdf` | `datasets/March_14_Adversarial.json` | Mar 14 | 200 |
| `SuitabilityOnly_P12_Standard.pdf` | `Suitability Only/suitability_only_P12.json` | Mar 22 (full regeneration) | 499 |
| `SuitabilityOnly_P13_Borderline.pdf` | `Suitability Only/suitability_only_P13.json` | Mar 22 | 250 |
| `SuitabilityOnly_P14_Conversations.pdf` | `Suitability Only/suitability_only_P14.json` | Mar 22 | 50 |
| `SuitabilityOnly_P15_RedFlags.pdf` | `Suitability Only/suitability_only_P15.json` | Mar 22 | 50 |
| `SuitabilityOnly_P16_Adversarial.pdf` | `Suitability Only/suitability_only_P16.json` | Mar 22 | 200 |
| `New_Folders_P12_Standard.pdf` | `New_Folders_Suitability/new_folders_P12.json` | Mar–Apr | 600 |
| `New_Folders_P13_Borderline.pdf` | `New_Folders_Suitability/new_folders_P13.json` | Mar–Apr | 600 |
| `Benchmark_Suitability_GPT4o_vs_GPT54.pdf` | benchmarks | Mar 23 | — |
| `Benchmark_GPT4o_vs_GPT54_Adversarial.pdf` | benchmarks | Mar 17 | — |
| `Benchmark_NewFolders_GPT4o_vs_GPT52.pdf` | benchmarks | — | — |

**Key point:** FINAL includes **both** generations of P12–P16 side by side — the Mar 21 `Phase_*` set (from `datasets/`) *and* the Mar 22 `SuitabilityOnly_*` set (from `Suitability Only/`). Those two generations share ID numbers but have **entirely different scenario content**.

### What `AI Suitability Training Materials/` contains

Curated **machine-readable export** assembled **Apr 5, 2026**. JSON is the deliverable; some companion PDFs are included.

| Path | Contents |
|------|----------|
| `23_Folders_Suitability/suitability_only_P12–P16.json` | Byte-identical copies of `training_material_generation/Suitability Only/` — the **Mar 22 suitability-only** generation only (not `datasets/`) |
| `23_Folders_Suitability/*.pdf` | Renamed PDFs (`Standard.pdf`, `Borderline.pdf`, etc.) of the Suitability Only set |
| `New_6_Folders_Suitability/<product>/Standard.json` + `Borderline.json` | **770 items** — a product-folder split of `new_folders_P12/P13.json` (1,200 items total in source; 430 items exist only in the full `New_Folders_Suitability/` files) |

### Side-by-side comparison

| | `training_material_FINAL/` | `AI Suitability Training Materials/` |
|---|---------------------------|--------------------------------------|
| **Format** | PDF only | JSON (+ some PDFs) |
| **Created** | Apr 4, 2026 | Apr 5, 2026 (JSON source: Mar 22) |
| **P12–P16 content** | Both `Phase_*` and `SuitabilityOnly_*` versions | **Suitability Only** version only |
| **New folders 24–29** | Combined P12 + P13 PDFs (1,200 items) | Split by product × Standard/Borderline (770 items) |
| **Benchmarks** | Included as PDFs | Not included |
| **Use for fine-tuning** | No — read/review only | **Yes** — primary training deliverable |

---

## `AI Suitability Training Materials/` — counts and compliant/non-compliant stats

### Where to find stats

| What you need | Where to look |
|---------------|---------------|
| **Export totals and per-file counts** | This README (table below) — recomputed from live JSON |
| **Broader catalog** (all of `training_material_generation/`, hypotheticals, benchmarks, topic clusters) | [`training_material_generation/README.md`](Data%20Collection%20and%20Training%20Material%20Generation/training_material_generation/README.md) — §§1–4, especially **§4 Compliant vs. Non-Compliant** |
| **Apr 5 audit-level analysis** (gaps, skew, recommendations) | [`AUDITS_AND_REVIEWS.md`](Data%20Collection%20and%20Training%20Material%20Generation/AUDITS_AND_REVIEWS.md) — Final Review section |
| **Per-item `compliant` field** | The JSON files themselves (`"compliant": true/false` on P12 and New_6 Standard items) |
| **30 topic bucket labels** | `"topic"` field on each JSON item (e.g. `inappropriate_risk_profile`, `senior_investor_exploitation`) — no separate taxonomy file |

> **Note on compliant labels:** P13 (borderline), P14 (conversations), P15 (red flags), P16 (adversarial), and all New_6 Borderline items **do not set** `compliant: true/false` — ambiguity or violation-finding is embedded in the answer/analysis fields by design. Only P12 and New_6 Standard carry explicit boolean labels.

### Export totals (1,819 items)

| File / group | Items |
|--------------|------:|
| `23_Folders_Suitability/suitability_only_P12.json` | 499 |
| `23_Folders_Suitability/suitability_only_P13.json` | 250 |
| `23_Folders_Suitability/suitability_only_P14.json` | 50 |
| `23_Folders_Suitability/suitability_only_P15.json` | 50 |
| `23_Folders_Suitability/suitability_only_P16.json` | 200 |
| `New_6_Folders_Suitability/` (6 products × Standard + Borderline) | 770 |
| **Total** | **1,819** |

New_6 per-product split: Variable Annuities 200 · Leveraged/Inverse ETFs 200 · Options 190 · Crypto 75 · Private Placements 60 · Structured Products 45.

### Compliant vs. non-compliant (export JSON, explicit `compliant` field only)

| Group | Compliant | Non-compliant | No explicit label |
|-------|----------:|--------------:|------------------:|
| P12 (`suitability_only_P12.json`) | 75 | 424 | 0 |
| P13–P16 (`suitability_only_P13–P16.json`) | 0 | 0 | 550 |
| New_6 Standard (420 items) | 1 | 419 | 0 |
| New_6 Borderline (350 items) | 0 | 0 | 350 |
| **Export total** | **76** | **843** | **900** |

**Reading these numbers:** Among items with an explicit label, the ratio is roughly **1 compliant : 11 non-compliant**. The 900 "no label" items are not unclassified errors — P13 borderline scenarios (250), conversations (50), red flags (50), adversarial (200), and New_6 borderline (350) intentionally omit a boolean verdict. For the **full** `training_material_generation/` tree (including hypotheticals and `datasets/`), see §4 of the training README for Apr 5 estimates across all sources (~1,455 non-compliant / ~430 compliant / ~1,463+ borderline-or-unknown).

---

## Raw corpus: 29 folders (`raw_data/`)

Each folder has a `00_ABOUT.md` (where present) and descriptive subfolders. Full subfolder tables are in [`RAW_MATERIALS.md`](Data%20Collection%20and%20Training%20Material%20Generation/RAW_MATERIALS.md).

| # | Folder | Contents |
|---|--------|----------|
| 01 | `01_SEC_Guidance_NoAction_Interpretive` | SEC IM guidance, no-action letters, interpretive releases, Reg BI / Form CRS adopting releases |
| 02 | `02_SEC_Enforcement_Actions` | SEC IA-series enforcement (2015–2025) |
| 03 | `03_FINRA_AWC` | FINRA Acceptance, Waiver & Consent disciplinary settlements |
| 04 | `04_RegBI_Compliance_Guidance` | Reg BI rule text, Form CRS, FINRA Reg BI AWCs, compliance FAQs |
| 05 | `05_SEC_Exam_Findings` | OCIE/EXAMS priorities and risk alerts |
| 06 | `06_DOL_Fiduciary` | DOL fiduciary rules, PTEs, FABs, Retirement Security Rule materials |
| 07 | `07_Practitioner_Commentary` | SIFMA, IAA, CFP Board, CFA Institute, industry guides |
| 08 | `08_State_Fiduciary_Rules` | NY Reg 187, state fiduciary legislation, NASAA model rules |
| 09 | `09_Law_Firm_Memos` | Dechert, Sidley, K&L Gates, Willkie, Groom Law client alerts |
| 10 | `10_Academic_Articles` | Law reviews, NBER, GAO, policy papers |
| 11 | `11_Case_Law` | Federal circuit, SCOTUS, landmark IA/ERISA decisions |
| 12 | `12_FINRA_Arbitration_Awards` | Customer arbitration awards (~2,600+ PDFs) |
| 13 | `13_FINRA_Rulebook` | FINRA rules 2010/2111/2330/5310, suitability FAQs, Reg BI checklist |
| 14 | `14_NASAA_State_Enforcement` | NASAA reports, state enforcement, model rules |
| 15 | `15_FINRA_Regulatory_Notices` | Regulatory notices and industry comment letters |
| 16 | `16_Congressional_Materials` | Dodd-Frank §913 study, CRS/GAO, hearing transcripts |
| 17 | `17_SEC_Staff_Bulletins` | SEC staff bulletins on Reg BI care, conflicts, disclosure |
| 18 | `18_OCC_Bank_Trust_Guidance` | OCC Comptroller's Handbook, bank fiduciary guidance |
| 19 | `19_CFTC_Guidance` | CTA/CPO disclosure, commodity adviser obligations |
| 20 | `20_SEC_ALJ_OALJ_Decisions` | SEC Administrative Law Judge initial decisions |
| 21 | `21_FINRA_OHO_Decisions` | FINRA Office of Hearing Officers contested decisions |
| 22 | `22_PIABA_Materials` | PIABA amicus briefs, research, comment letters |
| 23 | `23_International_Comparative` | UK FCA Consumer Duty, EU MiFID II, IOSCO, ASIC, etc. |
| 24 | `24_Variable_Annuities` | NAIC Model 275, state insurance bulletins, VA exam outlines |
| 25 | `25_Leveraged_Inverse_ETFs` | FINRA/SEC/ESMA leveraged-ETF guidance and research |
| 26 | `26_Options` | OCC disclosure docs, FINRA options notices, retail options cases |
| 27 | `27_Private_Placements_DPPs` | Reg D, non-traded REITs, DPP due diligence |
| 28 | `28_Structured_Products` | Structured notes, reverse convertibles, IOSCO/ESMA reports |
| 29 | `29_Crypto_Digital_Assets` | SEC/CFTC/FINRA crypto enforcement, MiCA, congressional hearings |

---

## Training materials (`training_material_generation/`)

### Primary datasets (`datasets/`)

| File | Phase | Records | Format |
|------|-------|---------|--------|
| `March_6_AI_Training.json` | P12 | 499 | Standard Q&A (fact pattern → question → answer) |
| `March_6_AI_Training_Borderline.json` | P13 | 250 | Borderline / ambiguous analysis |
| `March_7_AI_Training_Materials_Conversations.json` | P14 | 50 | Multi-turn advisor–client dialogues |
| `AI_training_materials_Red_Flags.json` | P15 | 50 | Red-flag identification scenarios |
| `March_14_Adversarial.json` | P16 | 200 | Adversarial / trick-question hypotheticals |

Each JSON has a matching `.txt` export. Parallel suitability-filtered copies live in `Suitability Only/` and in the export tree `AI Suitability Training Materials/23_Folders_Suitability/`.

### Hypotheticals (`hypotheticals/`)

| File | Description |
|------|-------------|
| `0001_*.txt` … `0983_*.txt` | One structured scenario per file (FACTS, QUESTION, ANSWER, APPLICABLE STANDARD) |
| `hypotheticals.json` | Full 983-item JSON |
| `all_hypotheticals.txt` | All scenarios concatenated |
| `hypotheticals_phase10_498.json` | Phase 10 Opus batch (498 items; ~90 overlap with P12 — dedupe before training) |
| `_older_versions/` | Earlier v2 batches kept for history |

### Product-folder training (`New_Folders_Suitability/`)

| File | Records | Scope |
|------|---------|-------|
| `new_folders_P12.json` | 600 | Standard Q&A grounded in folders 24–29 |
| `new_folders_P13.json` | 600 | Borderline analysis for folders 24–29 |

The export tree `AI Suitability Training Materials/New_6_Folders_Suitability/` splits these by product (Standard.json + Borderline.json per folder).

### Benchmarks (`benchmarks/`) — evaluation only, not training

| File | Notes |
|------|-------|
| `March_23_Suitability_Benchmark.json` | GPT-4o vs GPT-5.4 suitability bench (~1,990 items) |
| `March_17_GPT_Benchmark.json` | Superseded early adversarial bench |
| `New_Folders_Benchmark.json` | New-folders model comparison |
| `New_Folders_Benchmark_4o_vs_54.json` | GPT-4o vs GPT-5.4 on new folders |
| `March_23_P13_binary_cache.json` | Cached P13 yes/no verdicts for reruns |

### Backups — do not double-count

`backups_mar21/`, `backups_mar21_v2/`, `backups_mar21_v3/` hold pre–topic-filter snapshots of the March 21 filtering pass. They are historical, not additional unique items.

---

## Build phases 1–16

Phases **1–11** built the raw PDF corpus. **Phase 10** also generated synthetic hypotheticals (training layer). **Phases 12–16** produced the structured JSON training datasets (P12–P16). Folder numbers 12–23 refer to *corpus folders*, not phase numbers — phases 12–16 are training-generation phases.

| Phase | Dates | Type | What happened |
|-------|-------|------|---------------|
| **1** | Feb 25, 2026 | Corpus | Project kickoff. Scraper deployed; first live pulls for core sources — Reg BI guidance (folder 04), SEC exam findings (05), DOL fiduciary (06), then SEC guidance (01), enforcement (02), FINRA AWC (03), practitioner commentary (07). |
| **2** | Feb 25–26, 2026 | Corpus | Continued scraping and indexing of folders 01–07; initial no-action letters, IA enforcement releases, AWC settlements, and industry commentary. |
| **3** | Feb 27, 2026 | Corpus | Targeted gap-fill pass on folders **03, 07, 09, 10, 11** (FINRA AWC expansion, law firm memos, academic articles, case law). Log: `raw_data/logs/phase3_log.txt`. |
| **4** | Feb 28, 2026 | Corpus | Mid-build consolidation snapshot — 11 folders, ~1,171 PDFs, ~320 hypotheticals. Archived manifest: `_merged_sources/DATASET_MANIFEST.md`. |
| **5** | Feb 28 – Mar 1, 2026 | Corpus + training | Folders **01–11** completed (~1,271 PDFs). First **485** synthetic hypotheticals generated (legacy batch; many tagged `source_folder: unknown`). |
| **6** | Mar 1, 2026 | Corpus | Folders **12–19** added (+902 PDFs): FINRA arbitration awards, rulebook, NASAA, regulatory notices, congressional materials, SEC staff bulletins, OCC, CFTC. Sub-run **6b** filled folders 16–19. Logs: `phase6_log.txt`, `phase6b_log.txt`. |
| **7** | Mar 1, 2026 | Corpus | FINRA awards expansion; folders **20–22** added (SEC ALJ, FINRA OHO, PIABA). Sub-runs 7b/7c/7_alj/7_final refined ALJ and OHO collection. Logs: `phase7_*.txt`. |
| **8** | Mar 2, 2026 | Corpus | Quality and coverage pass; folder **23** (international comparative) added. Corpus total ~2,788 PDFs. Log: `phase8_log.txt`. |
| **9** | Mar 2, 2026 | Corpus | Large sweep: FINRA AWC 2020–2025 (+349 AWC via **9b**), arbitration awards 2016–2025 probe (**9a**, **9c**–**9e**). Corpus total **~5,102 PDFs**. Logs: `phase9_*.txt`. |
| **10** | Mar 3–7, 2026 | Training | **Opus hypotheticals generator** — grounded Q&A from folders 01–23. Grew hypotheticals from 485 → **983** (+498 Phase-10-tagged items). Outputs: `hypotheticals/`, `hypotheticals_phase10_498.json`. Log: `training_material_generation/logs/generation/phase10_log.txt`. |
| **11** | Mar 5, 2026 | Corpus | Gap-fill audit pass (+294 PDFs): law firm memos, academic articles, FINRA awards 2025, rulebook, congressional materials, OHO decisions. Corpus total **5,399 PDFs** across 23 folders. Report: `AUDITS_AND_REVIEWS.md` (Phase 11 section). |
| **12** | Mar 5–6, 2026 | Training | **P12 — Standard Q&A.** 499 JSON scenarios (`March_6_AI_Training.json`) from corpus PDFs via Opus. Log: `phase12_log.txt`. |
| **13** | Mar 6, 2026 | Training | **P13 — Borderline analysis.** 250 intentionally ambiguous scenarios (`March_6_AI_Training_Borderline.json`). Log: `phase13_log.txt`. |
| **14** | Mar 6–7, 2026 | Training | **P14 — Conversations.** 50 multi-turn advisor–client dialogues (`March_7_AI_Training_Materials_Conversations.json`). Log: `phase14_log.txt`. |
| **15** | Mar 6–7, 2026 | Training | **P15 — Red flags.** 50 red-flag identification scenarios (`AI_training_materials_Red_Flags.json`). Log: `phase15_log.txt`. |
| **16** | Mar 14, 2026 | Training | **P16 — Adversarial.** 200 trick-question hypotheticals designed to stump frontier models (`March_14_Adversarial.json`). Log: `phase16_log.txt`. |

### Post–phase-16 work (not numbered phases)

| Date | Work | Result |
|------|------|--------|
| Mar 21, 2026 | Topic filter + fill-up pass | Filtered `datasets/` to suitability topics; refilled P12 to 499. Log: `phase_fill_up_log.txt`. |
| Mar 22, 2026 | Suitability Only regeneration | `suitability_only_generator.py` — new P12–P16 JSON (source of export). Log: `Suitability Only/generation_log.txt`. |
| Apr 4, 2026 | `training_material_FINAL/` PDF export | Human-readable PDFs of both `Phase_*` and `SuitabilityOnly_*` sets + benchmarks. |
| Apr 5, 2026 | `AI Suitability Training Materials/` export | JSON copies + renamed PDFs + New_6 product-folder split. |
| Mar 23–25, 2026 | Suitability benchmarks | GPT-4o / GPT-5.4 evaluation runs. Logs in `logs/benchmark/`. |
| Mar 25, 2026 | Product-folder corpus (**24–29**) | Six specialty folders scraped (variable annuities, L/I ETFs, options, private placements, structured products, crypto). +616 PDFs. Logs: `phase24_log.txt` … `phase29_log.txt`. |
| Apr 5, 2026 | Comprehensive review | Training-vs-corpus gap analysis. See `AUDITS_AND_REVIEWS.md` (Final Review section). |
| Jun 19, 2026 | Tier-1 authority gap-fill | Added FINRA rule texts, SEC Care bulletin, DOL vacatur notice, etc. See `GAPFILL_ADDED_2026-06-19.md`. |
| Jun 22, 2026 | Repo reorganization | Merged docs (`RAW_MATERIALS.md`, `AUDITS_AND_REVIEWS.md`), training folder restructure, export tree under `AI Suitability Training Materials/`. |

---

## Key documentation index

| Document | Purpose |
|----------|---------|
| [`RAW_MATERIALS.md`](Data%20Collection%20and%20Training%20Material%20Generation/RAW_MATERIALS.md) | Live corpus inventory — 29 folders, subfolders, file counts, suitability relevance |
| [`AUDITS_AND_REVIEWS.md`](Data%20Collection%20and%20Training%20Material%20Generation/AUDITS_AND_REVIEWS.md) | Phase 11 report, completeness audit, final review, corpus coverage audit, gap-fill log |
| [`training_material_generation/README.md`](Data%20Collection%20and%20Training%20Material%20Generation/training_material_generation/README.md) | Training dataset catalog, compliant/non-compliant estimates, duplicate warnings |
| [`raw_data/logs/README.md`](Data%20Collection%20and%20Training%20Material%20Generation/raw_data/logs/README.md) | Corpus-build log index; `index.json` schema |
| [`_merged_sources/README.md`](Data%20Collection%20and%20Training%20Material%20Generation/_merged_sources/README.md) | Provenance for merged/archived source docs |

---

## Known caveats

1. **Deduplicate before training:** ~90 overlapping items between `March_6_AI_Training.json` and `hypotheticals_phase10_498.json`.
2. **Compliant/non-compliant skew:** Export JSON is ~76 compliant / 843 non-compliant where explicitly labeled; P13–P16 and borderline items omit the boolean. See **§ AI Suitability stats** above and training README §4 for the full-project picture.
3. **Benchmarks ≠ training:** Keep `benchmarks/` separate from fine-tuning data.
4. **DOL currency:** 2024 Retirement Security Rule materials in folder 06 are **vacated** as of Mar 2026 — treat as historical. See corpus coverage audit.
5. **Scripts not in repo:** Scraper and generator scripts (`phase10_hypotheticals_opus.py`, `phase12_opus.py`, etc.) live on the original author's machine; this repo holds outputs and logs only.
