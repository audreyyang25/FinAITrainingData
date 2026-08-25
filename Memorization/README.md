# Memorization

Tooling for the memorization-extraction replication (Cooper et al. 2025, *Extracting
Memorized Pieces of (Copyrighted) Books from Open-Weight Language Models*).

Code lives here. **All crawl output goes to
`Data Collection and Training Material Generation/crawl_manifest/`**, alongside the
case-ID manifest that drives it.

Three docs, and they answer different questions:

| | |
|---|---|
| `README.md` | how to run the tooling |
| `METHODS.md` | corpus, question generation, scoring, results |
| `PROMPTS.md` | every system prompt in run order, what it replaced, and why |

```
Memorization/
  crawl/
    common.py        HTTP session, per-host rate limiting, saves, JSONL log
    sources.py       per-category URL resolution (FINRA/SEC templates are UNVERIFIED)
    run_crawl.py     CLI - run --probe first
  extract/
    compare_extractors.py   head-to-head extractor benchmark
    normalize.py            paragraph-flow unwrap (the important one)
```

## Setup

```bash
export CRAWL_CONTACT="you@example.com"     # sec.gov's automated-access policy wants this in the UA
export COURTLISTENER_TOKEN="..."           # https://www.courtlistener.com/profile/api/
```

## Probe before crawling

```bash
python -m Memorization.crawl.run_crawl --probe -n 3
```

Reports what each source actually returns. **If a category comes back PDF-only, do not
bulk-crawl it** — those PDFs are already in `raw_data`, and re-downloading them adds
nothing. Extract locally instead.

The expected outcome is that only `court_opinions` is worth crawling. CourtListener
serves `plain_text` / `html` / `html_with_citations` as distinct fields, and FreeLaw in
the Pile is CourtListener-derived — so for Pythia that is not an approximation of the
training source, it *is* the training source. FINRA AWC, OHO, and arbitration documents
are PDF-native; SEC administrative orders likewise.

```bash
python -m Memorization.crawl.run_crawl --category court_opinions
```

Resumable (skips existing output), rate-limited per host, logs every fetch to
`crawl_manifest/crawl_log.jsonl` with status, final URL, sha256, and byte count.

## Extraction, for the PDF-native categories

No mainstream PDF extractor reflows paragraphs — they all emit one line per *visual*
line, because that is what the PDF encodes. Measured on AWCs, pdfplumber and pypdf land
within 0.2 points of each other on mid-sentence line breaks (39.3% vs 39.1%), and pypdf
is 3.6x faster. So pick the extractor on speed and reading order, then fix the wrapping
separately.

```bash
python Memorization/extract/compare_extractors.py --category finra_awc -n 5
python Memorization/extract/normalize.py "03_FINRA_AWC/Individual_AWC_Case_Files"
```

`normalize.unwrap()` joins intra-paragraph hard wraps while preserving list markers,
headings, and paragraph boundaries. Measured effect: 54.3% → 3.2% mid-sentence breaks on
AWCs, 65.4% → 3.3% on court opinions. Opinions retain more residual wrapping (15–23%)
because footnotes and citation blocks resist joining.

This matters because the paper's metric is teacher-forced: `p = ∏ P(yᵢ | x, y<i)`. An
artifact newline every ~11 tokens is a high-surprisal token at ~9% of positions, which
depresses the product enough to make a perfectly memorized document look unmemorized.

## Before trusting any result

Push text you are confident is memorized — the Constitution's preamble, a well-known
SCOTUS passage, the opening of a public-domain novel — through the identical pipeline.
A null result on obscure regulatory text looks exactly like a broken harness.

Note also that FINRA PDFs sit behind a search interface and are poorly linked, so
Common Crawl coverage is likely thin. Those categories may return nothing simply because
the documents were never in any pretraining corpus. Court opinions are the plausible
positive; OHO decisions make a natural negative control.
