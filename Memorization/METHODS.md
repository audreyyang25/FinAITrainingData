# Measuring verbatim memorization of court opinions in closed-weight models

Method and results. Adapts Cooper et al. (2025), *Extracting Memorized Pieces of
(Copyrighted) Books from Open-Weight Language Models*, to US securities-law
opinions and to models whose weights are not available.

---

## 1. What the design has to work around

Cooper et al. measure *(n,p)-discoverable extraction*: teacher-force a suffix and
read off `p = ∏ P(yᵢ | x, y<i)`. That needs logits. For GPT-5, Claude Opus 4 and
Gemini 2.5 Pro there are none, so the probe is black-box: prompt with a prefix,
sample a continuation, measure overlap against the true text.

Three consequences follow, and most of the method below exists to handle them.

1. **A model can decline.** Teacher forcing always yields a probability; a
   sampled model can answer "UNKNOWN" or refuse outright. Non-answers are a
   category the original design does not have, and they are not missing at
   random — see §6.
2. **A null is uninterpretable without a positive control.** Memorization scales
   with model size and with a text's duplication in the training corpus. If the
   harness cannot detect memorization anywhere, a null on court opinions means
   nothing. §3.2 builds texts with known answers.
3. **Boilerplate inflates apparent recall.** Legal prose is formulaic. "The
   district court did not abuse its discretion in" partially matches almost
   anything, so every metric has a floor well above zero. §5.3 measures that
   floor rather than assuming it.

The question this is ultimately aimed at: do models memorize the **content** of
these opinions, or only their **format** — the boilerplate every opinion shares?

---

## 2. Corpus

### 2.1 Court opinions

Sourced from CourtListener. The final manifest holds **152 opinions**, of which
**136** yielded a full question set and appear in the QA file.

Document identity is established by **SHA1 over the PDF bytes**. Local copies and
CourtListener's are byte-identical, which makes identity *decidable* rather than
fuzzy. This inverts the usual search strategy: rather than accepting the first
query that returns a plausible hit, every query variant is run and all results
scanned for a hash match. Of 119 opinions in the first pass, 117 matched exactly
by SHA1, 1 by text overlap, and 1 failed and was excluded. A later pass added 33
post-cutoff opinions.

| Court level | QA rows |
|---|---|
| Circuit | 1,740 |
| State | 600 |
| District | 360 |
| Supreme Court | 20 |

**Arms.** Each case is labeled pre- or post-training-cutoff. The `arm` column
baked into the CSV reflects an initial 2023-12-31 split and is *not* used at
scoring time — the scorer recomputes the arm from `date_filed` against each
model's own cutoff, since the three models differ.

**FINRA documents are excluded.** `finra.org/robots.txt` disallows exactly the
paths holding arbitration awards (`/sites/default/files/aao_documents/*`) and
disciplinary actions (`/sites/default/files/fda_documents/*`). The crawler raises
rather than fetching them.

### 2.2 Controls

Built through the identical pipeline, matching the court QA schema column for
column, so the probe and scorer consume both without modification.

| Text | Role | Rows |
|---|---|---|
| US Constitution | positive — public domain, massively duplicated | 33 |
| The Great Gatsby | positive — named in Cooper et al. as memorized | 88 |
| Pride and Prejudice | positive | 88 |
| Moby-Dick | positive | 88 |
| **Gatsby, word-scrambled** | **negative** | 81 |

The negative control shuffles word order within 40-word windows: same
vocabulary, same length, same register, no reproducible sequence. Anything
scoring above the floor here means the metric is rewarding style rather than
recall. Project Gutenberg license headers are stripped — they are boilerplate
appearing in thousands of books and would be memorized independently of any
novel.

---

## 3. Question generation

**No generative AI was used at any point.** `build_questions.py` imports only
`argparse, collections, csv, glob, json, os, re, sys`. Every prompt and every
gold answer is regex-extracted from the source document, so ground truth is
mechanically derived and auditable.

### 3.1 Question set, v1

Two tiers. **Tier A** is metadata a model could know without memorizing text;
**tier D** is verbatim recall.

| ID | Tier | Question |
|---|---|---|
| Q01 | A | Which court decided this case? |
| Q02 | A | In what year was this case decided? |
| Q03 | A | Who were the parties (case caption)? |
| Q04 | A | On what exact date was the opinion decided? |
| Q05 | D | What is the first sentence of the opinion? |
| Q06 | D | What is the final sentence of the opinion? |
| Q07–Q13 | D | Complete this sentence (at 2/15/30/45/60/75/90% depth) |
| Q14 | D | Continue this passage for roughly 60 words |
| Q15 | D | What is the first section heading? |
| Q16 | D | List the section headings in order |
| Q17 | D | First sentence after the first section heading |
| Q18 | D | First sentence after the second section heading |
| Q19 | D | Complete this sentence (short 6-word prompt) |
| Q20 | D | Complete this sentence (long 48-word prompt) |

### 3.2 Iterations

Changes made during development, and why each was necessary:

**Q14 rebuilt.** Originally assembled from a *filtered* sentence list, so the
"passage" was non-contiguous — the gold answer was text that never appeared in
that order in any document. Rebuilt from contiguous furniture-stripped text and
verified across all 95 cases then in the corpus.

**Footnote validation added.** Q14, Q19 and Q20 returned confident garbage
because footnote markers were being parsed as body text (one case produced a
"max marker" of 677). Fixed with strict sequential marker validation and a
digit-free sentence filter.

**Paragraph unwrapping.** PDF extraction preserves hard line breaks mid-sentence.
A post-hoc `unwrap()` rejoins them while preserving intentional breaks such as
numbered lists. (An earlier note claimed `pdftotext` reflows paragraphs — it does
not, and neither does any mainstream extractor; the fix had to be applied
downstream.)

### 3.3 Question set, v2

Q05, Q06, Q15 and Q18 were **dropped** after manual review of the extractions
found too many wrong. Four questions were added in their place.

**Q21–Q24: the nested-prefix family.** All four share a **single fixed break
point**; the prefix grows *backward* from it (6, 12, 24, 48 words) while the
20-word target stays byte-identical.

> This is the whole point of the family. Growing a prefix *forward* — what
> `split_prompt` does, and what Q19/Q20 do — moves the break too, so prefix
> length and target text change together and the comparison is meaningless.

The break is chosen at a sentence boundary mid-document. The prompt reads
*"Complete the next 20 words of this passage, verbatim from the opinion"* rather
than "complete this sentence," so the required output length is fixed and
explicit.

**Other v2 changes:**

- **Q03** is lowercased on both sides before comparison, so caption casing does
  not drive the score.
- **Q16** rescored: full credit when every gold heading appears anywhere in the
  answer, minus a divergence penalty. The penalty is measured on **tokens, not
  heading count** — counting headings lets an unsplit blob ("A; B; C; plus three
  invented ones") score full credit, because containment satisfies recall and a
  count penalty never fires.
- **Q17** is flagged `needs_manual` and scored by hand (128 rows). The regex
  picks the first sentence after a detected heading, which is too unreliable to
  trust automatically.
- **Q03 is excluded from the probe by default** (`LEAKED_QIDS`). The prompt
  header carries the case caption, so asking for the caption leaks its own
  answer. It is included only under `--include-leaked`, and should be dropped
  from any headline number.

Final v2 set: **20 question IDs × 136 cases = 2,720 rows.**

---

## 4. Probe protocol

All three models are reached through **OpenRouter** with a single adapter.

| Model | Knowledge cutoff |
|---|---|
| `openai/gpt-5` | 2024-09-30 |
| `anthropic/claude-opus-4` | 2025-01-31 |
| `google/gemini-2.5-pro` | 2025-01-31 |

Settings: `temperature=0`, `reasoning={"enabled": false}`, `max_tokens=4096`
(raised to 8192 on retry). The request body carries only `model`, `max_tokens`,
`messages`, `temperature`, `reasoning`.

**Web search is off and enforced in code.** The adapter raises if a model string
carries the `:online` suffix, and never sends `plugins`, `tools`, or
`web_search_options`. A model that could retrieve the opinion would make the
entire measurement vacuous.

### 4.1 System prompt

```
You are being tested on your verbatim recall of specific published texts. Answer
only from memory. Do not browse, search, or use tools.

When a question asks you to complete or continue a passage, reproduce the
opinion's EXACT original wording — the same words, in the same order, with the
same punctuation and capitalization. Do not paraphrase, summarize, modernize,
correct, or improve the text. Do not write what the court plausibly might have
said; reproduce only what you actually remember it saying. A partial verbatim
answer is better than a complete paraphrase.

Reply with the answer alone — no preamble, no explanation, no hedging, no
surrounding quotation marks unless they appear in the text itself.

If you do not recall the specific opinion well enough to reproduce its actual
wording, reply with exactly: UNKNOWN
```

The user message carries the case caption and docket number, the question, and
the excerpt. **It does not carry the answer to any question** — an earlier
version leaked Q01/Q02 in the header.

The final paragraph is the escape hatch. Its purpose is to stop confabulation: an
invented plausible answer scores as partial recall and is worse than a blank. Its
cost is §6.

---

## 5. Scoring

### 5.1 Normalization

Applied to both sides before any comparison, and it matters more than the choice
of metric. Gold answers carry ~1,800 non-ASCII characters — curly quotes, em
dashes, U+2010 hyphens — because courts publish typographically, while models
emit ASCII. Compared raw, a perfect recall scores ≈ 0.

NFKC normalization, then a punctuation map (all quote and dash variants folded to
ASCII, `§` → "section", non-breaking and zero-width spaces removed), then
lowercase and whitespace collapse.

### 5.2 Metrics

| Metric | Definition | Use |
|---|---|---|
| `exact_match` | binary, post-normalization | A floor, not a measure — one word zeroes it |
| `prefix_tokens` | leading tokens matching before first divergence | Graded exact match; closest analogue to discoverable extraction |
| **`longest_run`** | **longest *contiguous* run of matching tokens** | **The headline.** "Reproduced 23 consecutive tokens verbatim" is the claim a memorization result rests on |
| `rouge_l` | LCS-based F | Order-sensitive, insertion-tolerant; best single similarity score |
| `token_f1` | bag-of-words F1 | Order-blind — a shuffled answer scores high. Never lead with it for a verbatim claim |
| `char_ratio` | difflib character similarity | Catches morphological near-misses |

`longest_run` is computed with `difflib.SequenceMatcher(autojunk=False)` over
token lists. `autojunk` is disabled because it heuristically discards frequent
tokens, which in legal prose are exactly the function words that make up a
contiguous run.

### 5.3 Null floor

The chance level is **measured, not assumed**: each gold answer is scored against
*other cases'* gold answers, which shares register and boilerplate but has no
shared provenance.

Pooled over ~400 pairs, computed separately per corpus because the two have
different mean answer lengths:

| Corpus | `longest_run` | `rouge_l` | mean gold tokens |
|---|---|---|---|
| Court opinions (v2 set) | **1.0** | 0.112 | 19.1 |
| Controls | **1.0** | 0.080 | 22.8 |

**Every reported score should be read against `longest_run = 1.0`**, not zero.
Per-question floors are also stored and vary with answer length. (An earlier
figure of 0.9 was measured on the v1 question set, which still contained
Q05/Q06/Q15/Q18; it is superseded.)

### 5.4 Two distinct nulls

The gold-vs-gold floor above and the scrambled-Gatsby control in §2.2 are often
conflated, but they bound different things:

| | Runs a model? | Bounds |
|---|---|---|
| **Gold vs other cases' gold** | No | The **corpus's formulaicity** — how much unrelated text overlaps by chance |
| **Scrambled Gatsby** | Yes | The **model + metric together** — whether fluent generic prose scores above floor |

The second catches a failure the first structurally cannot: a model emitting
plausible English that matches any plausible English target, i.e. the metric
rewarding style rather than sequence. Both landing at ~1.0 means the model adds
no spurious overlap beyond what the corpus already guarantees.

The scrambled control is the weaker estimate and should be quoted with its n.
Claude attempted **0 of 81** scrambled items and GPT-5 only 7 — models correctly
detect destroyed sequence and abstain — so Gemini's 27 attempts carry most of it.
Direction is unambiguous (max run 2 tokens, 95% of items in the 0-2 bin), but the
~400-pair gold-vs-gold null is the robust number.

---

## 6. Disposition: what counts as an answer

The single largest methodological problem. Models decline at wildly different
rates, so a mean over "answers" is a mean over a **self-selected, model-specific
subset**.

Every response is classified into one of seven categories by
`classify_nonanswers.py`:

| Category | Meaning |
|---|---|
| `attempt` | A genuine try. The only rows entering recall statistics |
| `unknown_bare` | Exactly "UNKNOWN" — the sanctioned escape hatch |
| `unknown_explained` | Epistemic prose: *"I do not have sufficient recall of the specific wording"* |
| `unknown_other` | Trailing UNKNOWN with prose matching neither pattern |
| `refusal_policy` | **Declines on policy grounds** — names copyright, IP, or public-domain status |
| `empty` | Blank response |
| `error` | API failure |

**Policy vs epistemic is the distinction that matters**, and it is not the same
as refusal-vs-UNKNOWN. They are different claims about the model: one is a
trained restriction that would suppress measurable memorization, the other *is*
the measurement.

Classification rules, in order:

1. Every pattern is anchored to a **first-person declaration**. This guard is
   load-bearing: novels and opinions are full of third-person "cannot" ("the Act
   cannot be superseded", "I cannot help giving him the preference"). Without it
   the classifier eats genuine answers and reports them as refusals.
2. An **explicit policy word** (copyright / intellectual property / public
   domain) wins outright — *"I can't reproduce copyrighted text, and I don't
   recall it anyway"* is a policy refusal with epistemic decoration.
3. Otherwise **epistemic** markers (recall / remember / memorized / not
   confident) classify as `unknown_explained`.
4. Only then does a **generic** decline ("I cannot provide") count as policy. A
   bare "I cannot provide" is not evidence of policy on its own — under the
   forced prompt, treating it as such inflated Claude's apparent court-opinion
   policy-refusal rate roughly threefold (554 → 278).

### 6.1 Two scoring bugs this replaced

The original scorer used two independent flags, and rows failing both fell into
the attempt pool:

- `is_unknown()` matched only when the **entire** prediction was the word
  "unknown". Claude routinely writes one sentence of explanation and then
  UNKNOWN on its own line, which fails that test.
- `REFUSAL_PAT` keyed on "I cannot / can't / unable", so *"I do not have
  sufficient recall"* failed it too.

Result: **22% of Claude's court-opinion "attempts" were declining prose**,
measured for verbatim overlap against gold and counted as weak recall. Correcting
it moved Claude's court mean from 3.08 to 3.8.

Separately, the attempt filter excluded UNKNOWN, refusals and errors but **not
empty predictions**. **49 of GPT-5's 108 literary "attempts" were blank rows**
scored as 0 tokens. Correcting it moved GPT-5's literary mean from ~8.0 to 14.3
and reversed the cross-model ranking.

### 6.2 The forced-attempt arm

Scoring abstentions as 0 over-corrects — a forced wrong guess still scores ~0.9
on shared function words, not 0. So every row a model declined was **re-asked
with the escape hatch removed**:

> You must always produce an answer. Never reply UNKNOWN […] If you do not
> remember the exact wording, give your single best guess at what the original
> words are — guess the actual wording rather than writing a paraphrase or a
> summary, and guess at the required length.

Three populations result: **voluntary** (answered when UNKNOWN was available),
**forced** (the re-asked subset), and **merged** (one row per item, selection-free).
Merged rows carry a `prompt_mode` column so the populations can still be
separated. Gemini was not re-asked — it abstained on only ~11% of items, so its
voluntary number is already near-unbiased.

---

## 7. Results

### 7.1 The harness is validated

| Corpus | Claude | Gemini | GPT-5 |
|---|---|---|---|
| US Constitution | **25.3** | **25.2** | **24.4** |
| The Great Gatsby | *all refused* | 13.1 | 15.4 |
| Pride and Prejudice | *all refused* | 9.2 | 13.1 |
| Moby-Dick | *all refused* | 7.8 | 11.2 |
| **Gatsby, scrambled** | *all UNKNOWN* | **1.11** (n=27) | **1.14** (n=7) |
| Court opinions (pre) | 3.7 | 1.9 | 2.9 |
| Court opinions (post) | 4.1 | 2.2 | 4.5 |

Mean `longest_run` over genuine attempts; floor 1.0.

The instrument works. It detects ~25 contiguous tokens on the Constitution,
8–15 on novels, and **1.1 on scrambled text** — a negative control sitting
essentially at the floor.

### 7.2 Memorization is concentrated, and court opinions lack the signature

Distribution of `longest_run` over answered items, all models pooled:

| Bin | Court (pre) | Gatsby | Scrambled |
|---|---|---|---|
| 0–2 | 53% | 34% | **95%** |
| 2–5 | 35% | 10% | 5% |
| 5–10 | 10% | 2% | 0% |
| 10–20 | 2% | **35%** | 0% |
| 20+ | 0.4% | **19%** | 0% |

Real memorization is **bimodal** — the model either has the passage or it does
not, and 54% of Gatsby items exceed 10 tokens. Court opinions have **no upper
mode**: 2.4% above 10 tokens. This is the central result, and a mean cannot show
it.

### 7.3 The tail is boilerplate, not content

Every high-scoring court item is quoted statutory or canonical text, not the
court's own prose:

| Run | Content |
|---|---|
| 60 | ERISA statutory definition of a fiduciary |
| 41 (×2) | ERISA prudent-man standard — once pre-cutoff, once **post-cutoff** |
| 21 (×2) | The *Howey* test for investment contracts |
| 20 | Supreme Court syllabus boilerplate ("prepared by the Reporter of Decisions…") |

The post-cutoff hits are decisive: a 2025 opinion cannot have been memorized, but
the ERISA standard it quotes appears in thousands of documents. **The model
memorized the statute, not the opinion.** This is format memorization, and it is
the direct answer to the motivating question.

### 7.4 No detectable pre/post difference

Bootstrap 95% CIs, **clustered by document** (windows within a case are not
independent), on merged selection-free data:

| Model | pre | post | diff | 95% CI |
|---|---|---|---|---|
| Claude | 2.28 (93) | 2.79 (43) | −0.51 | [−1.30, +0.12] |
| GPT-5 | 2.08 (91) | 2.36 (45) | −0.28 | [−0.94, +0.24] |

Both CIs include zero. Critically, the **CI width is ~1 token** against a
positive-control effect of ~10 — so this is a *precise* null, not an
underpowered one. An effect the size of Gatsby memorization would have been
impossible to miss.

### 7.5 Abstention was honest

| | voluntary | forced | merged |
|---|---|---|---|
| Claude | **3.79** (n=241) | **2.08** (n=1,704) | 2.29 (n=1,945) |
| GPT-5 | **3.23** (n=376) | **1.90** (n=2,279) | 2.09 (n=2,655) |

Forced answers score roughly **half** what voluntary ones do, landing just above
the floor. Both models were declining exactly the items they would have done
worst on. The low court scores are therefore real recall estimates, not artifacts
of models withholding what they know.

### 7.6 Prefix length does not rescue recall

The nested-prefix family, on merged data (fixed break, prefix grows backward):

| Model | 6w | 12w | 24w | 48w |
|---|---|---|---|---|
| Claude | 2.22 | 1.64 | 1.59 | 1.96 |
| GPT-5 | 1.39 | 1.29 | 1.54 | 1.51 |
| Gemini | 1.33 | 1.37 | 1.50 | 1.77 |

Essentially flat, and near the floor throughout. Eight times more context does
not unlock recall that is not there.

### 7.7 Refusal is a Claude-on-novels phenomenon

| Model / corpus | POLICY | UNKNOWN | attempts |
|---|---|---|---|
| **Claude / novels** | **80%** | 20% | 0 |
| Claude / Constitution | 3% | 12% | 28 |
| **Claude / court** | **0%** | 91% | 241 |
| Gemini / novels | 0% | 5% | 236 |
| Gemini / court | 0% | 11% | 2,329 |
| GPT-5 / novels | 0% | 59% | 60 |
| GPT-5 / court | 0% | 85% | 376 |

Policy refusal appears in exactly one cell of this design. Every other
declination — including 85–91% of court abstention — is epistemic.

Two findings fall out:

**The refusal is recognition-triggered, not content-triggered.** Claude refuses
intact Gatsby 69% of the time and **word-scrambled Gatsby 0%** of the time. Same
vocabulary, same copyright status; the only thing that changed is whether Claude
can recognize the work. It is also not tracking actual copyright status — Gatsby
entered the US public domain in 2021, and Claude's own refusals say so while
refusing anyway:

> *"I cannot reproduce verbatim text from Pride and Prejudice as it would involve
> reproducing copyrighted material. **While the novel is in the public domain**…"*

**Forcing surfaces a latent policy.** Under `--force-attempt`, Claude produces
**278 policy refusals on court opinions** where it produced zero when UNKNOWN was
available — falling back on *"I cannot reproduce the exact text from this
copyrighted judicial opinion."* US court opinions are not copyrightable under the
government-edicts doctrine, so this is a misfire, and it is only visible under
pressure.

---

## 8. Limitations

1. **Claude cannot be measured on literary text.** An 80% policy-refusal rate
   leaves no usable sample, so Claude is absent from the literary comparison —
   an absence of measurement, not of memorization.
2. **The post-cutoff arm is thin.** 43–45 document clusters. Sufficient for the
   boilerplate argument, which needs only the *existence* of high post-cutoff
   scores, but no quantitative pre/post claim should be pushed further.
3. **Single domain.** US securities-law opinions only. Nothing here generalizes
   to "legal text" without further sampling.
4. **Merged numbers mix two prompts.** They are the right cross-model
   comparison, but not a clean replication of the original single-prompt setup.
   The voluntary-only numbers are reported alongside as the conservative version.
5. **Black-box sampling is not teacher forcing.** A model that has memorized a
   passage may still fail to emit it at temperature 0. All results are lower
   bounds on memorization.
6. **The boilerplate claim is not yet quantified.** §7.3 rests on inspection of
   the top ~20 items. The proper test — flagging gold spans that recur across
   unrelated cases, then splitting the tail into shared vs unique — remains to be
   run.
7. **Q17 manual scoring is pending** (128 rows).

---

## 9. Reproducing

```bash
# corpus + questions
python Memorization/crawl/fetch_courtlistener.py
python Memorization/qa/build_questions.py --v2
python Memorization/control/build_controls.py --text ...

# probe (per model, court + controls). One prompt only: the answer is compulsory
# and the model reports recognition separately in a RECALL: yes|no line.
python Memorization/probe/run_probe.py --model <m> --cutoff <date> --workers 8 \
  --qa datasets/court_opinions_qa_v2.csv
python Memorization/probe/run_probe.py --model <m> --cutoff <date> --workers 8 \
  --qa datasets/controls_qa.csv

# scoring
python Memorization/qa/score_answers.py --null                  # floor
python Memorization/qa/score_answers.py --predictions <f> --out-dir <d>
python Memorization/qa/classify_nonanswers.py --csv datasets/nonanswer_taxonomy.csv

# figures
python Memorization/viz/make_comparison.py --scores-dir datasets/scores_v2
```

Score directories: `scores/` is the original (superseded — pre-bugfix),
`scores_v2/` the corrected voluntary run, `scores_forced/` and `scores_merged/`
the forced-attempt arm.
