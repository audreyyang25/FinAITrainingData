<!--
EXAMPLE PAPER — generated as a template from the Data_Analysis pipeline.
Structure follows the NeurIPS format; drop the sections into the official
neurips_2024.sty LaTeX template to submit. Numbers are REAL outputs of the
pipeline in this repo (a 400-case pilot, single Llama extractor/judge).
References are illustrative and MUST be verified/replaced before use.
-->

# Reasoning Fingerprints: How LLM Families Differ Beyond Answer Correctness on Securities-Suitability Compliance

**Anonymous Author(s)**
Affiliation withheld for double-blind review

---

## Abstract

Large language models are increasingly deployed on high-stakes legal and compliance
reasoning, yet evaluation still centers on whether the final answer is *correct*. We
argue that in regulated domains *how* a model reasons—which factors it engages, which it
omits—matters as much as the verdict. Using 400 U.S. securities-suitability cases spanning
five task formats, we evaluate a grid of 12 models (four developer families × three
capability tiers) along three complementary axes: (1) **answer quality** against reference
answers via an LLM judge; (2) **feature coverage**, the fraction of a case's reasoning
factors a model engages, measured by an *independent* extractor rather than model
self-report; and (3) **feature distribution**, each model's engagement profile over a fixed
20-criterion legal codebook. We find that final-verdict correctness is uniformly high and
does *not* separate models (outcome-match rate 0.89–0.97 across all 12), while calibrated
quality and reasoning breadth track capability tier. Most notably, models cluster by
*developer family* in reasoning space: a model's nearest neighbour in Jensen–Shannon
divergence is a same-family model 75% of the time (permutation \(p<0.001\)), and this
structure **survives** collapsing the open extraction vocabulary onto a fixed, human-curated
codebook—the across/within-family divergence ratio *increases* from 1.25 to 1.69—indicating
the clustering reflects reasoning *substance*, not phrasing. We release the full pipeline.

---

## 1  Introduction

Suitability and "best-interest" obligations (FINRA Rule 2111; SEC Regulation Best Interest)
require a financial professional to reason about a recommendation across many dimensions—a
client's risk tolerance, liquidity needs, time horizon, the product's costs and mechanics,
conflicts of interest, and more. As LLMs are piloted for compliance triage and drafting, a
natural evaluation is: *does the model reach the right conclusion?* We show this question is
necessary but far from sufficient. On our benchmark, essentially every model reaches the
correct violation/no-violation verdict; the models differ instead in the **breadth and
composition of the reasoning** that supports it—exactly the part a compliance reviewer must
audit.

We make three contributions:

1. **A three-axis evaluation** of legal-compliance reasoning that separates *verdict
   correctness* (Part 1) from *reasoning coverage* (Part 2) and *reasoning composition*
   (Part 3), on a shared set of model generations.
2. **An independent-extractor coverage metric.** Rather than ask a model to self-report the
   factors it used (which conflates reasoning with self-description), a neutral third model
   extracts factors from every answer, which are canonicalized into a per-case factor
   *superset*; coverage is a model's share of that superset.
3. **Evidence that model families have reasoning fingerprints.** In divergence space, models
   cluster by developer family, and the clustering is robust to vocabulary coarsening and to
   swapping the open extraction for a fixed 20-criterion codebook—arguing the signal is
   substantive rather than an artifact of phrasing or extraction noise.

## 2  Related Work

**LLM-as-a-judge.** Using a strong model to score free-form answers is now standard for
open-ended tasks [Zheng et al., 2023]. We adopt it for Part 1 but treat its calibrated
dimensions as judge-dependent and its binary outcome dimension as judge-robust, and we report
agreement across judges. **Reasoning evaluation** typically scores rationales for correctness
or faithfulness; we instead measure *which* factors are engaged, independent of correctness.
**Model fingerprinting / behavioural similarity** has examined output-distribution or
representational similarity; we characterize similarity in a *human-interpretable* reasoning-
factor space and test whether developer family is recoverable. *(References are illustrative
in this template and should be completed for submission.)*

## 3  Experimental Setup

**Data.** Five securities-suitability datasets, each mapped to a common
`(prompt, reference)` form by a dataset adapter: `standard` (fact pattern + question),
`borderline` (contested cases with a documented likely outcome), `conversations`
(client–advisor transcripts), `redflags` (single-issue fact patterns), and `adversarial`
(questions engineered to elicit an incorrect verdict). We use the first 100 cases of each
(400 total, as two sets contain only 50).

**Model grid.** Four developer families × three capability tiers = 12 models (Table 1),
all routed through a single OpenAI-compatible endpoint. Each model answers every case;
the dataset's reference answer participates as a 13th source, `gold`.

**Generation.** Answers are produced *answer-only* (no self-reported factors), at temperature
0, so all three analyses run on one shared generations file.

| Family | Frontier | Mid | Small |
|---|---|---|---|
| Anthropic | claude-opus-4.8 | claude-sonnet-4.6 | claude-haiku-4.5 |
| OpenAI | gpt-5.5 | gpt-5.4 | gpt-5.4-mini |
| Gemini | gemini-3.1-pro | gemini-3.5-flash | gemini-3.1-flash-lite |
| Qwen | qwen3.5-397b | qwen3.6-35b | qwen3.5-9b |

*Table 1: The 4×3 model grid. Model versions are those configured in the released pipeline.*

## 4  Part 1 — Answer Quality Against Gold

**Method.** A single judge (Llama-3.3-70B, temperature 0, provider-pinned for
reproducibility) scores each answer against its dataset's reference along dimensions declared
per dataset: **content similarity** (holistic overlap with the reference reasoning),
**factor recall** (for datasets with an enumerated factor list), **outcome** (an *exact*
violation/no-violation match for datasets with clean ground truth, or *directional*
consistency otherwise), and **citation extraction** (authorities cited but not on a
whitelist, a proxy for fabricated citations). Every binary judgment requires a quoted span.

**Findings (Table 2).**
- **Verdict correctness does not separate models.** Exact outcome-match rate is 0.89–0.97 for
  all 12 models; directional consistency is ≥0.95 everywhere. On this axis alone the models
  are indistinguishable.
- **Calibrated quality tracks capability tier.** Content similarity and factor recall order
  the models by tier: frontier Anthropic/OpenAI lead (content ≈0.88, factor recall ≈0.97),
  small open-weight models trail (qwen3.5-9b: 0.67 / 0.77).
- **Unverified citations are a family trait.** Anthropic models emit far more off-whitelist
  authorities per response (sonnet-4.6 ≈4.0, opus-4.8 ≈2.1) than Gemini-lite or Qwen
  (≈0.3–0.5)—a fabrication-risk signal invisible to the outcome metric.

| Model | Content sim | Factor recall | Outcome-exact | Unverified cites/resp |
|---|---|---|---|---|
| claude-opus-4.8 | 0.888 | 0.970 | 0.899 | 2.06 |
| claude-sonnet-4.6 | 0.829 | 0.890 | 0.945 | 4.02 |
| claude-haiku-4.5 | 0.836 | 0.962 | 0.935 | 0.95 |
| gpt-5.5 | 0.872 | 0.966 | 0.893 | 1.30 |
| gpt-5.4 | 0.879 | 0.966 | 0.887 | 1.12 |
| gpt-5.4-mini | 0.763 | 0.891 | 0.903 | 0.61 |
| gemini-3.1-pro | 0.810 | 0.938 | 0.899 | 1.19 |
| gemini-3.5-flash | 0.852 | 0.961 | 0.925 | 1.66 |
| gemini-3.1-flash-lite | 0.743 | 0.804 | 0.954 | 0.27 |
| qwen3.5-397b | 0.706 | 0.775 | 0.970 | 0.38 |
| qwen3.6-35b | 0.783 | 0.896 | 0.933 | 0.46 |
| qwen3.5-9b | 0.673 | 0.766 | 0.954 | 0.29 |

*Table 2: Quality metrics, averaged over datasets. Outcome-exact is high and flat; content
similarity and factor recall track tier; unverified citations vary sharply by family.*

## 5  Part 2 — Reasoning Coverage via Independent Extraction

**Extraction and canonicalization.** For every (case, model) answer, an independent extractor
*E* (Llama-3.3-70B, pinned to a single bf16 provider so serving precision is constant across
calls) returns the reasoning factors the answer engages, as short natural-language phrases.
Extraction is deliberately *not* self-report: a model asked to enumerate its own factors
conflates reasoning with the willingness and ability to describe it, and stronger models
describe more. A separate high-reasoning arbiter *A* (Claude Opus 4.8) then canonicalizes in
two passes. *(i) Within each case*, it merges paraphrases of the same factor across all 13
sources into a deduplicated **superset** \(S_c\), tagging each raw factor with the superset
slot it maps to (or null if discarded). *(ii) Across cases*, it folds the per-case supersets
into a single **global vocabulary** *G* of 556 factors with a mapping phrase → *G*, so every
model on every case lives on a common alphabet.

**Coverage.** For case *c* and model *j*, coverage = (slots of \(S_c\) engaged by *j*) / |\(S_c\)|;
we report per-model means over cases. Because \(S_c\) is the union over all sources (gold
included), coverage is a relative **breadth** measure, independent of the verdict.

**Reasoning profiles.** Each model *j* is a distribution \(p_j\) over *G*, where \(p_j(f)\) is
the fraction of cases in which *j* engaged global factor *f* (**selection frequency**),
normalized to sum to one. We discard the extractor's importance weights and use selection only,
so no result depends on a weaker model's numeric scoring. Dissimilarity is the Jensen–Shannon
divergence \(D_{jk}=\mathrm{JSD}(p_j,p_k)\) in bits [Lin, 1991]. At *n*=400 cases and |*G*|=556,
absolute JSD is upward-biased—two halves of the *same* distribution already score well above
zero—so we never read magnitudes: every claim is a **ranking** on a shared case set (where the
common sampling bias cancels), checked against a null. We run four such tests.

**(1) Is the geometry real? (replication).** Split the cases into two disjoint halves, build the
full 13×13 divergence matrix on each, and correlate their upper triangles, averaging over 200
random splits. Signal reproduces across independent data; noise does not. The null shuffles one
half's model labels before correlating, preserving the distance *distribution* while destroying
the correspondence.

**(2) Verbosity or family? (the verbosity-gap regression).** Same-family models might look alike
only because they are equally *verbose*: models differ in factors engaged per case (\(v_j\), the
mean distinct factors/case), and verbosity itself clusters by family. Residualizing profiles
against verbosity would be wrong twice over—it changes the metric (residuals aren't
distributions, so JSD is undefined) and it controls away a *mediator*, deleting the family signal
we are testing. Instead we regress the \(\binom{12}{2}=66\) contestant-pair divergences on the
absolute verbosity gap and a same-family indicator,

\[ D_{jk} = \beta_0 + \beta_1\,|v_j - v_k| + \beta_2\,\mathbf{1}[\text{same family}] + \varepsilon, \]

and ask whether family explains variance *beyond* verbosity. We report the incremental \(R^2\)
of the family term and a permutation *p* from shuffling the 12 family labels and refitting
(20,000 draws).

**(3) Is family recoverable? (nearest-neighbour test + edge stability).** Holding `gold` out,
take each of the 12 contestant models' nearest neighbour in *D*; **family accuracy** is the
fraction whose nearest neighbour shares its developer family. The chance baseline is the mean
over models of \((n_{\mathrm{fam}(j)}-1)/(12-1)\)—the probability a random other model matches
under the observed family sizes—and significance is a permutation null over the 12 family labels
(20,000 draws of leave-one-out NN accuracy). Because a "hit" on a pair separated by a hair is
luck, we also **bootstrap edge stability**: resampling cases with replacement, we recompute each
model's nearest neighbour and report the fraction of bootstraps in which the observed edge
recurs, flagging edges below 65% as coin-flips rather than evidence.

**(4) Which model reasons like gold?** Treating `gold` as a 13th profile, rank the contestants
by JSD to gold, estimate uncertainty with a case bootstrap (2,000 resamples), and report for each
model the fraction of bootstraps in which it is gold's nearest neighbour (aggregated to a
per-family win fraction). The bootstrap plug-in JSD is biased upward near-uniformly across
models, so we display bootstrap *spread* (±1.96 sd) rather than a percentile interval and let the
bias-invariant win counts carry the inference.

**Missing data.** Generation fails on a few cases for some models; a model estimated from fewer
cases has a sparser profile and an inflated JSD to *everyone*—a systematic bias that replicates
across splits and so survives the replication test. A `--complete-cases` option restricts every
model to the intersection of cases where all generators succeeded, confirming the family results
are not an artifact of differential missingness.

**Results.** *Coverage:* frontier-heavy families cover more of the factor superset (Anthropic
0.394, OpenAI 0.383) than Gemini (0.317) or Qwen (0.313); `gold` sits mid-pack (0.374), i.e.
models frequently raise legitimate factors the reference omits (also surfaced as a per-case
audit list). *Geometry* (all statistics recorded in `stats_freq.json`): (1) the geometry
**replicates**, \(r=0.875\) vs. label-shuffled null 0.003 (beats the null in 100% of splits);
(2) **family, not verbosity**—verbosity alone explains \(R^2=0.045\) while same-family adds
\(R^2=0.152\) with \(\beta_2<0\) (same-family pairs closer), permutation \(p=0.0009\); (3)
**family is recoverable**—leave-one-out NN family accuracy is **75%** (9/12) vs. chance 18%,
permutation \(p=0.0006\), with within-family edges ≥76% bootstrap-stable and the two cross-family
misses ~50% coin-flips; (4) `gold` reasons most like Anthropic—its nearest neighbour is an
Anthropic model in 85.7% of bootstraps (closest single models sonnet-4.6, opus-4.8).

## 6  Part 3 — Reasoning Distribution over a Fixed Codebook

**Method.** To rule out that Part 2's clustering is a vocabulary artifact—families phrasing the
same concept differently, split into separate global factors—we re-score every answer against
a fixed **20-criterion legal codebook** (Appendix A) curated to be roughly mutually exclusive
(e.g. *risk alignment*, *concentration*, *conflicts of interest*, *standard of care*,
*quantitative suitability*, *senior vulnerability*). One labeler marks each criterion
*present* if the reasoning is substantively engaged—argued through the facts, not merely named.
Each model becomes a 20-dimensional presence-rate distribution; the same JSD machinery applies.

**Family tilts.** Relative to the four-family mean, families over- and under-emphasize
different criteria by a few points of engagement each; ordering criteria by cross-family
spread yields an interpretable "tilt" signature per family (and per model), consistent across
a family's three tiers more often than not.

**Clustering is substance, not vocabulary.** On the clean 20-axis codebook the across/within-
family divergence ratio is **1.69** (permutation \(p=0.011\)). Comparing granularities from
noisy to clean—556 open-vocabulary factors (ratio **1.25**, \(p<0.001\)) versus the fixed
20-criterion codebook (**1.69**, \(p=0.011\))—the ratio *grows* as the axes are cleaned
(Figure 1). If family clustering were a phrasing artifact, the fixed codebook would wash it
out; instead it sharpens, indicating the noise was *masking* a substantive family signal.

*Figure 1: across/within-family JSD ratio rises from the open vocabulary (556 factors, 1.25)
to the fixed 20-criterion codebook (1.69). Ratio \(>1\) = families diverge more than tiers
within a family. (See `outputs/part3_distribution/figures/separation_ratio.png`.)*

## 7  Discussion

Three patterns cut across the axes. **First, correctness saturates**: on clean-ground-truth
cases every model reaches the right verdict, so a verdict-only benchmark would call these
models equivalent. **Second, capability tier lives in the reasoning, not the verdict**:
frontier models cover more factors and match the reference's reasoning more closely, while
small models reach the same verdict on a thinner rationale—precisely the failure mode a
compliance reviewer cares about. **Third, developer family is a first-class axis of
variation**: models from the same lab reason alike—engaging and weighting legal factors
similarly—more than models of the same size across labs, and this holds on a fixed, human-
curated codebook. Practically, this cautions against treating "an LLM" as interchangeable in a
compliance workflow: model choice imports a house reasoning style, including which factors tend
to be under-weighted and how freely authorities are cited.

## 8  Limitations

Results are a **400-case pilot** from single-run generations. Extraction and rubric scoring use
a **single** labeler (Llama-3.3-70B); the calibrated dimensions (content similarity, factor
recall, extraction) shift with the labeler even where the outcome metric does not, so
family-level claims should be replicated across labelers (the pipeline supports multiple
extractors/judges). Absolute JSD is biased at this sample size—hence the reliance on rankings
and permutation nulls. The canonicalization arbiter is itself an Anthropic model, a possible
source of mild Anthropic-favouring vocabulary bias, which Part 3's model-free codebook is
designed to neutralize. Finally, roughly 0.1–2.5% of extractor/judge calls fail on malformed
JSON and are dropped; the losses are labeler-side and approximately random across models.

## 9  Conclusion

Verdict correctness is a weak lens on LLM legal-compliance reasoning: it saturates and hides
the differences that matter. Measuring *reasoning coverage* and *reasoning composition* on a
shared benchmark reveals that capability tier and—strikingly—developer family are legible in
how models reason, not just what they conclude. We release the three-axis pipeline to support
reasoning-level evaluation in regulated domains.

---

## References

*Illustrative — verify and complete before submission.*

[1] L. Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS*, 2023.
[2] J. Lin. "Divergence measures based on the Shannon entropy." *IEEE Trans. Information Theory*, 37(1), 1991.
[3] FINRA Rule 2111 (Suitability); U.S. SEC Regulation Best Interest (Reg BI), 17 CFR §240.15l-1.
[4] *(placeholder)* Survey on reasoning-faithfulness evaluation of LLMs.
[5] *(placeholder)* Behavioural / representational model-similarity methods.

## Appendix A — The 20-criterion codebook (abbreviated)

1 Risk alignment of product with client profile · 2 Concentration / position size ·
3 Time horizon & duration fit · 4 Conflicts of interest & disclosure · 5 Client objectives /
risk tolerance / KYC · 6 Affordability / capacity to bear loss · 7 Adequacy of information
conveyed · 8 Applicable standard of care · 9 Fees, costs & cost–benefit · 10 Client
understanding & informed consent · 11 Affirmative misstatement / misleading conduct ·
12 Investment experience & sophistication · 13 Reasonable alternatives · 14 Liquidity needs &
emergency reserves · 15 Client direction / unsolicited trades · 16 Quantitative suitability /
excessive trading · 17 Reasonable basis & due diligence · 18 Product characteristics &
complexity · 19 Firm supervision & recordkeeping · 20 Senior status & diminished capacity.

## Appendix B — Reproducibility

The full pipeline (generation → three parts) is released. Every stage is resumable and keyed
by `(dataset, case_id, model)`; all analyses run from a single `generations.json`. See the
repository `README` for exact commands.
