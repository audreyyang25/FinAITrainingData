# Teacher-forced logprobs (Cooper et al.'s metric)

Measures `p = ∏ P(yᵢ | x, y<ᵢ)` — the probability the model assigns to the *true*
continuation. This is the `(n,p)`-discoverable extraction metric from Cooper et
al., and the one `METHODS.md` §1 says is impossible for closed-weight models.
It needs local weights, so it runs on a rented GPU.

**Why it exists.** The black-box probe scored Llama 3.1 70B at **3.6 tokens** on
The Great Gatsby — but Gatsby is in the control set precisely *because* Cooper et
al. report that model substantially memorising it. Either the sampling proxy
misses memorisation the logprobs would show, or Llama-as-served doesn't carry the
text. These scripts tell the two apart.

---

## Run it

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # Llama weights are gated on HF
export HF_HOME=/workspace/hf   # put the cache on the big disk, not the boot volume
```

**1. Dry run — loads the tokenizer only, no weights and no GPU. Needs
`transformers` installed, but runs anywhere. Do this first.**

```bash
python score_logprobs.py --model meta-llama/Llama-3.1-8B --qa controls --dry-run
```

Prints the decoded prefix tail and suffix head for three pairs. If the suffix
head doesn't start where you expect, the alignment is wrong and nothing below
matters.

**2. Validate on the small model.** All three checks must pass.

```bash
python validate.py --model meta-llama/Llama-3.1-8B
```

**3. Score.** Base model, raw condition — this is the replication.

```bash
python score_logprobs.py --model meta-llama/Llama-3.1-70B --qa controls
python score_logprobs.py --model meta-llama/Llama-3.1-70B --qa court
```

Or `bash run_all.sh meta-llama/Llama-3.1-70B` for the full matrix
(base + instruct × controls + court).

**4. Figures — back on the laptop, no GPU.**

```bash
python make_curve.py \
  --logprobs "../../Data Collection and Training Material Generation/datasets/logprobs/logprobs__controls__*.csv" \
  --scores  "../../Data Collection and Training Material Generation/datasets/scores/scores_controls__meta-llama__llama-3.1-70b-instruct.csv"
```

---

## Sizing the box

| model | weights (bf16) | fits on |
|---|---|---|
| Llama-3.1-8B | ~16 GB | 1×24 GB (write and debug here) |
| Llama-3.1-70B | ~140 GB | 2×A100-80G or 2×H100 |

**The experiment is not the cost — the download is.** 2,170 pairs is ~15–30 min
of compute on 2×A100; pulling 140 GB of weights takes longer. Point `HF_HOME` at
a persistent volume if the provider offers one.

Everything is resumable on `(case_id, qid)`, so a killed pod costs only the rows
in flight. Re-issue the same command and it continues.

---

## Three decisions already made, and why

**Base, not Instruct.** Instruction tuning suppresses verbatim reproduction, and
the question is whether the model *carries* the text. Run Instruct too — the gap
between them is itself a result, and it's one of the two explanations for the 3.6.

**Raw text, not the chat template.** Cooper et al. score the plain
language-model probability of the continuation. `--condition chat` wraps the
prompt exactly as `probe/run_probe.py` does, which is the right comparison for
calibrating against the black-box run — but keep them as separate rows, never
averaged.

**No quantisation for the headline run.** Quantisation degrading verbatim recall
is one of the hypotheses being tested; measuring it with a quantised model would
confound the thing under test. Run fp8/int4 as a separate labelled condition if
you want it as a finding.

---

## Output

`datasets/logprobs/logprobs__<corpus>__<model>__<condition>.csv`, one row per
`(case_id, qid, trunc_n)`, joinable onto `datasets/scores/` on `(case_id, qid)`.

| column | |
|---|---|
| `trunc_n` | 10 / 25 / 50 / `full` — `logp_sum` is not comparable across different `n`, and these suffixes run 19–33 words, so the fixed lengths are what make the curve mean anything |
| `logp_sum` | Σ log P — the quantity thresholded for `(n,p)` |
| `logp_per_token`, `p_geomean` | comparable across rows of different length |
| `min_token_logprob`, `first_break_index` | memorised passages usually fail at one surprising token rather than degrading evenly; this says where |
| `align_warning` | `boundary_merge` when a BPE token spans the prefix/suffix seam. Span is still located by character offsets, but inspect a few before publishing |

---

## The traps this code already handles

1. **Off-by-one.** `logits[i]` predicts token `i+1`, so position `j` is read from
   `logits[j-1]`. Reversed, everything still runs and every number is wrong.
2. **Tokenizer boundary.** Prefix and suffix are never tokenised separately and
   concatenated — BPE merges across the seam. The span is found by character
   offsets and the prefix-prefix property is asserted.
3. **Padding.** Padded positions still produce logits. Check C in `validate.py`
   rescores a batch at `batch_size=1` and requires agreement.
4. **bf16 softmax.** `log_softmax` runs in float32; bf16 over a 128k vocab loses
   enough precision to matter once ~40 terms are summed.
