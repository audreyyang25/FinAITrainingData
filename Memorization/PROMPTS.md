# Prompt changelog

Every system prompt used in the memorization study, in the order it was run, with
what it replaced and why. Two experiments share this file because they share a
corpus and a failure mode — a prompt that lets a model decline turns the
post-cutoff arm into a self-selected subset — and reading them side by side is
the only way to see that the same lesson was learned twice.

Dates are **run** dates, taken from the `ts` column of the output files rather
than from commits, because several prompts were written and revised inside one
commit.

**Status vocabulary**

| | |
|---|---|
| **current** | in the code now, data on disk is from this version |
| **superseded** | replaced; its data was regenerated or discarded |
| **planned** | agreed, not yet run |

> **Note.** `METHODS.md` §4.1 quotes **V1**, not the current V4. It was written
> when V1 was current and has not been updated. Treat this file as authoritative
> for prompt text; treat METHODS.md as authoritative for corpus and scoring.

---

## Experiment 1 — verbatim recall

Does the model reproduce the opinion's exact wording? Scored by character ratio
and token F1 against the gold continuation.

### V1 — voluntary, with an `UNKNOWN` escape hatch · superseded

Quoted in full at `METHODS.md` §4.1.

```
Reply with the answer alone — no preamble, no explanation, no hedging …

If you do not recall the specific opinion well enough to reproduce its actual
wording, reply with exactly: UNKNOWN
```

The escape hatch was there to stop confabulation: an invented plausible answer
scores as partial recall, which is worse than a blank.

**Why it failed.** It destroyed the post-cutoff floor. A model cannot have
memorized an opinion filed after its cutoff, so whatever it scores there is what
verbatim text it can produce knowing nothing — but that only works if every
post-cutoff row yields a scoreable guess. Models declined *precisely* on
post-cutoff rows. **Claude Opus 4 answered 15 of 656 items.** The arm collapsed
to a self-selected handful and stopped being a floor.

### V2 — forced-attempt re-ask · superseded

Every row declined under V1 was re-asked with the escape hatch removed:

```
You must always produce an answer. Never reply UNKNOWN […] If you do not
remember the exact wording, give your single best guess at what the original
words are — guess the actual wording rather than writing a paraphrase or a
summary, and guess at the required length.
```

A patch, not a redesign: V1 and V2 together produced three populations —
**voluntary**, **forced**, and **merged** — separated by a `prompt_mode` column.
Gemini was never re-asked; it abstained on ~11% of items, so its voluntary number
was already near-unbiased.

**Why it was replaced.** Two prompts over one item set means two populations to
carry through every downstream figure, and the merged population still mixes them.

### V3 — compulsory answer, `RECALL: yes|no` asked *first* · superseded

Collapsed V1 and V2 into one compulsory prompt, and recovered the abstention
signal as a separate self-report field placed **before** the answer, asking
whether the model "recognized this opinion and remembered its wording".

**Why it failed** (recorded at `probe/run_probe.py:75`). **GPT-5 answered `no` on
10 of 10 pre-cutoff rows** — including ones where it then reproduced a plausible
27-word continuation — and its single `yes` was on a post-cutoff case it cannot
have seen. Two causes:

- **Order.** Asked before answering, the model reports a *prior* about whether it
  will succeed, not an observation about what it produced. It hasn't tried yet.
- **Framing.** "Do you remember the wording" is introspective, and models have
  poor access to whether their own output came from memorization or from fluent
  pattern completion. It also reads as an overclaim, which RLHF pushes against,
  making `no` the safe token regardless of capability.

### V4 — compulsory answer, `SOURCE: recalled|inferred` *after* · **current**

`probe/run_probe.py:33`. Fixes both V3 causes: the field moves after the answer,
making it a posterior, and it asks about the **provenance of text already on the
page** — a choice between two concrete alternatives rather than a confidence
confession.

```
ANSWER: <the continuation>
SOURCE: recalled   (the words you just wrote are ones you have actually seen in
                    this document)
SOURCE: inferred   (you reconstructed them from how documents like this usually
                    read)

The ANSWER line is MANDATORY on every reply, including when SOURCE is inferred.
```

The column stayed named `recall` so the scorer and score CSVs didn't churn; the
*values* are now `recalled` / `inferred` / `''`. `yes`/`no` are still accepted as
aliases so V3 rows parse.

**Runs:** 2026-08-12 → 2026-08-16, 7 models, `datasets/predictions/`
(1,968 court rows + 202 control rows each). No `prompt_mode` column — one
population, no selection to correct for.

---

## Experiment 2 — outcome knowledge

Does the model know what the court *held*? Semantic rather than verbatim, and
obtainable from a summary without ever seeing the opinion — which is the point of
running both.

### O1 — recall, `UNKNOWN` permitted · current (as the recall arm)

`outcome/run_outcome.py:42`. Asks for two things separately, because appellate
courts affirm 75–80% of the time so "affirmed" alone is guessable, while the
court's *rationale* is not.

```
1. THE OUTCOME — the court's disposition … and which party prevailed.
2. THE REASONING — the court's principal rationale …

If you do not know this specific case, you may reply with: UNKNOWN
If you are unsure, you can state your lack of certainty but still give it your
best attempt.
```

**Runs:** 2026-08-07, 6 models, `datasets/outcomes/`. Llama 3.1 70B was never run
under O1, so it has no recall arm.

**The V1 lesson repeated.** Decline rates ran 4–93%. **Claude Opus 4 attempted 9
of 93 pre-cutoff cases and 0 of 43 post-cutoff**, replying `UNKNOWN` verbatim 89
times. Its pre-cutoff scores are over the nine most famous cases in the corpus.
Post-cutoff, all three newer models declined 32 of 32 common-arm cases.

### O2 — forecast, declining forbidden · current

`outcome/run_outcome.py:67`. Built from O1 **by dropping its final paragraph**, so
the task definition, the two-part output shape and the "what the court actually
held" instruction stay byte-identical. Only the instruction about not knowing
changes, which is the whole manipulation.

```
You will not recognize many of these cases: some were decided after your
training data ends. Do not reply UNKNOWN and do not decline. If you do not
remember the case, or doubt it exists, predict what the court would most likely
have held and why — reason from the caption, the court, the procedural posture,
and how cases of this kind usually come out.
```

**Runs:** started 2026-08-13, completed 2026-08-18 (interrupted mid-run by an
OpenRouter credit exhaustion), 7 models, `datasets/outcomes_predict/`.

**Known limitation.** 110 of 952 rows still came back declined despite the
instruction. And because O1 and O2 differ in *both* arm and prompt, the pre/post
gap in figure `o4` is an upper bound, not a clean estimate — which is what O3
below exists to fix.

### O3 — one prompt for both arms · **planned, not yet run**

Make the pre-cutoff and post-cutoff arms differ *only* by arm. Today the pre bars
come from O1 (self-selected attempts, `UNKNOWN` allowed) and the post bars from O2
(everything, declining forbidden), so selection flatters the pre side.

Paired with the corpus restrictions adopted 2026-08-24 — federal courts of appeals
only, and `models.common_arm()` excluding cases between the earliest and latest
cutoff in the suite — this is the first design where a pre/post delta is
like-for-like across models.

Estimated **$80** for 3 models × 125 cases, generation plus judging. Requires
`--force` or a fresh `--out`, or old-prompt rows survive in the output files.

---

## Judge prompts

The judge grades a candidate answer against the full opinion text. Web search is
off: the judge already holds the ground truth, and retrieval would surface the
same secondary summaries the targets may have learned from.

### J1 — grading rubric, outcome and reasoning scored separately · current

`outcome/judge_outcome.py:55`. Five described anchors per axis rather than a free
0–1, because judges asked for "a number between 0 and 1" cluster at 0/0.5/0.8/1.0
and are unreliable between the modes. Anchoring makes it a classification task.

Outcome and reasoning are scored **independently and combined afterwards**: a
model that names the right disposition and invents the rationale is the signature
of knowledge acquired from a summary rather than from the opinion, and a single
blended score would hide it.

**Runs:** 2026-08-07 (recall, 815 rows); 2026-08-13 → 08-18 (predict, 952 rows).

### J2 — `SYSTEM_TRUTH`, disposition from the opinion alone · current

`outcome/judge_outcome.py:127`. Added 2026-08-15 to fix a contamination path.

`disposition_actual` is a fact about the opinion, but J1 re-derived it on every
call — six times per case, each time alongside a different candidate answer.
**16 of 136 cases came back with different "actual" labels depending on which
model was being graded.** Ground truth must not be able to drift toward the answer
being scored.

So it is determined **once per case, from the opinion, with no candidate in the
prompt** — 136 calls instead of 576, cheaper than what it replaced. There is
deliberately no catch-all category: `other` was retired because it merged granted
with denied, letting a model score a hit with the direction backwards.

### J3 — `SYSTEM_REGRADE`, grading against a fixed disposition · current

`outcome/judge_outcome.py:176`. Derived from J1 by replacing one paragraph.

J1's outcome anchors are phrased "disposition correct / partly correct", so the
judge must commit to a disposition in order to score — meaning a score carries the
judge's own reading of the label. When J2 disagreed, the two stopped matching: on
the 38 affected rows, corr(disposition hit, outcome_score) was **+0.435 against
the old label and −0.393 against the new one.** The sign flip is the
contamination, measured.

J3 supplies the established disposition and grades against it, so the judge scores
the candidate rather than re-litigating ground truth. Applied only to rows whose
label moved — regrading all 815 would have cost ~$126 to change nothing on the 90%
that never moved.

**Runs:** 2026-08-15 (recall, 38 rows); 2026-08-24 (predict, 64 rows).

---

## Timeline

| date | prompt | experiment | event |
|---|---|---|---|
| pre 08-07 | V1 | verbatim | voluntary prompt; post-cutoff floor collapses |
| pre 08-07 | V2 | verbatim | forced re-ask patches V1; three populations |
| pre 08-12 | V3 | verbatim | one compulsory prompt; `RECALL` asked first, fails |
| 08-07 | O1 | outcome | recall run, 6 models, `UNKNOWN` allowed |
| 08-07 | J1 | judge | recall run graded, 815 rows |
| 08-12→16 | V4 | verbatim | `SOURCE` after answer; 7 models rerun |
| 08-13→18 | O2 | outcome | forecast run, 7 models, 952 rows |
| 08-15 | J2 | judge | stable truth pass; 16 drifting labels fixed |
| 08-15 | J3 | judge | 38 recall rows regraded against fixed labels |
| 08-18 | O2 | outcome | forecast run completed after credit exhaustion |
| 08-24 | J2/J3 | judge | truth applied to forecast run; 64 rows regraded |
| 08-24 | — | design | federal appellate only; `common_arm` exclusion band |
| — | O3 | outcome | **planned:** one prompt across both arms |

---

## What has never been run

- **Llama 3.1 70B under O1.** It has a complete O2 forecast arm and no recall arm,
  so it appears in no outcome figure.
- **A judge other than Claude Opus 5 at full scale.** `models.JUDGE_ALT`
  (Gemini 3.1 Pro) is defined for a self-preference check on a subset — Claude is
  also a target family, so its verdicts carry a self-preference risk that is
  accepted rather than avoided. What matters is not absolute agreement but whether
  the Claude-vs-others gap moves when the judge changes family.
- **`--web` sensitivity check.** Supported by `judge_outcome.py`, never run.
