#!/usr/bin/env python3
"""Three checks that must pass before any logprob output is trusted.

  python validate.py --model meta-llama/Llama-3.1-8B

Every failure mode this catches is SILENT: the pass completes, the CSV looks
reasonable, and the numbers are wrong. Run it on the small model while writing,
and once more on the big model after it loads on the rented box.

  A  SAMPLING AGREEMENT   the real test of alignment. Take short suffixes whose
     computed p is in a measurable range, sample the model many times, and check
     the empirical hit rate against p. An off-by-one or a boundary-merge error
     fails this immediately; nothing else does.
  B  KNOWN ANSWER         the Constitution should approach p_geomean 1.0 and
     scrambled Gatsby should sit near the floor. The black-box run already puts
     these at 19.9 and 1.0 tokens of verbatim run, so the shape is known.
  C  BATCH INVARIANCE     rescore one batch at batch_size=1 and require the
     per-token logprobs to match. Catches padding/mask bugs, which are otherwise
     invisible because padded positions still produce logits.
"""
from __future__ import annotations
import argparse, math, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairs import load_pairs, CONTROLS
from score_logprobs import locate_suffix, build_prefix_text


def token_logprobs(model, tok, ids, k, dev, batch=None):
    """Per-token logprobs of the suffix span, unbatched reference path.

    Passes an explicit all-ones attention_mask. Without it transformers warns,
    and more importantly the two paths would then differ in TWO ways (batching
    and mask) when check C is trying to isolate one.
    """
    import torch
    t = torch.tensor([ids], device=dev)
    am = torch.ones_like(t)
    with torch.inference_mode():
        logits = model(input_ids=t, attention_mask=am).logits
        lp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tl = lp.gather(2, t[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tl[0, k - 1: len(ids) - 1].float().cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--samples", type=int, default=400,
                    help="draws per pair in check A; 400 gives a tight enough "
                         "binomial CI to catch a misalignment")
    ap.add_argument("--skip-a", action="store_true",
                    help="check A samples the model and is the slow one")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map=args.device_map)
    model.eval()
    dev = next(model.parameters()).device
    pairs = load_pairs(CONTROLS)
    fails = []

    # ---------------------------------------------------------------- C first
    # Cheapest, and a padding bug would corrupt A and B too.
    print("CHECK C — batch invariance (batched vs batch_size=1)")
    sample = pairs[:6]
    prepped = []
    for p in sample:
        pre = build_prefix_text(p, "raw", tok)
        ids, k, _ = locate_suffix(tok, pre, pre + " " + p["suffix"])
        prepped.append((p, ids, k))
    L = max(len(t[1]) for t in prepped)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ii = torch.full((len(prepped), L), pad, dtype=torch.long)
    am = torch.zeros((len(prepped), L), dtype=torch.long)
    for b, (_, ids, _) in enumerate(prepped):
        ii[b, :len(ids)] = torch.tensor(ids); am[b, :len(ids)] = 1
    ii, am = ii.to(dev), am.to(dev)
    with torch.inference_mode():
        lg = model(input_ids=ii, attention_mask=am).logits
        lp = torch.log_softmax(lg[:, :-1, :].float(), dim=-1)
        tl = lp.gather(2, ii[:, 1:].unsqueeze(-1)).squeeze(-1).float().cpu()
    # A misalignment DECORRELATES the two vectors; float noise does not. So the
    # verdict is correlation, with the magnitude reported alongside for context.
    # Batched matmuls take different kernel paths than batch-of-1, and in
    # fp16/bf16 that moves individual logprobs by ~0.01-0.05 -- an earlier
    # version of this check used a flat 1e-2 tolerance and failed on that noise
    # alone. Run --dtype float32 on a small model if you want a tight bound.
    tol = {"float32": 1e-3}.get(args.dtype, 0.08)
    worst, allg, allr = 0.0, [], []
    for b, (p, ids, k) in enumerate(prepped):
        got = tl[b, k - 1: len(ids) - 1].tolist()
        ref = token_logprobs(model, tok, ids, k, dev)
        worst = max(worst, max(abs(a - c) for a, c in zip(got, ref)))
        allg += got; allr += ref
    mg, mr = statistics.mean(allg), statistics.mean(allr)
    num = sum((a - mg) * (c - mr) for a, c in zip(allg, allr))
    den = (sum((a - mg) ** 2 for a in allg) * sum((c - mr) ** 2 for c in allr)) ** .5
    corr = num / den if den else 0.0
    mean_abs = statistics.mean(abs(a - c) for a, c in zip(allg, allr))
    ok = corr > 0.999 and worst < tol
    print(f"   corr(batched, unbatched) = {corr:.6f}   "
          f"mean|diff| = {mean_abs:.2e}   max|diff| = {worst:.2e}")
    print(f"   tolerance for dtype={args.dtype}: {tol:.0e}   {'PASS' if ok else 'FAIL'}")
    if corr <= 0.999:
        fails.append(f"C: vectors decorrelated (r={corr:.4f}) — this is a real "
                     f"alignment or mask bug, not float noise")
    elif worst >= tol:
        print(f"   NOTE: correlation is fine, so the span is aligned; the spread "
              f"is {args.dtype} kernel noise. Re-run --dtype float32 to confirm.")
        fails.append(f"C: max|diff| {worst:.2e} exceeds {tol:.0e} at dtype="
                     f"{args.dtype} (likely numerical — see note)")

    # ---------------------------------------------------------------- B
    print("\nCHECK B — known answers (p_geomean by corpus)")
    got = {}
    for work in ("constitution", "gatsby", "gatsby_shuf"):
        sub = [p for p in pairs if p["work"] == work][:12]
        vals = []
        for p in sub:
            pre = build_prefix_text(p, "raw", tok)
            ids, k, _ = locate_suffix(tok, pre, pre + " " + p["suffix"])
            v = token_logprobs(model, tok, ids, k, dev)
            if v:
                vals.append(math.exp(sum(v) / len(v)))
        got[work] = statistics.mean(vals) if vals else 0.0
        print(f"   {work:14s} p_geomean = {got[work]:.4f}  (n={len(vals)})")
    if not (got["constitution"] > got["gatsby_shuf"]):
        fails.append("B: Constitution does not beat scrambled Gatsby — "
                     "alignment or model loading is wrong")
    else:
        print(f"   ordering constitution > gatsby_shuf   PASS")

    # ---------------------------------------------------------------- A
    if args.skip_a:
        print("\nCHECK A — skipped")
    else:
        print(f"\nCHECK A — sampling agreement ({args.samples} draws/pair)")
        # Find a TESTABLE span per pair, do not assume one exists.
        #
        # Every suffix in controls_qa is 25-45 tokens, so a whole-suffix p is
        # ~1e-20 and no sampling experiment can measure it. Instead walk the
        # cumulative logprob and take the LONGEST prefix of the suffix whose p is
        # still above P_MIN. That is a real test of the alignment on a span the
        # sampler can actually hit.
        #
        # An earlier version fell back to "first pair, 4 tokens" with no range
        # check, which produced p~1e-8, empirical 0, and a PASS that verified
        # nothing. A vacuous pass is worse than a failure, so an unmeasurable
        # pair is now reported INCONCLUSIVE and counted as a failure.
        # P_MIN is set by POWER, not by convenience. The pass condition is
        # |p̂ − p| ≤ 4·se + 0.01, and a fully misaligned run yields p̂ ≈ 0, so
        # the test only has teeth when p itself EXCEEDS that tolerance. At
        # n=400 that needs p ≳ 0.1: at p=0.05 the tolerance is 0.054, and a
        # totally broken implementation would pass. Raise --samples to lower
        # this floor (tolerance shrinks as 1/√n).
        P_MIN, P_MAX = 0.10, 0.6
        cands = []
        for p in pairs:
            pre = build_prefix_text(p, "raw", tok)
            ids, k, _ = locate_suffix(tok, pre, pre + " " + p["suffix"])
            v = token_logprobs(model, tok, ids, k, dev)
            if len(v) < 2:
                continue
            run, best = 0.0, None
            for n in range(1, len(v) + 1):
                run += v[n - 1]
                pr = math.exp(run)
                if pr < P_MIN:
                    break
                if pr <= P_MAX and n >= 2:
                    best = (n, pr)          # keep extending while measurable
            if best:
                n, pr = best
                cands.append((p, ids[:k + n], k, pr, n))
            if len(cands) >= 3:
                break
        if not cands:
            print("   INCONCLUSIVE — no pair has a 2+ token span with "
                  f"p in [{P_MIN}, {P_MAX}]. The model assigns very low "
                  "probability to every continuation here, so sampling cannot "
                  "validate the alignment. Try a stronger model, or a corpus "
                  "the model actually knows (constitution).")
            fails.append("A: inconclusive — no measurable span, alignment UNVERIFIED")
        for p, ids, k, pr, n in cands:
            prompt = torch.tensor([ids[:k]], device=dev)
            target = ids[k:]
            hits = 0
            with torch.inference_mode():
                for _ in range(0, args.samples, 50):
                    b = min(50, args.samples - _)
                    inp = prompt.repeat(b, 1)
                    out = model.generate(inp, attention_mask=torch.ones_like(inp),
                                         do_sample=True,
                                         temperature=1.0, top_p=1.0, top_k=0,
                                         max_new_tokens=n,
                                         pad_token_id=pad)
                    for row in out[:, k:k + n].tolist():
                        hits += int(row == target)
            emp = hits / args.samples
            se = math.sqrt(max(pr * (1 - pr), 1e-12) / args.samples)
            tol = 4 * se + 0.01
            # A test that cannot distinguish "correct" from "p̂=0" is not a test.
            powered = pr > tol
            ok = abs(emp - pr) <= tol and powered
            print(f"   {p['case_id'][:26]:26s} {p['qid']}  n={n:2d}  "
                  f"p={pr:.4g}  empirical={emp:.4g}  (hits {hits}/{args.samples})  "
                  f"tol={tol:.4f}  {'PASS' if ok else 'FAIL'}"
                  + ("" if powered else "  <- UNDERPOWERED, p < tol"))
            if not powered:
                fails.append(f"A: {p['case_id']} {p['qid']} underpowered "
                             f"(p={pr:.4g} < tol={tol:.4f}); raise --samples")
            elif not ok:
                fails.append(f"A: {p['case_id']} {p['qid']} p={pr:.4g} vs "
                             f"empirical={emp:.4g} — MISALIGNMENT")

    print("\n" + ("ALL CHECKS PASSED" if not fails else "FAILURES:"))
    for f in fails:
        print(f"   {f}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
