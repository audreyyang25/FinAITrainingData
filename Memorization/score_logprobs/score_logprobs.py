#!/usr/bin/env python3
"""Teacher-forced logprobs: p = prod P(y_i | x, y_<i) for a true continuation.

This is the metric Cooper et al. use for (n,p)-discoverable extraction, and the
one METHODS.md section 1 says is impossible for closed-weight models. It needs
local weights, so it runs on a rented GPU.

  python score_logprobs.py --model meta-llama/Llama-3.1-8B --qa controls --dry-run
  python score_logprobs.py --model meta-llama/Llama-3.1-70B --qa controls
  python score_logprobs.py --model meta-llama/Llama-3.1-70B --qa court --condition chat

ONE FORWARD PASS PER PAIR. Nothing is sampled and nothing is generated; the true
suffix is fed in alongside the prefix and the model's probability for each real
token is read off. Cost is therefore trivial next to downloading the weights.

TWO ALIGNMENT TRAPS, both of which fail silently and produce plausible numbers:

  1. OFF BY ONE. logits[i] predicts token i+1, so the logprob of the token at
     position j is read from logits[j-1]. Reverse it and every number is wrong
     but nothing raises.
  2. TOKENIZER BOUNDARY. Tokenising prefix and suffix separately and
     concatenating ids gives different tokens than tokenising the joined string,
     because BPE merges across the seam. The suffix span is therefore located by
     character offsets, and the prefix-prefix property is asserted.

Run validate.py before trusting any output. It catches both of the above plus
padding-mask bugs, which are the third silent failure mode.
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairs import load_pairs, corpus_name, CONTROLS, COURT

COLS = ["case_id", "qid", "work", "arm", "date_filed", "jurisdiction",
        "model", "condition", "trunc_n",
        "n_prefix_tokens", "n_suffix_tokens",
        "logp_sum", "logp_per_token", "p_geomean",
        "min_token_logprob", "n_tokens_below_1e3", "first_break_index",
        "align_warning"]

# The chat wrapper for --condition chat. Kept byte-identical to the SYSTEM in
# probe/run_probe.py so the two conditions differ only in scoring method, not in
# what the model was asked. --condition raw uses no wrapper at all, which is what
# Cooper et al. measure: the plain language-model probability of the text.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "probe"))
    from run_probe import SYSTEM as PROBE_SYSTEM, build_user as probe_build_user
except Exception:                                   # probe imports providers ->
    PROBE_SYSTEM, probe_build_user = None, None     # openai sdk; optional here


def resolve_qa(name):
    return {"controls": CONTROLS, "court": COURT}.get(name, name)


def build_prefix_text(pair, condition, tok):
    """The text the suffix is conditioned on."""
    if condition == "raw":
        return pair["prefix"]
    if PROBE_SYSTEM is None:
        sys.exit("--condition chat needs probe/run_probe.py importable "
                 "(pip install openai, or copy SYSTEM across)")
    msgs = [{"role": "system", "content": PROBE_SYSTEM},
            {"role": "user", "content": probe_build_user(
                {**pair, "question": "Continue the passage.",
                 "prompt_normalized": pair["prefix"], "caption": pair["case_id"]}, {})}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def locate_suffix(tok, prefix_text, full_text):
    """-> (input_ids, k, warning). k is the first token index of the suffix.

    Character offsets, not token-count arithmetic: a BPE merge spanning the seam
    would silently shift the span by one otherwise.
    """
    enc = tok(full_text, return_offsets_mapping=True, add_special_tokens=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    n_pref_chars = len(prefix_text)
    k = None
    for i, (a, b) in enumerate(offs):
        if a >= n_pref_chars and b > a:      # skip zero-width special tokens
            k = i
            break
    warn = ""
    if k is None:                            # suffix tokenised into nothing
        return ids, len(ids), "no_suffix_tokens"
    pref_ids = tok(prefix_text, add_special_tokens=True)["input_ids"]
    if ids[:k] != pref_ids:
        # Recoverable: the span is still right by offsets, but the boundary
        # merged, so flag the row rather than dropping or silently trusting it.
        warn = "boundary_merge"
    return ids, k, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF id. Prefer the BASE model: instruction tuning "
                         "suppresses verbatim reproduction, which is the thing "
                         "being measured.")
    ap.add_argument("--qa", default="controls",
                    help="'controls', 'court', or a path to a QA csv")
    ap.add_argument("--out", help="default: logprobs__<corpus>__<model>__<cond>.csv")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Data Collection and Training Material Generation", "datasets", "logprobs"))
    ap.add_argument("--condition", default="raw", choices=["raw", "chat"])
    ap.add_argument("--truncate-n", default="10,25,50",
                    help="fixed suffix lengths to emit, plus the full suffix. "
                         "logp_sum is not comparable across different n, and "
                         "these suffixes run 19-33 words, so a fixed n is what "
                         "makes the (n,p) curve mean anything.")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="peak memory is dominated by the float32 log_softmax "
                         "over [B, L, vocab]; lower this if you OOM")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--qid", action="append", help="restrict to these question ids")
    ap.add_argument("--force", action="store_true", help="ignore existing rows")
    ap.add_argument("--dry-run", action="store_true",
                    help="tokenise and print the decoded suffix span. Loads the "
                         "tokenizer only -- no weights, no GPU -- so it is the "
                         "cheapest way to catch an alignment error")
    args = ap.parse_args()

    qa = resolve_qa(args.qa)
    pairs = load_pairs(qa, limit=args.limit, qids=set(args.qid) if args.qid else None)
    corpus = corpus_name(qa)
    mslug = args.model.replace("/", "__")
    out_path = args.out or os.path.join(
        args.out_dir, f"logprobs__{corpus}__{mslug}__{args.condition}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    trunc = sorted({int(x) for x in args.truncate_n.split(",") if x.strip()})

    # Resume on (case_id, qid) -- one API-free rerun after an interruption picks
    # up where it stopped, same contract as probe/run_probe.py.
    done = set()
    if os.path.exists(out_path) and not args.force:
        with open(out_path, newline="") as fh:
            done = {(r["case_id"], r["qid"]) for r in csv.DictReader(fh)}
    todo = [p for p in pairs if (p["case_id"], p["qid"]) not in done]

    print(f"model      {args.model}")
    print(f"corpus     {corpus}  ({qa})")
    print(f"condition  {args.condition}")
    print(f"pairs      {len(pairs)} total, {len(done)} done, {len(todo)} to score")
    print(f"trunc_n    {trunc} + full")
    print(f"out        {out_path}\n")
    if not todo:
        print("nothing to do"); return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.dry_run:
        for p in todo[:3]:
            pre = build_prefix_text(p, args.condition, tok)
            ids, k, warn = locate_suffix(tok, pre, pre + " " + p["suffix"])
            print("=" * 72)
            print(f"[{p['case_id']} {p['qid']}]  prefix_tokens={k}  "
                  f"suffix_tokens={len(ids)-k}  warn={warn or 'none'}")
            print(f"  prefix tail : {tok.decode(ids[max(0,k-8):k])!r}")
            print(f"  suffix head : {tok.decode(ids[k:k+10])!r}")
            print(f"  suffix tail : {tok.decode(ids[-6:])!r}")
        print(f"\n(dry run — {len(todo)} pairs would be scored)")
        return

    import torch
    from transformers import AutoModelForCausalLM
    dtype = getattr(torch, args.dtype)
    print("loading weights (this is the slow part) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device_map)
    model.eval()
    dev = next(model.parameters()).device
    print(f"loaded on {dev}\n", flush=True)

    # Pre-tokenise so batches can be length-sorted; padding waste otherwise
    # dominates runtime on a corpus with a 5x spread in sequence length.
    prepped = []
    for p in todo:
        pre = build_prefix_text(p, args.condition, tok)
        ids, k, warn = locate_suffix(tok, pre, pre + " " + p["suffix"])
        if k >= len(ids) or k == 0:
            print(f"  skip {p['case_id']} {p['qid']}: empty suffix span")
            continue
        prepped.append((p, ids, k, warn))
    prepped.sort(key=lambda t: len(t[1]))

    new = args.force or not os.path.exists(out_path)
    fh = open(out_path, "w" if new else "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    if new:
        w.writeheader()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0, n_done = time.time(), 0
    for i in range(0, len(prepped), args.batch_size):
        batch = prepped[i:i + args.batch_size]
        L = max(len(t[1]) for t in batch)
        input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), L), dtype=torch.long)
        for b, (_, ids, _, _) in enumerate(batch):
            input_ids[b, :len(ids)] = torch.tensor(ids)
            attn[b, :len(ids)] = 1                       # RIGHT padding
        input_ids, attn = input_ids.to(dev), attn.to(dev)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            # float() before log_softmax: bf16 softmax over a 128k vocab loses
            # enough precision to matter once ~40 of these are summed.
            lp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            tok_lp = lp.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            # tok_lp[b, i] is the logprob of input_ids[b, i+1];
            # so position j is tok_lp[b, j-1].  <-- the off-by-one, made explicit
        tok_lp = tok_lp.float().cpu()

        for b, (p, ids, k, warn) in enumerate(batch):
            v = tok_lp[b, k - 1: len(ids) - 1].tolist()   # suffix positions k..end
            if not v:
                continue
            for n in trunc + [None]:
                sub = v if n is None else v[:n]
                if not sub:
                    continue
                s = float(sum(sub))
                w.writerow({
                    **{c: p.get(c, "") for c in
                       ("case_id", "qid", "work", "arm", "date_filed", "jurisdiction")},
                    "model": args.model, "condition": args.condition,
                    "trunc_n": "full" if n is None else n,
                    "n_prefix_tokens": k, "n_suffix_tokens": len(sub),
                    "logp_sum": round(s, 6),
                    "logp_per_token": round(s / len(sub), 6),
                    "p_geomean": round(math.exp(s / len(sub)), 8),
                    "min_token_logprob": round(min(sub), 6),
                    "n_tokens_below_1e3": sum(1 for x in sub if x < math.log(1e-3)),
                    # Memorised passages usually fail at one surprising token
                    # rather than degrading evenly; this says where.
                    "first_break_index": int(min(range(len(sub)), key=lambda j: sub[j])),
                    "align_warning": warn,
                })
        fh.flush()
        n_done += len(batch)
        if (i // max(1, args.batch_size)) % 10 == 0 or n_done >= len(prepped):
            el = time.time() - t0
            rate = n_done / max(el, 1e-9)
            print(f"  {n_done}/{len(prepped)}  {el:6.0f}s  "
                  f"{rate:5.2f} pairs/s  eta {(len(prepped)-n_done)/max(rate,1e-9):5.0f}s",
                  flush=True)
    fh.close()
    nwarn = sum(1 for _, _, _, wn in prepped if wn)
    print(f"\ndone: {n_done} pairs -> {out_path}")
    if nwarn:
        print(f"  {nwarn} rows carry align_warning=boundary_merge; span located "
              f"by offsets, but inspect a few before trusting them")


if __name__ == "__main__":
    main()
