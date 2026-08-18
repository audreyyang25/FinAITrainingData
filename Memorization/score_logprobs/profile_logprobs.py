#!/usr/bin/env python3
"""Stage 1: per-token logprob profile of whole opinions, to locate hot spots.

  python profile_logprobs.py --model meta-llama/Llama-3.1-70B --limit 3 --dry-run
  python profile_logprobs.py --model meta-llama/Llama-3.1-70B
  python profile_logprobs.py --model meta-llama/Llama-3.1-70B --top-k 400

WHY THIS EXISTS. score_logprobs.py samples 4-15 fixed anchors per opinion. If
memorisation inside a court opinion is CONCENTRATED rather than absent -- and the
black-box run says it is, with a 41-token verbatim run that turned out to be
ERISA 404(a)(1)(B) quoted inside a 2025 opinion -- then fixed anchors can miss it
entirely, or hit one and misattribute it to the opinion rather than the statute.
This sweeps every token of every document instead.

CHEAP BY CONSTRUCTION. A naive fixed-prefix sweep at stride 256 is ~7,700 forward
passes over this corpus. Scoring each document in chunks and slicing the
resulting logprob vector is ~800, because one pass yields every position at once.

    THE PRICE OF THAT: a token here sees everything before it in its chunk, up to
    ~2k tokens. Cooper et al.'s design gives it 50. So these numbers are NOT
    comparable to score_logprobs.py output and will look better. Stage 1 LOCATES;
    stage 2 (fixed-prefix, on the hot spots this finds) MEASURES.

The control that makes it mean something: run it over both arms. If pre- and
post-cutoff opinions have hot spots in the SAME places -- syllabus, quoted
statutes, the mandate -- that is genre structure, not memorisation of the case,
and it is the direct test of the "boilerplate inflates apparent recall" problem
in METHODS.md section 1.3.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
DS = os.path.join(BASE, "datasets")
TEXT_DIR = os.path.join(DS, "court_opinions_courtlistener")
MANIFEST = os.path.join(BASE, "crawl_manifest", "case_ids.json")

COLS = ["case_id", "arm", "date_filed", "court_level", "jurisdiction",
        "model", "condition", "win_start_tok", "win_end_tok",
        "char_start", "char_end", "n_tokens",
        "mean_logp", "min_logp", "frac_below_1e3", "excerpt"]
HOT_COLS = ["rank", "case_id", "arm", "mean_logp", "n_tokens",
            "char_start", "char_end", "text"]


def load_cases(jurisdiction=None, limit=None, text_dir=None, manifest=None):
    """Opinions that exist on disk, with the manifest metadata joined on."""
    text_dir, manifest = text_dir or TEXT_DIR, manifest or MANIFEST
    for p, what in ((manifest, "manifest"), (text_dir, "opinion text dir")):
        if not os.path.exists(p):
            sys.exit(f"{what} not found: {p}\n"
                     f"  This path is derived from where this script lives "
                     f"({os.path.abspath(__file__)}),\n"
                     f"  which assumes <repo>/Memorization/score_logprobs/. "
                     f"Pass --manifest / --text-dir to override.")
    man = json.load(open(manifest))
    out = []
    for c in man["court_opinions"]["cases"]:
        if jurisdiction and c.get("jurisdiction") != jurisdiction:
            continue
        p = os.path.join(text_dir, f"{c['case_id']}.txt")
        if not os.path.exists(p):
            continue
        out.append({"case_id": c["case_id"], "path": p,
                    # NOTE: the manifest arm is the global 2023-12-31 split, not
                    # a per-model one. Correct for Llama 3.1 (cutoff 2023-12);
                    # recompute from date_filed vs the model cutoff for anything
                    # else, the way qa/score_answers.py does.
                    "arm": c.get("arm", ""), "date_filed": c.get("date_filed", ""),
                    "court_level": c.get("court_level", ""),
                    "jurisdiction": c.get("jurisdiction", "")})
    out.sort(key=lambda c: c["case_id"])
    return out[:limit] if limit else out


def chunk_spans(n_tok, chunk, overlap, warmup):
    """-> [(start, end, score_from)] covering [0, n_tok) with no gaps.

    `score_from` is where a chunk's logprobs become trustworthy: the first
    `warmup` tokens of a chunk have almost no in-chunk context and would read as
    spuriously unmemorised. Chunk 0 is exempt -- position 1 really is the start
    of the document, so there is nothing better available.
    """
    spans, start = [], 0
    while start < n_tok:
        end = min(start + chunk, n_tok)
        spans.append((start, end, start + (0 if start == 0 else warmup)))
        if end >= n_tok:
            break
        start = end - overlap
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--out-dir", default=os.path.join(DS, "logprobs"))
    # Overridable because the derived paths assume this file sits at
    # <repo>/Memorization/score_logprobs/. Extract the tarball one level off and
    # everything resolves to the wrong place, with a bare FileNotFoundError.
    ap.add_argument("--text-dir", default=TEXT_DIR,
                    help="directory of <case_id>.txt opinions")
    ap.add_argument("--manifest", default=MANIFEST,
                    help="crawl_manifest/case_ids.json")
    ap.add_argument("--condition", default="raw", choices=["raw"],
                    help="raw only: a chat wrapper is meaningless for a document sweep")
    ap.add_argument("--jurisdiction", default="federal",
                    help="'federal' matches the QA corpus; '' for all")
    ap.add_argument("--window", type=int, default=50,
                    help="tokens per reported window -- the unit of the heat map")
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--chunk-tokens", type=int, default=2048)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=128)
    ap.add_argument("--batch-chunks", type=int, default=2,
                    help="peak memory is [B, chunk, vocab] in float32; 2 x 2048 "
                         "x 128k is ~2GB. Lower if you OOM.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--top-k", type=int, default=0,
                    help="also write the k lowest-surprisal windows WITH THEIR "
                         "TEXT, which is the file you actually read")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = load_cases(args.jurisdiction or None, args.limit,
                       args.text_dir, args.manifest)
    mslug = args.model.replace("/", "__")
    out_path = args.out or os.path.join(
        args.out_dir, f"profile__{mslug}__{args.condition}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done = set()
    if os.path.exists(out_path) and not args.force:
        with open(out_path, newline="") as fh:
            done = {r["case_id"] for r in csv.DictReader(fh)}
    todo = [c for c in cases if c["case_id"] not in done]

    print(f"model      {args.model}")
    print(f"corpus     {len(cases)} opinions ({args.jurisdiction or 'all'})")
    print(f"window     {args.window} tok, stride {args.stride}")
    print(f"chunking   {args.chunk_tokens} tok, overlap {args.overlap}, warmup {args.warmup}")
    print(f"todo       {len(todo)} ({len(done)} already profiled)")
    print(f"out        {out_path}\n")
    if not todo:
        print("nothing to do"); return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.dry_run:
        for c in todo[:3]:
            txt = open(c["path"], errors="ignore").read()
            enc = tok(txt, add_special_tokens=True, return_offsets_mapping=True)
            n = len(enc["input_ids"])
            sp = chunk_spans(n, args.chunk_tokens, args.overlap, args.warmup)
            wins = sum(max(0, (e - sf - args.window) // args.stride + 1)
                       for _, e, sf in sp)
            print(f"{c['case_id'][:52]:52s} {n:7,} tok  {len(sp):3d} chunks  "
                  f"{wins:5,} windows  arm={c['arm']}")
        tot = 0
        for c in todo:
            n = len(tok(open(c['path'], errors='ignore').read())["input_ids"])
            tot += len(chunk_spans(n, args.chunk_tokens, args.overlap, args.warmup))
        print(f"\n(dry run — {tot:,} forward passes over {len(todo)} opinions)")
        return

    import torch
    from transformers import AutoModelForCausalLM
    print("loading weights ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map=args.device_map)
    model.eval()
    dev = next(model.parameters()).device
    print(f"loaded on {dev}\n", flush=True)

    new = args.force or not os.path.exists(out_path)
    fh = open(out_path, "w" if new else "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    if new:
        w.writeheader()
    hot = []
    t0 = time.time()

    for ci, c in enumerate(todo, 1):
        txt = open(c["path"], errors="ignore").read()
        enc = tok(txt, add_special_tokens=True, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        n = len(ids)
        spans = chunk_spans(n, args.chunk_tokens, args.overlap, args.warmup)

        # Absolute per-token logprobs; NaN where no chunk supplies a trusted value.
        prof = [float("nan")] * n
        for b0 in range(0, len(spans), args.batch_chunks):
            batch = spans[b0:b0 + args.batch_chunks]
            L = max(e - s for s, e, _ in batch)
            inp = torch.full((len(batch), L), tok.eos_token_id, dtype=torch.long)
            am = torch.zeros((len(batch), L), dtype=torch.long)
            for bi, (s, e, _) in enumerate(batch):
                inp[bi, :e - s] = torch.tensor(ids[s:e]); am[bi, :e - s] = 1
            inp, am = inp.to(dev), am.to(dev)
            with torch.inference_mode():
                lg = model(input_ids=inp, attention_mask=am).logits
                lp = torch.log_softmax(lg[:, :-1, :].float(), dim=-1)
                tl = lp.gather(2, inp[:, 1:].unsqueeze(-1)).squeeze(-1).float().cpu()
            for bi, (s, e, sf) in enumerate(batch):
                # tl[bi, i] is the logprob of the chunk's token i+1, i.e. of
                # absolute position s+i+1. Same off-by-one as score_logprobs.py.
                for absolute in range(max(sf, s + 1), e):
                    # FIRST WRITER WINS. Overlap positions are produced by two
                    # chunks: the earlier one saw ~1900 tokens of context, the
                    # later one only `warmup`. Overwriting would hand those
                    # positions the worse-conditioned value and put a systematic
                    # dip every (chunk - overlap) tokens, biasing which windows
                    # reach the top-k.
                    if math.isnan(prof[absolute]):
                        prof[absolute] = tl[bi, absolute - s - 1].item()

        rows, lo = [], math.log(1e-3)
        for st in range(0, n - args.window + 1, args.stride):
            v = [x for x in prof[st:st + args.window] if not math.isnan(x)]
            if len(v) < args.window * 0.8:      # mostly warmup/uncovered
                continue
            cs, ce = offs[st][0], offs[min(st + args.window, n) - 1][1]
            rows.append({**{k: c.get(k, "") for k in
                            ("case_id", "arm", "date_filed", "court_level", "jurisdiction")},
                         "model": args.model, "condition": args.condition,
                         "win_start_tok": st, "win_end_tok": st + args.window,
                         "char_start": cs, "char_end": ce, "n_tokens": len(v),
                         "mean_logp": round(sum(v) / len(v), 6),
                         "min_logp": round(min(v), 6),
                         "frac_below_1e3": round(sum(1 for x in v if x < lo) / len(v), 4),
                         "excerpt": " ".join(txt[cs:ce].split())[:120]})
        w.writerows(rows); fh.flush()
        if args.top_k:
            hot += [(r["mean_logp"], r, txt[r["char_start"]:r["char_end"]]) for r in rows]
            hot.sort(key=lambda t: -t[0])
            hot = hot[:args.top_k]
        el = time.time() - t0
        print(f"  {ci}/{len(todo)}  {c['case_id'][:44]:44s} {n:6,}tok "
              f"{len(rows):4d}win  {el:6.0f}s  eta {(len(todo)-ci)*el/ci:6.0f}s", flush=True)
    fh.close()
    print(f"\ndone -> {out_path}")

    if args.top_k and hot:
        hp = out_path.replace("profile__", "hotspots__")
        with open(hp, "w", newline="") as f2:
            w2 = csv.DictWriter(f2, fieldnames=HOT_COLS, extrasaction="ignore")
            w2.writeheader()
            for i, (mu, r, text) in enumerate(hot, 1):
                w2.writerow({"rank": i, "case_id": r["case_id"], "arm": r["arm"],
                             "mean_logp": mu, "n_tokens": r["n_tokens"],
                             "char_start": r["char_start"], "char_end": r["char_end"],
                             "text": " ".join(text.split())})
        print(f"  top {len(hot)} windows with full text -> {hp}")
        print("  READ THAT FILE. If the hot spots are quoted statutes, syllabus")
        print("  boilerplate and mandates -- and if they recur in the post-cutoff")
        print("  arm -- then what looks like memorised opinions is memorised")
        print("  STATUTE, which is the section 1.3 confound made concrete.")


if __name__ == "__main__":
    main()
