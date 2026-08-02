"""sample_judge_disagreement.py -- pull a reviewable sample from the archived
panel-of-judges run (5 generators x 5 judges, archive/results/pilot_run/).

Targets judge validity, NOT model quality: the question is whether a judge's
score is defensible given the answer in front of it, so the sample is enriched
for the two places a judge is most likely to be wrong --

  DISAGREE  the 5 judges span a wide range on the SAME answer; at most one of
            them can be right, so the spread is direct evidence of unreliability
  LOW       all judges agree the answer is poor; if the answer is in fact fine,
            the rubric (not one judge) is the problem
  RANDOM    unenriched baseline -- without it you can only find errors, never
            estimate how often judging is sound

gemini-as-generator is excluded by default: that run truncated its answers to a
median 455 chars (reasoning ate the token budget), so its judgments score
fragments and would swamp both enriched strata -- 33 of 46 high-disagreement
items and 54 of 67 low-score items -- without saying anything about the judges.

  python -m tools.sample_judge_disagreement --n 50
  python -m tools.sample_judge_disagreement --n 50 --include-gemini
"""

import argparse
import importlib.util
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "archive", "results", "pilot_run")
ARCHIVE_CONFIG = os.path.join(ROOT, "archive", "config.py")
JUDGES = ["claude", "deepseek", "gemini", "gpt", "qwen"]


def load_references(config_path=ARCHIVE_CONFIG):
    """(dataset, id) -> (task_prompt, reference_answer), exactly as the judges saw
    them. Loaded from archive/config.py rather than shared/config.py because the
    pilot's per-dataset `truth` adapters are what built the REFERENCE ANSWER block
    in the judge prompt (borderline composes analysis+factors+outcome, redflags
    dumps JSON, ...) -- the current pipeline's adapters may not match."""
    from shared.config import DATA_DIR  # archive/config.py's own DATA_DIR predates
                                        # the reorg and no longer resolves

    spec = importlib.util.spec_from_file_location("_archive_config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    refs = {}
    for name in cfg.ADAPTERS:
        for rid, prompt, truth in cfg.load_dataset(name, data_dir=DATA_DIR):
            refs[(name, rid)] = (prompt, truth)
    return refs


def load_panel(run_dir=RUN_DIR, condition="no_q", drop_generator="gemini"):
    """-> (wide scores+spread per item, {(dataset,id,generator): answer}, rationales)."""
    df = pd.DataFrame([json.loads(l) for l in open(os.path.join(run_dir, "judgments.jsonl"))])
    df = df[df["valid"].fillna(False)].copy()
    df = df[df.condition == condition]
    if drop_generator:
        df = df[df.generator != drop_generator]

    wide = df.pivot_table(index=["dataset", "id", "generator"],
                          columns="judge", values="score").dropna()
    wide["spread"] = wide[JUDGES].max(axis=1) - wide[JUDGES].min(axis=1)
    wide["mean"] = wide[JUDGES].mean(axis=1)

    rationales = {(r.dataset, r.id, r.generator, r.judge): r.rationale
                  for r in df.itertuples()}
    # A few generations in this run are null (the same truncation that hit
    # gemini); treat them as empty so the packet renders rather than crashing.
    answers = {(g["dataset"], g["id"], g["generator"]): (g.get("answer") or "")
               for g in (json.loads(l) for l in
                         open(os.path.join(run_dir, "generations.jsonl")))}
    return wide, answers, rationales


def build_sample(wide, n=50, seed=0):
    """Stratified: ~40% widest spread, ~25% lowest mean, remainder random."""
    n_dis, n_low = int(n * 0.4), int(n * 0.25)
    dis = wide.nlargest(n_dis, "spread")
    low = wide.drop(dis.index, errors="ignore").nsmallest(n_low, "mean")
    rest = wide.drop(dis.index.union(low.index), errors="ignore")
    rnd = rest.sample(min(n - len(dis) - len(low), len(rest)), random_state=seed)

    out = pd.concat([dis.assign(stratum="DISAGREE"),
                     low.assign(stratum="LOW"),
                     rnd.assign(stratum="RANDOM")])
    return out.reset_index()


def _quote(text, limit=None):
    text = text or ""
    clipped = text[:limit] if limit else text
    return "> " + clipped.replace("\n", "\n> ")


def write_outputs(sample, answers, rationales, refs, out_dir,
                  excerpt=1500, include_question=False):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "judge_disagreement_sample.csv")
    md_path = os.path.join(out_dir, "judge_disagreement_sample.md")

    sample.round(3).to_csv(csv_path, index=False)

    with open(md_path, "w") as f:
        f.write("# Panel-of-judges review sample\n\n"
                "Each judge was asked how well the CANDIDATE captures the REFERENCE's "
                "substantive legal conclusions, scored 0-1. Both texts are reproduced below, "
                "so the question is narrow: **given this reference, is the score defensible?**\n\n"
                "Suggested pass: read reference and candidate, decide your own score, "
                "then compare against the five judges.\n\n"
                "`DISAGREE` = judges spanned a wide range; `LOW` = judges agreed it was poor; "
                "`RANDOM` = unenriched control. Strata are labelled here for our own use -- "
                "**strip this column before sending to a reviewer.**\n\n---\n\n")
        for i, r in enumerate(sample.itertuples(), 1):
            key = (r.dataset, r.id, r.generator)
            prompt, reference = refs.get((r.dataset, r.id), ("", ""))
            f.write(f"## {i}. {r.dataset}/{r.id} — generated by `{r.generator}`\n\n")
            f.write(f"*stratum {r.stratum} · judge mean {r.mean:.2f} · spread {r.spread:.2f}*\n\n")

            if include_question:
                f.write(f"**Original task** (excerpt):\n\n{_quote(prompt, excerpt)}\n\n")

            f.write(f"**REFERENCE answer** — what the judges scored against "
                    f"({len(reference or '')} chars):\n\n{_quote(reference)}\n\n")

            ans = answers.get(key) or ""
            f.write(f"**CANDIDATE answer** ({len(ans)} chars"
                    f"{', excerpt' if len(ans) > excerpt else ''}):\n\n"
                    f"{_quote(ans, excerpt)}\n\n")

            f.write("**Judge scores**\n\n| judge | score | rationale |\n|---|---|---|\n")
            for j in JUDGES:
                rat = (rationales.get(key + (j,), "") or "").replace("\n", " ").replace("|", "\\|")
                f.write(f"| {j} | {getattr(r, j):.2f} | {rat} |\n")
            f.write("\n---\n\n")
    return csv_path, md_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="no_q", choices=["no_q", "with_q"])
    ap.add_argument("--include-gemini", action="store_true",
                    help="keep gemini-as-generator (truncated answers -- see module docstring)")
    ap.add_argument("--with-question", action="store_true",
                    help="also print the original fact pattern (needed only to audit the "
                         "reference itself, not to audit the judge)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "judge_validation"))
    args = ap.parse_args()

    wide, answers, rationales = load_panel(
        condition=args.condition,
        drop_generator=None if args.include_gemini else "gemini")
    sample = build_sample(wide, n=args.n, seed=args.seed)
    csv_path, md_path = write_outputs(sample, answers, rationales, load_references(),
                                      args.out_dir, include_question=args.with_question)

    print(f"{len(wide)} items with all 5 judges -> sampled {len(sample)}")
    print(sample.stratum.value_counts().to_string())
    print(f"\nwrote {csv_path}\n      {md_path}")
