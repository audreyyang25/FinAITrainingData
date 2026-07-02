"""Formatting-effect pipeline.

Question: does presenting the SAME standard case as a narrative vs a conversation
change how well models answer it (as scored by the judges)?

Runs generation + judging on both formats of the standard cases that have been
rewritten (paired by id -- only fact_pattern differs; question/answer identical),
then reports the PAIRED signed effect: delta = score(conversation) - score(narrative).

Reuses the main pipeline's generation/judging calls for comparability. Resumable.
Everything is judged WITH the question. Outputs under runs/v2/.

PREREQ: modified_data/standard_conversation.jsonl must exist (produced by
rewrite_format.py). Run that first if it's missing.
"""

import os
import json
import math
from config import GEN_SUITE, JUDGE_SUITE, DATA_DIR
from generations import model_call as gen_call
from judge import model_call as judge_call, judge_user, parse_judgment
from figures import save_table_fig, save_bar, save_heatmap

HERE = os.path.dirname(os.path.abspath(__file__))   # LLM_As_A_Judge
REPO = os.path.dirname(HERE)                         # repo root
OUT_DIR = os.path.join(HERE, "runs", "v2")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
GEN_PATH = os.path.join(OUT_DIR, "generations.jsonl")
JUD_PATH = os.path.join(OUT_DIR, "judgments.jsonl")

STD_SRC = os.path.join(DATA_DIR, "suitability_only_P12.json")
CONVO_SRC = os.path.join(REPO, "modified_data", "standard_conversation.jsonl")


def _prompt(fact_pattern, question):
    return f"{fact_pattern}\n\nQuestion: {question}"


def load_formats():
    """{format: {id: {"prompt", "truth"}}}, restricted to ids present in BOTH."""
    narrative = {}
    for r in json.load(open(STD_SRC)):
        narrative[r["id"]] = {"prompt": _prompt(r["fact_pattern"], r["question"]),
                              "truth": r["answer"]}
    conversation = {}
    for line in open(CONVO_SRC):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("_verified"):  # skip fact-drifted rewrites; add them back once fixed
            continue
        conversation[r["id"]] = {"prompt": _prompt(r["fact_pattern"], r["question"]),
                                 "truth": r["answer"]}
    paired = set(narrative) & set(conversation)
    return {"narrative": {i: narrative[i] for i in paired},
            "conversation": {i: conversation[i] for i in paired}}


def _load_done(path, keylen):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if keylen == 3:
                done.add((r["format"], r["id"], r["generator"]))
            else:
                done.add((r["format"], r["id"], r["generator"], r["judge"]))
    return done


def run_generation(formats):
    done = _load_done(GEN_PATH, 3)
    with open(GEN_PATH, "a") as out:
        for fmt, recs in formats.items():
            for rid, d in recs.items():
                for gen in GEN_SUITE:
                    key = (fmt, rid, gen["key"])
                    if key in done:
                        continue
                    try:
                        answer = gen_call(gen["model"], d["prompt"])
                    except Exception as e:
                        print(f"GEN FAIL {key}: {e}")
                        continue
                    if not answer:
                        print(f"GEN EMPTY {key}")
                        continue
                    out.write(json.dumps({
                        "format": fmt, "id": rid, "generator": gen["key"],
                        "model": gen["model"], "answer": answer}) + "\n")
                    out.flush()
                    done.add(key)
                    print(f"gen ok {key}")


def run_judging(formats):
    done = _load_done(JUD_PATH, 4)
    with open(JUD_PATH, "a") as out:
        for line in open(GEN_PATH):
            g = json.loads(line)
            d = formats[g["format"]][g["id"]]
            for spec in JUDGE_SUITE:
                key = (g["format"], g["id"], g["generator"], spec["key"])
                if key in done:
                    continue
                user = judge_user(d["truth"], g["answer"],
                                    task=d["prompt"])
                try:
                    raw = judge_call(spec["model"], user)
                except Exception as e:
                    print(f"JUDGE FAIL {key}: {e}")
                    continue
                score, rationale, gq, valid = parse_judgment(raw)
                out.write(json.dumps({
                        "format": g["format"], "id": g["id"], "generator": g["generator"],
                        "judge": spec["key"], "judge_model": spec["model"],
                        "score": score, "rationale": rationale, "valid": valid}) + "\n")
                out.flush()
                done.add(key)
                print(f"judge ok {key} -> {score}")


def aggregate():
    import pandas as pd
    df = pd.read_json(JUD_PATH, lines=True)
    df = df[df["valid"].fillna(False)].copy()

    # Pair conversation vs narrative on (id, generator, judge).
    wide = df.pivot_table(index=["id", "generator", "judge"],
                          columns="format", values="score", aggfunc="mean")
    wide = wide.dropna(subset=["narrative", "conversation"])
    wide["delta"] = wide["conversation"] - wide["narrative"]  # + => convo scored higher
    deltas = wide.reset_index()

    def summary(d):
        x = d["delta"].to_numpy()
        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1) if n > 1 else float("nan")
        t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        return pd.Series({"n_pairs": n, "mean_delta": round(mean, 4),
                          "std": round(sd, 4), "t": round(t, 3),
                          "pos": int((x > 0).sum()), "neg": int((x < 0).sum())})

    overall = summary(deltas).to_frame("overall").T
    by_judge = deltas.groupby("judge").apply(summary)
    by_generator = deltas.groupby("generator").apply(summary)
    gen_judge = deltas.pivot_table(index="generator", columns="judge",
                                   values="delta", aggfunc="mean").round(4)

    deltas.to_csv(os.path.join(OUT_DIR, "formatting_deltas.csv"), index=False)
    for name, tbl in [("overall", overall), ("by_judge", by_judge),
                      ("by_generator", by_generator), ("gen_x_judge", gen_judge)]:
        tbl.to_csv(os.path.join(OUT_DIR, f"formatting_{name}.csv"))

    print(f"\nFORMATTING EFFECT (conversation - narrative): overall mean_delta="
          f"{overall.loc['overall', 'mean_delta']}  n={int(overall.loc['overall', 'n_pairs'])}")

    try:
        from scipy.stats import wilcoxon
        x = deltas["delta"].to_numpy()
        if (x != 0).any():
            stat, p = wilcoxon(x)
            print(f"Wilcoxon signed-rank (overall): stat={stat:.1f}, p={p:.4g}")
    except ImportError:
        pass

    try:
        save_table_fig(overall, "Formatting effect (conversation - narrative) -- overall", os.path.join(FIG_DIR, "overall.png"))
        save_table_fig(by_judge, "Formatting effect by judge", os.path.join(FIG_DIR, "by_judge.png"))
        save_table_fig(by_generator, "Formatting effect by generator", os.path.join(FIG_DIR, "by_generator.png"))
        save_bar(by_generator["mean_delta"], "Formatting effect by generator", os.path.join(FIG_DIR, "bar_by_generator.png"), ylabel="mean delta (conversation - narrative)")
        save_heatmap(gen_judge, "Formatting effect: generator x judge (mean delta)", os.path.join(FIG_DIR, "heatmap_gen_judge.png"), cbar_label="conversation - narrative")
        print(f"figures + csvs saved under {OUT_DIR}")
    except ImportError:
        print(f"matplotlib not installed -- CSVs saved under {OUT_DIR}; pip install matplotlib for figures")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "agg":
        aggregate()  # stats + figures only, from existing judgments (no API calls)
    else:
        formats = load_formats()
        print(f"paired records: {len(formats['narrative'])}")
        run_generation(formats)
        run_judging(formats)
        aggregate()
