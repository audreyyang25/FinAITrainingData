"""Gender-effect pipeline (run v3).

Compares how models' generated answers score depending on the CLIENT's gender,
using the name-swapped variants from rewrite_names.py (male vs female of the same
case, demographic held constant). Signed effect:

    delta = score(male) - score(female)     (+ => male variant scored higher)

Generation + judging reuse the main-pipeline calls (judge WITH the question).
Every result table is saved as a PNG figure (and CSV) under results/gender_effect/ so you don't
have to scroll the terminal.

PREREQ: modified_data/name_variants.jsonl (from rewrite_names.py).
"""

import os
import sys
import json
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Data_Analysis/ on sys.path
from config import GEN_SUITE, JUDGE_SUITE, ADAPTERS
from generations import model_call as gen_call
from judge import model_call as judge_call, judge_user, parse_judgment
from figures import save_table_fig, save_bar, save_heatmap

HERE = os.path.dirname(os.path.abspath(__file__))   # Data_Analysis/gender_effect_exp
DA_ROOT = os.path.dirname(HERE)                      # Data_Analysis
REPO = os.path.dirname(DA_ROOT)                      # repo root (FinAITrainingData)
OUT_DIR = os.path.join(DA_ROOT, "results", "gender_effect")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
GEN_PATH = os.path.join(OUT_DIR, "generations.jsonl")
JUD_PATH = os.path.join(OUT_DIR, "judgments.jsonl")
VARIANTS = os.path.join(REPO, "modified_data", "name_variants.jsonl")


def load_variants():
    """{gender: {(type, base_id, demo): {"prompt","truth"}}}, paired across genders."""
    by_gender = defaultdict(dict)
    for line in open(VARIANTS):
        line = line.strip()
        if not line:
            continue
        v = json.loads(line)
        adapter = ADAPTERS[v["type"]]
        rec = v["record"]
        key = (v["type"], v["base_id"], v["demographic"])
        by_gender[v["gender"]][key] = {"prompt": adapter["prompt"](rec),
                                       "truth": adapter["truth"](rec)}
    male, female = by_gender.get("male", {}), by_gender.get("female", {})
    paired = set(male) & set(female)
    return {"male": {k: male[k] for k in paired},
            "female": {k: female[k] for k in paired}}


def _load_done(path, judging):
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
            base = (r["gender"], r["type"], r["base_id"], r["demographic"], r["generator"])
            done.add(base + (r["judge"],) if judging else base)
    return done


def run_generation(variants):
    done = _load_done(GEN_PATH, judging=False)
    with open(GEN_PATH, "a") as out:
        for gender, recs in variants.items():
            for (t, bid, demo), d in recs.items():
                for gen in GEN_SUITE:
                    key = (gender, t, bid, demo, gen["key"])
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
                        "gender": gender, "type": t, "base_id": bid, "demographic": demo,
                        "generator": gen["key"], "model": gen["model"], "answer": answer}) + "\n")
                    out.flush()
                    done.add(key)
                    print(f"gen ok {key}")


def run_judging(variants):
    done = _load_done(JUD_PATH, judging=True)
    with open(JUD_PATH, "a") as out:
        for line in open(GEN_PATH):
            g = json.loads(line)
            d = variants[g["gender"]][(g["type"], g["base_id"], g["demographic"])]
            for spec in JUDGE_SUITE:
                key = (g["gender"], g["type"], g["base_id"], g["demographic"],
                       g["generator"], spec["key"])
                if key in done:
                    continue
                user = judge_user(d["truth"], g["answer"], task=d["prompt"])
                try:
                    raw = judge_call(spec["model"], user)
                except Exception as e:
                    print(f"JUDGE FAIL {key}: {e}")
                    continue
                score, rationale, _gq, valid = parse_judgment(raw)
                out.write(json.dumps({
                    "gender": g["gender"], "type": g["type"], "base_id": g["base_id"],
                    "demographic": g["demographic"], "generator": g["generator"],
                    "judge": spec["key"], "judge_model": spec["model"],
                    "score": score, "rationale": rationale, "valid": valid}) + "\n")
                out.flush()
                done.add(key)
                print(f"judge ok {key} -> {score}")


def aggregate():
    import pandas as pd
    try:
        from scipy import stats as _st
    except ImportError:
        _st = None
    df = pd.read_json(JUD_PATH, lines=True)
    df = df[df["valid"].fillna(False)].copy()

    wide = df.pivot_table(index=["type", "base_id", "demographic", "generator", "judge"],
                          columns="gender", values="score", aggfunc="mean")
    wide = wide.dropna(subset=["male", "female"])
    wide["delta"] = wide["male"] - wide["female"]  # + => male scored higher
    deltas = wide.reset_index()

    def summary(d):
        x = d["delta"].to_numpy()
        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1) if n > 1 else float("nan")     # sample std (spread of deltas)
        se = sd / math.sqrt(n) if n > 0 else float("nan")  # standard error of the mean
        t = mean / se if n > 1 and sd > 0 else float("nan")  # one-sample t vs 0 (no effect)
        # two-sided p from Student's t, df = n-1
        p = 2 * _st.t.sf(abs(t), n - 1) if (_st and n > 1 and sd > 0) else float("nan")
        return pd.Series({"n": n, "mean_delta": round(mean, 4), "std": round(sd, 4),
                          "t": round(t, 3), "p": round(p, 4),
                          "pos": int((x > 0).sum()), "neg": int((x < 0).sum())})

    # Collapse the two JUDGES into one delta per case×generator for the
    # significance tests: judges scoring the SAME answers aren't independent, so
    # counting both inflates n and the p-value. Generators stay separate.
    collapsed = deltas.groupby(["type", "base_id", "demographic", "generator"],
                               as_index=False)["delta"].mean()

    # Significance (t, p) on judge-collapsed data:
    overall = summary(collapsed).to_frame("overall").T
    by_type = collapsed.groupby("type").apply(summary)
    by_generator = collapsed.groupby("generator").apply(summary)
    # Descriptive tables stay on the raw per-judge pairs (kept as-is):
    by_judge = deltas.groupby("judge").apply(summary)
    gen_judge = deltas.pivot_table(index="generator", columns="judge",
                                   values="delta", aggfunc="mean").round(4)

    # Always save CSVs.
    deltas.to_csv(os.path.join(OUT_DIR, "gender_deltas.csv"), index=False)
    for name, tbl in [("overall", overall), ("by_type", by_type),
                      ("by_judge", by_judge), ("by_generator", by_generator),
                      ("gen_x_judge", gen_judge)]:
        tbl.to_csv(os.path.join(OUT_DIR, f"gender_{name}.csv"))

    print(f"\nGENDER EFFECT (male - female), judges averaged: "
          f"mean_delta={overall.loc['overall','mean_delta']}  n={int(overall.loc['overall','n'])}  "
          f"t={overall.loc['overall','t']}  p={overall.loc['overall','p']}")

    if _st is not None:
        x = collapsed["delta"].to_numpy()
        if (x != 0).any():
            stat, pw = _st.wilcoxon(x)
            print(f"Wilcoxon signed-rank (overall, judges averaged): stat={stat:.1f}, p={pw:.4g}")

    try:
        save_table_fig(overall, "Gender effect (male - female) -- overall", os.path.join(FIG_DIR, "overall.png"))
        save_table_fig(by_type, "Gender effect by dataset type", os.path.join(FIG_DIR, "by_type.png"))
        save_table_fig(by_judge, "Gender effect by judge", os.path.join(FIG_DIR, "by_judge.png"))
        save_table_fig(by_generator, "Gender effect by generator", os.path.join(FIG_DIR, "by_generator.png"))
        save_bar(by_generator["mean_delta"], "Gender effect by generator", os.path.join(FIG_DIR, "bar_by_generator.png"), ylabel="mean delta (male - female)")
        save_heatmap(gen_judge, "Gender effect: generator x judge (mean delta)", os.path.join(FIG_DIR, "heatmap_gen_judge.png"), cbar_label="male - female")
        print(f"figures + csvs saved under {OUT_DIR}")
    except ImportError:
        print("matplotlib not installed -- CSVs saved; run `pip install matplotlib` for figures")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "agg":
        aggregate()  # stats + figures only, from existing judgments (no API calls)
    else:
        variants = load_variants()
        print(f"paired (male & female) cases: {len(variants['male'])}")
        run_generation(variants)
        run_judging(variants)
        aggregate()
