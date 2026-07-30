"""Paired A/B effect analysis shared by the effect experiments.

Each case appears under two arms (e.g. narrative vs conversation, or male vs
female). We pair the arms per (case, generator, judge), take the signed delta
    delta = score(hi) - score(lo),
then run the significance test on judge-collapsed data (the two judges score the
SAME answers, so they aren't independent -- counting both inflates n and the
p-value). Writes the deltas + summary tables as CSVs and figures.
"""

import os
import math


def _summary_fn(pd, _st):
    """Build the per-group summary (n, mean_delta, std, t, p, pos, neg)."""
    def summary(d):
        x = d["delta"].to_numpy()
        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1) if n > 1 else float("nan")      # sample std (spread of deltas)
        se = sd / math.sqrt(n) if n > 0 else float("nan")   # standard error of the mean
        t = mean / se if n > 1 and sd > 0 else float("nan")  # one-sample t vs 0 (no effect)
        # two-sided p from Student's t, df = n-1
        p = 2 * _st.t.sf(abs(t), n - 1) if (_st and n > 1 and sd > 0) else float("nan")
        return pd.Series({"n": n, "mean_delta": round(mean, 4), "std": round(sd, 4),
                          "t": round(t, 3), "p": round(p, 4),
                          "pos": int((x > 0).sum()), "neg": int((x < 0).sum())})
    return summary


def paired_delta(jud_path, out_dir, arm_col, hi, lo, case_cols, prefix,
                 effect_name, extra_group_tables=()):
    """Pair the two arms and analyze the effect.

    arm_col:  column holding the two arms (e.g. "format" or "gender").
    hi, lo:   the two arm values; delta = score(hi) - score(lo).
    case_cols: columns identifying a case (e.g. ["id"] or
               ["type", "base_id", "demographic"]).
    prefix:   output filename prefix ("formatting" / "gender").
    effect_name: human label for prints/titles ("Formatting effect").
    extra_group_tables: iterable of (table_name, groupby_col, fig_title) for extra
               by-X summary tables on the collapsed deltas (gender adds by_type).

    Returns the overall summary (a 1-row DataFrame).
    """
    import pandas as pd
    try:
        from scipy import stats as _st
    except ImportError:
        _st = None
    summary = _summary_fn(pd, _st)
    arm_desc = f"{hi} - {lo}"

    df = pd.read_json(jud_path, lines=True)
    df = df[df["valid"].fillna(False)].copy()

    # Pair the two arms on (case, generator, judge).
    wide = df.pivot_table(index=list(case_cols) + ["generator", "judge"],
                          columns=arm_col, values="score", aggfunc="mean")
    wide = wide.dropna(subset=[hi, lo])
    wide["delta"] = wide[hi] - wide[lo]     # + => hi arm scored higher
    deltas = wide.reset_index()

    # Collapse the two JUDGES into one delta per (case, generator) for the
    # significance tests; generators stay separate (they produce different answers).
    collapsed = deltas.groupby(list(case_cols) + ["generator"], as_index=False)["delta"].mean()

    overall = summary(collapsed).to_frame("overall").T
    by_generator = collapsed.groupby("generator").apply(summary)
    by_judge = deltas.groupby("judge").apply(summary)      # descriptive, per-judge pairs
    gen_judge = deltas.pivot_table(index="generator", columns="judge",
                                   values="delta", aggfunc="mean").round(4)
    extras = {name: (collapsed.groupby(col).apply(summary), title)
              for name, col, title in extra_group_tables}

    # ---- CSVs ----
    os.makedirs(out_dir, exist_ok=True)
    deltas.to_csv(os.path.join(out_dir, f"{prefix}_deltas.csv"), index=False)
    tables = {"overall": overall, "by_judge": by_judge,
              "by_generator": by_generator, "gen_x_judge": gen_judge}
    tables.update({name: tbl for name, (tbl, _t) in extras.items()})
    for name, tbl in tables.items():
        tbl.to_csv(os.path.join(out_dir, f"{prefix}_{name}.csv"))

    print(f"\n{effect_name.upper()} ({arm_desc}), judges averaged: "
          f"mean_delta={overall.loc['overall','mean_delta']}  n={int(overall.loc['overall','n'])}  "
          f"t={overall.loc['overall','t']}  p={overall.loc['overall','p']}")

    if _st is not None:
        x = collapsed["delta"].to_numpy()
        if (x != 0).any():
            stat, pw = _st.wilcoxon(x)
            print(f"Wilcoxon signed-rank (overall, judges averaged): stat={stat:.1f}, p={pw:.4g}")

    # ---- Figures (optional; CSVs are always written above) ----
    fig_dir = os.path.join(out_dir, "figures")
    try:
        from figures import save_table_fig, save_bar, save_heatmap
        os.makedirs(fig_dir, exist_ok=True)
        save_table_fig(overall, f"{effect_name} ({arm_desc}) -- overall",
                       os.path.join(fig_dir, "overall.png"))
        for name, (tbl, title) in extras.items():
            save_table_fig(tbl, title, os.path.join(fig_dir, f"{name}.png"))
        save_table_fig(by_judge, f"{effect_name} by judge",
                       os.path.join(fig_dir, "by_judge.png"))
        save_table_fig(by_generator, f"{effect_name} by generator",
                       os.path.join(fig_dir, "by_generator.png"))
        save_bar(by_generator["mean_delta"], f"{effect_name} by generator",
                 os.path.join(fig_dir, "bar_by_generator.png"),
                 ylabel=f"mean delta ({arm_desc})")
        save_heatmap(gen_judge, f"{effect_name}: generator x judge (mean delta)",
                     os.path.join(fig_dir, "heatmap_gen_judge.png"), cbar_label=arm_desc)
        print(f"figures + csvs saved under {out_dir}")
    except ImportError:
        print(f"matplotlib not installed -- CSVs saved under {out_dir}; "
              f"pip install matplotlib for figures")

    return overall
