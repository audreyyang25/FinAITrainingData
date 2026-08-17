#!/usr/bin/env python3
"""Figures for the outcome-knowledge probe.

  python Memorization/viz/make_outcome_figures.py

Plumbing is done; the three plot bodies are yours to fill in. Each stub says
what to build and names the trap in that particular figure.

Two decisions already made, because they constrain everything downstream:

  * COLOR ENCODES FAMILY, NOT MODEL. Six models would need a six-hue categorical
    palette, which means re-running scripts/validate_palette.js and finding six
    steps that clear every all-pairs CVD gate. The existing s1/s2/s3 are already
    validated, and the comparison you actually care about is within-family
    (Opus 4 -> Fable 5, GPT-5 -> 5.6 Sol, 2.5 Pro -> 3.1 Pro). So: hue = family,
    position within each adjacent pair = generation, and the y-axis label names
    the model outright. No new palette to validate, and the old/new contrast
    sits side by side where the eye can do the subtraction.

  * ATTEMPT/DECLINE COMES FROM summarize.graded AND THE declined COLUMN, not a
    re-derivation here. If a figure and the summary CSV disagree about what an
    attempt is, one of them is wrong and nobody will notice for weeks.
"""
from __future__ import annotations
import argparse, collections, csv, math, os, random, statistics, sys

VIZ = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(VIZ)
sys.path.insert(0, VIZ)
sys.path.insert(0, os.path.join(ROOT, "outcome"))

from make_figures import LIGHT, FIGS, style                 # palette + axis styling
from make_comparison import save                            # writes the PNG
from models import TARGETS, LABEL, CUTOFF                   # target suite + cutoffs
from summarize import graded                                # the one definition of "judged"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(VIZ))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")
JUDGED = os.path.join(DS, "outcome_scores", "judged__anthropic__claude-opus-5.csv")
JUDGED_PREDICT = os.path.join(DS, "outcome_scores",
                              "judged__anthropic__claude-opus-5__predict.csv")

# Family-grouped, older generation first inside each pair. This order is also the
# y-axis order in every figure -- keep it stable so the eye can move between them.
PAIRS = [("anthropic", "s1", ["anthropic/claude-opus-4", "anthropic/claude-fable-5"]),
         ("openai",    "s2", ["openai/gpt-5", "openai/gpt-5.6-sol"]),
         ("google",    "s3", ["google/gemini-2.5-pro", "google/gemini-3.1-pro-preview"])]
MODEL_ORDER = [m for _, _, ms in PAIRS for m in ms]
SLOT = {m: slot for _, slot, ms in PAIRS for m in ms}
# Generation carried by alpha, matching the memorization figures: older
# faded, newer full. Keyed on the "/" id used here, not the "__" slug.
ALPHA_GEN = {m: (0.55 if i == 0 else 1.0)
             for _, _, ms in PAIRS for i, m in enumerate(ms)}

# Must match the judge rubric's vocabulary exactly. There is no catch-all:
# `other` was retired because it merged opposite trial-court outcomes -- of the
# 20 cases it held, 10 were granted and 7 denied, so `other` == `other` scored a
# hit whichever direction the model guessed. It is now granted/denied/judgment.
DISPO = ["affirmed", "reversed", "vacated", "remanded", "dismissed",
         "granted", "denied", "judgment", "mixed"]


def load(path=JUDGED):
    """Judged rows only. Errored and unparsed rows are dropped, never zeroed --
    a 402 is a billing failure, not a model getting the case wrong."""
    rows = [r for r in csv.DictReader(open(path, newline="")) if graded(r)]
    for r in rows:
        r["_declined"] = bool(int(r.get("declined") or 0))
        r["_outcome"] = float(r["outcome_score"])
        r["_reasoning"] = float(r["reasoning_score"])
        r["_combined"] = float(r["combined"])
    return rows


def by(rows, **kw):
    """by(rows, model=..., arm=..., declined=False) -> filtered list."""
    out = rows
    for k, v in kw.items():
        key = f"_{k}" if f"_{k}" in (rows[0] if rows else {}) else k
        out = [r for r in out if r.get(key) == v]
    return out


def layout(per_model=1, model_gap=0.45, family_gap=0.55):
    """y positions for `per_model` bars on each model, grouped into families.

    Returns (ys, labels) where ys[(model, k)] is the y for the k-th bar of that
    model. Whitespace does the grouping: a small gap between the two generations
    of a family, a larger one between families, so the within-family comparison
    reads as adjacent before the cross-family one does.
    """
    ys, ticks, y = {}, [], 0.0
    for _, _, models in PAIRS:
        for m in models:
            for k in range(per_model):
                ys[(m, k)] = y + k
            ticks.append((y + (per_model - 1) / 2, m))
            y += per_model + model_gap
        y += family_gap
    return ys, ticks


def _bar_labels(ax, xs, ys, texts, C, pad):
    for x, y, t in zip(xs, ys, texts):
        ax.text(x + pad, y, t, va="center", color=C["ink2"], fontsize=7.5)


# --- paired-difference statistics -------------------------------------------
# outcome and reasoning are two judgements of the SAME answer, so the contrast
# is within-row. Var(o-r) = Var(o) + Var(r) - 2Cov(o,r), and the two correlate
# at .69-.86 here, which cancels 70-86% of the summed variance. Putting an
# interval on each bar instead and reading their overlap would test the wrong
# quantity -- it inflates the SE ~1.8-2.6x, and the overlap heuristic is itself
# stricter than p<.05. Hence: bars stay bare, the interval goes on the gap.
BOOT_N = 20000
BOOT_SEED = 0          # pinned: an unseeded bootstrap makes the figure differ
                       # between runs and there is no way to tell that from a
                       # data change.


def boot_ci(v, alpha=0.05):
    """Percentile bootstrap CI for the mean. Scores are bounded and lumpy, so a
    normal-theory interval is a worse fit than resampling."""
    rng = random.Random(BOOT_SEED)
    n = len(v)
    ms = sorted(statistics.mean(rng.choices(v, k=n)) for _ in range(BOOT_N))
    return ms[int(alpha / 2 * BOOT_N)], ms[int((1 - alpha / 2) * BOOT_N)]


def sign_test(d):
    """Exact two-sided sign test on non-tied pairs -> (p, n_up, n_down).

    The sign test, not the bootstrap, decides significance here. ~40% of rows
    are exact ties (both scores identical), the scale is discrete, and at n=9
    a percentile bootstrap is not trustworthy -- Opus 4's interval clears zero
    while only 7 of its rows are non-tied.
    """
    up = sum(1 for x in d if x > 0)
    dn = sum(1 for x in d if x < 0)
    n = up + dn
    if n == 0:
        return 1.0, up, dn
    tail = sum(math.comb(n, i) for i in range(min(up, dn) + 1)) / 2 ** n
    return min(1.0, 2 * tail), up, dn


# ---------------------------------------------------------------------------
def o1_disposition(rows, C):
    """FIGURE 1 — attempted vs declined, per model, split by arm.

    Form: 100% stacked horizontal bars. Two groups per model (pre, post), so 12
    bars. Fill `attempted` in the model's family hue and `declined` in C["grid"];
    this is a part-to-whole of one variable, so it is NOT a place for a second
    categorical hue.

    Build it as: for each model, for each arm -> pct_attempted, pct_declined.

    The trap: post-cutoff is ~100% declined for every model, so the figure looks
    broken unless the bars are labelled. Put `n=` on each bar. The near-empty
    post bars ARE the finding -- 3 attempts out of 238 items -- so make them
    legible rather than letting them read as missing data.
    """
    ys, ticks = layout(per_model=2)
    fg, ax = plt.subplots(figsize=(10.2, 6.4))
    fg.patch.set_facecolor(C["surface"])

    for m in MODEL_ORDER:
        for k, armname in enumerate(("pre_cutoff", "post_cutoff")):
            s = by(rows, model=m, arm=armname)
            y = ys[(m, k)]
            if not s:
                ax.text(1, y, "not run", va="center", color=C["ink2"],
                        fontsize=7, style="italic", alpha=0.8)
                continue
            att = 100 * sum(1 for r in s if not r["_declined"]) / len(s)
            ax.barh(y, att, height=0.78, color=C[SLOT[m]])
            ax.barh(y, 100 - att, left=att, height=0.78, color=C["grid"])
            n_att = sum(1 for r in s if not r["_declined"])
            # Label every bar. A post row is a 0.0%-wide fill; without the count
            # it reads as a rendering failure rather than as near-total refusal.
            ax.text(101.5, y, f"{att:.0f}% attempted   {n_att}/{len(s)}",
                    va="center", color=C["ink2"], fontsize=7.5)
            ax.text(1.5, y, "pre" if k == 0 else "post", va="center",
                    fontsize=7, color=C["surface"] if att > 12 else C["ink2"],
                    alpha=0.95)

    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([LABEL.get(m, m) for _, m in ticks], color=C["ink"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=C["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    style(ax, C, xlabel="% of judged cases")
    ax.set_title("Models answer before their cutoff and refuse after it\n"
                 "upper bar = pre-cutoff, lower = post-cutoff; filled = attempted",
                 color=C["ink"], fontsize=12, pad=12, loc="left")
    save(fg, "o1_attempt_vs_decline")


def o2_accuracy(rows, C):
    """FIGURE 2 — accuracy among attempts. PRE-CUTOFF ONLY.

    Form: grouped horizontal bars, two per model: mean outcome_score and mean
    reasoning_score over attempted pre-cutoff rows.

    Do not add a post-cutoff panel. There are 3 post-cutoff attempts in the whole
    dataset; a bar over n=0 or n=2 is noise rendered at the same visual weight as
    a bar over n=94. State the post-cutoff n in the subtitle instead.

    Encode the two measures by position, not by two new hues -- keep the family
    hue and let the pair (outcome above, reasoning below) carry the distinction,
    labelled directly. The outcome-minus-reasoning gap is the thing to make
    visible: every model knows *what* better than *why*.

    Annotate n per model. Claude Opus 4 has 9 attempts against Fable's 94, and
    an unlabelled 0.33 next to an unlabelled 0.67 invites a comparison the data
    does not support.

    RIGHT PANEL: the gap as a forest plot -- point = mean paired difference,
    whisker = 95% bootstrap CI, rule at 0 = the null. The uncertainty belongs
    here and not on the bars; see the note above boot_ci for why. Significance
    is the exact sign test, and the ↑/↓ counts are printed because they carry
    the result without needing any statistics: pooled across models, outcome
    beats reasoning on 215 answers and loses on 25.
    """
    ys, ticks = layout(per_model=2)
    fg, (ax, axg) = plt.subplots(1, 2, figsize=(13.4, 6.4), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 0.62]})
    fg.patch.set_facecolor(C["surface"])
    n_post = sum(1 for r in rows if r["arm"] == "post_cutoff" and not r["_declined"])

    stats = {}
    for m in MODEL_ORDER:
        att = [r for r in by(rows, model=m, arm="pre_cutoff") if not r["_declined"]]
        for k, (key, lbl) in enumerate((("_outcome", "outcome"), ("_reasoning", "reasoning"))):
            y = ys[(m, k)]
            if not att:
                ax.text(0.01, y, "no attempts", va="center", color=C["ink2"],
                        fontsize=7, style="italic", alpha=0.8)
                continue
            v = statistics.mean(r[key] for r in att)
            # Reasoning drawn at reduced alpha: same entity, lesser measure. A
            # second hue here would imply a second category.
            ax.barh(y, v, height=0.78, color=C[SLOT[m]], alpha=1.0 if k == 0 else 0.55)
            ax.text(v + 0.012, y, f"{v:.2f}  {lbl}", va="center",
                    color=C["ink2"], fontsize=7.5)
        if att:
            d = [r["_outcome"] - r["_reasoning"] for r in att]
            lo, hi = boot_ci(d)
            p, up, dn = sign_test(d)
            stats[m] = dict(gap=statistics.mean(d), lo=lo, hi=hi, p=p,
                            up=up, dn=dn, n=len(d))
            ax.text(0.985, ys[(m, 0)] + 0.5, f"n={len(att)}", va="center", ha="right",
                    color=C["ink2"], fontsize=7.5, style="italic")

    # --- right panel: the paired gap, with its own interval -----------------
    # A forest plot. The rule at 0 is the null (outcome == reasoning); an
    # interval clearing it is the result, readable without doing arithmetic.
    axg.axvline(0, color=C["ink"], linewidth=1.1, zorder=2)
    for m in MODEL_ORDER:
        s = stats.get(m)
        y = ys[(m, 0)] + 0.5
        if not s:
            continue
        # Underpowered rows are drawn faint. Opus 4 has 7 non-tied pairs; its
        # bootstrap interval clears zero but the exact sign test does not, and
        # the full-strength mark would assert a result the data cannot carry.
        ns = s["p"] >= 0.05
        a = 0.32 if ns else 1.0
        axg.plot([s["lo"], s["hi"]], [y, y], color=C[SLOT[m]], lw=2.2,
                 alpha=a, solid_capstyle="butt", zorder=3)
        for x in (s["lo"], s["hi"]):     # end caps
            axg.plot([x, x], [y - 0.17, y + 0.17], color=C[SLOT[m]], lw=2.2,
                     alpha=a, zorder=3)
        axg.plot([s["gap"]], [y], marker="o", markersize=7, color=C[SLOT[m]],
                 alpha=a, zorder=4)
        star = ("ns" if ns else "***" if s["p"] < .001
                else "**" if s["p"] < .01 else "*")
        # One right-aligned string per row, not a label beside the whisker:
        # Opus 4's interval runs to +0.44 and a floating number there collides
        # with this column.
        axg.text(0.985, y, f"{s['gap']:+.2f}   {s['up']}↑ / {s['dn']}↓   {star}",
                 transform=axg.get_yaxis_transform(), va="center", ha="right",
                 color=C["ink2"], fontsize=7.5, style="italic")

    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([LABEL.get(m, m) for _, m in ticks], color=C["ink"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.xaxis.grid(True, color=C["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    style(ax, C, xlabel="mean judge score (0-1), attempted cases only")
    ax.set_title("Every model knows the outcome better than the reasoning\n"
                 f"pre-cutoff attempts only — the post-cutoff arm has {n_post} attempts in total",
                 color=C["ink"], fontsize=12, pad=12, loc="left")

    axg.set_xlim(-0.06, 0.52)
    axg.xaxis.grid(True, color=C["grid"], linewidth=0.7)
    axg.set_axisbelow(True)
    style(axg, C, xlabel="outcome − reasoning, per answer (95% CI)")
    axg.set_title("The gap, tested within each answer\n"
                  "↑/↓ = answers favouring outcome / reasoning; ties omitted",
                  color=C["ink"], fontsize=10, pad=12, loc="left")
    save(fg, "o2_outcome_vs_reasoning")


def o3_mixed_collapse(rows, C, arm="pre_cutoff"):
    """FIGURE 3 — where the disposition errors actually are.

    This is the strongest single result in the outcome probe: on the stable
    labels, accuracy is 72% across the clean dispositions and 32% when the truth
    is `mixed` -- and `mixed` is the largest class at 183 of 356 pre-cutoff
    attempts. Models compress partial outcomes into a clean headline.

    The compression has a DIRECTION, which the matrix shows and the bar does not:
    of the mixed-truth attempts, 32% say `mixed` and 31% say `affirmed`. So a
    partial disposition is about as likely to be called a flat affirmance as to
    be called partial. That is a prior toward the modal clean outcome, not
    scattered confusion.

    Two forms work. Pick one:

      (a) Accuracy by TRUE disposition -- one bar per class in DISPO, pooled over
          models, with n on each. Simple, and it makes the mixed cell obvious.

      (b) The full confusion matrix as a heatmap: rows = disposition_actual,
          cols = disposition_claimed, cells = counts or row-normalised percent.
          Richer, and it shows the 2:1 skew toward "affirmed" when the truth is
          mixed. If you go this way use a SEQUENTIAL ramp (one hue, light->dark)
          -- a heatmap encodes magnitude, so a categorical palette or a rainbow
          would be wrong.

    Drop the `other` class from any headline number: both sides of that
    comparison are judge-assigned catch-alls, so its 95% is trivial agreement.
    """
    from matplotlib.colors import LinearSegmentedColormap
    # Every label is now a real disposition, so nothing is excluded as a truth
    # row. The old code dropped `other` because both sides of that diagonal were
    # judge-assigned catch-alls; retiring `other` in favour of
    # granted/denied/judgment removes the reason, and returns the ~20 trial-court
    # cases it was swallowing to the matrix.
    truth = list(DISPO)
    # Arm filter, matching o5 and accuracy_table. This was previously absent and
    # o3 silently pooled both arms -- harmless while it was, because the original
    # run's post-cutoff arm had 3 attempts in 238 rows, so it added exactly one
    # mixed row (n=184 vs 183). It stops being harmless the moment this is
    # pointed at the predict run, which has 232 post-cutoff attempts.
    att = [r for r in rows if r["arm"] == arm and not r["_declined"]
           and r["disposition_actual"].strip().lower() in truth]
    # Rows with no observations would render as an empty band; drop them.
    truth = [d for d in truth
             if any(r["disposition_actual"].strip().lower() == d for r in att)]
    # Columns likewise: `remanded` is in DISPO but no model has ever claimed it
    # and no case carries it as truth, so it rendered as a blank stripe. Dropped
    # by observation rather than by hardcoding, so a column reappears the moment
    # some model does use it -- which is what the DISPO comment was protecting.
    claimed = [d for d in DISPO
               if any(r["disposition_claimed"].strip().lower() == d for r in att)]
    classes = truth

    fg, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                  gridspec_kw={"width_ratios": [1, 1.35]})
    fg.patch.set_facecolor(C["surface"])

    # (a) accuracy by true disposition
    accs, ns = [], []
    for c in classes:
        s = [r for r in att if r["disposition_actual"].strip().lower() == c]
        hit = sum(1 for r in s
                  if r["disposition_claimed"].strip().lower() == c)
        accs.append(100 * hit / len(s) if s else 0)
        ns.append(len(s))
    order = sorted(range(len(classes)), key=lambda i: -ns[i])
    ys = list(range(len(order)))
    axA.barh(ys, [accs[i] for i in order], height=0.68, color=C["s1"])
    for y, i in zip(ys, order):
        axA.text(accs[i] + 1.5, y, f"{accs[i]:.0f}%   n={ns[i]}", va="center",
                 color=C["ink2"], fontsize=8)
    axA.set_yticks(ys)
    axA.set_yticklabels([classes[i] for i in order], color=C["ink"], fontsize=9)
    axA.invert_yaxis()
    axA.set_xlim(0, 100)
    axA.xaxis.grid(True, color=C["grid"], linewidth=0.7)
    axA.set_axisbelow(True)
    style(axA, C, xlabel="% of attempts naming the right disposition")
    axA.set_title("Accuracy collapses on partial outcomes", color=C["ink"],
                  fontsize=10.5, pad=8, loc="left")

    # (b) confusion, row-normalised. Sequential ramp: this encodes magnitude, so
    # one hue light->dark. A categorical palette here would be a category error.
    cmap = LinearSegmentedColormap.from_list("seq", [C["surface"], C["s1"]])
    M = []
    for a in truth:
        s = [r for r in att if r["disposition_actual"].strip().lower() == a]
        cnt = collections.Counter(r["disposition_claimed"].strip().lower() for r in s)
        M.append([100 * cnt[b] / len(s) if s else 0 for b in claimed])
    axB.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(truth)):
        for j in range(len(claimed)):
            if M[i][j] >= 1:
                axB.text(j, i, f"{M[i][j]:.0f}", ha="center", va="center", fontsize=7.5,
                         color=C["surface"] if M[i][j] > 55 else C["ink2"])
    axB.set_xticks(range(len(claimed)))
    axB.set_xticklabels(claimed, color=C["ink"], fontsize=8, rotation=30, ha="right")
    axB.set_yticks(range(len(truth)))
    axB.set_yticklabels([f"{t}  (n={sum(1 for r in att if r['disposition_actual'].strip().lower()==t)})"
                         for t in truth], color=C["ink"], fontsize=8)
    axB.set_xlabel("model claimed", color=C["ink2"], fontsize=9, labelpad=6)
    axB.set_ylabel("court actually held", color=C["ink2"], fontsize=9, labelpad=6)
    axB.tick_params(colors=C["ink2"], length=0)
    for sp in axB.spines.values():
        sp.set_visible(False)
    axB.set_title("Mixed outcomes get compressed into clean ones\n"
                  "row-normalised %; rows sum to 100 up to rounding",
                  color=C["ink"], fontsize=10.5, pad=8, loc="left")

    fg.suptitle(f"Where the disposition errors are "
                f"({arm.replace('_', ' ')} attempts, all models pooled)",
                color=C["ink"], fontsize=12, x=0.006, ha="left", y=0.995)
    save(fg, "o3_mixed_collapse")


def o5_by_disposition(rows, C, arm="pre_cutoff"):
    """FIGURE 5 — per-model hit-rate on each kind of disposition.

    "When the court affirmed, did the model say affirmed?" One group per true
    disposition, one bar per model, exact-match on the label.

    Labels with no cases in this arm are dropped automatically. There is no
    catch-all any more: `other` was retired because it merged granted with
    denied, so a model could get the direction backwards and still score a hit.

    THE THIN CELLS ARE THE POINT OF THE n LABELS. Pooled across models `vacated`
    and `dismissed` have 12 rows each, so a per-model rate rests on 0-4 cases and
    can only come out 0/50/100%. Those bars are drawn faint and their n called
    out; a solid bar at 100% over two cases would be the most misleading mark on
    the page. Claude Opus 4 attempted 9 pre-cutoff cases in total, so most of its
    row is absent by the same rule.
    """
    # Empty classes are dropped by observation, not by name: with `other` retired
    # the only reason to hardcode an exclusion is gone, and a label that gains
    # cases reappears on its own.
    keep = [d for d in DISPO
            if any(r["disposition_actual"].strip().lower() == d for r in rows
                   if r["arm"] == arm and not r["_declined"])]
    att = [r for r in rows if r["arm"] == arm and not r["_declined"]]
    n_models = len(MODEL_ORDER)
    h = 0.80 / n_models
    fg, ax = plt.subplots(figsize=(10.4, 1.25 * len(keep) + 2.2))
    fg.patch.set_facecolor(C["surface"])

    MIN_N = 5           # below this a percentage is not a rate, it is an anecdote
    for i, m in enumerate(MODEL_ORDER):
        for j, d in enumerate(keep):
            s = [r for r in att if r["model"] == m
                 and r["disposition_actual"].strip().lower() == d]
            y = j + (i - (n_models - 1) / 2) * h
            if not s:
                # "no cases" and "declined them all" are opposite claims: the
                # first says the corpus had none, the second says the model
                # refused. Opus 4 declined 13 reversed / 4 dismissed / 1 vacated
                # case here, so the blank is a behaviour, not missing data.
                nd = sum(1 for r in rows
                         if r["arm"] == arm and r["model"] == m and r["_declined"]
                         and r["disposition_actual"].strip().lower() == d)
                ax.text(1.0, y, f"declined all {nd}" if nd else "no cases in corpus",
                        va="center", color=C["ink2"], fontsize=6.5,
                        style="italic", alpha=0.75)
                continue
            hit = sum(1 for r in s
                      if r["disposition_claimed"].strip().lower() == d)
            v = 100 * hit / len(s)
            thin = len(s) < MIN_N
            ax.barh(y, v, height=h * 0.86, color=C[SLOT[m]],
                    alpha=(0.3 if thin else ALPHA_GEN[m]),
                    label=LABEL.get(m, m) if j == 0 else None)
            ax.text(v + 1.2, y, f"{v:.0f}%  n={len(s)}" + ("  ·thin" if thin else ""),
                    va="center", color=C["ink2"], fontsize=7,
                    alpha=0.7 if thin else 1.0)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([d + f"\n(n={sum(1 for r in att if r['disposition_actual'].strip().lower()==d)})"
                        for d in keep], color=C["ink"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=C["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    style(ax, C, xlabel="% of those cases where the model named this disposition")
    ax.set_title("Which verdicts each model gets right\n"
                 f"{arm.replace('_',' ')} attempts, exact match on the label; "
                 "faint bars rest on fewer than 5 cases",
                 color=C["ink"], fontsize=12, pad=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["ink2"], loc="lower right")
    save(fg, "o5_hit_rate_by_disposition")


def o6_vs_baselines(rows, C, arm="pre_cutoff"):
    """FIGURE 6 — disposition accuracy against four knowledge-free baselines.

    The bar is the model. The vertical rules are what you get for free:

      uniform random    guess evenly among the labels in use -> 1/k
      prior-matched     draw from the true verdict distribution -> sum(p^2).
                        This is the "weighted random" design, and on skewed
                        classes it is a WEAK baseline -- see the next one.
      always "affirmed" answer affirmed every time -> p(affirmed)
      always "mixed"    answer mixed every time -> p(mixed), the mode here

    Reporting only the random lines would flatter the models badly: prior-matched
    sits at ~31% and always-mixed at ~47%, so a model can clear "chance" by 20
    points and still lose to a one-line heuristic that never reads the case. The
    always-mode rule is the bar that matters.

    EVERY BASELINE IS PER MODEL, drawn as ticks on that model's own bar. An
    earlier version drew them as shared rules on the pooled set, on the grounds
    that per-model values varied by only 1-3 points. That is true for five of six
    models and badly false for the sixth: Claude Opus 4 attempted 9 pre-cutoff
    cases, they span just two labels, and its baselines are uniform 50.0 /
    prior-matched 50.6 / affirmed 44.4 / mixed 55.6 against pooled rules at
    14.3 / 31.3 / 26.4 / 46.9. Read against the shared rules its 33% looked like
    a pass; read against its own it is below all four. A baseline computed on a
    denominator the model never faced is not a baseline.
    """
    att = [r for r in rows if r["arm"] == arm and not r["_declined"]]

    def baselines(sub):
        """The four knowledge-free rates for one model's own attempted set."""
        a = [r["disposition_actual"].strip().lower() for r in sub]
        c = collections.Counter(a)
        n = len(a)
        return [("uniform random", 100 / len(c)),
                ("prior-matched (Σp²)", 100 * sum((v / n) ** 2 for v in c.values())),
                ('always "affirmed"', 100 * c.get("affirmed", 0) / n),
                ('always "mixed"', 100 * c.get("mixed", 0) / n)]

    # Ordered weakest to strongest, so the tick styles read as a ramp: the
    # faintest mark is the easiest bar to clear, the solid one is the real test.
    STYLE = [((0, (1, 2)), 0.40), ((0, (4, 2, 1, 2)), 0.60),
             ((0, (5, 3)), 0.80), ((0, ()), 1.00)]

    fg, ax = plt.subplots(figsize=(11.0, 6.0))
    fg.patch.set_facecolor(C["surface"])
    ys = list(range(len(MODEL_ORDER)))
    for y, m in zip(ys, MODEL_ORDER):
        s = [r for r in att if r["model"] == m]
        if not s:
            continue
        hit = sum(1 for r in s
                  if r["disposition_claimed"].strip().lower()
                  == r["disposition_actual"].strip().lower())
        v = 100 * hit / len(s)
        ax.barh(y, v, height=0.52, color=C[SLOT[m]], alpha=ALPHA_GEN[m], zorder=3)
        # Ticks stand taller than the bar so they stay visible where they fall
        # inside it -- which is the case that matters, a model below its baseline.
        for (lbl, x), (dash, alpha) in zip(baselines(s), STYLE):
            ax.plot([x, x], [y - 0.36, y + 0.36], color=C["ink"], lw=1.3,
                    linestyle=dash, alpha=alpha, zorder=5,
                    label=lbl if y == 0 else None)
        # Direct label carries n, because the denominators are not comparable:
        # Opus 4's 9 attempts against Fable 5's 94 is a selection effect, not a
        # sample-size quibble.
        # Name WHICH baselines it fails, not just that it failed the highest one.
        # An earlier version tested against max() but printed "below every
        # baseline", which libelled Gemini 2.5 Pro: at 37% it clears uniform,
        # prior-matched and always-"affirmed", and misses only always-"mixed".
        bl = baselines(s)
        fails = sorted([(x, lbl) for lbl, x in bl if v <= x])
        if not fails:
            note = ""
        elif len(fails) == len(bl):
            note = "   below every baseline"
        else:
            note = f"   below {fails[-1][1]}"
        ax.text(max(v, max(x for _, x in bl)) + 1.2, y,
                f"{v:.0f}%  n={len(s)}{note}",
                va="center", color=C["ink2"], fontsize=8, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL.get(m, m) for m in MODEL_ORDER], color=C["ink"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=C["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    style(ax, C, xlabel="% of attempts naming the right disposition")
    ax.set_title("Models against baselines that read nothing\n"
                 f"{arm.replace('_', ' ')} attempts; each tick is that model's OWN "
                 "baseline, on the cases it actually answered",
                 color=C["ink"], fontsize=12, pad=12, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=C["ink2"], ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.10))
    save(fg, f"o6_vs_baselines_{arm}")


def o4_recall_vs_predict(rows, C, predict_rows):
    """FIGURE 4 — knowing vs guessing, per model.

    Left panel OUTCOME, right panel REASONING; two bars per model, pre-cutoff
    recall above and post-cutoff prediction below. Measure is the panel because
    the two are on the same 0-1 scale but answer different questions, and a
    reader should compare within a measure before across.

    READ THE GAP, NOT THE LEVELS. This is not a clean A/B -- two things differ
    between the bars at once:
      * ARM. Pre-cutoff cases are inside training data, post-cutoff are not.
      * PROMPT. Pre-cutoff ran the default SYSTEM, which offers UNKNOWN as an
        out; post-cutoff ran SYSTEM_PREDICT, which forbids declining. So the
        pre bars are means over a self-selected subset (decline rates 38-93%)
        and the post bars are means over everything. Selection flatters the pre
        bars, which makes the drop an UPPER bound on how much knowledge is lost
        -- conservative in the direction of the claim, but not a clean estimate.
    n is on every bar for that reason.
    """
    ys, ticks = layout(per_model=2)
    fg, axes = plt.subplots(1, 2, figsize=(13.0, 6.6), sharey=True)
    fg.patch.set_facecolor(C["surface"])

    for ax, key, mlabel in ((axes[0], "_outcome", "outcome"),
                            (axes[1], "_reasoning", "reasoning")):
        for m in MODEL_ORDER:
            pre = [r for r in by(rows, model=m, arm="pre_cutoff") if not r["_declined"]]
            post = [r for r in by(predict_rows, model=m, arm="post_cutoff")
                    if not r["_declined"]]
            for k, (s, lbl) in enumerate(((pre, "pre-cutoff (recall)"),
                                          (post, "post-cutoff (predicted)"))):
                y = ys[(m, k)]
                if not s:
                    ax.text(0.01, y, "no data", va="center", color=C["ink2"],
                            fontsize=7, style="italic", alpha=0.8)
                    continue
                v = statistics.mean(r[key] for r in s)
                # Prediction drawn faint: same model, weaker epistemic footing.
                ax.barh(y, v, height=0.78, color=C[SLOT[m]],
                        alpha=1.0 if k == 0 else 0.45, label=lbl if m == MODEL_ORDER[0] else None)
                ax.text(v + 0.012, y, f"{v:.2f}  n={len(s)}", va="center",
                        color=C["ink2"], fontsize=7.5)
            if pre and post:
                d = statistics.mean(r[key] for r in post) - statistics.mean(r[key] for r in pre)
                ax.text(0.985, ys[(m, 0)] + 0.5, f"{d:+.2f}", transform=ax.get_yaxis_transform(),
                        va="center", ha="right", color=C["ink2"], fontsize=7.5, style="italic")
        ax.set_xlim(0, 1.0)
        ax.xaxis.grid(True, color=C["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        style(ax, C, xlabel=f"mean {mlabel} score (0-1), attempted cases only")
        ax.set_title(mlabel.upper(), color=C["ink"], fontsize=10.5, pad=8, loc="left")

    axes[0].set_yticks([t for t, _ in ticks])
    axes[0].set_yticklabels([LABEL.get(m, m) for _, m in ticks], color=C["ink"], fontsize=9)
    if not axes[0].yaxis_inverted():
        axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fg.legend(h, l, frameon=False, fontsize=8.5, labelcolor=C["ink2"], ncol=2,
              loc="upper right", bbox_to_anchor=(0.995, 1.005))
    fg.suptitle("What the models know vs what they can guess\n"
                "pre-cutoff bars are over self-selected attempts (UNKNOWN was allowed); "
                "post-cutoff bars are over everything (declining was not)",
                color=C["ink"], fontsize=12, x=0.006, ha="left", y=0.995)
    save(fg, "o4_recall_vs_predict")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default=JUDGED)
    ap.add_argument("--judged-predict", default=JUDGED_PREDICT,
                    help="the --predict/--arm post_cutoff run, for O4")
    args = ap.parse_args()
    C = LIGHT
    plt.rcParams.update({"font.family": "sans-serif", "axes.titleweight": "regular"})

    rows = load(args.judged)
    print(f"{len(rows)} judged rows")
    for m in MODEL_ORDER:
        pre = by(rows, model=m, arm="pre_cutoff")
        att = [r for r in pre if not r["_declined"]]
        print(f"  {LABEL.get(m,m):<16} pre={len(pre):>4} attempts={len(att):>4}")
    os.makedirs(FIGS, exist_ok=True)

    for fn in (o1_disposition, o2_accuracy, o3_mixed_collapse,
               o5_by_disposition, o6_vs_baselines):
        try:
            fn(rows, C)
        except NotImplementedError:
            print(f"  [ ] {fn.__name__} not written yet")

    # O4 needs the second experiment. Skipped with a note rather than crashing,
    # so this script still runs before the predict arm has been collected.
    if os.path.exists(args.judged_predict):
        pr = load(args.judged_predict)
        print(f"{len(pr)} judged rows in the predict run")
        o4_recall_vs_predict(rows, C, pr)
        # Same baseline view on the forecast arm: the mode shifts from
        # `mixed` to `affirmed` there, so the bar to clear is different.
        o6_vs_baselines(pr, C, arm='post_cutoff')
    else:
        print(f"  [ ] o4 skipped — no {os.path.basename(args.judged_predict)}")


if __name__ == "__main__":
    main()
