#!/usr/bin/env python3
"""Figures for the outcome-knowledge probe.

  python Memorization/viz/make_outcome_figures.py
  python Memorization/viz/make_outcome_figures.py --dark

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
import argparse, collections, csv, os, statistics, sys

VIZ = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(VIZ)
sys.path.insert(0, VIZ)
sys.path.insert(0, os.path.join(ROOT, "outcome"))

from make_figures import LIGHT, DARK, FIGS, style          # palette + axis styling
from make_comparison import save                            # writes PNG, handles --dark
from models import TARGETS, LABEL, CUTOFF                   # target suite + cutoffs
from summarize import graded                                # the one definition of "judged"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
REPO = os.path.dirname(os.path.dirname(VIZ))
DS = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")
JUDGED = os.path.join(DS, "outcome_scores", "judged__anthropic__claude-opus-5.csv")

# Family-grouped, older generation first inside each pair. This order is also the
# y-axis order in every figure -- keep it stable so the eye can move between them.
PAIRS = [("anthropic", "s1", ["anthropic/claude-opus-4", "anthropic/claude-fable-5"]),
         ("openai",    "s2", ["openai/gpt-5", "openai/gpt-5.6-sol"]),
         ("google",    "s3", ["google/gemini-2.5-pro", "google/gemini-3.1-pro-preview"])]
MODEL_ORDER = [m for _, _, ms in PAIRS for m in ms]
SLOT = {m: slot for _, slot, ms in PAIRS for m in ms}

# Must match the judge rubric's vocabulary exactly. `remanded` is currently
# never emitted, but omitting it would silently drop those rows if it were.
DISPO = ["affirmed", "reversed", "vacated", "remanded", "dismissed", "mixed", "other"]


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


# ---------------------------------------------------------------------------
def o1_disposition(rows, C, dark):
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
    save(fg, "o1_attempt_vs_decline", dark)


def o2_accuracy(rows, C, dark):
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
    """
    ys, ticks = layout(per_model=2)
    fg, ax = plt.subplots(figsize=(10.2, 6.4))
    fg.patch.set_facecolor(C["surface"])
    n_post = sum(1 for r in rows if r["arm"] == "post_cutoff" and not r["_declined"])

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
            gap = (statistics.mean(r["_outcome"] for r in att)
                   - statistics.mean(r["_reasoning"] for r in att))
            ax.text(0.985, ys[(m, 0)] + 0.5, f"gap {gap:+.2f}   n={len(att)}",
                    va="center", ha="right", color=C["ink2"], fontsize=7.5, style="italic")

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
    save(fg, "o2_outcome_vs_reasoning", dark)


def o3_mixed_collapse(rows, C, dark):
    """FIGURE 3 — where the disposition errors actually are.

    This is the strongest single result in the outcome probe: accuracy is 73% on
    clean dispositions and 35% when the truth is `mixed`, and `mixed` is the
    largest class. Models compress partial outcomes into a clean headline.

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
    # `other` is excluded as a ROW: both sides of that diagonal cell are
    # judge-assigned catch-alls, so its agreement rate measures the rubric rather
    # than the models. It stays as a COLUMN, because columns are what the model
    # claimed and dropping one stops the rows summing to 100 -- 19 of the 168
    # mixed-outcome answers were classed `other`, so that row displayed 89%.
    truth = [d for d in DISPO if d != "other"]
    claimed = list(DISPO)
    att = [r for r in rows if not r["_declined"]
           and r["disposition_actual"].strip().lower() in truth]
    # Rows with no observations would render as an empty band; drop them.
    truth = [d for d in truth
             if any(r["disposition_actual"].strip().lower() == d for r in att)]
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

    fg.suptitle("Where the disposition errors are (all models pooled, attempts only)",
                color=C["ink"], fontsize=12, x=0.006, ha="left", y=0.995)
    save(fg, "o3_mixed_collapse", dark)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--judged", default=JUDGED)
    args = ap.parse_args()
    C = DARK if args.dark else LIGHT
    plt.rcParams.update({"font.family": "sans-serif", "axes.titleweight": "regular"})

    rows = load(args.judged)
    print(f"{len(rows)} judged rows")
    for m in MODEL_ORDER:
        pre = by(rows, model=m, arm="pre_cutoff")
        att = [r for r in pre if not r["_declined"]]
        print(f"  {LABEL.get(m,m):<16} pre={len(pre):>4} attempts={len(att):>4}")
    os.makedirs(FIGS, exist_ok=True)

    for fn in (o1_disposition, o2_accuracy, o3_mixed_collapse):
        try:
            fn(rows, C, args.dark)
        except NotImplementedError:
            print(f"  [ ] {fn.__name__} not written yet")


if __name__ == "__main__":
    main()
