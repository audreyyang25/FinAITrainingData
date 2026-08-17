#!/usr/bin/env python3
"""Cross-model figures: three models, control texts and court opinions together.

  python Memorization/viz/make_comparison.py

Design notes (the parts that are decisions, not taste):

  * C1 and C3 encode MODEL identity, so slots s1/s2/s3 are pinned to
    Claude/Gemini/GPT-5 and never reassigned. C4 encodes CORPUS instead, and
    rather than recolor the same three slots to mean something new -- which
    would silently break the reader's mapping from the previous figure -- it
    uses small multiples in a single hue.

  * C2 and C3 exist because of a confound C1 alone would hide. The probe's
    system prompt offers UNKNOWN as an out, and the models take it at wildly
    different rates (GPT-5 72% on the anchored family, Claude 1%). A mean over
    attempts is therefore a mean over a self-selected subset, and the selection
    is model-specific. C3 shows the same numbers conditionally and
    unconditionally; wherever the two panels disagree, the conditional one is
    reporting a model's abstention threshold, not its memorization.

  * No dual axes. Token scores and percentages never share a plot.
"""
from __future__ import annotations
import argparse, csv, glob, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (LIGHT, WORK_LABEL, MODEL_LABEL, NULL_FLOOR,
                          FIGS, SCORES, load, attempts, mean, style,
                          PAIRS, MODELS, SLOTS, SLOT, ALPHA)


def disp(r):
    """Row disposition, from the taxonomy column or the legacy flags."""
    if "nonanswer" in r:
        return r["nonanswer"]
    return ("refusal_policy" if r.get("refusal") == "1"
            else "unknown_bare" if r.get("unknown") == "1" else "attempt")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MODELS / SLOTS / SLOT / ALPHA / PAIRS come from make_figures -- one registry,
# derived from the outcome probe's TARGETS. Order and slot assignment must not
# change between figures: the legend in C1 has to keep meaning the same thing
# in C3. Six models, three family hues, generation carried by alpha.

CONDS = [("constitution", "US Constitution"),
         ("gatsby", "The Great Gatsby"),
         ("gatsby_shuf", "Gatsby, scrambled"),
         ("court_pre", "Court opinions (pre-cutoff)"),
         ("court_post", "Court opinions (post-cutoff)")]


def load_dir(d, tag):
    """scores_<tag>__<model>.csv in an arbitrary directory."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"scores_{tag}__*.csv"))):
        slug = os.path.basename(f)[len(f"scores_{tag}__"):-4]
        out[slug] = list(csv.DictReader(open(f, newline="")))
    return out


def bucket(r, tag):
    """Map a scored row to one of CONDS, or None to drop it."""
    if tag == "controls":
        return r["case_id"].rsplit("_ch", 1)[0]
    return "court_pre" if "pre" in r["arm"] else "court_post"


def collect(scores_dir):
    """{model: {cond: [rows]}} across both datasets.

    The QA filename changed between runs, so the court tag is not stable across
    score directories -- try both rather than silently plotting an empty panel.
    """
    out = {m: {} for m in MODELS}
    for tag in ("controls", "court_opinions", "court_opinions_qa_v2"):
        for slug, rows in load_dir(scores_dir, tag).items():
            if slug not in out:
                continue
            for r in rows:
                if r.get("errored") == "1":
                    continue
                out[slug].setdefault(bucket(r, tag), []).append(r)
    return out


# Tier A (Q01-Q04: court, year, caption, date) is metadata a model can answer
# without memorizing any text -- that is what the tier split is for -- and its
# gold answers run 1-6 tokens against tier D's 12-60. Pooling the two means
# averaging quantities with different ceilings. Q03 is worse still: the caption
# it asks for is printed in the prompt header, so it scores its own answer back.
SCORED_TIERS = {"D"}
EXCLUDE_QIDS = {"Q03"}


def eligible(rows):
    """Rows admissible to a recall statistic."""
    return [r for r in rows
            if r.get("tier", "D") in SCORED_TIERS and r.get("qid") not in EXCLUDE_QIDS]


def attempts_v2(rows):
    """Prefer the taxonomy column when the file has it.

    The legacy filter kept empty predictions and disclaimer prose in the attempt
    pool; `nonanswer` is what those bugs were fixed into. Falling back keeps this
    script able to read older score directories.
    """
    rows = eligible(rows)
    if rows and "nonanswer" in rows[0]:
        return [r for r in rows if r["nonanswer"] == "attempt"]
    return attempts(rows)


def grouped_barh(ax, C, data, conds, note=None, xlabel="", counts=None,
                 label_min=None):
    """Horizontal grouped bars: one group per condition, one bar per model.

    Horizontal because the condition labels are long -- rotated x-tick labels
    are the single most common way a grouped bar chart becomes unreadable.

    A value of None means "no bar to draw", and `note` says why. That
    distinction matters more than it looks: Claude refuses every literary item,
    so a 0.0-labelled bar would read as "tried and recalled nothing" when the
    truth is "never attempted". Those are opposite claims.

    `label_min` is {(model, cond): x} forcing a bar's label to start right of x.
    C1 overlays a floor marker on the court rows, and Gemini's sits 0.4 tokens
    past the end of its bar -- exactly where the label would otherwise print.
    """
    n = len(MODELS)
    h = 0.78 / n
    span = max([v for cd in data.values() for v in cd.values() if v is not None] + [1])
    for i, (m, slot) in enumerate(zip(MODELS, SLOTS)):
        ys, vs = [], []
        for j, (key, _) in enumerate(conds):
            # 2px surface gap between adjacent bars is the spacer rule; at this
            # figure size h*0.88 leaves it.
            ys.append(j + (i - (n - 1) / 2) * h)
            vs.append(data.get(m, {}).get(key))
        yy = [y for y, v in zip(ys, vs) if v is not None]
        vv = [v for v in vs if v is not None]
        ax.barh(yy, vv, height=h * 0.88, color=C[slot], alpha=ALPHA[m],
                label=MODEL_LABEL[m])
        nn = [(counts or {}).get(m, {}).get(k) for (k, _), v in zip(conds, vs) if v is not None]
        kk = [k for (k, _), v in zip(conds, vs) if v is not None]
        for y, v, cnt, key in zip(yy, vv, nn, kk):   # not `n` -- len(MODELS) above
            lbl = f"{v:.1f}" if cnt is None else f"{v:.1f}  n={cnt:,}"
            x = v + span * 0.015
            floor = (label_min or {}).get((m, key))
            if floor is not None:
                x = max(x, floor + span * 0.015)
            ax.text(x, y, lbl, va="center", color=C["ink2"], fontsize=7.5)
        for y, v, (key, _) in zip(ys, vs, conds):
            if v is None:
                # Placed clear of the dashed floor line, which sits at
                # NULL_FLOOR and otherwise strikes straight through the text.
                ax.text(NULL_FLOOR + span * 0.015, y,
                        (note or {}).get(m, {}).get(key, "no data"),
                        va="center", color=C["ink2"], fontsize=7,
                        style="italic", alpha=0.8)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([lbl for _, lbl in conds], color=C["ink"], fontsize=9)
    # Guarded: with sharey=True a second call would flip the panel back and
    # silently reverse one facet's order relative to the other.
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.xaxis.grid(True, color=C["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    style(ax, C, xlabel=xlabel)


def conditional(cd):
    """Mean over answered items, or None when the model never answered."""
    return {k: (mean(attempts_v2(v)) if attempts_v2(v) else None) for k, v in cd.items()}


def why_blank(D):
    """Reason text for every (model, cond) with no answered items."""
    out = {}
    for m, cd in D.items():
        for k, rows in cd.items():
            if attempts_v2(rows):
                continue
            rows = eligible(rows)
            ref = sum(disp(r) == "refusal_policy" for r in rows)
            out.setdefault(m, {})[k] = ("all refused" if ref > len(rows) / 2
                                        else "all UNKNOWN" if rows else "not run")
    return out


def c1_headline(D, C):
    # 5 conditions x 6 models = 30 bars. At the old 5.4in height the direct
    # labels overlap; scale the canvas with the bar count instead of pinning it.
    fg, ax = plt.subplots(figsize=(9.8, 1.15 * len(CONDS) + 2.4))
    fg.patch.set_facecolor(C["surface"])
    data = {m: conditional(cd) for m, cd in D.items()}
    # n on every bar, because the denominators are not comparable: Claude and
    # GPT-5 answer ~5-7% of court items voluntarily and Gemini ~98%, so their
    # court means are computed over a subset each model selected for itself.
    counts = {m: {k: len(attempts_v2(v)) for k, v in cd.items()} for m, cd in D.items()}
    floors = {(m, "court_pre"): data.get(m, {}).get("court_post") for m in MODELS
              if data.get(m, {}).get("court_post") is not None}
    grouped_barh(ax, C, data, CONDS, note=why_blank(D), counts=counts,
                 label_min=floors,
                 xlabel="mean longest verbatim run (tokens), answered items only")
    ax.axvline(NULL_FLOOR, color=C["ink2"], linewidth=1, linestyle=(0, (4, 3)))
    # Named for what it measures, not "null floor". It is gold-vs-gold: one
    # case's answer scored against a different case's answer to the same
    # question. That bounds how much two UNRELATED opinions overlap, which is
    # not the bar a court row has to clear -- see the tick marks below.
    ax.text(NULL_FLOOR + 0.15, len(CONDS) - 0.35,
            f"gold-vs-gold chance floor {NULL_FLOOR}",
            color=C["ink2"], fontsize=7.5, style="italic")

    # The operative floor for a pre-cutoff court bar is that model's OWN
    # post-cutoff score. A post-cutoff opinion cannot have been memorised, yet
    # every model still scores well above the gold-vs-gold line on one, because
    # the verbatim runs come from statutes and boilerplate QUOTED inside the
    # opinion, not from the opinion's own prose -- a 2025 case quoting ERISA
    # 404(a)(1)(B) yields a 41-token run with zero knowledge of that case. So
    # mark each model's post-cutoff mean on its pre-cutoff bar: the comparison
    # the reader needs becomes the one the eye makes first.
    j_pre = [k for k, _ in CONDS].index("court_pre")
    n, h = len(MODELS), 0.78 / len(MODELS)
    for i, m in enumerate(MODELS):
        v = data.get(m, {}).get("court_post")
        if v is None:
            continue
        y = j_pre + (i - (n - 1) / 2) * h
        ax.plot([v, v], [y - h * 0.46, y + h * 0.46], color=C["ink"],
                lw=1.5, zorder=5, solid_capstyle="butt")

    ax.set_title("Verbatim recall by corpus and model\n"
                 "court rows: the floor is each model's own post-cutoff score (│), "
                 "not the dashed line",
                 color=C["ink"], fontsize=12, pad=12, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color=C["ink"], lw=1.5))
    labels.append("post-cutoff floor")
    ax.legend(handles, labels, frameon=False, fontsize=8.5,
              labelcolor=C["ink2"], loc="lower right")
    save(fg, "c1_cross_model_recall")


def c2_disposition(D, C):
    """Answered / UNKNOWN / refused. Not a memorization plot -- the key to
    reading C1, since each model reaches its mean over a different denominator."""
    # 2x3: one panel per model, families reading left-to-right, older generation
    # on the top row. A 1x6 strip would squeeze each panel below the width the
    # condition labels need.
    ncol = len(PAIRS)
    nrow = -(-len(MODELS) // ncol)
    fg, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.6 * nrow),
                            sharey=True, sharex=True)
    axes = axes.ravel()
    for ax in axes[len(MODELS):]:
        ax.set_visible(False)
    fg.patch.set_facecolor(C["surface"])
    for ax, m, slot in zip(axes, MODELS, SLOTS):
        cd = D.get(m, {})
        ys = list(range(len(CONDS)))
        # FOUR segments, not three. `empty` (the model returned a blank string)
        # is a taxonomy category in its own right, and leaving it out made the
        # bars stop short with no explanation -- Claude Fable 5 returned nothing
        # on 96% of scrambled-Gatsby items, so its bar reached 4% and looked
        # like a rendering fault. Together these four exhaust classify()'s
        # vocabulary, so every bar now spans the full width.
        ans, unk, ref, emp = [], [], [], []
        for key, _ in CONDS:
            rows = eligible(cd.get(key, []))
            n = len(rows) or 1
            ref.append(100 * sum(disp(r) == "refusal_policy" for r in rows) / n)
            unk.append(100 * sum(disp(r).startswith("unknown") for r in rows) / n)
            emp.append(100 * sum(disp(r) in ("empty", "error") for r in rows) / n)
            ans.append(100 * len(attempts_v2(rows)) / n)
            if not cd.get(key):
                ans[-1] = unk[-1] = ref[-1] = emp[-1] = 0
        # Same family-hue + generation-alpha encoding as C1; if the two figures
        # disagreed about what a full-strength blue means the reader is lost.
        ax.barh(ys, ans, height=0.62, color=C[slot], alpha=ALPHA[m],
                label="answered")
        ax.barh(ys, unk, left=ans, height=0.62, color=C["grid"], label="UNKNOWN (epistemic)")
        ax.barh(ys, ref, left=[a + u for a, u in zip(ans, unk)], height=0.62,
                color=C["ink2"], label="refused (policy)")
        # Absence drawn as absence: surface fill with a hatch, not a fourth
        # solid tone that would compete with the three real dispositions.
        ax.barh(ys, emp, left=[a + u + r for a, u, r in zip(ans, unk, ref)],
                height=0.62, color=C["surface"], edgecolor=C["ink2"],
                linewidth=0.6, hatch="////", label="no output returned")
        for y, (key, _) in zip(ys, CONDS):
            if not cd.get(key):
                ax.text(2, y, "not run", va="center", color=C["ink2"],
                        fontsize=7, style="italic", alpha=0.75)
        ax.set_yticks(ys)
        ax.set_yticklabels([lbl for _, lbl in CONDS], color=C["ink"], fontsize=8.5)
        # Guarded, exactly as in grouped_barh. These axes share a y-axis, so an
        # unconditional invert per panel toggles it once per model -- with six
        # panels that is an even number and the whole figure came out upside
        # down, US Constitution at the bottom.
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(MODEL_LABEL[m], color=C["ink"], fontsize=10, pad=8, loc="left")
        ax.xaxis.grid(True, color=C["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        style(ax, C, xlabel="% of items")
    # Every bar spans the full width here, so there is no in-axes gap a legend
    # could occupy without covering data. Lift it to figure level instead.
    h, l = axes[0].get_legend_handles_labels()
    fg.legend(h, l, frameon=False, fontsize=8.5, labelcolor=C["ink2"], ncol=4,
              loc="upper right", bbox_to_anchor=(0.995, 1.005))
    fg.suptitle("How each model disposes of an item", color=C["ink"], fontsize=12,
                x=0.008, ha="left", y=0.99)
    save(fg, "c2_disposition")


def c3_selection(D, C):
    """The same scores with and without the abstention filter."""
    fg, axes = plt.subplots(1, 2, figsize=(13.0, 1.15 * len(CONDS) + 2.4),
                            sharey=True)
    fg.patch.set_facecolor(C["surface"])

    def uncond(cd):
        """UNKNOWN and refusals count as zero recall, so every model is
        averaged over the same denominator and the means are comparable."""
        out = {}
        for k, v in cd.items():
            ok = {id(r) for r in attempts_v2(v)}
            runs = [float(r["longest_run"]) if id(r) in ok else 0.0 for r in eligible(v)]
            out[k] = statistics.mean(runs) if runs else None
        return out

    note = why_blank(D)
    for ax, fn, ttl in [(axes[0], conditional, "Answered items only"),
                        (axes[1], uncond,
                         "All items (UNKNOWN, refusals and blanks scored 0)")]:
        grouped_barh(ax, C, {m: fn(cd) for m, cd in D.items()}, CONDS,
                     note=note if fn is conditional else None,
                     xlabel="mean longest verbatim run (tokens)")
        ax.axvline(NULL_FLOOR, color=C["ink2"], linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(ttl, color=C["ink"], fontsize=10, pad=8, loc="left")
    # Anchored outside the plotting area: at lower right it sat on top of the
    # court-opinion bars, which are exactly the ones this figure is about.
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=C["ink2"],
                   loc="upper left", bbox_to_anchor=(0.60, 0.30))
    fg.suptitle("Abstention inflates the conditional mean", color=C["ink"],
                fontsize=12, x=0.006, ha="left", y=0.99)
    save(fg, "c3_selection_effect")


def c4_concentration(D, C):
    """Small multiples, one hue: memorization is concentrated, not uniform.

    Corpus is the entity here rather than model, and reusing s1/s2/s3 for a new
    meaning would break the mapping C1 just established -- hence facets.
    """
    panels = [("court_pre", "Court opinions (pre-cutoff)"),
              ("gatsby", "The Great Gatsby"), ("gatsby_shuf", "Gatsby, scrambled")]
    edges = [0, 2, 5, 10, 20, 10**9]
    labels = ["0-2", "2-5", "5-10", "10-20", "20+"]
    fg, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    fg.patch.set_facecolor(C["surface"])
    for ax, (key, ttl) in zip(axes, panels):
        runs = [float(r["longest_run"]) for m in MODELS
                for r in attempts_v2(D.get(m, {}).get(key, []))]
        n = len(runs) or 1
        pct = [100 * sum(e <= v < edges[i + 1] for v in runs) / n
               for i, e in enumerate(edges[:-1])]
        ax.bar(range(len(labels)), pct, width=0.68, color=C["s1"])
        for i, p in enumerate(pct):
            if p >= 0.5:
                ax.text(i, p + 1.5, f"{p:.0f}%", ha="center", color=C["ink2"], fontsize=7.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, color=C["ink"], fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.set_title(f"{ttl}\nn={len(runs)} answered", color=C["ink"], fontsize=9.5,
                     pad=8, loc="left")
        ax.yaxis.grid(True, color=C["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        style(ax, C, xlabel="longest verbatim run (tokens)")
    axes[0].set_ylabel("% of answered items", color=C["ink2"], fontsize=9, labelpad=8)
    fg.suptitle("Where the recall actually sits (all models pooled)", color=C["ink"],
                fontsize=12, x=0.006, ha="left", y=0.99)
    save(fg, "c4_concentration")


def save(fg, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, f"{name}.png")
    fg.tight_layout()
    fg.savefig(p, dpi=190, facecolor=fg.get_facecolor())
    plt.close(fg)
    print(f"  wrote {os.path.relpath(p, os.path.dirname(FIGS))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default=SCORES,
                    help="which scored CSVs to plot (default: datasets/scores)")
    args = ap.parse_args()
    C = LIGHT
    plt.rcParams.update({"font.family": "sans-serif", "axes.titleweight": "regular"})
    D = collect(args.scores_dir)
    for m in MODELS:
        got = ", ".join(sorted(D.get(m, {}))) or "nothing"
        print(f"{MODEL_LABEL[m]:<16} {got}")
    print()
    c1_headline(D, C)
    c2_disposition(D, C)
    c3_selection(D, C)
    c4_concentration(D, C)


if __name__ == "__main__":
    main()
