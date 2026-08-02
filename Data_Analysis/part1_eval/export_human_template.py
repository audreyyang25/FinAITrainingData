"""export_human_template.py -- build the human rating packet for the cross-model
judging subset.

Emits one row per (case, generator) pair -- the same 50 the panel judged -- carrying
everything a human rater needs to reproduce and critique the LLM judgments:

    item_id, dataset, case_id, generator
    case_question    what the GENERATOR was asked (shared.config ADAPTERS prompt)
    case_answer      the reference the JUDGES scored against (dataset_specs content_ref)
    llm_answer       that one generator's answer -- not all twelve
    human_anchor     <- rater fills: 1.0 / 0.7 / 0.5 / 0.3 / 0.0
    human_score      <- rater fills: optional finer value within the anchor band
    human_why        <- rater fills: why that rating
    judges_comparison <- rater fills: which judges read the answer better, and why
    <judge>_score / <judge>_anchor / <judge>_reasoning   one trio per judge

Three artifacts, same item_id in each:
    human_rating_template.csv   the fillable sheet
    human_rating_packet.md      the same content laid out for reading
    human_rating_rubric.md      the anchor definitions the judges were given

Column order puts the rater's own fields BEFORE the judge columns on purpose. A rater
who reads the judge scores first is anchored by them, and the agreement number stops
measuring independent judgment. Preferred workflow is two passes: --blind first (judge
columns omitted entirely) to collect human_anchor/human_score/human_why, then the full
template to collect judges_comparison. If you only have one pass in you, hide the judge
columns until the rating column is filled.

  python -m part1_eval.export_human_template
  python -m part1_eval.export_human_template --blind      # pass 1: no judge columns
"""

import argparse
import csv
import json
from pathlib import Path

from part1_eval.dataset_specs import SPECS, load_gold
from part1_eval.general_judge import QUALITY_ANCHORS
from shared.config import ADAPTERS, load_dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SUBSET = ROOT / "outputs" / "part1_eval" / "cross_model_subset"
DEFAULT_GENS = ROOT / "generations.json"

HUMAN_FIELDS = ["human_anchor", "human_score", "human_why", "judges_comparison"]


def load_questions(datasets):
    """(dataset, case_id) -> the prompt the generator actually received."""
    q = {}
    for ds in datasets:
        if ds not in ADAPTERS:
            continue
        for case_id, prompt, _truth in load_dataset(ds):
            q[(ds, case_id)] = prompt
    return q


def load_references(datasets):
    """(dataset, case_id) -> the reference text the judges scored against."""
    refs = {}
    for ds in datasets:
        spec = SPECS.get(ds)
        if spec is None:
            continue
        for case_id, gold in load_gold(ds).items():
            refs[(ds, case_id)] = spec["content_ref"](gold)
    return refs


def load_judgments(subset_dir, metric="overall_quality"):
    """-> ({judge: {(ds,case,model): {...}}}, [judge names])."""
    per_judge, names = {}, []
    for detail in sorted(Path(subset_dir).glob("*/judgments_detail.jsonl")):
        judge = detail.parent.name
        names.append(judge)
        rows = {}
        for line in detail.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            block = r.get(metric)
            if isinstance(block, dict):
                rows[(r["dataset"], r["case_id"], r["model"])] = block
        per_judge[judge] = rows
    return per_judge, names


def load_answers(gens_path):
    out = {}
    for g in json.loads(Path(gens_path).read_text()):
        if all(k in g for k in ("dataset", "case_id", "model")):
            out[(g["dataset"], g["case_id"], g["model"])] = g.get("answer") or ""
    return out


def build_rows(manifest, questions, references, answers, per_judge, judges, metric):
    rows = []
    for i, m in enumerate(manifest, 1):
        ds, cid, model = m["dataset"], int(m["case_id"]), m["model"]
        key = (ds, cid, model)
        row = {
            "item_id": i, "dataset": ds, "case_id": cid, "generator": model,
            "case_question": questions.get((ds, cid), ""),
            "case_answer": references.get((ds, cid), ""),
            "llm_answer": answers.get(key, ""),
        }
        row.update({f: "" for f in HUMAN_FIELDS})
        for j in judges:
            block = per_judge.get(j, {}).get(key) or {}
            row[f"{j}_score"] = block.get(metric, "")
            row[f"{j}_anchor"] = block.get("anchor", "")
            row[f"{j}_reasoning"] = (block.get("explanation")
                                     or block.get("reasoning") or "")
        rows.append(row)
    return rows


def write_csv(rows, judges, path, blind):
    cols = ["item_id", "dataset", "case_id", "generator",
            "case_question", "case_answer", "llm_answer", *HUMAN_FIELDS]
    if not blind:
        for j in judges:
            cols += [f"{j}_score", f"{j}_anchor", f"{j}_reasoning"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def write_packet(rows, judges, path, blind):
    with open(path, "w") as f:
        f.write("# Human rating packet — cross-model judging subset\n\n"
                "One entry per (case, generator) pair. For each: read the question, the "
                "reference answer, and the model's answer, then record your own anchor "
                "in the CSV against the matching `item_id`.\n\n")
        if blind:
            f.write("Judge scores are withheld in this pass by design — rate "
                    "independently first.\n\n")
        else:
            f.write("Judge scores follow each answer. If you are still filling in your "
                    "own rating, do that before reading them.\n\n")
        f.write("Anchors:\n\n```\n" + QUALITY_ANCHORS + "\n```\n\n---\n\n")
        for r in rows:
            f.write(f"## Item {r['item_id']} — {r['dataset']}/{r['case_id']} "
                    f"· generator `{r['generator']}`\n\n")
            f.write("### Question asked\n\n> "
                    + r["case_question"].replace("\n", "\n> ") + "\n\n")
            f.write("### Reference answer\n\n> "
                    + r["case_answer"].replace("\n", "\n> ") + "\n\n")
            f.write("### Model's answer\n\n> "
                    + r["llm_answer"].replace("\n", "\n> ") + "\n\n")
            if not blind:
                f.write("### Judge scores\n\n| judge | anchor | score | reasoning |\n"
                        "|---|---|---|---|\n")
                for j in judges:
                    reason = str(r.get(f"{j}_reasoning", "")).replace("\n", " ").replace("|", "\\|")
                    f.write(f"| {j} | {r.get(f'{j}_anchor','')} | "
                            f"{r.get(f'{j}_score','')} | {reason} |\n")
                f.write("\n")
            f.write("---\n\n")
    return path


def write_rubric(path):
    Path(path).write_text(
        "# Rating rubric\n\n"
        "Give the answer ONE cumulative quality score — how good an answer it is to "
        "this case, taking together its legal grounding, the correctness of its "
        "conclusion, and its coverage of the issues that matter.\n\n"
        "**Treat the reference analysis (`case_answer`) as correct.** A response that "
        "reaches a different bottom-line conclusion is wrong, however well argued. The "
        "case facts are given so you can judge whether the response's reasoning actually "
        "holds here — not merely whether it echoes the reference's wording. The reference "
        "is authoritative but not exhaustive: an additional point that is correct on these "
        "facts should not be penalised.\n\n"
        "Pick the closest anchor, then optionally adjust by up to 0.1 within that band "
        "(`human_score`). These are the exact anchors the LLM judges were given.\n\n"
        "```\n" + QUALITY_ANCHORS + "\n```\n\n"
        "Judge substance, not wording, length, tone, or formatting. A short answer that "
        "is correct and well grounded outranks a long one that is not.\n\n"
        "`human_why` — 1-3 sentences on what drove your rating.\n\n"
        "`judges_comparison` — after you have recorded your own rating: which judges "
        "read this answer well, which misread it, and what they missed or over-weighted.\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset-dir", default=str(DEFAULT_SUBSET))
    ap.add_argument("--generations", default=str(DEFAULT_GENS))
    ap.add_argument("--metric", default="overall_quality",
                    choices=["overall_quality", "content_similarity"])
    ap.add_argument("--blind", action="store_true",
                    help="omit judge columns (pass 1: independent human rating)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    subset_dir = Path(args.subset_dir)
    out_dir = Path(args.out_dir) if args.out_dir else subset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = subset_dir / "subset_manifest.csv"
    with open(manifest_path) as fh:
        manifest = list(csv.DictReader(fh))
    datasets = {m["dataset"] for m in manifest}

    per_judge, judges = load_judgments(subset_dir, args.metric)
    if not judges and not args.blind:
        print(f"[warn] no judgments under {subset_dir} -- judge columns will be blank. "
              "Run part1_eval.cross_judge_subset first, or pass --blind.")

    rows = build_rows(manifest, load_questions(datasets), load_references(datasets),
                      load_answers(args.generations), per_judge, judges, args.metric)

    suffix = "_blind" if args.blind else ""
    csv_path = write_csv(rows, judges, out_dir / f"human_rating_template{suffix}.csv", args.blind)
    md_path = write_packet(rows, judges, out_dir / f"human_rating_packet{suffix}.md", args.blind)
    rub_path = write_rubric(out_dir / "human_rating_rubric.md")

    missing_q = sum(1 for r in rows if not r["case_question"])
    missing_r = sum(1 for r in rows if not r["case_answer"])
    missing_a = sum(1 for r in rows if not r["llm_answer"])
    print(f"{len(rows)} items · judges: {judges or '(none yet)'}")
    print(f"missing question/reference/answer: {missing_q}/{missing_r}/{missing_a}")
    for p in (csv_path, md_path, rub_path):
        print("wrote", p)


if __name__ == "__main__":
    main()
