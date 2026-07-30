"""CLI entry point for the general suitability eval.

Judges an existing generations file across whatever datasets it contains (P12-P16),
using each dataset's own gold and the applicable judge dimensions. Resumable.

Examples
  # The 12 feature-importance models, all datasets present, gold answer excluded:
  python run_eval.py \
      --generations ../feature_importance_exp/outputs/generations.jsonl \
      --out ../results/feature_importance_eval

  # Only the borderline + adversarial sets:
  python run_eval.py --generations <path> --out <dir> --datasets borderline adversarial
"""

import argparse
from pathlib import Path

from general_judge import run_eval, DEFAULT_JUDGE, DEFAULT_WORKERS

HERE = Path(__file__).resolve().parent
DEFAULT_GENS = HERE.parent / "feature_importance_exp" / "outputs" / "generations.jsonl"
DEFAULT_OUT = HERE.parent / "results" / "feature_importance_eval"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", default=str(DEFAULT_GENS),
                    help="generations JSONL/JSON (default: feature_importance outputs)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory (default: results/feature_importance_eval)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these dataset names (default: all present)")
    ap.add_argument("--exclude", nargs="*", default=["gold"],
                    help="model labels to skip (default: gold)")
    ap.add_argument("--limit", type=int, default=0,
                    help="judge only the first N generations after filtering (0 = all)")
    ap.add_argument("--judge", default=DEFAULT_JUDGE,
                    help=f"judge model slug (default: {DEFAULT_JUDGE}). Known: "
                         "google/gemini-3.5-flash, meta-llama/llama-3.3-70b-instruct. "
                         "Use a separate --out per judge.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"concurrent judge threads (default: {DEFAULT_WORKERS})")
    args = ap.parse_args()

    run_eval(
        args.generations,
        out_dir=args.out,
        exclude_models=set(args.exclude),
        datasets=args.datasets,
        limit=args.limit,
        judge=args.judge,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
