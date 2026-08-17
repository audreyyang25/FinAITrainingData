#!/usr/bin/env bash
# Full matrix on a rented GPU:  bash run_all.sh meta-llama/Llama-3.1-70B
#
# Validate FIRST on the small model -- every failure mode here is silent:
#   python validate.py --model meta-llama/Llama-3.1-8B
#
# Resumable on (case_id, qid): re-issue after a killed pod and it continues.
set -uo pipefail
BASE="${1:-meta-llama/Llama-3.1-70B}"
INSTRUCT="${2:-${BASE}-Instruct}"
BS="${BATCH_SIZE:-4}"

for model in "$BASE" "$INSTRUCT"; do
  for qa in controls court; do
    echo "=============================================================="
    echo ">>> $model  |  $qa  |  raw   $(date '+%F %T')"
    python score_logprobs.py --model "$model" --qa "$qa" \
      --condition raw --batch-size "$BS" \
      || echo "!!! $model/$qa exited non-zero -- rerun to resume"
  done
done

echo
echo "done. figures (no GPU needed):"
echo "  python make_curve.py --logprobs '<datasets>/logprobs/logprobs__controls__*.csv' \\"
echo "                       --scores  '<datasets>/scores/scores_controls__meta-llama__llama-3.1-70b-instruct.csv'"
