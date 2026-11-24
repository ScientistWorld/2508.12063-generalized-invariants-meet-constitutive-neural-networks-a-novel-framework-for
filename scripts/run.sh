#!/bin/bash
# CPU scoring job: export model-agnostic prediction artifacts from the packaged
# checkpoints, regenerate all scoring files, and run structural checks.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs outputs/predictions scoring

echo "========================================="
echo "GI-CANN artifact export and validation"
echo "========================================="

rm -f outputs/predictions/*.json

python3 method/export_predictions.py \
  --checkpoints-dir checkpoints \
  --predictions-dir outputs/predictions \
  --device cpu \
  2>&1 | tee logs/export_predictions.log

echo "--- Evaluating full benchmark from prediction artifacts ---"
bash scripts/evaluate.sh 2>&1 | tee logs/evaluate.log

echo "--- Evaluating train slice from prediction artifacts ---"
bash scripts/evaluate_train.sh 2>&1 | tee logs/evaluate_train.log

echo "--- Evaluating held-out test slice from prediction artifacts ---"
bash scripts/evaluate_test.sh 2>&1 | tee logs/evaluate_test.log

echo "--- Validating generated score files ---"
python3 validate.py --compare 2>&1 | tee logs/validate_compare.log

echo ""
echo "========================================="
echo "Done! Results in scoring/scores.json, scoring/scores_train.json, and scoring/scores_test.json"
echo "========================================="
