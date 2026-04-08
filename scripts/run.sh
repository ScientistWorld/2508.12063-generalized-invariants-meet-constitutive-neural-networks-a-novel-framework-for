#!/bin/bash
# Job script for a clean checkpoint refresh after the audit fixes.
# This retrains GI-CANN and SI-CANN on both materials under the current
# constitutive equations, then regenerates scoring/scores.json.

set -e

cd /home/user
mkdir -p logs checkpoints

echo "========================================="
echo "Full GI-CANN/SI-CANN checkpoint refresh"
echo "========================================="

rm -f checkpoints/*.pt checkpoints/*results.json

python3 run_all_training.py \
  --device cuda \
  --materials rubber,brain \
  --modes positive,negative,combined \
  --families gi,si \
  --grad-clip 0.0 \
  2>&1 | tee logs/brain_retrain.log

echo ""
echo "--- Evaluating results ---"
python3 eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json

echo ""
echo "========================================="
echo "Done! Results in scoring/scores.json"
echo "========================================="
