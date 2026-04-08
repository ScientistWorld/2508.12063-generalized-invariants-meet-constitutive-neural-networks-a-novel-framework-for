#!/bin/bash
# Fresh checkpoint refresh using the current robust trainer and evaluator.

set -e

cd /home/user
mkdir -p logs checkpoints scoring

echo "========================================="
echo "Full GI-CANN/SI-CANN checkpoint refresh"
echo "========================================="

rm -f checkpoints/*.pt checkpoints/training_summary.json scoring/scores.json

python3 run_all_training.py \
  --device cuda \
  --materials rubber,brain \
  --modes positive,negative,combined \
  --families gi,si \
  --grad-clip 0.0 \
  2>&1 | tee logs/full_refresh.log

echo ""
echo "--- Evaluating results ---"
bash scripts/evaluate.sh

echo ""
echo "========================================="
echo "Done! Results in scoring/scores.json"
echo "========================================="
