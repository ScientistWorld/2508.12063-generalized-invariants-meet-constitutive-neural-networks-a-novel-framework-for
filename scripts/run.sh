#!/bin/bash
# Targeted retraining after fixing the mode-specific stress computation.

set -e

cd /home/user
mkdir -p logs checkpoints scoring

echo "========================================="
echo "Brain GI-CANN recovery run"
echo "========================================="

rm -f \
  checkpoints/brain_negative.pt \
  checkpoints/brain_combined.pt \
  checkpoints/training_summary.json \
  scoring/scores.json

python3 run_all_training.py \
  --device cuda \
  --materials brain \
  --modes negative,combined \
  --families gi \
  --epochs 30000 \
  --grad-clip 0.0 \
  2>&1 | tee logs/brain_gicann_recovery.log

echo ""
echo "--- Evaluating all checkpoints ---"
bash scripts/evaluate.sh

echo ""
echo "========================================="
echo "Done! Results in scoring/scores.json"
echo "========================================="
