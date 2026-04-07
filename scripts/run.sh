#!/bin/bash
# Job script for GI-CANN hyperelastic materials reproduction.
#
# Fixes in this run:
# - Rubber combined SI-CANN: clamp LN and negative-ID branches to prevent divergence
# - Brain models: reduced L1 (0.01→0.001), exp branch warm-start for all configs
# - Brain GI-CANN: all branches (id/exp/ln) initialized for negative mode

set -e

cd /home/user
mkdir -p logs checkpoints

echo "========================================="
echo "GI-CANN Training Run (with fixes)"
echo "========================================="

python3 run_all_training.py --epochs 30000 --lr 0.001 --device cuda --grad-clip 0.5 2>&1 | tee logs/training.log

echo ""
echo "--- Evaluating results ---"
python3 eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json

echo ""
echo "========================================="
echo "Done! Results in scoring/scores.json"
echo "========================================="
