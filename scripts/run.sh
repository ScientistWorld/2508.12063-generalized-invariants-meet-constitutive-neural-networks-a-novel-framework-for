#!/bin/bash
# Job script for brain-only GI-CANN/SI-CANN retraining after correcting
# the mode-specific stress equations to match Eqs. (15)–(18) in the paper.

set -e

cd /home/user
mkdir -p logs checkpoints

echo "========================================="
echo "Brain-only GI-CANN/SI-CANN retraining"
echo "========================================="

rm -f checkpoints/brain*.pt checkpoints/*brain*.pt

python3 run_all_training.py \
  --device cuda \
  --materials brain \
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
