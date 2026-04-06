#!/bin/bash
# Run the GI-CANN method from the paper.
#
# This script trains the GI-CANN model (the paper's proposed method) on both
# rubber and brain tissue datasets, in all three configurations:
# positive (R+), negative (R-), and combined (R).
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user
PYTHON=${PYTHON:-python3}

mkdir -p checkpoints

echo "=== Training GI-CANN (positive) on rubber ==="
${PYTHON} method/train.py \
    --material rubber \
    --mode positive \
    --epochs 5000 \
    --lr 0.001 \
    --r 0.01 \
    --seed 42 \
    --checkpoint-dir checkpoints \
    --output checkpoints/rubber_positive_results.json

echo ""
echo "=== Training GI-CANN (negative) on rubber ==="
${PYTHON} method/train.py \
    --material rubber \
    --mode negative \
    --epochs 5000 \
    --lr 0.001 \
    --r 0.01 \
    --seed 42 \
    --checkpoint-dir checkpoints \
    --output checkpoints/rubber_negative_results.json

echo ""
echo "=== Training GI-CANN (combined) on rubber ==="
${PYTHON} method/train.py \
    --material rubber \
    --mode combined \
    --epochs 5000 \
    --lr 0.001 \
    --r 0.01 \
    --seed 42 \
    --checkpoint-dir checkpoints \
    --output checkpoints/rubber_combined_results.json

echo ""
echo "=== Training GI-CANN (negative) on brain ==="
${PYTHON} method/train.py \
    --material brain \
    --mode negative \
    --epochs 5000 \
    --lr 0.001 \
    --r 0.01 \
    --seed 42 \
    --checkpoint-dir checkpoints \
    --output checkpoints/brain_negative_results.json

echo ""
echo "=== Training GI-CANN (positive) on brain ==="
${PYTHON} method/train.py \
    --material brain \
    --mode positive \
    --epochs 5000 \
    --lr 0.001 \
    --r 0.01 \
    --seed 42 \
    --checkpoint-dir checkpoints \
    --output checkpoints/brain_positive_results.json

echo ""
echo "=== Training GI-CANN (combined) on brain ==="
${PYTHON} method/train.py \
    --material brain \
    --mode combined \
    --epochs 5000 \
    --lr 0.001 \
    --r 0.01 \
    --seed 42 \
    --checkpoint-dir checkpoints \
    --output checkpoints/brain_combined_results.json

echo ""
echo "=== All GI-CANN training complete ==="
