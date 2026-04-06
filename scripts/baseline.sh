#!/bin/bash
# Run SI-CANN baseline methods from the paper.
#
# SI-CANN uses fixed classical invariants (α=2 for I₁, α=-2 for I₂).
# The exponents are NOT trainable — only the branch weights are learned.
# This is the paper's main baseline for comparison.
#
# Usage:
#   bash scripts/baseline.sh

set -e

cd /home/user
PYTHON=${PYTHON:-python3}

mkdir -p checkpoints

echo "=== SI-CANN: rubber, positive (α=2) ==="
${PYTHON} method/baseline_train.py --material rubber --mode positive --epochs 5000 --lr 0.001 --seed 42

echo ""
echo "=== SI-CANN: rubber, negative (α=-2) ==="
${PYTHON} method/baseline_train.py --material rubber --mode negative --epochs 5000 --lr 0.001 --seed 42

echo ""
echo "=== SI-CANN: rubber, combined ==="
${PYTHON} method/baseline_train.py --material rubber --mode combined --epochs 5000 --lr 0.001 --seed 42

echo ""
echo "=== SI-CANN: brain, positive (α=2) ==="
${PYTHON} method/baseline_train.py --material brain --mode positive --epochs 5000 --lr 0.001 --seed 42

echo ""
echo "=== SI-CANN: brain, negative (α=-2) ==="
${PYTHON} method/baseline_train.py --material brain --mode negative --epochs 5000 --lr 0.001 --seed 42

echo ""
echo "=== SI-CANN: brain, combined ==="
${PYTHON} method/baseline_train.py --material brain --mode combined --epochs 5000 --lr 0.001 --seed 42

echo ""
echo "=== All SI-CANN baseline training complete ==="
