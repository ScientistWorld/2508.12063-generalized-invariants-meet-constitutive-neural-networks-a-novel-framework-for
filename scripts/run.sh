#!/bin/bash
# Job script for GI-CANN hyperelastic materials reproduction.
#
# This runs inside the compute container with:
#   - Workspace mounted at /home/user
#   - Python with PyTorch installed
#   - No internet access
#
# The full reproduction pipeline: baseline + GI-CANN + evaluation

set -e

cd /home/user
PYTHON=${PYTHON:-python3}
export PYTHON

echo "========================================="
echo "GI-CANN Hyperelastic Reproduction"
echo "========================================="

# Quick smoke test first
echo ""
echo "--- Smoke test: data loading ---"
/usr/bin/python3 -c "
import sys
sys.path.insert(0, '/home/user')
from data.load_data import get_rubber_dataset, get_brain_dataset, compute_r2
from method.gicann import GICANN
train_ds, test_ds = get_rubber_dataset()
print(f'Rubber train: {len(train_ds)} samples')
print(f'Rubber test: {len(test_ds)} samples')
train_ds2, test_ds2 = get_brain_dataset()
print(f'Brain train: {len(train_ds2)} samples')
print(f'Brain test: {len(test_ds2)} samples')
model = GICANN({'mode': 'positive', 'r': 0.01}).double()
F = train_ds.F[:3]
S_pred = model(F)
print(f'Smoke test: model forward pass OK, output shape: {S_pred.shape}')
print('Data loading and model: OK')
"

echo ""
echo "--- Downloading data (embedded, no-op) ---"
bash scripts/download.sh

echo ""
echo "--- Running SI-CANN baselines ---"
bash scripts/baseline.sh

echo ""
echo "--- Running GI-CANN method ---"
bash scripts/method.sh

echo ""
echo "--- Evaluating results ---"
bash scripts/evaluate.sh

echo ""
echo "========================================="
echo "All done! Results in scoring/scores.json"
echo "========================================="
