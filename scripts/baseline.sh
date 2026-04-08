#!/bin/bash
# Run SI-CANN baseline experiments through the current robust trainer.
#
# Usage:
#   bash scripts/baseline.sh

set -e

cd /home/user
PYTHON=${PYTHON:-python3}

mkdir -p checkpoints scoring

${PYTHON} run_all_training.py \
  --device "${DEVICE:-cpu}" \
  --materials rubber,brain \
  --modes positive,negative,combined \
  --families si \
  --grad-clip "${GRAD_CLIP:-0.5}"

bash scripts/evaluate.sh
