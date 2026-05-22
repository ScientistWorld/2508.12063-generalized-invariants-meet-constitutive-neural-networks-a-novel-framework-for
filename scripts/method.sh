#!/bin/bash
# Run the paper's GI-CANN experiments through the current robust trainer.
#
# Usage:
#   bash scripts/method.sh

set -e

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

mkdir -p checkpoints scoring
rm -f outputs/predictions/*.json 2>/dev/null || true

${PYTHON} method/training_runner.py \
  --device "${DEVICE:-cpu}" \
  --materials rubber,brain \
  --modes positive,negative,combined \
  --families gi \
  --grad-clip "${GRAD_CLIP:-0.5}"

bash scripts/evaluate.sh
