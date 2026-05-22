#!/bin/bash
# End-to-end reproduction entrypoint.
#
# Assumes the environment is installed.

set -e

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

bash scripts/download.sh

rm -f checkpoints/*.pt checkpoints/training_summary.json scoring/scores.json
rm -f outputs/predictions/*.json 2>/dev/null || true

${PYTHON} method/training_runner.py \
  --device "${DEVICE:-cpu}" \
  --materials rubber,brain \
  --modes positive,negative,combined \
  --families gi,si \
  --grad-clip "${GRAD_CLIP:-0.5}"

bash scripts/evaluate.sh
