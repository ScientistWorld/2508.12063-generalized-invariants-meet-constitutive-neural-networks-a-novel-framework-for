#!/bin/bash
# End-to-end reproduction entrypoint.
#
# Assumes the environment is installed and the workspace is mounted at /home/user.

set -e

cd /home/user
PYTHON=${PYTHON:-python3}

bash scripts/download.sh

rm -f checkpoints/*.pt checkpoints/training_summary.json scoring/scores.json

${PYTHON} run_all_training.py \
  --device "${DEVICE:-cpu}" \
  --materials rubber,brain \
  --modes positive,negative,combined \
  --families gi,si \
  --grad-clip "${GRAD_CLIP:-0.5}"

bash scripts/evaluate.sh
