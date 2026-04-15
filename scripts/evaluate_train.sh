#!/bin/bash
# Train-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh but writes scoring/scores_train.json.

set -e

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

mkdir -p scoring

${PYTHON} eval/train/evaluate.py \
    --checkpoints-dir checkpoints \
    --scores-json scoring/scores_train.json \
    "$@"
