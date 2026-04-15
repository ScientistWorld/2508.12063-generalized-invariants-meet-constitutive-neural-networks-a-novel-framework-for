#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh but writes scoring/scores_test.json.

set -e

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

mkdir -p scoring

${PYTHON} eval/test/evaluate.py \
    --checkpoints-dir checkpoints \
    --scores-json scoring/scores_test.json \
    "$@"
