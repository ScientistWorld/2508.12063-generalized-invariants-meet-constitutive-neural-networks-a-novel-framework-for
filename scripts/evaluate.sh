#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script evaluates all checkpoints in checkpoints/ and produces scoring/scores.json.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, matching the structure in scoring/reference.json.
#
# Usage:
#   ./evaluate.sh                           # evaluate all checkpoints
#   ./evaluate.sh --checkpoint ckpt.pt       # evaluate specific checkpoint

set -e

cd /home/user
PYTHON=${PYTHON:-python3}

mkdir -p scoring

python eval/evaluate.py \
    --checkpoints-dir checkpoints \
    --scores-json scoring/scores.json \
    "$@"
