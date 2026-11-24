#!/bin/bash
# Development-side evaluation: scores every law file in outputs/laws/ (plus the fixed SI-CANN
# baseline) on the development materials and writes scoring/scores_train.json.
# Usage: bash scripts/evaluate_train.sh        (no arguments)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
${PYTHON} eval/train/evaluate.py
