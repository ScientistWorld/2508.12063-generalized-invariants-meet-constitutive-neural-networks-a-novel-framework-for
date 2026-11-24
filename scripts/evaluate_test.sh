#!/bin/bash
# Hidden-test evaluation: scores every law file in outputs/laws/ (plus the fixed SI-CANN
# baseline) on the test materials and writes scoring/scores_test.json. Never trains anything.
# Usage: bash scripts/evaluate_test.sh         (no arguments)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
${PYTHON} eval/test/evaluate.py
