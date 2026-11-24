#!/bin/bash
# Train the paper's method (GI-CANN) on every material and exponent family and write
# outputs/laws/GI-CANN.json, then score it on the paper's materials and the development side.
#
# Usage: bash scripts/method.sh            (about 20 CPU-minutes; uses every core)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
${PYTHON} method/train.py --model GI-CANN --out outputs/laws/GI-CANN.json "$@"
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
