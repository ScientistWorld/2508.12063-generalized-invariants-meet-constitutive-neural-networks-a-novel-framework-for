#!/bin/bash
# Train the SI-CANN baseline (the same network with every exponent fixed at +-2) and install its
# laws as the fixed baseline of both evaluators (eval/train/baselines, eval/test/baselines).
#
# Usage: bash scripts/baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
${PYTHON} method/train.py --model SI-CANN --out outputs/baselines/SI-CANN.json "$@"
${PYTHON} scripts/install_baselines.py outputs/baselines/SI-CANN.json
