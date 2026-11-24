#!/bin/bash
# End-to-end reproduction: data, the paper's method, the SI-CANN baseline, all score files.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/download.sh
bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate_test.sh
