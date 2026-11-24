#!/bin/bash
# CPU job: reproduce the reference end to end from the final code, then run the hardening checks
# (variants and shortcuts through the hidden test). Resumable: training runs are cached in
# outputs/runs/ and outputs/variant_runs/, finished checks carry a DONE file.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
LOG=redteam/hardening/logs
mkdir -p "$LOG"
JOBS=$(nproc)
echo "cpus: $JOBS  start: $(date)"

# 1. Reference (runs alone): SI-CANN baseline and GI-CANN, best of 20 paper runs per law.
python3 method/train.py --model SI-CANN --jobs "$JOBS" --out outputs/baselines/SI-CANN.json > "$LOG/train_sicann.log" 2>&1
python3 scripts/install_baselines.py outputs/baselines/SI-CANN.json
python3 method/train.py --model GI-CANN --jobs "$JOBS" --out outputs/laws/GI-CANN.json > "$LOG/train_gicann.log" 2>&1
tail -n 1 "$LOG/train_sicann.log" "$LOG/train_gicann.log"
bash scripts/evaluate.sh       > "$LOG/evaluate_paper.log" 2>&1
bash scripts/evaluate_train.sh > "$LOG/evaluate_train.log" 2>&1
bash scripts/evaluate_test.sh  > "$LOG/evaluate_test.log" 2>&1
cat "$LOG/evaluate_paper.log" "$LOG/evaluate_test.log"
echo "reference done: $(date)"

# 2. Determinism: a few reference runs repeated alone must be bit-identical.
python3 redteam/hardening/lib/determinism.py 2>&1 | tee "$LOG/determinism.log"

# 3. Hardening checks through the hidden test.
bash redteam/hardening/run_checks.sh > "$LOG/run_checks.log" 2>&1 || true
tail -n 60 "$LOG/run_checks.log"

# 4. Structural validation.
python3 validate.py 2>&1 | tee "$LOG/validate.log" || true
echo "end: $(date)"
