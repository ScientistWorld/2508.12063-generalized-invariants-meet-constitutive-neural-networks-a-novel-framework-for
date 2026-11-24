#!/bin/bash
# CHECK (reproduction + determinism): retrain the reference GI-CANN family from scratch
# with the unmodified trainer (python3 method/training_runner.py --families gi), exactly
# as scripts/method.sh does, then run the hidden test.  The trainer has no random
# component (constant init, full-batch Adam, hard-coded seed 42), so this also measures
# whether the shipped artifacts are reproducible bit-for-bit.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
rm -f "$SB"/outputs/predictions/*_gicann.json "$SB"/checkpoints/*.pt
(cd "$SB" && $PY method/training_runner.py --device cpu --families gi) > "$HERE/train.log" 2>&1 || echo "training rc=$?" >> "$HERE/train.log"
hidden_test "$SB" "$HERE"
tool "$SB" compare_preds "$SB/outputs/predictions" "$WS/outputs/predictions" > "$HERE/determinism.txt" 2>&1 || true
