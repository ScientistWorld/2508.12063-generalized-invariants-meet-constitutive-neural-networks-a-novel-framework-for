#!/bin/bash
# WEAKER VARIANT (less training): the reference trainer capped at 500 epochs per model
# (reference: 20000 rubber / 30000 brain).  Everything else unchanged.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
rm -f "$SB"/outputs/predictions/*_gicann.json "$SB"/checkpoints/*.pt
(cd "$SB" && $PY method/training_runner.py --device cpu --families gi --epochs 500) > "$HERE/train.log" 2>&1 || echo "training rc=$?" >> "$HERE/train.log"
hidden_test "$SB" "$HERE"
