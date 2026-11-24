#!/bin/bash
# CHECK (does the reference discover alpha, or is it handed the paper's answer?):
# brain negative / combined GI-CANN start alpha_minus at the class default -2 instead of
# the paper's -18 (and train alpha at the base lr, since the 1e-4 warm-start lr cannot
# move it by more than ~3).  All other slots keep the reference artifacts.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
rm -f "$SB"/outputs/predictions/brain_negative_gicann.json "$SB"/outputs/predictions/brain_combined_gicann.json
(cd "$SB" && $PY "$RT/lib/train_variant.py" --device cpu --families gi --materials brain --modes negative,combined --brain-generic-alpha) > "$HERE/train.log" 2>&1 || echo "training rc=$?" >> "$HERE/train.log"
hidden_test "$SB" "$HERE"
