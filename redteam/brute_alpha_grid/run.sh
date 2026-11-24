#!/bin/bash
# BRUTE FORCE the paper's method exists to avoid (the two-step approach): enumerate the
# exponent on a dense grid (and w_exp1 on a log grid), solve the linear weights exactly for
# the paper's loss on the training curves, keep the grid point with the lowest TRAINING
# loss.  Same constitutive family and stress equations as GI-CANN.  All six GI slots.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
(cd "$SB" && $PY "$RT/lib/brute_alpha.py" --select train) > "$HERE/fit.log" 2>&1
hidden_test "$SB" "$HERE"
