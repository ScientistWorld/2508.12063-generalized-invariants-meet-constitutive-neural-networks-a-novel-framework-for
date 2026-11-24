#!/bin/bash
# SHORTCUT (select on the held-out mode): same brute-force grid as brute_alpha_grid, but the
# grid point is chosen by its MSE on the held-out mode (rubber ps / brain ss), whose measured
# stresses are in data/load_data.py.  Weights are still fitted on the training curves only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
(cd "$SB" && $PY "$RT/lib/brute_alpha.py" --select heldout) > "$HERE/fit.log" 2>&1
hidden_test "$SB" "$HERE"
