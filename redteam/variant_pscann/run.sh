#!/bin/bash
# STRONGER VARIANT (paper's dense-exponent comparison model): PS-CANN, an identity-branch
# strain energy on 60 generalized invariants with fixed exponents k = -30..30, w >= 0,
# the paper's loss with L1 r = 0.01, fitted on brain ut + uc only (no held-out selection).
# The paper reports it generalizes better to simple shear than GI-CANN (R2_ss 0.988 vs 0.970).
# Rubber slots keep the reference artifacts.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
(cd "$SB" && $PY "$RT/lib/pscann.py" --slots brain_negative) > "$HERE/fit.log" 2>&1
hidden_test "$SB" "$HERE"
