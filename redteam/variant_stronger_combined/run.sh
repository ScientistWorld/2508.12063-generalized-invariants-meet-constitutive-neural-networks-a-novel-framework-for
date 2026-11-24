#!/bin/bash
# STRONGER VARIANT (richer model): use the reference's two-invariant GI-CANN
# (alpha+ and alpha- learned jointly, the paper's best rubber model) as the method for
# the two main experiments, whose protocol does not restrict the exponent sign.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
tool "$SB" copy_slot rubber_combined_gicann.json rubber_positive_gicann.json --with-meta > "$HERE/apply.log" 2>&1
tool "$SB" copy_slot brain_combined_gicann.json brain_negative_gicann.json --with-meta >> "$HERE/apply.log" 2>&1
hidden_test "$SB" "$HERE"
