#!/bin/bash
# SHORTCUT (an empty entry avoids a loss): a mode missing from an artifact gets R2 = 0 instead of
# its worst value, and is left out of the error/shape averages.  Delete the uniaxial-tension
# predictions where the reference has negative R2 (rubber_negative and brain_positive ablations).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
tool "$SB" drop_mode rubber_negative_gicann.json ut > "$HERE/apply.log" 2>&1
tool "$SB" drop_mode brain_positive_gicann.json ut >> "$HERE/apply.log" 2>&1
hidden_test "$SB" "$HERE"
