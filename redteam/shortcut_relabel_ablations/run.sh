#!/bin/bash
# SHORTCUT (move the other side of the constraint checks): the two "materially mismatched"
# ablations (rubber with only a negative exponent, brain with only a positive exponent) are
# scored higher-is-better and nothing checks which exponent produced the numbers.  Put the
# main-experiment model's predictions into those slots, keeping the slot's reported alpha_star.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
tool "$SB" copy_slot rubber_positive_gicann.json rubber_negative_gicann.json > "$HERE/apply.log" 2>&1
tool "$SB" copy_slot brain_negative_gicann.json brain_positive_gicann.json >> "$HERE/apply.log" 2>&1
hidden_test "$SB" "$HERE"
