#!/bin/bash
# SHORTCUT attempts with degenerate outputs:
#   constant    every prediction = the mean of the visible curve
#   copy_input  every prediction = the deformation input (stretch / shear strain)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "${NAME}_constant" 1 0); tool "$SB" constant --gi-only > /dev/null; hidden_test "$SB" "$HERE/constant"
SB=$(new_sandbox "${NAME}_copy_input" 1 0); tool "$SB" copy_input --gi-only > /dev/null; hidden_test "$SB" "$HERE/copy_input"
