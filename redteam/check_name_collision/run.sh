#!/bin/bash
# CHECK (does the hidden test write every method on disk?): add a second method's artifact for
# brain/negative next to the reference GI-CANN one.  The evaluator maps every model name other
# than "SI-CANN" to the "GI-CANN" entry, so one of the two is silently dropped (file-name order).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
tool "$SB" add_collision > "$HERE/apply.log" 2>&1
hidden_test "$SB" "$HERE"
