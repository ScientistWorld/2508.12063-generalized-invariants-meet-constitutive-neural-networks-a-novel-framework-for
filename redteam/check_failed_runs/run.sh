#!/bin/bash
# CHECK (what does a failed run get?): five ways a run can fail, each scored by the hidden test.
#   empty                 GI artifacts present but with no predictions
#   nan                   GI artifacts full of NaN
#   missing_gi            no GI artifacts at all (SI baseline artifacts only)
#   crash_leftover_ckpts  no artifacts, the reference checkpoints still in checkpoints/
#   crash_nothing         no artifacts, no checkpoints, no score file
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "${NAME}_empty" 1 0); tool "$SB" empty --gi-only > /dev/null; hidden_test "$SB" "$HERE/empty"
SB=$(new_sandbox "${NAME}_nan" 1 0); tool "$SB" nan --gi-only > /dev/null; hidden_test "$SB" "$HERE/nan"
SB=$(new_sandbox "${NAME}_missing_gi" 1 0); rm -f "$SB"/outputs/predictions/*_gicann.json; hidden_test "$SB" "$HERE/missing_gi"
SB=$(new_sandbox "${NAME}_crash_leftover_ckpts" 0 1); hidden_test "$SB" "$HERE/crash_leftover_ckpts"
SB=$(new_sandbox "${NAME}_crash_nothing" 0 0); hidden_test "$SB" "$HERE/crash_nothing"
