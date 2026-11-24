#!/bin/bash
# WEAKER VARIANT (key idea removed): the exponent is not learned.  The reference's own
# fixed-exponent SI-CANN models (alpha = +-2, same three branches, same trainer) are
# submitted as the method, i.e. placed in the GI-CANN slots.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
tool "$SB" si_into_gi > "$HERE/apply.log" 2>&1
hidden_test "$SB" "$HERE"
