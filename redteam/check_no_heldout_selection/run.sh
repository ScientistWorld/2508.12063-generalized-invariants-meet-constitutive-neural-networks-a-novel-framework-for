#!/bin/bash
# CHECK (a real fix of a protocol leak): the reference trainer keeps the checkpoint with
# the lowest MSE on the HELD-OUT mode (ps / ss) among its 20 logged checkpoints.  Here
# the same trainer keeps the checkpoint with the lowest training-mode MSE instead.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
train_gi "$SB" "$HERE/train.log" --select-on-train || echo "training rc=$?" >> "$HERE/train.log"
hidden_test "$SB" "$HERE"
