#!/bin/bash
# SEED SPREAD: the reference GI-CANN trainer, unchanged except that every trainable
# parameter is multiplied by (1 + 0.05 * N(0,1)) with seed 3 right before training
# (after the trainer's own warm starts).  The reference has no seed-dependent code path,
# so a 5% init perturbation is how a "seed" is defined for this gym.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
train_gi "$SB" "$HERE/train.log" --jitter-seed 3 --jitter-scale 0.05 || echo "training rc=$?" >> "$HERE/train.log"
hidden_test "$SB" "$HERE"
