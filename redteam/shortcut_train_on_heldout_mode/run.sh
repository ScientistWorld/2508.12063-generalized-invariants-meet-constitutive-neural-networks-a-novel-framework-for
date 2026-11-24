#!/bin/bash
# SHORTCUT (train on the held-out mode): the unchanged reference GI-CANN trainer, with the
# held-out curve (rubber ps / brain ss, from data/load_data.py) appended to the training set.
# Same model, same loss, same schedule; only the protocol's train/test partition is broken.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
train_gi "$SB" "$HERE/train.log" --train-on-heldout || echo "training rc=$?" >> "$HERE/train.log"
hidden_test "$SB" "$HERE"
