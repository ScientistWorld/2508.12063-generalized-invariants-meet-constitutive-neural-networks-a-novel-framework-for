#!/bin/bash
# CHECK: the hidden test applied to the reference's own artifacts reproduces
# scoring/scores_test.json exactly (sanity check of this harness).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"
NAME=$(basename "$HERE")
SB=$(new_sandbox "$NAME" 1 1)
hidden_test "$SB" "$HERE"
$PY -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); print("identical to scoring/scores_test.json" if a==b else "DIFFERENT from scoring/scores_test.json")' \
  "$HERE/scores_test.json" "$RT/lib/reference_scores_test.json" | tee "$HERE/compare.txt"
