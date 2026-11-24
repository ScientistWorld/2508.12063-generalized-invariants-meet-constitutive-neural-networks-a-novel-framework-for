#!/bin/bash
# Paper view: score every law file in outputs/laws/ on the paper's own two materials
# (Treloar rubber at 20 C, human cortex) and write scoring/scores.json, comparable to the paper's
# Tables 3 and 4 (scoring/reference.json). Not ranked.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
${PYTHON} eval/train/evaluate.py --materials rubber_T20,brain_CX --scores-json scoring/scores.json "$@"
