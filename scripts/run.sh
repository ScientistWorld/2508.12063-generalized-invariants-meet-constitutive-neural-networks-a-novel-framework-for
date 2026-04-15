#!/bin/bash
# Designer job: produce train/test split evaluation scores.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs scoring

echo "========================================="
echo "Train/test split evaluation"
echo "========================================="

echo "--- Evaluating train slice ---"
bash scripts/evaluate_train.sh 2>&1 | tee logs/evaluate_train.log

echo "--- Evaluating held-out test slice ---"
bash scripts/evaluate_test.sh 2>&1 | tee logs/evaluate_test.log

echo "--- Validating generated score files ---"
python3 - <<'PY'
import json
import validate
from pathlib import Path

reference = json.loads(Path("scoring/reference.json").read_text())
for name in ("scores_train.json", "scores_test.json"):
    path = Path("scoring") / name
    scores = json.loads(path.read_text())
    errors = validate.validate_scores(scores, reference)
    if errors:
        raise SystemExit(f"{name} validation failed:\n" + "\n".join(errors))
    print(f"{name}: ok")
PY

echo ""
echo "========================================="
echo "Done! Results in scoring/scores_train.json and scoring/scores_test.json"
echo "========================================="
