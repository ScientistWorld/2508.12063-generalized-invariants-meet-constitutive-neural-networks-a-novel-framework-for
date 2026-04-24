#!/bin/bash
# Designer job: produce train/test split evaluation scores.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs scoring
rm -f scoring/scores_train.json scoring/scores_test.json

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
baseline_scores = json.loads(Path("scoring/scores.json").read_text())
score_payloads = {}

for name in ("scores_train.json", "scores_test.json"):
    path = Path("scoring") / name
    scores = json.loads(path.read_text())
    errors = validate.validate_scores(scores, reference)
    if errors:
        raise SystemExit(f"{name} validation failed:\n" + "\n".join(errors))
    score_payloads[name] = validate._extract_scores_experiments(scores)
    print(f"{name}: ok")

train_exps = set(score_payloads["scores_train.json"])
test_exps = set(score_payloads["scores_test.json"])
baseline_exps = set(validate._extract_scores_experiments(baseline_scores))

if train_exps != test_exps:
    raise SystemExit(
        "train/test experiment mismatch:\n"
        f"  train-only: {sorted(train_exps - test_exps)}\n"
        f"  test-only: {sorted(test_exps - train_exps)}"
    )

missing_from_split = baseline_exps - train_exps
if missing_from_split:
    raise SystemExit(f"split scores missing experiments present in scoring/scores.json: {sorted(missing_from_split)}")

for exp_name in sorted(train_exps):
    train_methods = set(score_payloads["scores_train.json"][exp_name])
    test_methods = set(score_payloads["scores_test.json"][exp_name])
    if train_methods != test_methods:
        raise SystemExit(
            f"{exp_name}: train/test method mismatch:\n"
            f"  train-only: {sorted(train_methods - test_methods)}\n"
            f"  test-only: {sorted(test_methods - train_methods)}"
        )
    print(f"{exp_name}: methods ok ({len(train_methods)})")
PY

echo ""
echo "========================================="
echo "Done! Results in scoring/scores_train.json and scoring/scores_test.json"
echo "========================================="
