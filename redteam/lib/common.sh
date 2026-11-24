#!/bin/bash
# Shared helpers for red-team runs. Source this file from a run.sh.
#
# Every run works in its own sandbox copy of the workspace, laid out the way a
# future agent's workspace looks when it signals done:
#   data/ method/ baseline/ scripts/ eval/ (agent copy) and scoring/ with the four
#   agent documents (no reference scores), plus optionally the reference's own
#   trained outputs (outputs/predictions/*.json, checkpoints/*.pt) as if the agent
#   had run `bash scripts/method.sh` and `bash scripts/baseline.sh`.
# The hidden test is then run exactly the way the harness runs it: the gym's own
# eval/ and scripts/evaluate_test.sh are put back and `bash scripts/evaluate_test.sh`
# runs with no arguments from the sandbox root.

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RT=$WS/redteam
PY=${PYTHON:-python3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# new_sandbox NAME [copy_ref_outputs=1] [copy_ref_checkpoints=1]  -> prints sandbox path
new_sandbox() {
  local name=$1 outs=${2:-1} ckpts=${3:-1}
  local root=${RT_SANDBOX_ROOT:-$RT/.sandboxes}
  local sb=$root/$name
  rm -rf "$sb"
  mkdir -p "$sb/scoring" "$sb/outputs/predictions" "$sb/checkpoints"
  cp -r "$WS/data" "$WS/method" "$WS/baseline" "$WS/scripts" "$WS/eval" "$sb/"
  cp "$WS"/scoring/TARGETS.md "$WS"/scoring/CONSTRAINTS.md \
     "$WS"/scoring/DIRECTION.md "$WS"/scoring/EXPERIMENTS.md "$sb/scoring/"
  if [ "$outs" = 1 ]; then cp "$WS"/outputs/predictions/*.json "$sb/outputs/predictions/"; fi
  if [ "$ckpts" = 1 ]; then cp "$WS"/checkpoints/*.pt "$sb/checkpoints/"; fi
  find "$sb" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  echo "$sb"
}

# hidden_test SANDBOX OUTDIR
# 1) the agent's own train-slice evaluation (what it would have run last),
# 2) the harness puts back the gym's eval/ and scripts/evaluate_test.sh,
# 3) `bash scripts/evaluate_test.sh` with no arguments,
# 4) keep scores_test.json, logs, the predictions that were scored, and the raw rank.
hidden_test() {
  local sb=$1 out=$2
  mkdir -p "$out"
  (cd "$sb" && bash scripts/evaluate_train.sh) > "$out/evaluate_train.log" 2>&1 \
    || echo "evaluate_train.sh exited non-zero" >> "$out/evaluate_train.log"
  rm -rf "$sb/eval"
  cp -r "$WS/eval" "$sb/eval"
  find "$sb/eval" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  cp "$WS/scripts/evaluate_test.sh" "$sb/scripts/evaluate_test.sh"
  local rc=0
  (cd "$sb" && bash scripts/evaluate_test.sh) > "$out/evaluate_test.log" 2>&1 || rc=$?
  echo "$rc" > "$out/evaluate_test.rc"
  rm -f "$out/scores_test.json" "$out/scores_train.json"
  if [ -f "$sb/scoring/scores_test.json" ]; then
    cp "$sb/scoring/scores_test.json" "$out/scores_test.json"
  else
    echo "no scoring/scores_test.json after the hidden test" >> "$out/evaluate_test.log"
  fi
  if [ -f "$sb/scoring/scores_train.json" ]; then cp "$sb/scoring/scores_train.json" "$out/scores_train.json"; fi
  rm -rf "$out/predictions"
  mkdir -p "$out/predictions"
  cp "$sb"/outputs/predictions/*.json "$out/predictions/" 2>/dev/null || true
  $PY "$RT/lib/rank.py" "$out/scores_test.json" > "$out/rank.txt" 2>&1 || true
  echo "[hidden_test] $out rc=$rc"
}

# train_gi SANDBOX LOGFILE [train_variant.py args...]
# Retrain the GI-CANN family with the reference trainer (method/training_runner.py, via
# lib/train_variant.py for optional patches) from scratch inside the sandbox.  The
# reference's GI artifacts and checkpoints are removed first so a failed config cannot
# hide behind a stale reference file; the reference SI-CANN baseline artifacts stay.
train_gi() {
  local sb=$1 log=$2; shift 2
  rm -f "$sb"/outputs/predictions/*_gicann.json "$sb"/checkpoints/*.pt
  (cd "$sb" && $PY "$RT/lib/train_variant.py" --device cpu --families gi "$@") > "$log" 2>&1
}

# tool SANDBOX SUBCOMMAND [args...] : run lib/shortcut_tools.py inside the sandbox
tool() {
  local sb=$1; shift
  (cd "$sb" && $PY "$RT/lib/shortcut_tools.py" "$@")
}
