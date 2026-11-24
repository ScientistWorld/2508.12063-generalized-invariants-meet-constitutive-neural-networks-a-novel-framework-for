#!/bin/bash
# Hardening checks (iteration 11): the reference, its noise, better and weaker variants and the
# shortcuts, each run through the hidden test exactly as the harness runs it.
#
# Every run gets a sandbox laid out like an agent's workspace (data/, method/, scripts/, eval/ and
# outputs/laws/ with whatever the agent left). The gym's own eval/ and scripts/evaluate_test.sh are
# then put back and `bash scripts/evaluate_test.sh` runs with no arguments from the sandbox root.
# Results: redteam/hardening/results/<run>/{scores_test.json,evaluate_test.log,laws/}.
# Resumable: a run with a DONE file is skipped. Training runs are cached in outputs/*runs*/.
set -uo pipefail
WS=$(cd "$(dirname "$0")/../.." && pwd)
HD=$WS/redteam/hardening
LIB=$HD/lib
RES=$HD/results
SB=${SB_ROOT:-/tmp/hardening_sandboxes}
JOBS=${JOBS:-$(nproc)}
PY=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
REF_LAWS=$WS/outputs/laws/GI-CANN.json
mkdir -p "$RES" "$SB"

new_sandbox() {  # name -> path
  local sb=$SB/$1
  rm -rf "$sb"; mkdir -p "$sb/outputs/laws" "$sb/scoring"
  cp -r "$WS/data" "$WS/method" "$WS/scripts" "$WS/eval" "$sb/"
  find "$sb" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
  echo "$sb"
}

hidden_test() {  # name sandbox
  local name=$1 sb=$2 out=$RES/$1
  mkdir -p "$out"
  rm -rf "$sb/eval"; cp -r "$WS/eval" "$sb/eval"
  cp "$WS/scripts/evaluate_test.sh" "$sb/scripts/evaluate_test.sh"
  local rc=0
  (cd "$sb" && bash scripts/evaluate_test.sh) > "$out/evaluate_test.log" 2>&1 || rc=$?
  echo "$rc" > "$out/evaluate_test.rc"
  rm -f "$out/scores_test.json"
  [ -f "$sb/scoring/scores_test.json" ] && cp "$sb/scoring/scores_test.json" "$out/scores_test.json"
  (cd "$sb" && $PY eval/train/evaluate.py --scores-json "$out/scores_train.json") > "$out/evaluate_train.log" 2>&1
  rm -rf "$out/laws"; mkdir -p "$out/laws"; cp "$sb"/outputs/laws/*.json "$out/laws/" 2>/dev/null
  echo "[hidden_test] $name rc=$rc"
}

run() {  # name command...   (command runs with cwd = sandbox, $SBX set)
  local name=$1; shift
  if [ -n "${ONLY:-}" ] && ! [[ $name =~ $ONLY ]]; then return; fi
  if [ -f "$RES/$name/DONE" ]; then echo "skip $name"; return; fi
  local sbx; sbx=$(new_sandbox "$name")
  local t0; t0=$(date +%s)
  mkdir -p "$RES/$name"
  (cd "$sbx" && SBX="$sbx" eval "$@") > "$RES/$name/build.log" 2>&1
  hidden_test "$name" "$sbx"
  echo "$(( $(date +%s) - t0 ))s" > "$RES/$name/DONE"
}

TRAIN="$PY $WS/method/train.py --runs-dir $WS/outputs/runs --jobs $JOBS"
VAR="$PY $LIB/train_variant.py --jobs $JOBS --runs-dir $WS/outputs/variant_runs"

# --- reference and its noise -------------------------------------------------------------------
run reference_rescore           "cp $REF_LAWS outputs/laws/GI-CANN.json"
run noise_ref_seeds_20_39       "$TRAIN --model GI-CANN --seeds 20-39 --out outputs/laws/GI-CANN.json"
for s in 0 1 2 3 4; do
  run "single_paper_run_seed$s" "$TRAIN --model GI-CANN --seeds $s --select first-seed --out outputs/laws/GI-CANN.json"
done
# --- variants (honest) -------------------------------------------------------------------------
run variant_l1_tuned_on_dev     "$VAR rtuned outputs/laws/GI-CANN.json --seeds 0-19"
run variant_scalefree_loss      "$VAR scalefree outputs/laws/GI-CANN.json --seeds 0-19"
run variant_no_converge_phase   "$VAR noconverge outputs/laws/GI-CANN.json --seeds 0-19"
run variant_two_invariants      "$VAR twoinv outputs/laws/GI-CANN.json --seeds 0-19"
run variant_pscann_dense        "$PY $LIB/pscann.py outputs/laws/GI-CANN.json"
run brute_alpha_grid            "$PY $LIB/brute_alpha.py outputs/laws/GI-CANN.json"
run weak_short_training         "$VAR short outputs/laws/GI-CANN.json --seeds 0-19 --epochs 300"
run weak_fixed_alpha            "cp $WS/outputs/baselines/SI-CANN.json outputs/laws/GI-CANN.json"
# --- shortcuts ---------------------------------------------------------------------------------
for sc in copy_targets predictions_in_law relabel_ablations drop_slots nan_weights negative_weight_term \
          huge_exponent zero_exponent_only dense_law empty_outputs name_collision override_baseline \
          stale_scores_file edit_ground_truth; do
  run "shortcut_$sc" "$PY $LIB/shortcuts.py $sc \$SBX $REF_LAWS"
done
# --- the shortcut left to the grader: fitting held-out curves obtained outside the workspace ----
run left_fit_heldout_answers    "$PY $LIB/brute_alpha.py outputs/laws/GI-CANN.json --extra-gt $WS/eval/test/ground_truth.json"

$PY "$LIB/summarize.py" > "$HD/SUMMARY.md"
cat "$HD/SUMMARY.md"
