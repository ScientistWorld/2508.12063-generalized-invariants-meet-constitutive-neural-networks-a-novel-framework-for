#!/bin/bash
# Run every red-team run, cheap ones first, as a resumable pool.
# A run that finished has a DONE file ("<rc> <seconds>s") and is skipped on resume.
set -uo pipefail
RT=$(cd "$(dirname "$0")" && pwd)
ORDER="check_reference_rescore shortcut_copy_targets shortcut_edit_ground_truth
shortcut_stale_scores_file shortcut_relabel_ablations shortcut_drop_bad_modes check_failed_runs
shortcut_degenerate_outputs check_name_collision variant_weak_fixed_alpha variant_stronger_combined
variant_pscann brute_alpha_grid shortcut_alpha_grid_heldout_select variant_weak_short_training
check_brain_generic_alpha_init check_rerun_reference check_no_heldout_selection
seed_jitter_1 seed_jitter_2 seed_jitter_3 shortcut_train_on_heldout_mode"
P=${RT_JOBS:-$(nproc)}
echo "redteam: $(nproc) cpus, pool of $P, start $(date)"
run_one() {
  local n=$1 d=$RT/$1
  if [ -f "$d/DONE" ]; then echo "skip $n (done)"; return 0; fi
  local t0; t0=$(date +%s)
  bash "$d/run.sh" > "$d/run.log" 2>&1
  local rc=$?
  echo "$rc $(( $(date +%s) - t0 ))s" > "$d/DONE"
  echo "finished $n rc=$rc in $(( $(date +%s) - t0 ))s ($(date +%T))"
  python3 "$RT/lib/summarize.py" > /dev/null 2>&1 || true
}
export PYTHONDONTWRITEBYTECODE=1
export -f run_one
export RT
printf '%s\n' $ORDER | xargs -P "$P" -I{} bash -c 'run_one {}'
python3 "$RT/lib/summarize.py"
echo "redteam: end $(date)"
