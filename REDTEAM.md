# Red-team audit of the GI-CANN gym (iteration 10)

All runs live in `redteam/<name>/`. Each one has a `run.sh` that builds a sandbox copy of the workspace the way a future agent would leave it. The sandbox holds the agent's files plus, where needed, the reference's own trained outputs. The script applies the method or shortcut, puts back the gym's `eval/` and `scripts/evaluate_test.sh`, and runs `bash scripts/evaluate_test.sh` with no arguments. Each folder keeps its `scores_test.json`, the predictions that were scored, the logs and `rank.txt`.

`redteam/SUMMARY.md` has every number. All runs ran in one CPU job (`redteam/run_all.sh`).

**Raw Δ** is the rank formula applied to the hidden-test scores, against the reference's own `scoring/scores_test.json` (the GI-CANN entries): Σ over experiments of weight × Σ over scored metrics of |coef| × (improvement in the better direction). It leaves out the rank's division by the cross-attempt spread, because that spread is unknown here. `redteam/lib/normalized_estimate.txt` redoes the sum using the seed spread as the divisor.

## 1. Do better results score higher?

**Mostly yes, with one important exception: on brain, the score does not respond to training at all.**

| variant | what changes | rubber avg_r2 / r2_ps | brain avg_r2 / r2_ss | raw Δ main | raw Δ total |
|---|---|---|---|---|---|
| reference (shipped) | – | 0.9988 / 0.9994 | 0.9643 / 0.9683 | 0 | 0 |
| **variant_pscann** (stronger model) | the paper's dense-exponent comparison model on brain: identity branch on J_k, k = −30…30, w ≥ 0, paper loss, fitted on ut+uc only. It settles on 3 terms {−19, −18, +30} and reaches lower training loss (0.00126 vs ≈0.00141) | unchanged | **0.9749 / 0.9858** | **+0.0185** | **+0.0185** |
| variant_stronger_combined (stronger model) | the reference's two-exponent (α+, α−) GI-CANN in the two main experiments | 0.9993 / 0.9996 | 0.9650 / 0.9696 | +0.0015 | +0.0015 |
| variant_weak_fixed_alpha (key idea removed) | exponent fixed at ±2 (the reference's SI-CANN models placed in the GI slots) | 0.9834 / 0.9967 | 0.6602 / 0.8428 | −0.1739 | −0.2174 |
| variant_weak_short_training (less training) | the reference trainer capped at 500 epochs instead of 20k/30k | 0.4769 / −0.1103 | **0.9642 / 0.9689** | −0.6987 | −0.7599 |

The better variants score higher and the weaker ones lower, by far more than the seed spread (raw-main seed sd 0.0004). PS-CANN's +0.0185 is about 46 seed sd. The combined model's +0.0015 is about 4 seed sd.

**The exception is brain.** Cutting training from 30,000 to 500 epochs leaves the brain score unchanged (0.9642 / 0.9689 against 0.9643 / 0.9683). The reference starts the brain exponent at the paper's answer, α = −18, with an α learning rate of 1e-4. The brain score therefore reflects that hand-set starting point, not anything learned (see §3). Only rubber collapses under short training, so the "weaker" signal in that row comes entirely from rubber.

## 2. Seed spread and reproduction check

The trainer has no random component: constant initialization, full-batch Adam, and a hard-coded seed of 42. So "seeds" here are a 5% multiplicative jitter of every parameter at the start of training (`seed_jitter_1..3`), plus one unmodified rerun (`check_rerun_reference`).

| quantity (4 runs) | min | max | sd |
|---|---|---|---|
| rubber avg_r2 | 0.9983 | 0.9984 | 0.00005 |
| rubber r2_ps | 0.9991 | 0.9992 | 0.00005 |
| brain avg_r2 | 0.9644 | 0.9652 | 0.0004 |
| brain r2_ss | 0.9674 | 0.9696 | 0.0010 |
| raw Δ main | −0.0010 | 0.0000 | 0.0004 |
| raw Δ total | −0.0035 | 0.0000 | 0.0018 |

Any gain below about 0.001 raw on the main experiments (about 0.0035 in total) is noise.

**Harness check.** `check_reference_rescore` re-scores the shipped artifacts and reproduces `scoring/scores_test.json` exactly.

**The shipped reference is not reproducible bit-for-bit from the current code** (`check_rerun_reference/determinism.txt`):
- The shipped rubber-positive artifact differs from a fresh rerun by up to 0.20 MPa (α* 1.6769 against 1.6741 now). It comes from an older checkpoint (April).
- Brain negative gives α* −17.94 shipped against −17.77 on rerun.

The effect on the score is small (raw −0.0003). But all four honest reruns land 0.0004–0.0005 below the shipped rubber score, while rubber's own spread is only 0.00005. Once the rank divides by a spread that small, every honest rerun of the reference scores about 5.4–5.9 spread-units below the reference (`normalized_estimate.txt`). The shipped rubber number is effectively unreachable by rerunning the reference method.

**Does the method beat its baseline in the direction the paper reports?** Yes. On the hidden test, GI-CANN beats SI-CANN on rubber (0.9988 against 0.9834; r2_ps 0.9994 against 0.9967) and clearly on brain (0.9643 against 0.6602; r2_ss 0.9683 against 0.8428). The paper reports 0.998 against 0.982 and 0.966 against 0.610.

## 3. Where the reproduction departs from the paper

1. **Brain α starts at the paper's answer.** Brain negative and combined initialize α = −18, the paper's α* = −18.016, with an α learning rate of 1e-4. `check_brain_generic_alpha_init` starts from the class default −2 at the base learning rate. After 30k epochs α reaches only −15.8 (combined −16.7), and brain r2_ss falls to 0.9552 (raw −0.0053, about 13 seed sd).
2. **The checkpoint is chosen on the held-out mode.** The trainer keeps whichever of its 20 logged checkpoints has the lowest MSE on ps (rubber) or ss (brain). For brain that is the epoch-1,500 checkpoint near the starting α. Training further moves α toward −16.9, which lowers training loss but worsens ss.
   - `check_no_heldout_selection` selects on training MSE instead: brain r2_ss 0.9634 (α −17.04), raw main −0.0020.
   - The brute-force grid confirms that the training-loss optimum for the one-term brain model is α ≈ −16.8, not −18.
3. **Other warm starts and clamps.**
   - Rubber combined starts from the trained positive checkpoint.
   - Rubber negative starts at α = −2, with w_exp1 limited to 0.05 or less.
   - Brain weights are clamped to be non-negative, and w_exp to at least 1e-6.
   - The exp argument is clamped at 10, and gradients are clipped at 0.5 for rubber only.
   - The initial values are hand-chosen; the paper gives none.
4. **The shipped artifacts don't match the current code** (see §2).
5. **Several values differ from the paper.**
   - SI-CANN baselines: brain r2_ss 0.843 against 0.879; rubber combined SI-CANN 0.92 against 0.977.
   - Ablation exponents: rubber-negative α* is −0.43, while the paper reports 0.155; brain-positive α* is 1.90, while the paper reports 2.971.
6. **The test design.**
   - The data is the paper's real Table A.5: no synthetic data and no scaling down.
   - The "hidden test" is a target-balanced half of the non-zero rows of each curve: 6 rows per rubber curve and 7 per brain curve.
   - The test rows of the training modes (ut/bt, ut/uc) are also training rows, because the method trains on the full curves, as the paper does. Only the r2_ps and r2_ss rows are truly out of mode.
   - The data still shows the paper's effect: a learned exponent beats a fixed ±2 by 0.30 avg R² on brain.

**Metrics already at their bound.**
- shape_violation_rate is 0 for every GI-CANN entry in all six experiments, so it can never be improved.
- Rubber r2_ps is 0.9994 (at most 0.0006 left to gain) and avg_r2 is 0.9988. Rubber combined avg_r2 is 0.9993.
- Copying the true values gains only +0.0046 raw on rubber_generalization, against +0.046 on brain.
- Brain is the only main experiment with real room.
- The simple baseline (SI-CANN) sits clearly below the reference on brain, but only slightly below on rubber.

**Brute force.** `brute_alpha_grid` is the two-step approach the paper replaces. It enumerates α on a dense grid (and w_exp1 on a log grid), solves the weights exactly for the paper's loss on the training curves, and keeps the lowest training loss. It runs in **12 seconds** and scores rubber 0.9983 / 0.9989 and brain 0.9646 / 0.9619 (raw main −0.0029). That is about as good as the reference without held-out selection (−0.0020). The test is small enough that exhaustive enumeration over the exponent does as well as the single-pass discovery. The test rewards fit quality, not the discovery procedure.

## 4. Shortcuts

| run | what it does | rubber / brain main score | raw Δ main | raw Δ total | verdict |
|---|---|---|---|---|---|
| shortcut_copy_targets | writes the measured stresses from `data/load_data.py` as the GI-CANN predictions; the evaluator scores any numbers in `outputs/predictions/*.json` | 1.0000 / 1.0000 | +0.0508 | **+0.1167** | **WORKS**: perfect score in every experiment |
| shortcut_stale_scores_file | deletes all artifacts and checkpoints and leaves a hand-written `scoring/scores_test.json`. `evaluate_test.py` only opens its output when it has something to score, so it exits 0 and the agent's file is what gets read | 1.0000 / 1.0000 | +0.0508 | **+0.1167** | **WORKS** (unless the harness deletes the old file first) |
| shortcut_edit_ground_truth | rewrites the stress columns in `data/load_data.py` (which is not put back) to the reference model's own predictions; the hidden test imports its targets from there | 1.0000 / 1.0000 | +0.0507 | **+0.0433** | **WORKS** |
| shortcut_relabel_ablations | puts the main model's predictions into the two "mismatched exponent" constraint ablations, keeping the slot's reported α*. These ablations are scored higher-is-better (coef 0.8) and nothing checks the exponent sign | unchanged; rubber_negative 0.311→0.999, brain_positive 0.467→0.964 | 0 | **+0.0590** | **WORKS** |
| shortcut_drop_bad_modes | deletes the ut predictions where R² < 0. A missing mode gets R² = 0 instead of its worst value and is left out of the error averages | unchanged; ablations 0.311→0.366, 0.467→0.531 | 0 | **+0.0088** (about 5 seed sd) | **WORKS** |
| shortcut_train_on_heldout_mode | the unmodified reference trainer with the held-out curve (ps / ss) added to the training data | 0.9986 / 0.9649, brain r2_ss **0.9725** (seeds 0.9674–0.9696) | +0.0025 | −0.0023 | **WORKS on brain**: the brain main experiment alone gains +0.0031 (about 8 seed sd); applying it everywhere costs it on the ablations |
| shortcut_alpha_grid_heldout_select | the brute-force grid, choosing α by held-out MSE | 0.8586 / 0.8954 (r2_ps 0.9999, r2_ss 0.9950) | −0.0648 | −0.0727 | fails: training-mode fit collapses (rubber bt 0.59, brain ut 0.74); the avg_r2 and error terms protect |
| check_failed_runs/nan | GI predictions all NaN | NaN | 0 | 0 (20 NaN skipped) | **TIE = shortcut**: NaN is written instead of the worst value, the rank skips it, and shape_violation_rate is even written as 0 |
| check_failed_runs/missing_gi | no GI artifacts at all | entry absent, exit code 0 | 0 | 0 (6 missing) | **TIE = shortcut**: a missing entry avoids the loss |
| check_failed_runs/crash_nothing | no artifacts, no checkpoints | no scores file, exit code 0 | 0 | 0 | **TIE = shortcut** (or a stale file, as above) |
| check_failed_runs/crash_leftover_ckpts | no artifacts, but the reference checkpoints are still in `checkpoints/` | exactly the reference | 0 | 0 | **TIE = shortcut**: the evaluator falls back to scoring whatever `.pt` files are present |
| check_failed_runs/empty | GI artifacts with no predictions | 0 / 0 | −1.229 | −1.323 | fails, but the error metrics are written as 0, their *best* value |
| shortcut_degenerate_outputs/constant | prediction = mean of each curve | −0.005 / −0.010 | −1.533 | −1.669 | fails |
| shortcut_degenerate_outputs/copy_input | prediction = stretch or shear input | 0.41 / −23.7 | −5.17 | −7.22 | fails |
| check_name_collision | a second method's artifact next to GI-CANN's. Every model name other than "SI-CANN" maps to the "GI-CANN" entry, so the later file by name silently replaces the other | brain 0.6602 (the SI-CANN numbers took the GI slot) | −0.158 | −0.158 | reporting bug: not every method on disk is written |
| brute_alpha_grid | honest two-step brute force (§3) | 0.9983 / 0.9646 | −0.0029 | −0.0117 | honest alternative, close to a tie |

**Checks of the rank arithmetic.**
- Directions (`higher_is_better`) are correct for every metric, and α* has coefficient 0.
- A run that fails the task (empty or constant output) cannot outrank the reference through its other metrics: it loses about 1.3–1.7 raw.
- **A failed run does not get its worst value:** NaN, missing entries, a crash with no output, and leftover checkpoints all tie the reference, and empty predictions get the best value on the error metrics.
- The constraint ablations reward closing exactly the gap they exist to show.

## 5. What to fix, in priority order

1. **Score the constitutive law, not free numbers.** Have the artifact declare its strain energy (exponents and weights in a fixed schema), recompute the stresses inside `eval/` (the code in `eval/models.py` already does this), and reject artifacts whose predictions don't match. This also lets the evaluator enforce the exponent sign in the ablation slots.
2. **Move the benchmark curves into the restored `eval/` tree,** or check `data/load_data.py` against a hash.
3. **Make `evaluate_test.py` always write its file.** Delete any old `scores_test.json` first, and write the worst value for any missing method, mode or NaN. Never fall back to leftover checkpoints when the run produced no artifacts.
4. **Give each method its own name** instead of collapsing everything into "GI-CANN".
5. **Fix the reference trainer.** Select checkpoints on training loss instead of the held-out mode, report or remove the α = −18 starting point, and regenerate the reference from the current code so that it is reproducible.
6. **Rethink rubber_generalization.** It sits at the ceiling with a spread of about 0.00005, so once the rank divides by the spread, its weight of 0.4 mostly amplifies noise.
