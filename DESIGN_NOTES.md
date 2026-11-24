> **Superseded (iteration 11).** This file describes the gym before the hardening pass. The current design (declared strain-energy laws scored by `eval/`, the authors' exact data, development and test materials) is described in `HARDENING.md`, `briefing/` and `scoring/`.

# Evaluation Split Design Notes

## Checked

- Compared `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`; they call separate slice evaluators and write `scoring/scores_train.json` / `scoring/scores_test.json`.
- Checked for hardcoded method defaults in the slice scripts; both enumerate artifacts/checkpoints rather than forcing a fixed method name.
- Checked for train/test score stubs; none are present.
- Checked `scoring/reference.json` for paper-agreement metric names; no nonzero metric rewards matching a published value.
- Checked eval attachment to method artifacts; both slices prefer `outputs/predictions/*.json` and only fall back to legacy checkpoints.

## Changed

- Added anti-shortcut metrics to both train and held-out test scoring:
  - `avg_normalized_rmse` catches magnitude errors that can hide behind high R2.
  - `tail_normalized_mae` stresses largest-deformation rows, where hyperelastic behavior is most informative.
  - `shape_violation_rate` penalizes local trend reversals against observed stress trends.
- Updated checkpoint fallback scoring in both slice evaluators to emit the same new metrics as artifact scoring.
- Replaced the initial every-other-row split with a target-balanced complementary row split after validation showed the small rubber negative ablation was too sensitive to an unrepresentative held-out subset.
- Updated `scoring/reference.json` so every metric has `higher_is_better`, `coefficient`, and `description`.
- Reduced split-shape leakage through the shared artifact helper: slice row selection now lives in `eval/train/evaluate.py` and `eval/test/evaluate.py`, while `eval/artifacts.py` only scores caller-selected rows.
- Kept `alpha_star` as a zero-coefficient audit field so the benchmark does not reward matching the paper's exponent value.

## Runtime Concerns For Runner Iteration

- The split still uses a small tabular dataset, so train/test gaps may be subtle for smooth methods. The added magnitude, tail, and shape metrics should help, but runners should watch for saturation near perfect scores.
- `scripts/run.sh` exports prediction artifacts before scoring, so the new full-score metrics should be generated through the artifact path. The legacy full-evaluator checkpoint fallback was not the focus of this design pass.
- I only ran static checks (`bash -n`, grep, JSON parsing for `reference.json`) and did not run evaluators or compute jobs, per instruction.
