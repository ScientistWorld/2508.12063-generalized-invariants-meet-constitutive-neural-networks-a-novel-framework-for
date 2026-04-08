# Constraints

These constraints define what must stay fixed so that improvements come from better constitutive modeling, not from changing the task.

## Data Constraints

Use the paper’s Table A.5 stress-stretch values as loaded by [`data/load_data.py`](/home/user/data/load_data.py).

- Rubber:
  - Train on `ut` + `bt`
  - Test on `ps`
- Brain cortex:
  - Train on `ut` + `uc`
  - Test on `ss`

Do not change the numeric data, deformation modes, or train/test partition.

## Model Constraints

The task is constitutive-law discovery for isotropic incompressible hyperelasticity.

- Predictions must come from a valid constitutive model, not a direct black-box mapping from input curve coordinates to output stresses.
- The model should stay compact and interpretable enough that a future scientist can inspect the learned constitutive law.
- Improvements should come from better constitutive modeling or better training, not from adding large generic function approximators.

## Physics Constraints

- Respect incompressibility in the deformation descriptions used by the benchmark.
- Predict the appropriate Cauchy stress component for each loading mode.
- Preserve the same deformation definitions for `ut`, `uc`, `bt`, `ps`, and `ss`.

## Evaluation Constraints

Evaluate with [`scripts/evaluate.sh`](/home/user/scripts/evaluate.sh), which writes [`scoring/scores.json`](/home/user/scoring/scores.json).

- Rubber metric set: `r2_ut`, `r2_bt`, `r2_ps`, `avg_r2`, `alpha_star`
- Brain metric set: `r2_ut`, `r2_uc`, `r2_ss`, `avg_r2`, `alpha_star`

`avg_r2` remains the primary metric. `alpha_star` is included as a report-only descriptor because interpretability matters, but it is not scored directly.

## Resource Constraints

- No pretrained external model weights
- Keep the method small enough that gains reflect modeling quality rather than raw capacity
