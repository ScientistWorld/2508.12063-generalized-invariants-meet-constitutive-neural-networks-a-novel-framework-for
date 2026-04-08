# Constraints

These constraints define what must stay fixed so that improvements come from better GI-CANN reproduction, not from changing the task.

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

Stay within the GI-CANN / SI-CANN family from the paper.

- Generalized invariant:
  - `J_alpha = sum_i lambda_i^alpha`
- GI-CANN strain energy:
  - `psi = w_id (J_alpha - 3) + w_exp2 (exp(w_exp1 (J_alpha - 3)) - 1) + w_ln2 ln(1 - w_ln1 (J_alpha - 3))`
- SI-CANN baseline:
  - Same branch structure, but with fixed exponents `alpha=2` and/or `alpha=-2`

The current implementation covers the paper’s core one-positive / one-negative settings and the combined two-branch variant used in the benchmark tables. Do not replace this with a generic black-box regressor.

## Stress-Calculation Constraints

Stress must be computed from the hyperelastic energy, using the paper’s closed-form mode-specific equations.

- `ut` / `uc`: use the incompressible uniaxial relation
- `bt`: use the equibiaxial relation
- `ps`: use the pure-shear relation implied by the chosen deformation gradient
- `ss`: use the simple-shear formula based on `lambda_± = (sqrt(gamma^2 + 4) ± gamma) / 2`

## Evaluation Constraints

Evaluate with [`scripts/evaluate.sh`](/home/user/scripts/evaluate.sh), which writes [`scoring/scores.json`](/home/user/scoring/scores.json).

- Rubber metric set: `r2_ut`, `r2_bt`, `r2_ps`, `avg_r2`, `alpha_star`
- Brain metric set: `r2_ut`, `r2_uc`, `r2_ss`, `avg_r2`, `alpha_star`

`avg_r2` remains the primary metric, but `alpha_star` is also a constraint because the paper claims interpretable exponent discovery, not only fit quality.

## Resource Constraints

- No pretrained external model weights
- No architecture scaling outside the GI-CANN/SI-CANN family
- Keep the method small and interpretable rather than increasing capacity to memorize the benchmark
