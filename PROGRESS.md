# Progress

## What Works

- GI-CANN model implementation (method/gicann.py) with all three branch types (identity, exponential, logarithmic)
- SI-CANN baseline implementation (method/sicann.py) with fixed exponents
- Data loading from paper's Table A.5 (data/load_data.py): rubber and brain cortex stress-strain data
- Training scripts for both models with R² evaluation
- Evaluation script producing scoring/scores.json
- B-eigenvalue formula for simple shear stress (corrects SVD-based approach)
- All shell scripts for the full pipeline

## Results (2026-04-07 Run)

### Rubber — FULLY WORKING
| Config | Reference avg_r2 | Our avg_r2 | Status |
|--------|-----------------|------------|--------|
| GI-CANN positive (α*≈1.677) | 0.998 | 0.999 | ✓ |
| SI-CANN positive (α=2) | 0.982 | 0.991 | ✓ |
| GI-CANN combined | 0.999 | 0.998 | ✓ |
| SI-CANN combined | 0.977 | -1689.68 | ✗ DIVERGED |
| GI-CANN negative (α≈-0.52) | 0.264 | 0.647 | ~ |
| SI-CANN negative (α=-2) | 0.318 | 0.573 | ~ |

### Brain — NOT WORKING
| Config | Reference avg_r2 | Our avg_r2 | Status |
|--------|-----------------|------------|--------|
| GI-CANN negative (α*≈-18) | 0.610 | -0.023 | ✗ |
| SI-CANN negative (α=-2) | 0.610 | -0.506 | ✗ |
| GI-CANN positive (α≈0.60) | 0.526 | -0.101 | ✗ |
| SI-CANN positive (α=2) | 0.527 | -0.109 | ✗ |
| GI-CANN combined (α+≈3.4, α-≈-21) | 0.966 | 0.554 | ✗ |
| SI-CANN combined (α=-2) | 0.617 | -0.502 | ✗ |

## Issues & Fixes Applied

### Bug Fixes (2026-04-06)
1. **gicann.py**: `_stress_uniaxial` and `_stress_equibiaxial` — negative mode was calling `dpsi_plus` instead of `dpsi_minus`
2. **sicann.py**: wrong imports (`get_principal_stretches_batch` → `get_principal_stretches`, `detect_mode` → `detect_deformation_mode`)
3. **sicann.py**: `compute_stress` didn't accept `gamma_override` parameter for simple shear
4. **sicann.py**: simple shear gamma indexing fixed (`indices_t` → proper list indexing)
5. **train.py**: `gamma_override=` → `gamma=` in loss method call
6. **baseline_train.py**: `gamma_override=` → `gamma=` in loss method call
7. **eval/evaluate.py**: scores.json format fixed to include `experiments` wrapper
8. **container.def**: updated to python:3.12-slim

### Fixes Applied (2026-04-07)
1. **B-eigenvalue formula for SS**: Updated `_stress_simple_shear()` to use B-eigenvalues (l1=√(1+γ), l3=√(1-γ)) instead of SVD. This improved SS r2 from 0.626 to 0.922 for brain combined GI-CANN.
2. **Rubber combined SI-CANN divergence**: Added clamps for w_ln1_plus ≤ -1e-6 and w_ln2_plus/w_ln1_minus/w_ln2_minus/w_id_minus ≤ 0.01 to prevent LN singularity and negative-ID divergence for BT.
3. **Brain L1 regularization**: Reduced from r=0.01 to r=0.001 to allow exp branch to activate.
4. **Brain exp branch warm-start**: Initialized w_exp1=0.1, w_exp2=0.1 for all brain configs.
5. **Brain GI-CANN negative warm-start**: All branches (id/exp/ln) initialized for negative mode.

## Remaining Issues

1. **Brain models still fail**: Despite fixes, brain models may still have fundamental issues:
   - The exp branch still tends to collapse even with reduced L1
   - The α=-18 learned by GI-CANN negative doesn't match the paper's claim that it "discovers" this
   - Training dynamics may need hyperparameter tuning (learning rate, patience, etc.)
2. **Rubber combined SI-CANN**: Clamping to prevent divergence reduces it to essentially the positive-only model (avg_r2≈0.99 vs reference 0.977)
3. **Brain combined GI-CANN**: SS improved but UT/UC still poor (r2_ut=0.321, r2_uc=0.420 vs reference 0.930, 0.997)

## Key Insights from Analysis

1. **Rubber is stretch-dominated** (α*≈1.675): Positive α works perfectly with identity branch
2. **Brain is shear-dominated** (paper claims α*≈-18): Negative α should work but exp branch collapses
3. **B-eigenvalues vs SVD for SS**: SVD gives l1 = √((γ²+γ√(γ²+4))/2) while B-eigenvalues give √(1+γ). These differ significantly at moderate γ.
4. **LN branch instability**: For rubber BT with J-3 up to 10, w_ln1_plus must stay ≤ 0 to keep ln argument positive
5. **Exponential overflow**: For rubber UT at λ1=7.25 with α=-2: w_exp1_minus*J ≈ 5.2, causing exp(5.2) overflow
6. **Negative α stress formula**: For UT/UC where l3 > 1 > l1, the stress uses (l3^α - l1^α) not (l1^α - l3^α) to get correct sign
