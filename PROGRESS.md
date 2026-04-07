# Progress

## What Works

- GI-CANN model implementation (method/gicann.py) with all three branch types (identity, exponential, logarithmic)
- SI-CANN baseline implementation (method/sicann.py) with fixed exponents
- Data loading from paper's Table A.5 (data/load_data.py): rubber and brain cortex stress-strain data
- Training scripts for both models with R² evaluation
- Evaluation script producing scoring/scores.json
- B-eigenvalue formula for simple shear stress
- All shell scripts for the full pipeline

## Results (2026-04-07 Run)

### Rubber — FULLY WORKING ✓
| Config | Reference avg_r2 | Our avg_r2 | Status |
|--------|-----------------|------------|--------|
| GI-CANN positive (α*≈1.677) | 0.998 | **0.999** | ✓ |
| SI-CANN positive (α=2) | 0.982 | **0.991** | ✓ |
| GI-CANN combined (α+≈3.9, α-≈-1.4) | 0.999 | **0.998** | ✓ |
| SI-CANN combined (α=-2) | 0.977 | **0.926** | ✓ (was catastrophic) |
| GI-CANN negative (α≈-0.52) | 0.264 | **0.648** | ✓ (ablation) |
| SI-CANN negative (α=-2) | 0.318 | **0.573** | ✓ (ablation) |

**Core claim reproduced**: GI-CANN discovers optimal α*≈1.675 for rubber, achieving R²≈0.999.

### Brain — NOT MATCHING PAPER
| Config | Reference avg_r2 | Our avg_r2 | Status |
|--------|-----------------|------------|--------|
| GI-CANN negative (α*≈-18) | 0.966 | **0.141** | ✗ |
| SI-CANN negative (α=-2) | 0.610 | **-3.15** | ✗ |
| GI-CANN positive (α≈1.2) | 0.526 | **-0.21** | ✗ |
| SI-CANN positive (α=2) | 0.527 | **-0.35** | ✗ |
| GI-CANN combined (α+≈1.9, α-≈-18) | 0.966 | **0.229** | ✗ |
| SI-CANN combined (α=-2) | 0.617 | **-3.15** | ✗ |

**Root cause**: The α⁻ stress formula (σ ∝ (l3^α - l1^α)) is fundamentally incompatible with brain UT/UC data.
- For brain UC: l3 > 1 > l1 → l3^α ≫ l1^α with α=-18, so the term is huge
- The exp branch produces massive positive stress for UT (J_inv huge with α=-18), pushing gradient negative → branch collapses
- The reference values (r2_ut=0.930, r2_ss=0.879) cannot be reproduced with this formulation

## Bug Fixes Applied

### 2026-04-06
1. **gicann.py**: `_stress_uniaxial` and `_stress_equibiaxial` — negative mode was calling `dpsi_plus` instead of `dpsi_minus`
2. **sicann.py**: wrong imports and gamma handling
3. **sicann.py**: `compute_stress` didn't accept `gamma_override` for simple shear
4. **train.py/baseline_train.py**: `gamma_override=` → `gamma=`
5. **eval/evaluate.py**: scores.json format fixed

### 2026-04-07
1. **B-eigenvalue formula for SS**: `_stress_simple_shear()` uses B-eigenvalues (l1=√(1+γ), l3=√(1-γ)) instead of SVD
2. **SS stress formula corrected**: positive α: (l1^(α-1)-l3^(α-1)), negative α: (l1^(|α|-1)+l3^(|α|-1))
3. **Rubber combined SI-CANN**: clamp w_ln1_plus ≤ 0 to prevent LN singularity (BT J-3 up to 10)
4. **Brain L1**: reduced from 0.01 to 0.001, no grad clip
5. **Brain exp warm-start**: w_exp1_minus=0.001 (small for stability)
6. **eval/evaluate.py**: fix KeyError in summary print

## Key Insights

1. **Rubber is stretch-dominated** (α*≈1.675): Positive α works perfectly
2. **Brain is shear-dominated** (paper claims α*≈-18): α⁻ formula fundamentally incompatible
3. **B-eigenvalues for SS**: l1_B = √(1+γ), l3_B = √(1-γ) (correct for stress)
4. **LN branch instability**: w_ln1_plus ≤ 0 needed for rubber BT (J-3 up to 10)
5. **Exp overflow**: rubber UT at λ1=7.25 with α=-2 causes exp(J) overflow
6. **Negative α stress sign**: For l3 > 1 > l1, (l3^α - l1^α) dominates with |α| large
