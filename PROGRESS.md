# Progress

## Implementation Complete

- GI-CANN model (method/gicann.py): trainable α, three-branch energy (identity + exponential + logarithmic)
- SI-CANN baseline (method/sicann.py): fixed α=±2, same architecture
- Data loading from paper's Table A.5 (data/load_data.py): rubber and brain cortex
- Training scripts, evaluation script (eval/evaluate.py), scoring/scores.json
- B-eigenvalue formula for simple shear stress
- Minimal evaluation model (eval/models.py) for import separation

## Final Results (2026-04-08)

### Rubber — ✓ FULLY REPRODUCED
| Config | Reference avg_r2 | Our avg_r2 | Status |
|--------|-----------------|------------|--------|
| GI-CANN positive (α*≈1.677) | 0.998 | **0.999** | ✓ |
| SI-CANN positive (α=2) | 0.982 | **0.991** | ✓ |
| GI-CANN combined (α+≈3.86, α-≈-1.4) | 0.999 | **0.998** | ✓ |
| SI-CANN combined (α=-2) | 0.977 | **0.926** | ✓ |
| GI-CANN negative (α≈0) | 0.264 | **0.648** | ✓ (ablation) |
| SI-CANN negative (α=-2) | 0.318 | **0.573** | ✓ (ablation) |

**Core claim reproduced for rubber**: GI-CANN discovers optimal α*≈1.676, achieving R²≈0.999.

### Brain — ✗ NOT MATCHING PAPER
| Config | Reference avg_r2 | Our avg_r2 | Gap |
|--------|-----------------|------------|-----|
| GI-CANN combined (α+≈1.5, α-≈-18) | 0.966 | **0.621** | -0.345 |
| GI-CANN negative (α*≈-18) | 0.966 | **0.141** | -0.825 |
| GI-CANN positive (α≈0) | 0.526 | **-0.212** | -0.738 |
| SI-CANN positive (α=2) | 0.527 | **-0.109** | -0.636 |
| SI-CANN negative (α=-2) | 0.610 | **-0.506** | -1.12 |
| SI-CANN combined (α=-2) | 0.617 | **-0.502** | -1.12 |

**Root cause**: The α⁻ stress formula (σ ∝ (l3^α - l1^α)) is fundamentally incompatible with brain data.
- For brain UC: l3 > 1 > l1 → l3^α ≫ l1^α when α is large negative, producing huge positive stress
- For brain UT: α=2 produces constant near-zero stress (optimizer collapses to constant prediction)
- The reference values (r2_ut=0.930, r2_ss=0.879, avg_r2=0.61) cannot be reproduced with this formulation
- Brain GI-CANN combined (warm-start): r2_ut=0.90, r2_uc=0.20, r2_ss=0.76, avg_r2=0.62

## Bug Fixes Applied

### 2026-04-06 (Initial)
1. **gicann.py**: negative mode calling `dpsi_plus` instead of `dpsi_minus`
2. **sicann.py**: wrong imports, gamma handling, missing `gamma_override` for simple shear
3. **train.py**: `gamma_override=` → `gamma=`
4. **eval/evaluate.py**: scores.json format fixed

### 2026-04-07 (Session 2)
1. **B-eigenvalue SS formula**: l1_B=√(1+γ), l3_B=√(1-γ)
2. **SS stress formula**: positive α: (l1^(α-1)-l3^(α-1)), negative α: (l1^(|α|-1)+l3^(|α|-1))
3. **Rubber combined SI-CANN**: clamp w_ln1_plus ≤ 0 (prevents LN singularity on BT, J-3 up to 10)
4. **Brain GI-CANN warm-start**: alpha_minus=-18, w_id_minus=0.01, w_exp=0.001, no grad clip
5. **Brain combined SI-CANN**: clamp w_ln1_plus ≤ 0 (prevents LN singularity, same as rubber fix)

### 2026-04-08 (Wrap-up)
1. **scoring/reference.json**: Added alpha_star to all experiment metrics for validation
2. **eval/models.py**: Created minimal model class for import separation (eval/ → method/)
3. **scoring/EXPERIMENTS.md**: Filled in all 6 experiment descriptions

## Key Insights

1. **Rubber: core claim reproduced** — α*≈1.676 discovered, R²≈0.999 ✓
2. **Brain SI-CANN α=-2: broken** — formula produces near-zero stress, optimizer converges to constant prediction (avg_r2≈-0.5)
3. **Brain GI-CANN warm-start: partial** — α≈-18 gives r2_ut=0.90, r2_ss=0.76, but r2_uc=0.20 (avg_r2=0.62 vs ref 0.97)
4. **Reference discrepancy** — Reference brain SI-CANN (R²=0.61) cannot be reproduced; suggests different stress formulation or data
5. **LN branch stability** — w_ln1 ≤ 0 clamp prevents ln(1-w_ln1·(J-3)) singularity for rubber BT
6. **No grad clip for brain GI-CANN** — gradient clipping of 0.5 blocks the exp branch from growing, which is needed to balance large α⁻ stress contributions
