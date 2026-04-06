# Progress

## What Works

- GI-CANN model implementation (method/gicann.py) with all three branch types (identity, exponential, logarithmic)
- SI-CANN baseline implementation (method/sicann.py) with fixed exponents
- Data loading from paper's Table A.5 (data/load_data.py): rubber and brain cortex stress-strain data
- Training scripts for both models with R² evaluation
- Evaluation script producing scoring/scores.json (fixed format)
- All shell scripts for the full pipeline
- Container definition with PyTorch
- Briefing documentation (problem.md, evaluation.md, method.md, overview.md)
- Reference.json with paper's reported numbers

## Bug Fixes Applied (2026-04-06)

1. **gicann.py**: `_stress_uniaxial` and `_stress_equibiaxial` — negative mode was calling `dpsi_plus` instead of `dpsi_minus`
2. **sicann.py**: wrong imports (`get_principal_stretches_batch` → `get_principal_stretches`, `detect_mode` → `detect_deformation_mode`)
3. **sicann.py**: `compute_stress` didn't accept `gamma_override` parameter for simple shear
4. **sicann.py**: simple shear gamma indexing fixed (`indices_t` → proper list indexing)
5. **train.py**: `gamma_override=` → `gamma=` in loss method call
6. **baseline_train.py**: `gamma_override=` → `gamma=` in loss method call
7. **eval/evaluate.py**: scores.json format fixed to include `experiments` wrapper
8. **container.def**: updated to python:3.12-slim

## Results

Not yet run — submitting corrected job.

Expected results (from paper):
- GI-CANN on rubber (positive, α*≈1.675): avg R² ≈ 0.998
- SI-CANN on rubber (positive, α=2): avg R² ≈ 0.982
- GI-CANN on brain (negative, α*≈-18): avg R² ≈ 0.966
- SI-CANN on brain (negative, α=-2): avg R² ≈ 0.610

## Remaining

- Run the full training pipeline (baseline + GI-CANN)
- Validate scoring/scores.json matches reference.json structure
- Verify R² values match paper's reported numbers
- Optional: expand to additional configurations
