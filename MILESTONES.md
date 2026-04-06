# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-06] - method_runs
- Fixed critical bugs in GI-CANN and SI-CANN implementations:
  - gicann.py `_stress_uniaxial` and `_stress_equibiaxial`: negative mode was using wrong branch (dpsi_plus instead of dpsi_minus)
  - sicann.py: imports referenced non-existent functions (`get_principal_stretches_batch`, `detect_mode` → fixed to `get_principal_stretches`, `detect_deformation_mode`)
  - sicann.py `compute_stress`: added missing `gamma_override` parameter for simple shear
  - sicann.py `compute_stress`: fixed SS gamma indexing (`indices_t` → `[[i for i in indices]]`)
  - train.py: fixed `gamma_override=` → `gamma=` parameter name in loss call
  - baseline_train.py: fixed `gamma_override=` → `gamma=` parameter name in loss call
  - eval/evaluate.py: fixed scores.json format to include `experiments` wrapper
  - Updated container.def from python:3.11-slim to python:3.12-slim
- All code verified, ready for training run

### [2026-04-05 14:30] - method_runs
- Implemented GI-CANN neural network model (method/gicann.py) following the paper's architecture:
  - Trainable exponent α (positive, negative, or combined mode)
  - Three-branch strain energy function (identity, exponential, logarithmic)
  - Cauchy stress computation for each deformation mode (ut, bt, ps, ss)
- Implemented SI-CANN baseline model (method/sicann.py) with fixed exponents
- Created data loading code (data/load_data.py) with all rubber and brain tissue data from Table A.5
- Created training scripts for both GI-CANN and SI-CANN
- Created evaluation script (eval/evaluate.py) with R² metric
- Created all shell scripts (download.sh, evaluate.sh, method.sh, baseline.sh, reproduce.sh, run.sh)
- Created container definition and setup script
- Filled in all briefing files (problem.md, evaluation.md, method.md, overview.md)
- Created scoring/reference.json with paper's reported numbers

### [2026-04-05 14:18] - none
- Read the paper and identified key components
- Initial workspace analysis
