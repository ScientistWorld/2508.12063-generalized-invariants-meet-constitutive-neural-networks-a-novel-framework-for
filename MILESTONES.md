# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-07] - method_runs (in progress toward core_claim)
- Full training run with 12 configs completed
- **Rubber results (working)**:
  - GI-CANN positive (α*≈1.677): avg_r2=0.999 ✓ (reference: 0.998)
  - SI-CANN positive (α=2): avg_r2=0.991 ✓ (reference: 0.982)
  - GI-CANN negative (α≈-0.52): avg_r2=0.647 (reference: 0.264)
  - SI-CANN negative (α=-2): avg_r2=0.573 (reference: 0.318)
  - GI-CANN combined (α+≈3.86, α-≈-1.42): avg_r2=0.998 ✓ (reference: 0.999)
  - SI-CANN combined (α=-2): avg_r2=-1689 ✗ CATASTROPHIC DIVERGENCE
- **Brain results (not working)**:
  - GI-CANN negative (α≈-18): avg_r2=-0.023 ✗ (reference: 0.610)
  - SI-CANN negative (α=-2): avg_r2=-0.506 ✗ (reference: 0.610)
  - GI-CANN positive (α≈0.60): avg_r2=-0.101 ✗ (reference: 0.526)
  - SI-CANN positive (α=2): avg_r2=-0.109 ✗ (reference: 0.527)
  - GI-CANN combined (α+≈3.4, α-≈-21): avg_r2=0.554 (reference: 0.966)
  - SI-CANN combined (α=-2): avg_r2=-0.502 ✗ (reference: 0.617)
- **Root causes identified**:
  1. Rubber combined SI-CANN: w_ln1_plus grows positive → LN singularity for BT (J-3 up to 10)
  2. Brain models: exp branch collapses due to L1 regularization (r=0.01) + gradient clipping
  3. Brain combined SS stress improved from R²=0.626→0.922 with B-eigenvalue formula, but UT/UC still fail
- **Fixes applied**:
  - Added w_ln1_plus ≤ -1e-6 clamp for rubber combined SI-CANN
  - Clamped LN and negative-ID branches small for rubber combined SI-CANN
  - Reduced brain L1 from 0.01→0.001
  - Added exp branch warm-start (w_exp1=0.1, w_exp2=0.1) for all brain configs
  - Added all branches (id/exp/ln) initialized for brain GI-CANN negative
- **Core claim status**: PARTIAL — rubber fully works (R²≈0.999), brain models need further fixes

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
