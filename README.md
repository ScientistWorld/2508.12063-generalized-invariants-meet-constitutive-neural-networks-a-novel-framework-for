# Overview

- **Paper ID:** 2508.12063
- **Title:** Generalized invariants meet constitutive neural networks: A novel framework for hyperelastic materials
- **Domain:** Computational Mechanics / Constitutive Modeling
- **TL;DR:** A neural network (GI-CANN) that simultaneously discovers the optimal exponent for generalized invariants and the strain energy function from stress-strain data, enabling automated interpretable model discovery for hyperelastic materials.

## Short Summary

The paper addresses the challenge of automatically discovering physically meaningful constitutive models for isotropic incompressible hyperelastic materials from experimental stress-strain data. The proposed GI-CANN (Generalized Invariant Constitutive Artificial Neural Network) learns a trainable exponent α in the generalized invariant J_α = Σ λᵢ^α, and combines it with a structured neural network that outputs the strain energy function via three parallel branches (identity, exponential, logarithmic). Unlike prior approaches that require fixed invariants or sequential two-step fitting, GI-CANN discovers both the invariant and the strain energy function in a single training procedure. The method is validated on rubber (Treloar data) and brain cortex (Budday et al. data), demonstrating that it automatically identifies the appropriate invariant formulation for each material — stretch-dominated behavior (rubber) prefers positive exponents near α=1.7, while shear-dominated behavior (brain tissue) prefers large negative exponents α≈-18.

## Key Results

- **Rubber**: GI-CANN with single positive exponent achieves average R² = 0.998 (trained on uniaxial + equibiaxial, tested on pure shear), compared to SI-CANN's R² = 0.982
- **Brain tissue**: GI-CANN with single negative exponent achieves average R² = 0.966 (trained on uniaxial tension + compression, tested on simple shear), significantly outperforming SI-CANN's R² = 0.527 with α+=2
- **Interpretability**: The discovered models correspond to closed-form strain energy functions (e.g., ψ = 0.227 MPa · (J_1.675 - 3) + 0.297 MPa · [exp(0.107 · (J_1.675 - 3)) - 1] for rubber)

---

# Problem Definition

## Research Question

The central challenge is: given experimental stress-strain data for an isotropic incompressible hyperelastic material (such as rubber or brain tissue), how do we automatically discover an accurate and interpretable strain-energy law without hand-designing the final constitutive form in advance?

Classical hyperelastic constitutive models (neo-Hookean, Mooney-Rivlin, Ogden, etc.) require an engineer to manually choose a strain energy function form and invariants a priori. This choice is often based on intuition or trial-and-error and may not suit all materials. Data-driven approaches using neural networks can fit stress-strain data but often lack interpretability or require the user to pre-specify which invariants to use.

## Why It Matters

Hyperelastic materials (rubber, silicone, brain tissue, etc.) are used in critical applications: automotive components, biomedical devices, soft robotics, and personalized medical simulations. Accurate constitutive models are essential for predictive simulations. Automatically discovering interpretable material models from experimental data would dramatically accelerate the model development process and improve predictive accuracy across different loading conditions.

## Success Criteria

A successful solution should:
1. **Discover interpretable models** directly from experimental stress-strain data, without requiring manual specification of the strain energy function form
2. **Predict accurately** across multiple deformation modes — training on some loading conditions (e.g., uniaxial tension) and testing on unseen ones (e.g., pure shear or simple shear)
3. **Outperform classical approaches** that rely on fixed, pre-specified invariants (I₁ or I₂)
4. **Handle diverse materials** — both rubber-like materials dominated by stretch and soft biological tissues with shear-sensitive behavior
5. **Output physically meaningful** strain energy function formulations that can be written in closed form, not just black-box neural network predictions

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper proposes **GI-CANN (Generalized Invariant Constitutive Artificial Neural Networks)**, which simultaneously discovers the optimal generalized invariant exponent α and the corresponding strain energy function directly from experimental stress-strain data. Unlike prior methods that require manual selection of invariants or sequential two-step fitting, GI-CANN integrates discovery into a single end-to-end neural network training procedure.

## Approach

### Generalized Invariants

The method uses generalized invariants defined as:

    J_α = λ₁^α + λ₂^α + λ₃^α

where λᵢ are the principal stretches and α ∈ ℝ is a trainable parameter. For incompressible materials (λ₁λ₂λ₃ = 1):
- α = 2 recovers the first invariant: J₂ = I₁ = tr(C)
- α = -2 recovers the second invariant: J_{-2} = I₂

The key innovation is making α a trainable parameter that the network learns from data, rather than fixing it to classical values.

### Network Architecture

The GI-CANN has a structured architecture with three parallel branches computing the strain energy function ψ:

1. **Identity branch**: ψ_id = w_id · (J_α - 3)
2. **Exponential branch**: ψ_exp = w_exp2 · [exp(w_exp1 · (J_α - 3)) - 1]
3. **Logarithmic branch**: ψ_ln = w_ln2 · ln(1 - w_ln1 · (J_α - 3))

The total strain energy is the sum of all active branches: ψ = ψ_id + ψ_exp + ψ_ln

The Cauchy stress is computed by differentiating ψ with respect to the deformation gradient F.

### Loss Function

    L(α, w; F) = (1/n_data) Σ ||σ^α(J_α(F), w) - σ_measured||² + r Σ |w_i|

The loss is MSE between predicted and measured Cauchy stress, plus L1 regularization (r = 0.01) on network weights to induce sparsity and improve interpretability.

### Training Configuration

Three variants are tested:
- **GI-CANN (R+)**: Only positive exponents (stretch-dominated, I₁-like behavior)
- **GI-CANN (R-)**: Only negative exponents (shear-dominated, I₂-like behavior)
- **GI-CANN (R)**: Both positive and negative exponents simultaneously

## Main Claims

1. **Primary claim**: GI-CANN discovers interpretable strain energy functions with optimal exponents α* directly from data, achieving average R² > 0.99 on rubber and R² > 0.96 on brain tissue
2. **Secondary claim**: For rubber, the best exponent is α* ≈ 1.675 (close to I₁), consistent with stretch-dominated entropic elasticity
3. **Secondary claim**: For brain tissue, the best exponent is α* ≈ -18 (large negative, I₂-like), capturing shear-sensitive nonlinear behavior
4. **Generalization claim**: Models trained on uniaxial/equibiaxial tension generalize to pure shear for rubber; models trained on uniaxial tension/compression generalize to simple shear for brain tissue

---

# Evaluation

## Metrics

**Primary metric: Coefficient of Determination (R²)**

For each deformation mode, R² measures how well the predicted Cauchy stress components match the experimentally measured stress:

R² = 1 - (SS_res / SS_tot)

where:
- SS_res = Σ (σ_predicted - σ_measured)² (residual sum of squares)
- SS_tot = Σ (σ_measured - σ_mean)² (total sum of squares)

Higher R² is better. R² = 1.0 means perfect prediction. R² = 0 means the prediction is no better than predicting the mean. Negative R² means the prediction is worse than the mean.

R² is computed separately for each deformation mode (e.g., uniaxial tension R², pure shear R², etc.), then averaged across modes.

## Targets

The benchmark is centered on cross-mode generalization.

- A strong method should fit the observed loading modes well and generalize to the held-out loading mode.
- Rubber is the easier of the two settings and strong methods are expected to approach near-perfect `R²`.
- Brain cortex is harder and more sensitive to modeling choices; it should not be treated as solved unless the held-out simple-shear score is also strong.

## Evaluation Protocol

### Datasets

Two experimental datasets are used:

**Rubber (Treloar 1944):**
- Training: uniaxial tension (ut), equibiaxial tension (bt)
- Testing: pure shear (ps)
- Units: MPa

**Brain cortex (Budday et al. 2017):**
- Training: uniaxial tension (ut), uniaxial compression (uc)
- Testing: simple shear (ss)
- Units: kPa

### Protocol

1. Load the experimental stress-strain data for the specified training modes
2. Train the model to predict Cauchy stress components from deformation data
3. Evaluate on the held-out test deformation mode
4. Compute R² for each deformation mode (training modes + test mode)
5. Report average R² across all modes
6. Report any additional scalar descriptors needed to summarize the fitted constitutive response

---

# Reproduction Log

### Iteration 1: MiniMax-M2.7
- **Milestone**: `core_claim` | **Status**: done
- **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 1: MiniMax-M2.7
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 40m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 2: glm-5.1
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 13m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08] - Wrap-up Complete
- Fixed validation errors (added alpha_star to reference.json metrics, created eval/models.py for import separation)
- Updated scoring/EXPERIMENTS.md with all 6 experiments
- Updated scoring/TARGETS.md with Primary/Constraint/Ablation target classifications
- All validation checks passed
- Final commit prepared

### [2026-04-08] - Audit Complete
**Auditor verified:**
- Implementation is real and complete (GI-CANN algorithm correctly implemented)
- Evaluation is honest (results computed from checkpoints, not hardcoded)
- All deliverables present (briefing/, scoring/, scripts/, environment/)
- Evaluation script re-ran successfully, reproduced all scores

**Audit result: `core_claim` milestone restored.** Previous agent's claim was valid.

### [2026-04-07] - core_claim
- Reproduced GI-CANN for rubber material (VHB 4910)
- GI-CANN discovers α*≈1.677, achieving R²=0.999 (paper: 0.998/1.675) ✓
- All rubber configurations working: positive (R²=0.999), combined (R²=0.998), negative ablation (R²=0.648)
- Baseline SI-CANN for rubber also working (R²=0.991 with α=2)
- Brain tissue: NOT reproduced — α⁻ stress formula incompatible with brain UT/UC data
  - Brain GI-CANN achieves R²=0.14-0.62 vs reference 0.966
  - Root cause: α⁻ stress formula (σ ∝ (l3^α - l1^α)) produces incorrect stress for brain data
  - SI-CANN baseline for brain also fails (R²≈-0.5), confirming formulation issue not specific to GI-CANN

### [2026-04-06] - method_runs
- Implemented GI-CANN and SI-CANN models with three-branch strain energy
- Data loading from paper's Table A.5 for rubber and brain cortex
- Training and evaluation pipeline working

</details>

### Iteration 3: glm-5.1
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

</details>

### Iteration 4: glm-5.1 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 11m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

</details>

### Iteration 5: glm-5.1 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 27m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 6: gpt-5.4 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 7: gpt-5.4 (designer)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 17m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 7: MiniMax-M2.7 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 8: MiniMax-M2.7 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 9: gpt-5.4 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

### [2026-04-24 00:00] - core_claim
- Audited the first real train/test split run and found checkpoint loading failed before any scores were produced because the container NumPy exposes `numpy.core` while the frozen checkpoints unpickle `numpy._core`
- Patched all evaluation entry points to install backward-compatible NumPy module aliases before `torch.load`, preserving the frozen method artifacts while unblocking benchmark scoring
- Kept the milestone at `core_claim`; this turn advances evaluation robustness but does not change the reproduced method quality

### [2026-04-24 00:15] - core_claim
- Completed the real train/test split evaluation and produced validated `scoring/scores_train.json` and `scoring/scores_test.json` with matching experiment/method coverage
- Verified that the split is not a copy-paste bug: every experiment differs numerically across slices, although the strongest rubber benchmark has only a very small gap because the alternating split stays on the same smooth curve family
- Closed the design phase at `core_claim` because the evaluation artifact is now complete and usable, with the main caveat that the shipped brain checkpoints remain weak on both slices

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
