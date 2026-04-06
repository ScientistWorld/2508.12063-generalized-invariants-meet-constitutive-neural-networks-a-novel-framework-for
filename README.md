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

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

The central challenge is: given experimental stress-strain data for an isotropic incompressible hyperelastic material (such as rubber or brain tissue), how do we automatically discover both the most appropriate mathematical form of the strain energy function and the best set of invariants on which to express it?

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

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocol for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

**Primary metric: Coefficient of Determination (R²)**

For each deformation mode, R² measures how well the predicted Cauchy stress components match the experimentally measured stress:

R² = 1 - (SS_res / SS_tot)

where:
- SS_res = Σ (σ_predicted - σ_measured)² (residual sum of squares)
- SS_tot = Σ (σ_measured - σ_mean)² (total sum of squares)

Higher R² is better. R² = 1.0 means perfect prediction. R² = 0 means the prediction is no better than predicting the mean. Negative R² means the prediction is worse than the mean.

R² is computed separately for each deformation mode (e.g., uniaxial tension R², pure shear R², etc.), then averaged across modes.

## Baselines and Targets

The paper compares multiple approaches. Key reference points:

1. **SI-CANN** (Standard Invariant Constitutive ANN): Uses fixed classical invariants I₁ (α=2) and/or I₂ (α=-2). Average R² ≈ 0.318–0.982 depending on material and invariant choice.
2. **PS-CANN** (Principal Stretch CANN): Uses a dense grid of exponents from -30 to 30 with identity function only. Average R² ≈ 0.974 for brain tissue.
3. **FNN** (Feedforward Neural Network): Unconstrained neural network with fixed exponents. Average R² ≈ 0.460–0.984.
4. **FICNN** (Fully Input Convex Neural Network): Convex-constrained neural network. Average R² ≈ 0.195–0.993.
5. **Traditional models**: Neo-Hookean, Mooney-Rivlin, Gent, Ogden with various exponents.

The target for a good method is average R² > 0.96 on both rubber and brain tissue, across all three deformation modes per material.

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

### Experiment Configuration

Three invariant configurations are tested for each material:

1. **Positive exponent only (Jα+, α+)** — I₁-like behavior
2. **Negative exponent only (Jα-, α-)** — I₂-like behavior
3. **Both positive and negative (Jα+, Jα-)** — mixed formulation

### Protocol

1. Load the experimental stress-strain data for the specified training modes
2. Train the model to predict Cauchy stress components from deformation data
3. Evaluate on the held-out test deformation mode
4. Compute R² for each deformation mode (training modes + test mode)
5. Report average R² across all modes
6. Report the discovered exponent values (α*) and strain energy function form

### Hyperparameters

Standard training configuration:
- Optimizer: Adam
- Learning rate: 0.001 (or similar, tuned for convergence)
- L1 regularization coefficient: r = 0.01
- Training until convergence (loss plateau) or fixed epoch count
- Multiple random seeds recommended for reproducibility

---

# Reproduction Milestones

**Current: method_runs**

## Progress Log

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
