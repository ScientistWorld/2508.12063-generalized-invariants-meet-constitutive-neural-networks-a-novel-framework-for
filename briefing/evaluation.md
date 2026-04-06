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
