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
