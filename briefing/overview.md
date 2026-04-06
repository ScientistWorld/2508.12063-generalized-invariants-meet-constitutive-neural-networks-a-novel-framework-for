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
