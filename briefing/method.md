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
