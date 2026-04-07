# Research Direction

Future agents must work within the paper's problem space and approach family.

## Problem Space

**Domain**: Hyperelastic constitutive modeling for soft materials.

**Challenge**: Discovering appropriate invariant exponents α that characterize material behavior without prior knowledge.

**Materials**: Rubber (stretch-dominated, prefers positive α) and brain tissue (shear-sensitive, paper claims prefers negative α).

## Approach Family

**Must Use**: Generalized invariant-based constitutive neural networks (CI-CANN/GI-CANN framework).

**Key Idea**: Replace fixed invariants (I1, I2, I3) with generalized invariants J_α = Σ λ_i^α where α is learnable.

**Required Components**:
1. Generalized invariant computation with trainable exponent α
2. Three-branch strain energy (identity + exponential + logarithmic)
3. Stress derivation via hyperelasticity theory (∂ψ/∂C)

**Cannot Switch To**:
- Black-box neural networks that directly map λ → σ
- Physics-uninformed learning without hyperelasticity constraints
- Different invariants or constitutive formulations
- Data-driven surrogate models that bypass the physics

## Innovation Directions (Within Framework)

While staying within the GI-CANN family, future agents can:

1. **Multi-branch architectures**: Explore different branch structures (additive vs multiplicative)
2. **Initialization strategies**: Better warm-start for negative α regimes
3. **Regularization**: L1/L2 on weights to improve generalization
4. **Training stability**: Alternative optimizers, learning rate schedules, gradient clipping strategies
5. **Invariant combinations**: J_α+ + J_α- with separate learnable exponents
6. **Material-specific formulations**: Different architectures for stretch vs shear dominated materials

## Out of Bounds

Cannot claim improvement by:
- Using a completely different neural architecture (e.g., MLP, Transformer)
- Adding model capacity to memorize data
- Switching to a regression problem without hyperelasticity constraints
- Using external priors or embeddings not derived from deformation data
- Simplifying the problem to trivial toy cases

The goal is methodological advancement within the GI-CANN framework, not beating benchmarks at any cost.
