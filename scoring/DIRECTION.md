# Research Direction

Future agents must work within the paper's problem space and approach family.

## Problem Space

**Domain**: Hyperelastic constitutive modeling for soft materials.

**Challenge**: Discovering appropriate invariant exponents α that characterize material behavior without prior knowledge.

**Materials**: Rubber (stretch-dominated, prefers positive α) and brain tissue (shear-sensitive, paper claims prefers negative α).

## Approach Family

Work within physics-aware constitutive model discovery for isotropic incompressible hyperelasticity.

Suitable directions include:
1. Better invariant choices or invariant parameterizations
2. Better compact constitutive-law representations
3. Better training or regularization for cross-mode generalization
4. Better ways to preserve interpretability while improving held-out stress prediction

**Cannot Switch To**:
- Black-box neural networks that directly map λ → σ
- Physics-uninformed learning without hyperelasticity constraints
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

The goal is methodological advancement in compact, interpretable constitutive-law discovery under the same benchmark protocol, not beating the benchmark by leaving the problem setting.
