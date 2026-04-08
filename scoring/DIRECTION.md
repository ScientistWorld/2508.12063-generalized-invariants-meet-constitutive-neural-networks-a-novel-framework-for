# Research Direction

Future agents must work within the paper's problem space and a closely related approach family.

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

While staying within compact physics-aware constitutive modeling, future agents can:

1. **Invariant design**: Different invariant families or parameterizations with the same benchmark protocol
2. **Initialization strategies**: Better warm-starts for difficult exponent regimes
3. **Regularization**: Sparsity or smoothness constraints that improve transfer
4. **Training stability**: Alternative optimizers, schedules, or stabilization methods
5. **Constitutive combinations**: Different compact combinations of physically meaningful terms
6. **Material-specific formulations**: Variants tailored to stretch-dominated versus shear-dominated materials

## Out of Bounds

Cannot claim improvement by:
- Using a completely different neural architecture (e.g., MLP, Transformer)
- Adding model capacity to memorize data
- Switching to a regression problem without hyperelasticity constraints
- Using external priors or embeddings not derived from deformation data
- Simplifying the problem to trivial toy cases

The goal is methodological advancement in compact, interpretable constitutive-law discovery under the same benchmark protocol, not beating the benchmark by leaving the problem setting.
