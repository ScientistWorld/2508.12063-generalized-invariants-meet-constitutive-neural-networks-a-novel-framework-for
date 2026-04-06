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
