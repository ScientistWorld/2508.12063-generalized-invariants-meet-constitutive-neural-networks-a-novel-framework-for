# Generalized invariants meet constitutive neural networks: A novel framework for hyperelastic materials

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2508.12063"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2508.12063-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2508.12063](https://arxiv.org/abs/2508.12063) |
|:---|:---|
| **Domain** | Computational Mechanics / Constitutive Modeling |
| **TL;DR** | A neural network (GI-CANN) that simultaneously discovers the optimal exponent for generalized invariants and the strain energy function from stress-strain data, enabling automated interpretable model discovery for hyperelastic materials. |

## Summary

The paper addresses the challenge of automatically discovering physically meaningful constitutive models for isotropic incompressible hyperelastic materials from experimental stress-strain data. The proposed GI-CANN (Generalized Invariant Constitutive Artificial Neural Network) learns a trainable exponent α in the generalized invariant J_α = Σ λᵢ^α, and combines it with a structured neural network that outputs the strain energy function via three parallel branches (identity, exponential, logarithmic). Unlike prior approaches that require fixed invariants or sequential two-step fitting, GI-CANN discovers both the invariant and the strain energy function in a single training procedure. The method is validated on rubber (Treloar data) and brain cortex (Budday et al. data), demonstrating that it automatically identifies the appropriate invariant formulation for each material — stretch-dominated behavior (rubber) prefers positive exponents near α=1.7, while shear-dominated behavior (brain tissue) prefers large negative exponents α≈-18.

## Problem

The central challenge is: given experimental stress-strain data for an isotropic incompressible hyperelastic material (such as rubber or brain tissue), how do we automatically discover an accurate and interpretable strain-energy law without hand-designing the final constitutive form in advance?

Classical hyperelastic constitutive models (neo-Hookean, Mooney-Rivlin, Ogden, etc.) require an engineer to manually choose a strain energy function form and invariants a priori. This choice is often based on intuition or trial-and-error and may not suit all materials. Data-driven approaches using neural networks can fit stress-strain data but often lack interpretability or require the user to pre-specify which invariants to use.

## The paper's method

The paper proposes **GI-CANN (Generalized Invariant Constitutive Artificial Neural Networks)**, which simultaneously discovers the optimal generalized invariant exponent α and the corresponding strain energy function directly from experimental stress-strain data. Unlike prior methods that require manual selection of invariants or sequential two-step fitting, GI-CANN integrates discovery into a single end-to-end neural network training procedure.

## Key results of the paper

- **Rubber**: GI-CANN with single positive exponent achieves average R² = 0.998 (trained on uniaxial + equibiaxial, tested on pure shear), compared to SI-CANN's R² = 0.982
- **Brain tissue**: GI-CANN with single negative exponent achieves average R² = 0.966 (trained on uniaxial tension + compression, tested on simple shear), significantly outperforming SI-CANN's R² = 0.527 with α+=2
- **Interpretability**: The discovered models correspond to closed-form strain energy functions (e.g., ψ = 0.227 MPa · (J_1.675 - 3) + 0.297 MPa · [exp(0.107 · (J_1.675 - 3)) - 1] for rubber)

## Run it

```bash
pip install sciworld
sw research 2508.12063 --mode innovate --agent claude-code
sw evaluate 2508.12063
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

