# Overview

- **Paper ID:** 2508.12063
- **Title:** Generalized invariants meet constitutive neural networks: A novel framework for hyperelastic materials
- **Domain:** Computational mechanics / constitutive modeling
- **TL;DR:** GI-CANN makes the exponent of the generalized invariant J_alpha = sum l_i^alpha a trainable parameter of a constitutive neural network, so the invariant and the strain energy are discovered together from stress-stretch data.

## Short Summary

Classical hyperelastic models fix their invariants (I1, I2) and functional form in advance.
GI-CANN learns the generalized-invariant exponent together with a sparse strain energy built from
identity, exponential and logarithmic terms, trained on some loading modes and tested on another.
On Treloar's rubber it finds an I1-like law (alpha* = 1.675, avg R2 0.998); on human cortex a
single strongly I2-like term (alpha* = -18.016, avg R2 0.966), both clearly better than the same
network with alpha fixed at +-2 (0.982, 0.610).

## The gym

Laws are scored, not stress predictions: a method declares its strain energy per material and
exponent family, the evaluator computes the stresses. The paper's two materials and the corpus
callosum are the development side; Treloar rubber at 50 C and the corona radiata and basal
ganglia (same studies, same protocol) form the hidden test, whose held-out-mode curves are not in
the workspace. Metrics: per-mode R2 (clipped at 0, as in the paper), mean relative error on the
held-out mode, and the number of active terms. See `briefing/evaluation.md`.
