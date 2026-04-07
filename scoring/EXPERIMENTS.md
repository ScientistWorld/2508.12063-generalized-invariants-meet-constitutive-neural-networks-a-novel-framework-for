# Experiment Overview

This document describes what each experiment in `scores.json` is trying to establish.

## rubber_generalization

Tests whether the discovered strain energy function, trained on two deformation modes (uniaxial tension and equibiaxial tension), can accurately predict stress on an unseen deformation mode (pure shear) for rubber material. The rubber data (Treloar 1944) represents a stretch-dominated material where deformation is primarily governed by large principal stretches. This experiment validates that the model captures the underlying material physics rather than simply memorizing the training deformation.

## rubber_negative_ablation

Tests whether a negative exponent (I₂-like invariant) can adequately model rubber material behavior. This is an ablation study showing that rubber — as a stretch-dominated material — performs poorly when restricted to negative invariants. The poor expected performance confirms that rubber's constitutive behavior is not captured by I₂-type invariants, supporting the claim that stretch-dominated materials require positive exponents.

## rubber_combined_ablation

Tests whether using both positive and negative exponents simultaneously improves performance over a single positive exponent for rubber. This examines whether the multi-invariant formulation captures additional physics that a single invariant might miss. The expectation is that combined invariants may offer marginal improvements, but the single positive invariant already achieves near-perfect performance, suggesting that the positive invariant alone is sufficient for rubber.

## brain_generalization

Tests whether the discovered strain energy function, trained on two deformation modes (uniaxial tension and uniaxial compression), can accurately predict stress on an unseen deformation mode (simple shear) for brain cortex tissue. The brain data (Budday et al. 2017) represents a shear-sensitive material where deformation behavior is dominated by the coupling between normal and shear responses. This experiment validates that the model captures the complex nonlinear shear behavior characteristic of soft biological tissues.

## brain_positive_ablation

Tests whether a positive exponent (I₁-like invariant) can adequately model brain tissue behavior. This is an ablation study showing that brain cortex — as a shear-sensitive material — performs poorly when restricted to positive invariants. The poor expected performance confirms that brain's constitutive behavior is not captured by I₁-type invariants, supporting the claim that shear-sensitive materials require negative exponents.

## brain_combined_ablation

Tests whether using both positive and negative exponents simultaneously improves performance over a single negative exponent for brain tissue. This examines whether the multi-invariant formulation captures additional physics that a single invariant might miss. The expectation is that combined invariants may offer marginal improvements, but the single negative invariant already achieves good performance, suggesting that the negative invariant alone captures the essential shear-sensitive behavior.
