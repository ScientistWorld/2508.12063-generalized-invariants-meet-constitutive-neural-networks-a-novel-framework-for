# Experiment Overview

This document describes what each experiment in `scores.json` is trying to establish.

## rubber_generalization

Tests whether a constitutive model trained on rubber uniaxial tension and equibiaxial tension can generalize to held-out pure shear. This is the main rubber benchmark and measures whether the learned constitutive law transfers across loading modes rather than only matching the training curves.

## rubber_negative_ablation

Tests a restricted rubber setting intended to check whether a shear-dominated descriptor family is sufficient for this material. Poor performance is expected and serves as a sanity check that the benchmark can distinguish materially inappropriate constitutive choices.

## rubber_combined_ablation

Tests whether a richer rubber constitutive description improves on the main single-family setting. This experiment asks whether combining complementary descriptors yields a measurable gain beyond the already strong primary rubber benchmark.

## brain_generalization

Tests whether a constitutive model trained on brain-cortex uniaxial tension and compression can generalize to held-out simple shear. This is the hardest experiment in the workspace and is the most informative check of cross-mode transfer for soft biological tissue.

## brain_positive_ablation

Tests a restricted brain setting intended to check whether a stretch-dominated descriptor family is sufficient for this material. Poor performance is expected and helps verify that the benchmark remains sensitive to constitutive mismatch.

## brain_combined_ablation

Tests whether a richer brain constitutive description improves on the main single-family setting. This experiment checks whether extra flexibility materially helps once the model already captures the key shear-sensitive response.
