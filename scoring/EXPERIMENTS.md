# Experiment Overview

Every experiment asks the same question for one material class and one exponent family: fitted
on the training loading modes only, how well does the discovered constitutive law reproduce
those modes and predict the held-out mode, and how compact is it? Scores average over the
experiment's materials: the paper's material(s) on the development side, other materials from
the same studies on the hidden test (see `briefing/evaluation.md`).

## rubber_generalization (weight 0.30)

Rubber, exponents > 0 only (I1-like generalized invariants). Fit uniaxial + equibiaxial tension,
predict pure shear. The main rubber benchmark: stretch-dominated entropic elasticity at large
stretches (to 7-8), where small errors in the invariant choice show up in the held-out mode and
in the relative error at small stretches.

## brain_generalization (weight 0.30)

Brain tissue, exponents < 0 only (I2-like). Fit uniaxial tension + compression, predict simple
shear. The main soft-tissue benchmark: a small-strain, tension/compression-asymmetric material
whose shear response is not determined by a fixed classical invariant.

## rubber_combined_ablation (weight 0.15) and brain_combined_ablation (weight 0.15)

Exponents of either sign. Asks whether mixing I1-like and I2-like generalized invariants gives a
better (or equally good, more general) law than the single-family setting.

## rubber_negative_ablation (weight 0.05) and brain_positive_ablation (weight 0.05)

The mismatched family for each material (I2-like only for rubber, I1-like only for brain). Laws
are expected to be poor here; the experiments check that the benchmark separates appropriate from
inappropriate invariant families, and reward a better law even within the restricted family.
