# Constraints

What must stay fixed so that a better score means a better constitutive law.

## Data

- Fit each material's law on its training modes only: rubber uniaxial + equibiaxial tension,
  brain uniaxial tension + compression (`data/curves.json`, `data/load_data.py`).
- The held-out curves of the development materials (rubber_T20 pure shear, brain_CX and brain_CC
  simple shear) may be used to develop and compare methods, never as fitting data for a test
  material's law.
- Never use held-out curves of the test materials (rubber_T50 pure shear, brain_CR and brain_BG
  simple shear) from any source (for instance the authors' public data files).

## Model

- Submit laws in the schema of `briefing/evaluation.md`: sums of identity, exponential and
  logarithmic terms of generalized invariants J_alpha (optional power of J_alpha - 3 in [1, 3]).
  Any procedure may produce them.
- Weights non-negative; exponents obey the family (positive: > 0, negative: < 0, combined: any);
  |alpha| <= 60; the ln term's argument must stay positive wherever the law is evaluated.
- At most 12 active terms; fewer is better (`n_terms` is scored).

## Physics

Isotropic, incompressible hyperelasticity. The evaluator derives the stress of every loading mode
from the declared energy: uniaxial tension/compression and pure shear (sigma_11), equibiaxial
tension (sigma_11 = sigma_22), simple shear (sigma_12). Every admissible law is stress-free in the
reference state.

## Evaluation

`bash scripts/evaluate_train.sh` (development) and `bash scripts/evaluate_test.sh` (hidden test)
take no arguments and score every file in `outputs/laws/`; `outputs/laws/GI-CANN.json` is the
ranked entry. The SI-CANN baseline is fixed inside `eval/`.

## Resources

No pretrained weights or external data. CPU is enough: the reference trains in about 20
CPU-minutes.
