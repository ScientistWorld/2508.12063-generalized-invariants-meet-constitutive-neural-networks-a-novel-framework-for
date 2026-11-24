# Research Direction

## Problem Space

Automated discovery of hyperelastic constitutive laws for isotropic incompressible soft materials
(rubber, brain tissue) from a few loading modes, judged on a loading mode never fitted and on
materials whose held-out response is hidden.

## Approach Family

Physics-based constitutive model discovery with generalized invariants J_a = sum_i l_i^a:
the choice of invariants (exponents), the choice of functions of them, and how both are learned
from data. Suitable directions:

1. Better discovery of exponents and terms (optimization, initialization, multi-start, sparsity,
   model selection on the training modes or on the development materials).
2. Better loss design (mode weighting, scale-free errors, regularization that transfers across
   loading modes).
3. Better compact term libraries within the schema (functions, powers, number of invariants).
4. Using the development materials (rubber_T20, brain_CX, brain_CC, whose held-out curves are
   visible) to choose procedures that generalize, then applying the procedure to the test
   materials' training modes.

## Out of Bounds

- Fitting any held-out curve of a test material, from any source.
- Black-box stress regressors: only declared strain-energy laws are scored.
- Dense expansions that buy accuracy with many terms: `n_terms` is scored and more than 12 active
  terms is invalid.
- Laws that break the family sign rule, use negative weights or leave the ln domain (invalid).

The goal is laws that generalize better across loading modes while staying compact and readable.
