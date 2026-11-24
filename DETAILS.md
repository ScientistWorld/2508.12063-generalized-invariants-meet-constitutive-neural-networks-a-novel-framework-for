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

---

# Problem Definition

## Research Question

Given stress-stretch measurements of an isotropic, incompressible hyperelastic material in a few
loading modes, discover a strain energy function that (1) reproduces those measurements, (2)
predicts the material's response in a loading mode it was never fitted to, and (3) stays a
compact, closed-form, physically admissible constitutive law that an engineer can read.

Classical models (neo-Hookean, Mooney-Rivlin, Ogden, Gent, ...) fix the invariants and the
functional form in advance; generic regressors fit curves but give no constitutive law. The task
is automated discovery of the law itself.

## Data

Real experimental data, the curves used by the paper (its authors' own data files):

| material | source | training modes (fit) | held-out mode (predict) |
|---|---|---|---|
| rubber_T20 | Treloar (1944), 20 C | uniaxial tension, equibiaxial tension | pure shear |
| rubber_T50 | Treloar (1944), 50 C | uniaxial tension, equibiaxial tension | pure shear |
| brain_CX | Budday et al. (2017), cortex | uniaxial tension, uniaxial compression | simple shear |
| brain_CC | Budday et al. (2017), corpus callosum | uniaxial tension, uniaxial compression | simple shear |
| brain_CR | Budday et al. (2017), corona radiata | uniaxial tension, uniaxial compression | simple shear |
| brain_BG | Budday et al. (2017), basal ganglia | uniaxial tension, uniaxial compression | simple shear |

Rubber stretches reach 5-8 (stresses in MPa); brain tissue is tested to 10% stretch and 0.2 shear
(stresses in kPa) and is soft, nonlinear and stiffer in compression than in tension.

`data/curves.json` (loaded by `data/load_data.py`) holds every training-mode curve, and the
held-out curve of the development materials (rubber_T20, brain_CX, brain_CC). For the test
materials (rubber_T50, brain_CR, brain_BG) only the stretches / shears at which the held-out
mode is evaluated are given; their held-out stresses are the hidden test.

## Exponent families

The task is posed for three families of laws built on generalized invariants
`J_a = l1^a + l2^a + l3^a` (principal stretches l_i, l1 l2 l3 = 1): exponents a > 0 only
("positive", I1-like; a = 2 gives I1), a < 0 only ("negative", I2-like; a = -2 gives I2), or
either sign ("combined"). Each material is studied in all three.

## Why It Matters

Hyperelastic laws drive simulations of tyres, seals, soft robots, medical devices and brain
mechanics. A law that only fits the tests it was calibrated on fails in the multiaxial states that
matter in practice; one that generalizes across loading modes, and stays interpretable, is what
model discovery is for.

## Success Criteria

1. High accuracy on the held-out loading mode of materials whose held-out mode was never seen.
2. Faithful fit of the training modes.
3. Small relative stress errors, including at small strains.
4. A compact law: few active terms, every term a closed-form function of a generalized invariant.
5. Physical admissibility: non-negative energy contributions, a stress-free reference state, a
   well-defined energy wherever the law is evaluated.

---

# Paper's Method

## Key Contribution

GI-CANN (generalized-invariant constitutive artificial neural network) learns the invariant
itself. Instead of fixing I1 or I2, it uses generalized invariants J_alpha = l1^alpha + l2^alpha
+ l3^alpha with the exponent alpha as a trainable network parameter, and learns it together with
the strain energy function in one training run (no two-step "find alpha, then fit psi").

## Architecture (Eq. 13)

For every exponent alpha_k in the set S:

    psi = sum_k  wx2_k1 * w11_k * (J_k - 3)
               + wx2_k2 * [exp(w21_k (J_k - 3)) - 1]
               + wx2_k3 * [-ln(1 - w31_k (J_k - 3))]

All weights are non-negative. alpha = 2 gives J = I1 and alpha = -2 gives J = I2, so the
SI-CANN (S = {2}, {-2} or {2, -2}) is the special case with fixed exponents. Stresses follow from
psi in closed form for uniaxial tension/compression, equibiaxial tension, pure shear and simple
shear (Eqs. 15-18).

## Training (as in the authors' notebooks, LivingMatterLab/CANN/GICANN)

- Loss (Eq. 14): sum over training modes of the mean squared stress error, plus L1 with r = 0.01
  on every weight.
- Adam, learning rate 1e-3, mini-batches of 12, 30,000 epochs (40,000 for the brain
  two-invariant model), early stopping with patience 3,000 on the training loss, best epoch
  restored.
- Exponent starts: rubber 2.0 (two invariants: 1.7 and -0.1), brain 5.0 in magnitude (two
  invariants: 2 and -2).
- Families: positive (one alpha > 0), negative (one alpha < 0), combined (one of each).

## Reproduction in this gym (method/gicann.py, method/train.py)

Pure-Python implementation of the above with analytic gradients. Two additions make the reference
reach the law the paper reports instead of a random local stop:
- after the mini-batch phase, full-batch Adam with a decaying learning rate finishes minimizing the
  same objective (the objective is flat along an (alpha, weight) valley and early stopping halts
  at random places along it);
- each law is the best of 20 runs from random initializations, selected by the training objective
  (on the paper's cortex data only some starts escape a local optimum near alpha = -15).
On the paper's cortex data this gives alpha* = -18.11 with one identity term, w = 0.0092 kPa
(paper: -18.016, 0.009 kPa); on its rubber data alpha* = 1.676 with w_id = 0.231 MPa and
exponential weights 0.109, 0.282 MPa (paper Eq. 19: 1.675, 0.227, 0.107, 0.297).

## Main Claims

1. A learned exponent beats the fixed classical invariant: rubber avg R2 0.998 vs 0.982 (SI-CANN),
   brain 0.966 vs 0.610.
2. Rubber prefers positive (I1-like) exponents, alpha* ~ 1.7; negative exponents fail (0.264).
3. Brain tissue prefers a single large negative exponent, alpha* ~ -18; positive ones fail (0.526).
4. Combining both signs gives the best or an equally good fit (rubber 0.999, brain 0.966).
5. The discovered laws are sparse and interpretable (one or two active terms), close to dense
   PS-CANN or FNN accuracy.

---

# Evaluation

## What a method submits

A method submits constitutive laws, never stresses. One JSON file per method in `outputs/laws/`;
the file `outputs/laws/GI-CANN.json` is the ranked entry (other files are scored too and reported
under their file name). The file holds a law for every material and family:

```json
{"schema": "generalized_invariant_laws_v1",
 "laws": {"brain_CR": {"negative": {"terms": [{"fn": "id", "alpha": -24.2, "w": 0.0024}]},
                       "positive": {"terms": [...]}, "combined": {"terms": [...]}},
          "rubber_T50": {...}, ...}}
```

Every term is a function of one generalized invariant `J_alpha` (x = (J_alpha - 3)^power,
power in [1, 3], default 1):

| fn | energy | weights |
|---|---|---|
| `id` | `w * x` | `w >= 0` |
| `exp` | `w2 * (exp(w1 * x) - 1)` | `w1, w2 >= 0` |
| `ln` | `-w2 * ln(1 - w1 * x)` | `w1, w2 >= 0`, `1 - w1 x > 0` wherever evaluated |

The evaluator computes the Cauchy stress of every loading mode from the declared energy (closed
form for uniaxial tension/compression, equibiaxial tension, pure shear and simple shear) and
compares it with the measured curves. Nothing a method writes besides the law is read.

## Validity (a failed law scores as a failed run)

A law is invalid if it: breaks its family's sign rule (positive: every exponent > 0; negative:
every exponent < 0), has a negative or non-finite weight, |alpha| > 60, leaves the ln domain,
gives a non-finite stress, has no active term, or has more than 12 active terms. An experiment
whose law is missing or invalid for any of its materials receives the worst value of every
metric (R2 = 0, relative error = 1, n_terms = 12). A missing or unreadable `GI-CANN.json` scores
the worst value everywhere.

## Metrics (per experiment, averaged over its materials)

- `r2_<mode>`: coefficient of determination of each loading mode's stress, clipped to [0, 1]
  (a negative R2 counts as 0). The held-out mode's R2 (`r2_ps` / `r2_ss`) is scored.
- `avg_r2` (primary): mean of the three clipped per-mode R2.
- `heldout_rel_error`: mean of |sigma - sigma_hat| / |sigma_hat| over the loaded points of the
  held-out mode, each point capped at 1 (lower is better). Sensitive to small-strain response.
- `n_terms`: number of active terms, a term being active if its stress reaches 1% of a mode's
  measured stress range (lower is better): accuracy should not come from piling up terms.
- `alpha_star`: exponent of the dominant term, reported only.

R2 is computed on every point of each curve, including the undeformed state.

## Experiments

| experiment | material class | family | weight |
|---|---|---|---|
| rubber_generalization | rubber | positive | 0.30 |
| brain_generalization | brain | negative | 0.30 |
| rubber_combined_ablation | rubber | combined | 0.15 |
| brain_combined_ablation | brain | combined | 0.15 |
| rubber_negative_ablation | rubber | negative | 0.05 |
| brain_positive_ablation | brain | positive | 0.05 |

## Splits

- `scoring/scores.json` (`bash scripts/evaluate.sh`): the paper's own materials (rubber_T20,
  brain_CX), comparable with the paper's tables in `scoring/reference.json`.
- `scoring/scores_train.json` (`bash scripts/evaluate_train.sh`): development materials
  (rubber experiments: rubber_T20; brain experiments: brain_CX and brain_CC). Everything they
  are scored on is visible.
- `scoring/scores_test.json` (`bash scripts/evaluate_test.sh`, hidden test): rubber_T50 and
  brain_CR + brain_BG. Their held-out-mode stresses are not in the workspace.

Laws for the test materials must be fitted from their visible training modes only. Fitting any
held-out curve of a test material (for example one obtained from outside the workspace) is a
protocol violation.

## Ranking

The rank sums, over experiments, weight x coefficient x improvement over the reference in each
scored metric (in its better direction), each improvement divided by that metric's spread across
ranked attempts. Coefficients are in `scoring/reference.json`.

---

# Reproduction Log

### Iteration 1: MiniMax-M2.7
- **Milestone**: `core_claim` | **Status**: done
- **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-07] - method_runs (in progress toward core_claim)
- Full training run with 12 configs completed
- **Rubber results (working)**:
  - GI-CANN positive (α*≈1.677): avg_r2=0.999 ✓ (reference: 0.998)
  - SI-CANN positive (α=2): avg_r2=0.991 ✓ (reference: 0.982)
  - GI-CANN negative (α≈-0.52): avg_r2=0.647 (reference: 0.264)
  - SI-CANN negative (α=-2): avg_r2=0.573 (reference: 0.318)
  - GI-CANN combined (α+≈3.86, α-≈-1.42): avg_r2=0.998 ✓ (reference: 0.999)
  - SI-CANN combined (α=-2): avg_r2=-1689 ✗ CATASTROPHIC DIVERGENCE
- **Brain results (not working)**:
  - GI-CANN negative (α≈-18): avg_r2=-0.023 ✗ (reference: 0.610)
  - SI-CANN negative (α=-2): avg_r2=-0.506 ✗ (reference: 0.610)
  - GI-CANN positive (α≈0.60): avg_r2=-0.101 ✗ (reference: 0.526)
  - SI-CANN positive (α=2): avg_r2=-0.109 ✗ (reference: 0.527)
  - GI-CANN combined (α+≈3.4, α-≈-21): avg_r2=0.554 (reference: 0.966)
  - SI-CANN combined (α=-2): avg_r2=-0.502 ✗ (reference: 0.617)
- **Root causes identified**:
  1. Rubber combined SI-CANN: w_ln1_plus grows positive → LN singularity for BT (J-3 up to 10)
  2. Brain models: exp branch collapses due to L1 regularization (r=0.01) + gradient clipping
  3. Brain combined SS stress improved from R²=0.626→0.922 with B-eigenvalue formula, but UT/UC still fail
- **Fixes applied**:
  - Added w_ln1_plus ≤ -1e-6 clamp for rubber combined SI-CANN
  - Clamped LN and negative-ID branches small for rubber combined SI-CANN
  - Reduced brain L1 from 0.01→0.001
  - Added exp branch warm-start (w_exp1=0.1, w_exp2=0.1) for all brain configs
  - Added all branches (id/exp/ln) initialized for brain GI-CANN negative
- **Core claim status**: PARTIAL — rubber fully works (R²≈0.999), brain models need further fixes

### [2026-04-06] - method_runs
- Fixed critical bugs in GI-CANN and SI-CANN implementations:
  - gicann.py `_stress_uniaxial` and `_stress_equibiaxial`: negative mode was using wrong branch (dpsi_plus instead of dpsi_minus)
  - sicann.py: imports referenced non-existent functions (`get_principal_stretches_batch`, `detect_mode` → fixed to `get_principal_stretches`, `detect_deformation_mode`)
  - sicann.py `compute_stress`: added missing `gamma_override` parameter for simple shear
  - sicann.py `compute_stress`: fixed SS gamma indexing (`indices_t` → `[[i for i in indices]]`)
  - train.py: fixed `gamma_override=` → `gamma=` parameter name in loss call
  - baseline_train.py: fixed `gamma_override=` → `gamma=` parameter name in loss call
  - eval/evaluate.py: fixed scores.json format to include `experiments` wrapper
  - Updated container.def from python:3.11-slim to python:3.12-slim
- All code verified, ready for training run

### [2026-04-05 14:30] - method_runs
- Implemented GI-CANN neural network model (method/gicann.py) following the paper's architecture:
  - Trainable exponent α (positive, negative, or combined mode)
  - Three-branch strain energy function (identity, exponential, logarithmic)
  - Cauchy stress computation for each deformation mode (ut, bt, ps, ss)
- Implemented SI-CANN baseline model (method/sicann.py) with fixed exponents
- Created data loading code (data/load_data.py) with all rubber and brain tissue data from Table A.5
- Created training scripts for both GI-CANN and SI-CANN
- Created evaluation script (eval/evaluate.py) with R² metric
- Created all shell scripts (download.sh, evaluate.sh, method.sh, baseline.sh, reproduce.sh, run.sh)
- Created container definition and setup script
- Filled in all briefing files (problem.md, evaluation.md, method.md, overview.md)
- Created scoring/reference.json with paper's reported numbers

### [2026-04-05 14:18] - none
- Read the paper and identified key components
- Initial workspace analysis

</details>

### Iteration 1: MiniMax-M2.7
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 40m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-07] - method_runs (in progress toward core_claim)
- Full training run with 12 configs completed
- **Rubber results (working)**:
  - GI-CANN positive (α*≈1.677): avg_r2=0.999 ✓ (reference: 0.998)
  - SI-CANN positive (α=2): avg_r2=0.991 ✓ (reference: 0.982)
  - GI-CANN negative (α≈-0.52): avg_r2=0.647 (reference: 0.264)
  - SI-CANN negative (α=-2): avg_r2=0.573 (reference: 0.318)
  - GI-CANN combined (α+≈3.86, α-≈-1.42): avg_r2=0.998 ✓ (reference: 0.999)
  - SI-CANN combined (α=-2): avg_r2=-1689 ✗ CATASTROPHIC DIVERGENCE
- **Brain results (not working)**:
  - GI-CANN negative (α≈-18): avg_r2=-0.023 ✗ (reference: 0.610)
  - SI-CANN negative (α=-2): avg_r2=-0.506 ✗ (reference: 0.610)
  - GI-CANN positive (α≈0.60): avg_r2=-0.101 ✗ (reference: 0.526)
  - SI-CANN positive (α=2): avg_r2=-0.109 ✗ (reference: 0.527)
  - GI-CANN combined (α+≈3.4, α-≈-21): avg_r2=0.554 (reference: 0.966)
  - SI-CANN combined (α=-2): avg_r2=-0.502 ✗ (reference: 0.617)
- **Root causes identified**:
  1. Rubber combined SI-CANN: w_ln1_plus grows positive → LN singularity for BT (J-3 up to 10)
  2. Brain models: exp branch collapses due to L1 regularization (r=0.01) + gradient clipping
  3. Brain combined SS stress improved from R²=0.626→0.922 with B-eigenvalue formula, but UT/UC still fail
- **Fixes applied**:
  - Added w_ln1_plus ≤ -1e-6 clamp for rubber combined SI-CANN
  - Clamped LN and negative-ID branches small for rubber combined SI-CANN
  - Reduced brain L1 from 0.01→0.001
  - Added exp branch warm-start (w_exp1=0.1, w_exp2=0.1) for all brain configs
  - Added all branches (id/exp/ln) initialized for brain GI-CANN negative
- **Core claim status**: PARTIAL — rubber fully works (R²≈0.999), brain models need further fixes

### [2026-04-06] - method_runs
- Fixed critical bugs in GI-CANN and SI-CANN implementations:
  - gicann.py `_stress_uniaxial` and `_stress_equibiaxial`: negative mode was using wrong branch (dpsi_plus instead of dpsi_minus)
  - sicann.py: imports referenced non-existent functions (`get_principal_stretches_batch`, `detect_mode` → fixed to `get_principal_stretches`, `detect_deformation_mode`)
  - sicann.py `compute_stress`: added missing `gamma_override` parameter for simple shear
  - sicann.py `compute_stress`: fixed SS gamma indexing (`indices_t` → `[[i for i in indices]]`)
  - train.py: fixed `gamma_override=` → `gamma=` parameter name in loss call
  - baseline_train.py: fixed `gamma_override=` → `gamma=` parameter name in loss call
  - eval/evaluate.py: fixed scores.json format to include `experiments` wrapper
  - Updated container.def from python:3.11-slim to python:3.12-slim
- All code verified, ready for training run

### [2026-04-05 14:30] - method_runs
- Implemented GI-CANN neural network model (method/gicann.py) following the paper's architecture:
  - Trainable exponent α (positive, negative, or combined mode)
  - Three-branch strain energy function (identity, exponential, logarithmic)
  - Cauchy stress computation for each deformation mode (ut, bt, ps, ss)
- Implemented SI-CANN baseline model (method/sicann.py) with fixed exponents
- Created data loading code (data/load_data.py) with all rubber and brain tissue data from Table A.5
- Created training scripts for both GI-CANN and SI-CANN
- Created evaluation script (eval/evaluate.py) with R² metric
- Created all shell scripts (download.sh, evaluate.sh, method.sh, baseline.sh, reproduce.sh, run.sh)
- Created container definition and setup script
- Filled in all briefing files (problem.md, evaluation.md, method.md, overview.md)
- Created scoring/reference.json with paper's reported numbers

### [2026-04-05 14:18] - none
- Read the paper and identified key components
- Initial workspace analysis

</details>

### Iteration 2: glm-5.1
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 13m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08] - Wrap-up Complete
- Fixed validation errors (added alpha_star to reference.json metrics, created eval/models.py for import separation)
- Updated scoring/EXPERIMENTS.md with all 6 experiments
- Updated scoring/TARGETS.md with Primary/Constraint/Ablation target classifications
- All validation checks passed
- Final commit prepared

### [2026-04-08] - Audit Complete
**Auditor verified:**
- Implementation is real and complete (GI-CANN algorithm correctly implemented)
- Evaluation is honest (results computed from checkpoints, not hardcoded)
- All deliverables present (briefing/, scoring/, scripts/, environment/)
- Evaluation script re-ran successfully, reproduced all scores

**Audit result: `core_claim` milestone restored.** Previous agent's claim was valid.

### [2026-04-07] - core_claim
- Reproduced GI-CANN for rubber material (VHB 4910)
- GI-CANN discovers α*≈1.677, achieving R²=0.999 (paper: 0.998/1.675) ✓
- All rubber configurations working: positive (R²=0.999), combined (R²=0.998), negative ablation (R²=0.648)
- Baseline SI-CANN for rubber also working (R²=0.991 with α=2)
- Brain tissue: NOT reproduced — α⁻ stress formula incompatible with brain UT/UC data
  - Brain GI-CANN achieves R²=0.14-0.62 vs reference 0.966
  - Root cause: α⁻ stress formula (σ ∝ (l3^α - l1^α)) produces incorrect stress for brain data
  - SI-CANN baseline for brain also fails (R²≈-0.5), confirming formulation issue not specific to GI-CANN

### [2026-04-06] - method_runs
- Implemented GI-CANN and SI-CANN models with three-branch strain energy
- Data loading from paper's Table A.5 for rubber and brain cortex
- Training and evaluation pipeline working

</details>

### Iteration 3: glm-5.1
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

</details>

### Iteration 4: glm-5.1 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 11m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

</details>

### Iteration 5: glm-5.1 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 27m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 6: gpt-5.4 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 7: gpt-5.4 (designer)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 17m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 7: MiniMax-M2.7 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 8: MiniMax-M2.7 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

</details>

### Iteration 9: gpt-5.4 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

### [2026-04-24 00:00] - core_claim
- Audited the first real train/test split run and found checkpoint loading failed before any scores were produced because the container NumPy exposes `numpy.core` while the frozen checkpoints unpickle `numpy._core`
- Patched all evaluation entry points to install backward-compatible NumPy module aliases before `torch.load`, preserving the frozen method artifacts while unblocking benchmark scoring
- Kept the milestone at `core_claim`; this turn advances evaluation robustness but does not change the reproduced method quality

### [2026-04-24 00:15] - core_claim
- Completed the real train/test split evaluation and produced validated `scoring/scores_train.json` and `scoring/scores_test.json` with matching experiment/method coverage
- Verified that the split is not a copy-paste bug: every experiment differs numerically across slices, although the strongest rubber benchmark has only a very small gap because the alternating split stays on the same smooth curve family
- Closed the design phase at `core_claim` because the evaluation artifact is now complete and usable, with the main caveat that the shipped brain checkpoints remain weak on both slices

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `majority` | **Status**: done
- **Working time**: 12m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-04-08 00:00] - method_runs
- Existing GI-CANN and SI-CANN pipelines execute end-to-end on the paper's Table A.5 data
- Evaluation and scoring infrastructure are in place and validate successfully

### [2026-04-08 00:00] - core_claim
- Rubber reproduction supports the paper's central claim: GI-CANN learns a positive exponent near 1.675 and achieves near-perfect held-out pure-shear accuracy
- Brain reproduction remains below paper numbers, but the workspace is usable and the current turn patches the stress equations to better match the paper's exact mode-specific formulas

### [2026-04-08 00:45] - core_claim
- Audited the previous reproduction and confirmed the code implements the paper's GI-CANN/SI-CANN family rather than a surrogate regressor
- Fixed the standalone evaluator so `eval/` no longer drops SI-CANN checkpoints and `scores.json` mirrors the `reference.json` experiment layout through a `results` block
- Restored trust only for the rubber core claim; the current brain and multi-invariant checkpoints need retraining under the corrected constitutive formulas

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

### [2026-04-24 00:00] - core_claim
- Audited the first real train/test split run and found checkpoint loading failed before any scores were produced because the container NumPy exposes `numpy.core` while the frozen checkpoints unpickle `numpy._core`
- Patched all evaluation entry points to install backward-compatible NumPy module aliases before `torch.load`, preserving the frozen method artifacts while unblocking benchmark scoring
- Kept the milestone at `core_claim`; this turn advances evaluation robustness but does not change the reproduced method quality

### [2026-04-24 00:15] - core_claim
- Completed the real train/test split evaluation and produced validated `scoring/scores_train.json` and `scoring/scores_test.json` with matching experiment/method coverage
- Verified that the split is not a copy-paste bug: every experiment differs numerically across slices, although the strongest rubber benchmark has only a very small gap because the alternating split stays on the same smooth curve family
- Closed the design phase at `core_claim` because the evaluation artifact is now complete and usable, with the main caveat that the shipped brain checkpoints remain weak on both slices

### [2026-05-22 07:19] - core_claim
- Audited the current package and confirmed the core weakness is still the brain checkpoints, not the rubber GI-CANN reproduction
- Patched SI-CANN training to use the same observed stress component loss as GI-CANN and scoring
- Patched the main evaluator to recompute scores from checkpoint weights instead of trusting checkpoint-embedded metrics
- Prepared a targeted CPU job to retrain all brain GI-CANN/SI-CANN checkpoints and regenerate `scores.json`, `scores_train.json`, and `scores_test.json`

### [2026-05-22 07:22] - core_claim
- Final pre-submit audit confirmed `scripts/run.sh` and `action.yaml` are aligned on the brain retraining job
- Fixed executable bits on the main shell entrypoints so the scheduler can invoke them directly

### [2026-05-22 07:50] - core_claim_plus
- CPU retraining job completed successfully and regenerated full/train/test score artifacts
- Brain GI-CANN now matches the paper-level core behavior: negative-mode avg R²=0.9590 with α*=-17.9399 versus paper avg R²=0.966 and α*=-18.016
- Brain combined GI-CANN also reproduces the secondary brain result with avg R²=0.9604
- Prepared a follow-up CPU job for the remaining weak rubber ablation checkpoints

### [2026-05-22 07:58] - core_claim_plus
- Rubber ablation job failed before evaluation because the trainer's warm-start checkpoint loader lacked the NumPy pickle compatibility alias already used in evaluation
- Partial rubber negative training reached avg R²=0.3486, better than the paper's low-ablation reference of 0.264, but no milestone increase is claimed because the job did not finish
- Patched trainer checkpoint loading and resubmitted the rubber ablation job

### [2026-05-22 07:59] - majority
- Rubber ablation retraining completed after one scheduler preemption/restart and regenerated `scores.json`, `scores_train.json`, and `scores_test.json`
- Rubber combined GI-CANN now reaches avg R²=0.9994 versus the paper's 0.999, while rubber negative GI-CANN remains a low-performing contrastive ablation at avg R²=0.3486 versus the paper's 0.264
- Full validation passes with both core GI-CANN claims reproduced: rubber avg R²=0.9988 with α*=1.6769, and brain avg R²=0.9590 with α*=-17.9399
- Stopped at `majority` rather than `near_complete` because SI-CANN baselines and some per-mode ablation details are still uneven relative to the paper, although the proposed-method claims and main constraints are covered

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `majority` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:00] - majority
- Implemented the paper's GI-CANN method with trainable generalized-invariant exponents and the identity, exponential, and logarithmic strain-energy branches from Eq. (13).
- Implemented SI-CANN fixed-exponent baselines and paper-table rubber/brain data loaders.
- Completed compute jobs that regenerated checkpoints and scoring artifacts for rubber and brain core generalization plus positive/negative/combined ablations.
- Reproduced both central proposed-method claims: rubber GI-CANN `avg_r2=0.9988`, `alpha_star=1.6769`; brain GI-CANN `avg_r2=0.9590`, `alpha_star=-17.9399`.
- Reproduced most proposed-method ablations, including rubber combined `avg_r2=0.9994` and brain combined `avg_r2=0.9604`.

### [2026-05-22 08:20] - majority
- Re-audited the continued workspace after milestone reset to `secondary_claims`.
- Confirmed the implementation is the paper's GI-CANN algorithm rather than a surrogate, and that evaluation reads checkpoint artifacts without importing `method/`.
- Fixed split evaluator score construction so `scores_train.json` and `scores_test.json` can be regenerated reliably.
- Re-ran syntax checks and `python3 validate.py --compare`; all checks passed, so `majority` is restored.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `majority` | **Status**: done
- **Working time**: 11m | **GPU**: 0.0h
- **Jobs**: 8 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `majority` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 8 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `majority` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 8 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `majority` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 9 total (6 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

### [2026-05-22 09:10] - majority
- Reviewed completed split-evaluation job `6487ae0e-790`: the full benchmark and both train/test split score files validate successfully.
- Confirmed the actual method supports the core claim in `scoring/scores.json`: GI-CANN improves over SI-CANN on rubber generalization (`avg_r2` 0.9988 vs 0.9847, learned `alpha_star` 1.6769) and brain generalization (`avg_r2` 0.9590 vs 0.6974, learned `alpha_star` -17.9399).

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `majority` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 9 total (6 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

### [2026-05-22 09:10] - majority
- Reviewed completed split-evaluation job `6487ae0e-790`: the full benchmark and both train/test split score files validate successfully.
- Confirmed the actual method supports the core claim in `scoring/scores.json`: GI-CANN improves over SI-CANN on rubber generalization (`avg_r2` 0.9988 vs 0.9847, learned `alpha_star` 1.6769) and brain generalization (`avg_r2` 0.9590 vs 0.6974, learned `alpha_star` -17.9399).

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `majority` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 9 total (6 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

### [2026-05-22 09:10] - majority
- Reviewed completed split-evaluation job `6487ae0e-790`: the full benchmark and both train/test split score files validate successfully.
- Confirmed the actual method supports the core claim in `scoring/scores.json`: GI-CANN improves over SI-CANN on rubber generalization (`avg_r2` 0.9988 vs 0.9847, learned `alpha_star` 1.6769) and brain generalization (`avg_r2` 0.9590 vs 0.6974, learned `alpha_star` -17.9399).

### [2026-05-22 09:20] - majority
- Moved the active trainer from the repository root to `method/training_runner.py` and updated `scripts/method.sh`, `scripts/baseline.sh`, `scripts/reproduce.sh`, and `method/export_predictions.py`.
- Re-ran syntax checks, artifact-based `scripts/evaluate.sh`, and `python3 validate.py --compare`; all checks passed with the same `majority` results.

### [2026-05-22 09:35] - majority
- Completed final verification of folder separation, method-agnostic documentation, scoring coefficients, artifact-based evaluation, and train/test split integrity.
- Re-ran `python3 validate.py --compare`; validation passed and the `majority` milestone remains justified.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `majority` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 9 total (6 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

### [2026-05-22 09:10] - majority
- Reviewed completed split-evaluation job `6487ae0e-790`: the full benchmark and both train/test split score files validate successfully.
- Confirmed the actual method supports the core claim in `scoring/scores.json`: GI-CANN improves over SI-CANN on rubber generalization (`avg_r2` 0.9988 vs 0.9847, learned `alpha_star` 1.6769) and brain generalization (`avg_r2` 0.9590 vs 0.6974, learned `alpha_star` -17.9399).

### [2026-05-22 09:20] - majority
- Moved the active trainer from the repository root to `method/training_runner.py` and updated `scripts/method.sh`, `scripts/baseline.sh`, `scripts/reproduce.sh`, and `method/export_predictions.py`.
- Re-ran syntax checks, artifact-based `scripts/evaluate.sh`, and `python3 validate.py --compare`; all checks passed with the same `majority` results.

### [2026-05-22 09:35] - majority
- Completed final verification of folder separation, method-agnostic documentation, scoring coefficients, artifact-based evaluation, and train/test split integrity.
- Re-ran `python3 validate.py --compare`; validation passed and the `majority` milestone remains justified.

### [2026-05-22 08:46] - majority
- Continuation audit re-read the paper's core GI-CANN algorithm and checked the packaged code against the actual method: trainable generalized invariant exponents, Eq. 13 strain-energy branches, loading-mode stress formulas, and L1 sparsity are present.
- Re-ran `python3 validate.py --compare`; validation passed using the existing artifact-based score files and the `majority` milestone remains justified.

### [2026-05-22 08:49] - majority
- Final verifier reviewed folder separation, method-agnostic docs, scoring coefficients, split evaluators, download portability, and reproduced numbers.
- Re-ran artifact-based full/train/test evaluation and `python3 validate.py --compare`; all checks passed and the `majority` milestone remains the final judgment.

</details>

### Iteration 10: claude-opus-5-5 (redteam)
- **Milestone**: `majority` | **Status**: done
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

### [2026-05-22 09:10] - majority
- Reviewed completed split-evaluation job `6487ae0e-790`: the full benchmark and both train/test split score files validate successfully.
- Confirmed the actual method supports the core claim in `scoring/scores.json`: GI-CANN improves over SI-CANN on rubber generalization (`avg_r2` 0.9988 vs 0.9847, learned `alpha_star` 1.6769) and brain generalization (`avg_r2` 0.9590 vs 0.6974, learned `alpha_star` -17.9399).

### [2026-05-22 09:20] - majority
- Moved the active trainer from the repository root to `method/training_runner.py` and updated `scripts/method.sh`, `scripts/baseline.sh`, `scripts/reproduce.sh`, and `method/export_predictions.py`.
- Re-ran syntax checks, artifact-based `scripts/evaluate.sh`, and `python3 validate.py --compare`; all checks passed with the same `majority` results.

### [2026-05-22 09:35] - majority
- Completed final verification of folder separation, method-agnostic documentation, scoring coefficients, artifact-based evaluation, and train/test split integrity.
- Re-ran `python3 validate.py --compare`; validation passed and the `majority` milestone remains justified.

### [2026-05-22 08:46] - majority
- Continuation audit re-read the paper's core GI-CANN algorithm and checked the packaged code against the actual method: trainable generalized invariant exponents, Eq. 13 strain-energy branches, loading-mode stress formulas, and L1 sparsity are present.
- Re-ran `python3 validate.py --compare`; validation passed using the existing artifact-based score files and the `majority` milestone remains justified.

### [2026-05-22 08:49] - majority
- Final verifier reviewed folder separation, method-agnostic docs, scoring coefficients, split evaluators, download portability, and reproduced numbers.
- Re-ran artifact-based full/train/test evaluation and `python3 validate.py --compare`; all checks passed and the `majority` milestone remains the final judgment.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `majority` | **Status**: done
- **Working time**: 51m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

### [2026-05-22 09:10] - majority
- Reviewed completed split-evaluation job `6487ae0e-790`: the full benchmark and both train/test split score files validate successfully.
- Confirmed the actual method supports the core claim in `scoring/scores.json`: GI-CANN improves over SI-CANN on rubber generalization (`avg_r2` 0.9988 vs 0.9847, learned `alpha_star` 1.6769) and brain generalization (`avg_r2` 0.9590 vs 0.6974, learned `alpha_star` -17.9399).

### [2026-05-22 09:20] - majority
- Moved the active trainer from the repository root to `method/training_runner.py` and updated `scripts/method.sh`, `scripts/baseline.sh`, `scripts/reproduce.sh`, and `method/export_predictions.py`.
- Re-ran syntax checks, artifact-based `scripts/evaluate.sh`, and `python3 validate.py --compare`; all checks passed with the same `majority` results.

### [2026-05-22 09:35] - majority
- Completed final verification of folder separation, method-agnostic documentation, scoring coefficients, artifact-based evaluation, and train/test split integrity.
- Re-ran `python3 validate.py --compare`; validation passed and the `majority` milestone remains justified.

### [2026-05-22 08:46] - majority
- Continuation audit re-read the paper's core GI-CANN algorithm and checked the packaged code against the actual method: trainable generalized invariant exponents, Eq. 13 strain-energy branches, loading-mode stress formulas, and L1 sparsity are present.
- Re-ran `python3 validate.py --compare`; validation passed using the existing artifact-based score files and the `majority` milestone remains justified.

### [2026-05-22 08:49] - majority
- Final verifier reviewed folder separation, method-agnostic docs, scoring coefficients, split evaluators, download portability, and reproduced numbers.
- Re-ran artifact-based full/train/test evaluation and `python3 validate.py --compare`; all checks passed and the `majority` milestone remains the final judgment.

### [2026-11-24 01:30] - majority (iteration 11, hardening)
- Gym rebuilt around declared strain-energy laws scored by `eval/` (no method-written numbers), the authors' exact data files, and a hidden test on Treloar rubber at 50 C and the corona radiata / basal ganglia (held-out curves only in `eval/test/`).
- Reference GI-CANN retrained from the final code following the authors' notebooks (plus convergence and best-of-20 by training objective): paper materials reproduce Tables 3/4 (rubber 0.9987 vs 0.998, cortex 0.9669 vs 0.966; alpha* 1.676 and -18.11 vs 1.675 and -18.016); GI beats SI on the hidden test (rubber 0.9939 vs 0.9751, brain 0.9544 vs 0.5429).
- All red-team shortcuts closed (fail or no effect); a dev-tuned L1 variant beats the reference by +0.22 normalized against a reference-procedure noise of +0.001; brute force and PS-CANN do not outrank. Details in `HARDENING.md`.

</details>


---

# Reproduction Milestones

**Current: majority**

## Stop Justification
- Completed at milestone `majority`.
