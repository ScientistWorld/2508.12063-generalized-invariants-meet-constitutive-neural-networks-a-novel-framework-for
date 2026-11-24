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
