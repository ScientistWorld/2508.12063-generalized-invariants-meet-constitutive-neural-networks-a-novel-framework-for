# Optimization Targets

Scored metrics and coefficients are in `scoring/reference.json`; every experiment has the same
metric set. The ranked file is `scoring/scores_test.json` (hidden test: rubber_T50, brain_CR,
brain_BG), entry `GI-CANN` (the law file `outputs/laws/GI-CANN.json`).

## Benefit metrics (every experiment)

| metric | direction | coefficient | what it measures |
|---|---|---|---|
| `avg_r2` (primary) | higher | 1.0 | mean clipped R2 over the two training modes and the held-out mode (paper Tables 3/4) |
| `r2_ps` (rubber) / `r2_ss` (brain) | higher | 1.0 | clipped R2 of the held-out loading mode |
| `heldout_rel_error` | lower | 0.5 | mean relative stress error on the held-out mode (paper Figs. 4/5) |

## Constraint metric (every experiment)

| metric | direction | coefficient | what it measures |
|---|---|---|---|
| `n_terms` | lower | 0.25 | active terms in the law: the paper's claim is accuracy *with* a compact, interpretable law |

Hard constraints (a violation fails the experiment, worst value in every metric): non-negative
weights, the family's exponent sign, a defined energy wherever evaluated, finite stresses, 1 to 12
active terms.

## Report-only

`r2_ut`, `r2_bt` / `r2_uc` (included in `avg_r2`) and `alpha_star` (exponent of the dominant
term) have coefficient 0.

## Per experiment

Reference GI-CANN on the hidden test (avg_r2 / held-out R2 / held-out relative error / n_terms),
with the fixed SI-CANN baseline for scale:

| experiment | family | weight | GI-CANN (reference) | SI-CANN (baseline) |
|---|---|---|---|---|
| rubber_generalization | positive | 0.30 | 0.9939 / 0.9966 / 0.111 / 2 | 0.9751 / 0.9724 / 0.114 / 3 |
| brain_generalization | negative | 0.30 | 0.9544 / 0.9037 / 0.201 / 1 | 0.5429 / 0.8695 / 0.431 / 1 |
| rubber_combined_ablation | combined | 0.15 | 0.9974 / 0.9946 / 0.078 / 3 | 0.9969 / 0.9963 / 0.111 / 3 |
| brain_combined_ablation | combined | 0.15 | 0.9544 / 0.9037 / 0.201 / 1 | 0.5429 / 0.8695 / 0.431 / 1 |
| rubber_negative_ablation | negative | 0.05 | 0.3287 / 0.1250 / 0.623 / 2 | 0.3321 / 0.0000 / 0.923 / 2 |
| brain_positive_ablation | positive | 0.05 | 0.4835 / 0.8967 / 0.383 / 1 | 0.4987 / 0.8872 / 0.398 / 1 |

## Commands

```bash
bash scripts/evaluate_train.sh   # development side -> scoring/scores_train.json
bash scripts/evaluate.sh         # paper's materials -> scoring/scores.json (compare with reference.json)
```
