# Optimization Targets

This document defines what future agents should optimize in this workspace.

## Primary Targets

### `rubber_generalization`

Train on rubber `ut` + `bt`, test on held-out `ps`.

- Primary metric: `avg_r2`
- Component metrics: `r2_ut`, `r2_bt`, `r2_ps`
- Report-only descriptor: `alpha_star`
- Main success condition: improve held-out `r2_ps` without hurting overall fit

### `brain_generalization`

Train on brain cortex `ut` + `uc`, test on held-out `ss`.

- Primary metric: `avg_r2`
- Component metrics: `r2_ut`, `r2_uc`, `r2_ss`
- Report-only descriptor: `alpha_star`
- Main success condition: improve held-out `r2_ss` without overfitting the training modes

## Constraint / Interpretation Targets

### `rubber_negative_ablation`

- Expect low `avg_r2` when the model is restricted to a negative exponent branch
- This serves as a contrastive sanity check, not the main optimization target

### `brain_positive_ablation`

- Expect low `avg_r2` when the model is restricted to a positive exponent branch
- This serves as a contrastive sanity check, not the main optimization target

## Optional Ablation Targets

### `rubber_combined_ablation`

- Compare combined `alpha+`/`alpha-` training against the single-positive rubber core experiment

### `brain_combined_ablation`

- Compare combined `alpha+`/`alpha-` training against the single-negative brain core experiment

## Evaluation Command

```bash
bash scripts/evaluate.sh
```

## Milestone Guidance

- `core_claim` requires at least `rubber_generalization` or `brain_generalization` to support the paper’s central claim with the actual GI-CANN algorithm.
- `core_claim_plus` requires both generalization experiments to be reproduced or one generalization experiment plus additional ablations/coverage that materially strengthen the claim.
