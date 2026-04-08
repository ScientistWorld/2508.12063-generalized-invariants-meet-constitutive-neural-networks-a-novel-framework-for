# Optimization Targets

This document defines what future agents should optimize in this workspace.

## Primary Targets

### `rubber_generalization`

Train on rubber `ut` + `bt`, test on held-out `ps`.

- Primary metric: `avg_r2`
- Component metrics: `r2_ut`, `r2_bt`, `r2_ps`
- Constraint metric: `alpha_star`
- Paper target: GI-CANN should learn a positive exponent near `1.675` and outperform fixed-`alpha=2` SI-CANN

### `brain_generalization`

Train on brain cortex `ut` + `uc`, test on held-out `ss`.

- Primary metric: `avg_r2`
- Component metrics: `r2_ut`, `r2_uc`, `r2_ss`
- Constraint metric: `alpha_star`
- Paper target: GI-CANN should learn a strongly negative exponent near `-18.016` and outperform fixed-`alpha=-2` SI-CANN

## Constraint / Interpretation Targets

### `rubber_negative_ablation`

- Expect low `avg_r2` when the model is restricted to a negative exponent branch
- This tests the paper’s claim that rubber is stretch-dominated and prefers positive generalized invariants

### `brain_positive_ablation`

- Expect low `avg_r2` when the model is restricted to a positive exponent branch
- This tests the paper’s claim that brain tissue is better represented by negative-exponent, `I2`-like behavior

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
