# Optimization Targets

Note: after the April 8 audit, `scoring/scores.json` was regenerated through the patched evaluator. Treat the current `scores.json` file as authoritative for audited numbers; some narrative text below still reflects pre-audit checkpoint summaries and should be refreshed after the next retraining run.

This document defines what future agents should optimize to improve upon the GI-CANN reproduction.

## Primary Targets

### rubber_generalization — Primary

**What it tests**: Whether a method trained on uniaxial tension and equibiaxial tension can generalize to pure shear deformation for rubber material. This is the paper's core claim for rubber.

**Primary metric**: `avg_r2` — average R² across UT, BT, PS deformation modes. Higher is better.

**Direction**: Maximize avg_r2 (target > 0.99). The paper reports 0.998 for GI-CANN and 0.982 for SI-CANN baseline.

**How to evaluate**:
```bash
python eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json
# Check rubber_generalization.GI-CANN.avg_r2
```

**Reference in scores.json**:
- `rubber_generalization.GI-CANN.avg_r2`: 0.9993 (our result) vs 0.998 (paper)
- `rubber_generalization.SI-CANN.avg_r2`: 0.9906 (our result) vs 0.982 (paper)

### brain_generalization — Primary

**What it tests**: Whether a method trained on uniaxial tension and compression can generalize to simple shear deformation for brain cortex tissue. This is the paper's core claim for brain.

**Primary metric**: `avg_r2` — average R² across UT, UC, SS deformation modes. Higher is better.

**Direction**: Maximize avg_r2 (target > 0.96). The paper reports 0.966 for GI-CANN and 0.610 for SI-CANN baseline.

**Current status**: NOT REPRODUCED. Our results: GI-CANN = 0.141, SI-CANN = -0.506. Root cause: α⁻ stress formula incompatible with brain UT/UC data.

**How to evaluate**:
```bash
python eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json
# Check brain_generalization.GI-CANN.avg_r2
```

**Reference in scores.json**:
- `brain_generalization.GI-CANN.avg_r2`: 0.1411 (our result) vs 0.966 (paper) — GAP
- `brain_generalization.SI-CANN.avg_r2`: -0.5063 (our result) vs 0.610 (paper) — GAP

## Constraint Targets

### brain_positive_ablation — Constraint

**What it tests**: Verifies that positive exponents (I₁-like invariants) perform poorly for brain tissue. This is a constraint — a method that claims brain uses positive α should perform poorly on this test.

**Primary metric**: `avg_r2` — should be low (< 0.6) for brain with positive α.

**Direction**: This is an ablation that should FAIL (low R² is expected). The paper reports R²≈0.53 for both GI-CANN and SI-CANN with positive α.

**How to evaluate**:
```bash
python eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json
# Check brain_positive_ablation.GI-CANN.avg_r2
```

**Reference in scores.json**:
- `brain_positive_ablation.GI-CANN.avg_r2`: -0.2121 (our result) vs 0.526 (paper) — negative indicates poor fit
- `brain_positive_ablation.SI-CANN.avg_r2`: -0.1092 (our result) vs 0.527 (paper) — negative indicates poor fit

### rubber_negative_ablation — Constraint

**What it tests**: Verifies that negative exponents (I₂-like invariants) perform poorly for rubber. This is a constraint — a method that claims rubber uses negative α should perform poorly on this test.

**Primary metric**: `avg_r2` — should be low (< 0.7) for rubber with negative α.

**Direction**: This is an ablation that should FAIL (low R² is expected). The paper reports R²≈0.26-0.32 for negative α on rubber.

**How to evaluate**:
```bash
python eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json
# Check rubber_negative_ablation.GI-CANN.avg_r2
```

**Reference in scores.json**:
- `rubber_negative_ablation.GI-CANN.avg_r2`: 0.6475 (our result) vs 0.264 (paper) — still low but better than paper
- `rubber_negative_ablation.SI-CANN.avg_r2`: 0.5727 (our result) vs 0.318 (paper) — still low as expected

## Ablation Targets

### rubber_combined_ablation — Ablation

**What it tests**: Whether using both positive and negative exponents improves over a single positive exponent for rubber. This is an ablation showing whether multi-invariant formulation adds value.

**Primary metric**: `avg_r2` — should be comparable or better than single positive α.

**Direction**: Maximize avg_r2. Paper reports 0.999 for combined vs 0.998 for single positive — marginal gain.

**How to evaluate**:
```bash
python eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json
# Compare rubber_combined_ablation.GI-CANN.avg_r2 vs rubber_generalization.GI-CANN.avg_r2
```

**Reference in scores.json**:
- `rubber_combined_ablation.GI-CANN.avg_r2`: 0.9980 (our result) vs 0.999 (paper) — matches paper

### brain_combined_ablation — Ablation

**What it tests**: Whether using both positive and negative exponents improves over a single negative exponent for brain. This is an ablation showing whether multi-invariant formulation adds value.

**Primary metric**: `avg_r2` — should be comparable or better than single negative α.

**Direction**: Maximize avg_r2. Paper reports 0.966 for combined vs 0.966 for single negative — no significant difference.

**Current status**: NOT REPRODUCED. Our result: 0.62 vs paper 0.966.

**How to evaluate**:
```bash
python eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json
# Compare brain_combined_ablation.GI-CANN.avg_r2 vs brain_generalization.GI-CANN.avg_r2
```

**Reference in scores.json**:
- `brain_combined_ablation.GI-CANN.avg_r2`: 0.6206 (our result) vs 0.966 (paper) — GAP

## Success Criteria

To claim `core_claim_plus` or higher, a method must:
1. Achieve avg_r2 > 0.99 on rubber_generalization
2. Achieve avg_r2 > 0.95 on brain_generalization (currently not reproduced)
3. Maintain constraint ablations (brain_positive_ablation < 0.6, rubber_negative_ablation < 0.7)
