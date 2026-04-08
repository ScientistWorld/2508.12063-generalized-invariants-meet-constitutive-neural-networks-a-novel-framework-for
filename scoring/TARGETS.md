# Optimization Targets

This document defines the scored targets for the experiments currently present in [`scoring/scores.json`](/home/user/scoring/scores.json).

## Primary Targets

### `rubber_generalization`

- What it tests: cross-mode generalization on rubber, training on `ut` + `bt` and evaluating transfer to held-out `ps`
- Primary metric: `avg_r2` (`higher_is_better`)
- Important held-out metric: `r2_ps` (`higher_is_better`)
- Report-only descriptor: `alpha_star`
- Evaluation: `bash scripts/evaluate.sh --experiment rubber_generalization`

### `brain_generalization`

- What it tests: cross-mode generalization on brain cortex, training on `ut` + `uc` and evaluating transfer to held-out `ss`
- Primary metric: `avg_r2` (`higher_is_better`)
- Important held-out metric: `r2_ss` (`higher_is_better`)
- Report-only descriptor: `alpha_star`
- Evaluation: `bash scripts/evaluate.sh --experiment brain_generalization`

## Constraint Targets

### `rubber_negative_ablation`

- What it tests: whether the rubber benchmark still penalizes a materially mismatched constitutive description
- Primary metric: `avg_r2` (`higher_is_better`, but the benchmark should preserve a visible gap versus the main rubber setting)
- Evaluation: `bash scripts/evaluate.sh --experiment rubber_negative_ablation`

### `brain_positive_ablation`

- What it tests: whether the brain benchmark still penalizes a materially mismatched constitutive description
- Primary metric: `avg_r2` (`higher_is_better`, but the benchmark should preserve a visible gap versus the main brain setting)
- Evaluation: `bash scripts/evaluate.sh --experiment brain_positive_ablation`

## Ablation Targets

### `rubber_combined_ablation`

- What it tests: whether a richer rubber constitutive description changes performance relative to the main rubber benchmark
- Primary metric: `avg_r2` (`higher_is_better`)
- Evaluation: `bash scripts/evaluate.sh --experiment rubber_combined_ablation`

### `brain_combined_ablation`

- What it tests: whether a richer brain constitutive description changes performance relative to the main brain benchmark
- Primary metric: `avg_r2` (`higher_is_better`)
- Evaluation: `bash scripts/evaluate.sh --experiment brain_combined_ablation`

## Run All Evaluation

```bash
bash scripts/evaluate.sh
```
