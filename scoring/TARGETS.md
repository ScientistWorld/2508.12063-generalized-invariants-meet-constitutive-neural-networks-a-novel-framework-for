# Optimization Targets

This document defines what future agents should optimize to improve upon the GI-CANN reproduction.

## Primary Goals

### 1. Rubber Material (VHB 4910) — Solved
**Target**: Reproduce or improve upon GI-CANN's R²≈0.998 performance with discovered α*≈1.675.

**Current Status**: Achieved R²=0.999 with α*≈1.677. ✓

**Evaluation**: Run `python eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json` and verify `rubber_generalization` experiment shows high avg_r2 for GI-CANN.

### 2. Brain Tissue (Brain Cortex) — Open Problem
**Target**: Achieve R²≈0.966 with α*≈-18 as reported in the paper.

**Current Status**: R²=0.14-0.62 — NOT REPRODUCED.

**Root Cause**: The α⁻ stress formula (σ ∝ (l3^α - l1^α)) appears incompatible with brain UT/UC data. For uniaxial compression with α=-18, the term l3^α ≫ l1^α produces huge positive stress.

**Challenge**: The paper reports excellent results for brain, but our faithful reproduction fails. Possible directions:
- Investigate if the paper used a different stress formulation for brain
- Verify the brain data matches the paper's source
- Explore alternative invariant definitions or stress derivations
- Check for data preprocessing steps not described in the paper

## Secondary Goals

### Ablation Studies
- **Negative exponent for rubber**: Should show poor performance (R²<0.7) — confirms stretch-dominated materials prefer positive α
- **Positive exponent for brain**: Should show poor performance (R²<0.6) — confirms shear-sensitive materials prefer negative α

## Baselines

**SI-CANN** (fixed α=±2) serves as the baseline. Any proposed method should:
1. Exceed SI-CANN performance on the primary material (rubber: R²>0.99)
2. Demonstrate learnable α values differ from the fixed ±2
3. Maintain similar parameter count (not just a larger model)

## Evaluation Protocol

Run `bash scripts/evaluate.sh` to:
1. Load all trained checkpoints from `checkpoints/`
2. Evaluate on test data (pure shear for rubber, simple shear for brain)
3. Compute R² scores for train and test modes
4. Write results to `scoring/scores.json`

The system compares `scoring/scores.json` against `scoring/reference.json` (paper's reported values).
