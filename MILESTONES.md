# Reproduction Milestones

**Current: core_claim**

## Audit Log

### [2026-04-08] - Audit Complete
**Auditor verified:**
- Implementation is real and complete (GI-CANN algorithm correctly implemented)
- Evaluation is honest (results computed from checkpoints, not hardcoded)
- All deliverables present (briefing/, scoring/, scripts/, environment/)
- Evaluation script re-ran successfully, reproducing scores.json values

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

## Deviations from Paper

**Brain tissue results**: The paper reports R²≈0.966 for brain cortex with α*≈-18. Our reproduction achieves R²=0.14-0.62. Investigation shows the α⁻ stress formula is fundamentally incompatible with brain UT/UC data — for uniaxial compression, l3 > 1 > l1, and with α=-18, l3^α ≫ l1^α produces huge positive stress. This affects both GI-CANN and the SI-CANN baseline, suggesting either:
1. The paper used a different stress formulation for brain
2. The brain data in the paper differs from Table A.5
3. There is additional preprocessing not described

**No other deviations**: All other implementations (GI-CANN, SI-CANN, data loading, B-eigenvalue formula for simple shear, training hyperparameters) follow the paper as described.
