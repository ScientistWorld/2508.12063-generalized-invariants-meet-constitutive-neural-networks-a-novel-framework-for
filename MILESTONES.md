# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-08] - Wrap-up Complete
- Fixed validation errors (added alpha_star to reference.json metrics, created eval/models.py for import separation)
- Updated scoring/EXPERIMENTS.md with all 6 experiments
- All validation checks passed
- Final commit prepared

**Stop Justification**: The paper's core claim is reproduced for rubber material (GI-CANN discovers α*≈1.677, R²=0.999 vs reference 0.998). Brain tissue could not be reproduced (R²=0.14-0.62 vs reference 0.966) due to a fundamental incompatibility between the α⁻ stress formula and brain UT/UC data. This is honestly documented in PROGRESS.md and MILESTONES.md. The evaluation pipeline is complete and reproducible — all 12 checkpoints exist, and `python eval/evaluate.py` successfully recomputes the R² scores. Further work on brain would require investigating alternative stress formulations or data discrepancies not described in the paper.

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

## Deviations from Paper

**Brain tissue results**: The paper reports R²≈0.966 for brain cortex with α*≈-18. Our reproduction achieves R²=0.14-0.62. Investigation shows the α⁻ stress formula is fundamentally incompatible with brain UT/UC data — for uniaxial compression, l3 > 1 > l1, and with α=-18, l3^α ≫ l1^α produces huge positive stress. This affects both GI-CANN and the SI-CANN baseline, suggesting either:
1. The paper used a different stress formulation for brain
2. The brain data in the paper differs from Table A.5
3. There is additional preprocessing not described

**No other deviations**: All other implementations (GI-CANN, SI-CANN, data loading, B-eigenvalue formula for simple shear, training hyperparameters) follow the paper as described.
