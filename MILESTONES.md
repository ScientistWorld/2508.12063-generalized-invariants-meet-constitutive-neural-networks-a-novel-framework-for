# Reproduction Milestones

**Current: core_claim**

## Progress Log

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

## Stop Justification
