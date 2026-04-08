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

### [2026-04-08 04:05] - core_claim
- Verified the paper's public code release exists at `LivingMatterLab/CANN`
- Realigned all packaged shell entrypoints with the current robust trainer so future submissions and `scripts/reproduce.sh` use the same training path as `scripts/run.sh`
- Reset the workspace from an incorrect permanent `done` state to a fresh submission path for checkpoint regeneration

## Stop Justification
- Final state: `core_claim`.
- The workspace is being stopped here because the reproduced rubber benchmark supports the paper's central claim and the environment has been packaged, validated, and documented for future scientist agents.
- The main unfinished item is stronger reproduction on the brain benchmarks. The current audited brain scores remain materially below the paper's reported values, and closing that gap would require additional retraining and iteration rather than more packaging work.
- The workspace is still a valid gym because the method implementation is real, `eval/` is method-independent, the scoring files are complete, and at least one primary generalization experiment plus its contrastive ablations are reproduced.
