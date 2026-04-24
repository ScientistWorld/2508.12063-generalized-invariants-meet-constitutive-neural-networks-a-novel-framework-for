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

### [2026-04-08 05:05] - core_claim
- Fixed the brain constitutive path so compression uses the correct loading mode and diagonal stress evaluation preserves the physical principal-stretch ordering
- Verified locally that the paper's reported one-term brain GI-CANN parameters now achieve avg R²≈0.9608, close to the paper's 0.966
- Prepared a targeted submission to retrain only the brain GI-CANN negative and combined checkpoints under the corrected formulas

### [2026-04-08 05:35] - core_claim
- Final gym verification confirmed that `data/` and `eval/` remain method-independent and that `python3 validate.py --compare` passes
- Added `--experiment` support to the evaluator so the documented `scripts/evaluate.sh` interface matches the actual CLI
- Tightened the gym-facing direction docs to stay method-safe while keeping the benchmark scope clear

### [2026-04-08 06:15] - core_claim
- Regenerated the full `scoring/scores.json` after the filtered evaluation smoke test so all six benchmark experiments are present in the final package
- Finalized `PROGRESS.md`, `VERIFICATION.md`, and the scoring guidance files for gym handoff
- Stopped at `core_claim` because rubber is credibly reproduced and the workspace is clean, but the brain checkpoints still lag the paper and need another training iteration rather than more packaging work

### [2026-04-24 00:00] - core_claim
- Audited the first real train/test split run and found checkpoint loading failed before any scores were produced because the container NumPy exposes `numpy.core` while the frozen checkpoints unpickle `numpy._core`
- Patched all evaluation entry points to install backward-compatible NumPy module aliases before `torch.load`, preserving the frozen method artifacts while unblocking benchmark scoring
- Kept the milestone at `core_claim`; this turn advances evaluation robustness but does not change the reproduced method quality

### [2026-04-24 00:15] - core_claim
- Completed the real train/test split evaluation and produced validated `scoring/scores_train.json` and `scoring/scores_test.json` with matching experiment/method coverage
- Verified that the split is not a copy-paste bug: every experiment differs numerically across slices, although the strongest rubber benchmark has only a very small gap because the alternating split stays on the same smooth curve family
- Closed the design phase at `core_claim` because the evaluation artifact is now complete and usable, with the main caveat that the shipped brain checkpoints remain weak on both slices

## Stop Justification
- Completed at milestone `core_claim`.
