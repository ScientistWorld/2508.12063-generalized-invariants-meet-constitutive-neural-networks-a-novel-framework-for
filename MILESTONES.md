# Reproduction Milestones

**Current: majority**

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

### [2026-05-22 07:19] - core_claim
- Audited the current package and confirmed the core weakness is still the brain checkpoints, not the rubber GI-CANN reproduction
- Patched SI-CANN training to use the same observed stress component loss as GI-CANN and scoring
- Patched the main evaluator to recompute scores from checkpoint weights instead of trusting checkpoint-embedded metrics
- Prepared a targeted CPU job to retrain all brain GI-CANN/SI-CANN checkpoints and regenerate `scores.json`, `scores_train.json`, and `scores_test.json`

### [2026-05-22 07:22] - core_claim
- Final pre-submit audit confirmed `scripts/run.sh` and `action.yaml` are aligned on the brain retraining job
- Fixed executable bits on the main shell entrypoints so the scheduler can invoke them directly

### [2026-05-22 07:50] - core_claim_plus
- CPU retraining job completed successfully and regenerated full/train/test score artifacts
- Brain GI-CANN now matches the paper-level core behavior: negative-mode avg R²=0.9590 with α*=-17.9399 versus paper avg R²=0.966 and α*=-18.016
- Brain combined GI-CANN also reproduces the secondary brain result with avg R²=0.9604
- Prepared a follow-up CPU job for the remaining weak rubber ablation checkpoints

### [2026-05-22 07:58] - core_claim_plus
- Rubber ablation job failed before evaluation because the trainer's warm-start checkpoint loader lacked the NumPy pickle compatibility alias already used in evaluation
- Partial rubber negative training reached avg R²=0.3486, better than the paper's low-ablation reference of 0.264, but no milestone increase is claimed because the job did not finish
- Patched trainer checkpoint loading and resubmitted the rubber ablation job

### [2026-05-22 07:59] - majority
- Rubber ablation retraining completed after one scheduler preemption/restart and regenerated `scores.json`, `scores_train.json`, and `scores_test.json`
- Rubber combined GI-CANN now reaches avg R²=0.9994 versus the paper's 0.999, while rubber negative GI-CANN remains a low-performing contrastive ablation at avg R²=0.3486 versus the paper's 0.264
- Full validation passes with both core GI-CANN claims reproduced: rubber avg R²=0.9988 with α*=1.6769, and brain avg R²=0.9590 with α*=-17.9399
- Stopped at `majority` rather than `near_complete` because SI-CANN baselines and some per-mode ablation details are still uneven relative to the paper, although the proposed-method claims and main constraints are covered

## Stop Justification
- Finalized at `majority`. The reusable gym is complete, validation passes, both primary GI-CANN claims are reproduced, and the remaining discrepancies are calibration gaps in secondary SI-CANN baselines rather than blockers for the core reproduction.
