# Reproduction Milestones

**Current: majority**

## Progress Log

### [2026-05-22 08:45] - majority
- Re-verified that `method/` implements GI-CANN with trainable generalized invariant exponent(s), Eq. 13 derivative branches, paper loading-mode stresses, and L1 sparsity.
- `scoring/scores.json` reproduces the rubber and brain GI-CANN core claims and the main positive/negative/combined ablation grid.
- Added a model-agnostic prediction-artifact scoring path so future methods can be evaluated without matching GI-CANN checkpoint internals.

### [2026-05-22 08:50] - majority
- Audited the latest job records and confirmed the artifact-export validation job has not yet run in the container.
- Keeping the milestone at `majority` and submitting a CPU test job to export neutral prediction artifacts, regenerate full/train/test scores from those artifacts, and run `validate.py --compare`.

### [2026-05-22 08:55] - majority
- Reviewed completed job `47926f28-2ce`: artifact export produced 12 prediction JSON files, artifact-based full/train/test evaluation completed, and `validate.py --compare` passed.
- Final verified score files reproduce both paper core claims and the main ablation grid through the reusable artifact evaluator.

### [2026-05-22 09:10] - majority
- Reviewed completed split-evaluation job `6487ae0e-790`: the full benchmark and both train/test split score files validate successfully.
- Confirmed the actual method supports the core claim in `scoring/scores.json`: GI-CANN improves over SI-CANN on rubber generalization (`avg_r2` 0.9988 vs 0.9847, learned `alpha_star` 1.6769) and brain generalization (`avg_r2` 0.9590 vs 0.6974, learned `alpha_star` -17.9399).

## Stop Justification
- Completed at milestone `majority`.
