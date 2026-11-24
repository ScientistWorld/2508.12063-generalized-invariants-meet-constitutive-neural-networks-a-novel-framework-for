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

### [2026-05-22 09:20] - majority
- Moved the active trainer from the repository root to `method/training_runner.py` and updated `scripts/method.sh`, `scripts/baseline.sh`, `scripts/reproduce.sh`, and `method/export_predictions.py`.
- Re-ran syntax checks, artifact-based `scripts/evaluate.sh`, and `python3 validate.py --compare`; all checks passed with the same `majority` results.

### [2026-05-22 09:35] - majority
- Completed final verification of folder separation, method-agnostic documentation, scoring coefficients, artifact-based evaluation, and train/test split integrity.
- Re-ran `python3 validate.py --compare`; validation passed and the `majority` milestone remains justified.

### [2026-05-22 08:46] - majority
- Continuation audit re-read the paper's core GI-CANN algorithm and checked the packaged code against the actual method: trainable generalized invariant exponents, Eq. 13 strain-energy branches, loading-mode stress formulas, and L1 sparsity are present.
- Re-ran `python3 validate.py --compare`; validation passed using the existing artifact-based score files and the `majority` milestone remains justified.

### [2026-05-22 08:49] - majority
- Final verifier reviewed folder separation, method-agnostic docs, scoring coefficients, split evaluators, download portability, and reproduced numbers.
- Re-ran artifact-based full/train/test evaluation and `python3 validate.py --compare`; all checks passed and the `majority` milestone remains the final judgment.

### [2026-11-24 01:30] - majority (iteration 11, hardening)
- Gym rebuilt around declared strain-energy laws scored by `eval/` (no method-written numbers), the authors' exact data files, and a hidden test on Treloar rubber at 50 C and the corona radiata / basal ganglia (held-out curves only in `eval/test/`).
- Reference GI-CANN retrained from the final code following the authors' notebooks (plus convergence and best-of-20 by training objective): paper materials reproduce Tables 3/4 (rubber 0.9987 vs 0.998, cortex 0.9669 vs 0.966; alpha* 1.676 and -18.11 vs 1.675 and -18.016); GI beats SI on the hidden test (rubber 0.9939 vs 0.9751, brain 0.9544 vs 0.5429).
- All red-team shortcuts closed (fail or no effect); a dev-tuned L1 variant beats the reference by +0.22 normalized against a reference-procedure noise of +0.001; brute force and PS-CANN do not outrank. Details in `HARDENING.md`.

## Stop Justification
- Completed at milestone `majority`.
