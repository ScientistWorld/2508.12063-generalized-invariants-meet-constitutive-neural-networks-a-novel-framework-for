# Verification Report

## Initial Assessment

The workspace started at a credible `core_claim`. The method implementation is a real GI-CANN/SI-CANN reproduction rather than a surrogate regressor, the folder separation between `data/`, `eval/`, and `method/` is intact, and the reproduced rubber result still supports the claimed milestone. The remaining verification work was mostly about gym usability: checking for method leakage in benchmark-facing docs, confirming the scoring rubric is sensible for the experiments actually present in `scores.json`, and making sure the documented evaluation interface matches the actual CLI.

## Issues Found

- Minor: [`scoring/DIRECTION.md`](/home/user/scoring/DIRECTION.md) still used paper-specific invariant-exponent language more directly than necessary for a gym-facing “direction” document.
- Minor: [`scoring/TARGETS.md`](/home/user/scoring/TARGETS.md) documented `bash scripts/evaluate.sh --experiment ...`, but [`eval/evaluate.py`](/home/user/eval/evaluate.py) did not actually support `--experiment`.
- Minor: [`scripts/evaluate.sh`](/home/user/scripts/evaluate.sh) advertised a configurable Python interpreter but still hardcoded `python`, which fails on this machine.
- Minor: [`VERIFICATION.md`](/home/user/VERIFICATION.md) itself was stale and described older fixed issues rather than the current verification state.

## Fixes Applied

- Updated [`eval/evaluate.py`](/home/user/eval/evaluate.py) and [`scripts/evaluate.sh`](/home/user/scripts/evaluate.sh) so the documented `--experiment` interface now works for selected experiment evaluation.
- Fixed [`scripts/evaluate.sh`](/home/user/scripts/evaluate.sh) to honor `${PYTHON}` instead of calling a missing `python` binary directly.
- Reworded [`scoring/DIRECTION.md`](/home/user/scoring/DIRECTION.md) to keep the benchmark scope physics-aware and interpretable without overexposing the paper’s specific invariant-exponent framing.
- Replaced this report with the current verification findings and judgment.

## Final Judgment

The workspace is suitable as a research gym. A future scientist can use the problem/evaluation/data side without importing `method/`, and the benchmark-facing markdown is sufficiently method-safe for from-scratch use. The scoring rubric is also reasonable for the experiments actually present in `scores.json`: both core generalization experiments dominate at 40% each, the held-out mode is emphasized inside each core experiment, and the four ablations collectively remain secondary at 20%.

The train/test split is now also exercised against the frozen final checkpoints rather than only designed on paper. The run produced complete `scores_train.json` and `scores_test.json` files with matching experiment/method coverage and non-identical values across slices. The main caveat is calibration rather than correctness: the alternating split yields a very small train/test gap on `rubber_generalization` (`0.9986` vs `0.9989` for GI-CANN), so the anti-shortcut signal is present but mild on the strongest rubber result. By contrast, the weak brain and ablation checkpoints remain poor on both slices, which gives future improve-mode agents room to improve but also reflects that the shipped checkpoint set is uneven across claims.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: high
- Key limitation: brain-side reproduction remains below the paper’s reported numbers, and the visible/held-out gap is especially small on the strongest rubber benchmark because the split uses alternating samples from the same smooth curves
