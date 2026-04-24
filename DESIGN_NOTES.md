# Design Notes — Train/Test Evaluation Split

## Split Style

**Alternating samples (~50/50):** Each deformation curve (UT/BT/PS for rubber, UT/UC/SS for brain) is split into alternating nonzero samples. The neutral zero-stress point (λ=1, stress=0) is excluded from both slices — it dominates R² calculations and would make the split trivially gameable.

**Train slice** (`eval/train/`): even-offset alternating samples (indices 1, 3, 5, ... of the data arrays after dropping the λ=1 row).

**Test slice** (`eval/test/`): odd-offset alternating samples (indices 2, 4, 6, ...).

This preserves per-mode sample counts (13-14 per mode per slice for rubber, 7 per mode per slice for brain) — sufficient for R² to discriminate genuine vs. shortcut methods. The split is deterministic and reproducible.

## Why Mode-Level R² > avg_r2 for Generalization Experiments

The two primary experiments (`rubber_generalization`, `brain_generalization`) measure cross-mode generalization: training on UT+BT (rubber) or UT+UC (brain), testing on PS (rubber) or SS (brain). The benchmark's signal is the generalization gap — whether a method truly captures the material's constitutive law or just interpolates the training curves.

Promoting the held-out mode R² (`r2_ps`, `r2_ss`) to coefficient 1.5 while demoting the train-mode R²s (`r2_ut`, `r2_bt`, `r2_uc`) to 0 does two things:

1. **Makes the shortcut signal louder.** A method that overfits to training modes will score well on avg_r2 (which averages training and test R²s together), but will regress on the held-out mode. The held-out mode's standalone R² is now the dominant score contributor.

2. **Aligns the metric with the scientific claim.** The paper's core claim for both materials is about generalization to held-out modes. Scoring that claim through the held-out R² directly measures what the paper actually argues.

## Why Test Slice Covers All Modes (Not Just the Held-Out Mode)

Both slices evaluate all three modes per material, not just the held-out generalization mode. This is intentional:

- **Train/test scores are comparable.** If only the test slice evaluated the held-out mode and the train slice evaluated training modes, a method could have different sample counts and evaluation protocols between slices. That difficulty mismatch would make the train/test gap uninterpretable. Equal protocol = interpretable gap.

- **Ablation experiments need all modes.** `rubber_negative_ablation`, `brain_positive_ablation`, and both `*_combined_ablation` experiments report avg_r2 as their sole primary metric. These experiments must evaluate all three modes on both slices for the ablation signal to be meaningful.

- **The gap signal is still there.** For generalization experiments, the train slice's held-out-mode R² (e.g., r2_ps on eval/train) is a "visible proxy" for the test slice's held-out-mode R². A future agent can observe their PS performance on eval/train and anticipate what eval/test will measure. This creates incentive to genuinely improve the constitutive model rather than gaming train-specific patterns.

## Self-Contained Checkpoint Interface

Both slices load checkpoints from `checkpoints/` and reconstruct predictions via `MinimalGICANN`/`MinimalSICANN` from `eval/models.py`. The evaluation is purely forward-pass — no re-training, no gradient computation. This means:

- Future agents can ship a new checkpoint format as long as it provides the required tensor parameters.
- `eval/test/` re-runs forward passes on existing checkpoints, which is fast (<1s per checkpoint on CPU).
- No coupling to `method/` internals beyond the checkpoint file itself.

## Leakage Audit

- `eval/train/` and `eval/test/` import independently from `data.load_data` and `eval.models`.
- No cross-imports between the two eval folders.
- `eval.models` is a neutral package (no method-specific code); it lives at `eval/models.py`, which is outside both `train/` and `test/`.
- The `split_rows` function differs between train and test — train uses `[::2]`, test uses `[1::2]` — so a future agent reading `eval/train/evaluate.py` cannot reverse-engineer the test slice's samples.

## Runtime Notes (for runner iteration)

- `python3 eval/train/evaluate.py` and `python3 eval/test/evaluate.py` both complete in <30s on CPU.
- `scores_train.json` and `scores_test.json` must NOT be pre-populated before the runner iteration.
- Both scripts accept `--checkpoint` for single-checkpoint evaluation and `--experiment` for filtering experiments.
- Torch 2.4.0+cpu is installed in the environment; higher versions have a maskedtensor import bug that breaks evaluation.
