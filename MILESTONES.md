# Reproduction Milestones

**Current: majority**

## Progress Log

### [2026-05-22 08:00] - majority
- Implemented the paper's GI-CANN method with trainable generalized-invariant exponents and the identity, exponential, and logarithmic strain-energy branches from Eq. (13).
- Implemented SI-CANN fixed-exponent baselines and paper-table rubber/brain data loaders.
- Completed compute jobs that regenerated checkpoints and scoring artifacts for rubber and brain core generalization plus positive/negative/combined ablations.
- Reproduced both central proposed-method claims: rubber GI-CANN `avg_r2=0.9988`, `alpha_star=1.6769`; brain GI-CANN `avg_r2=0.9590`, `alpha_star=-17.9399`.
- Reproduced most proposed-method ablations, including rubber combined `avg_r2=0.9994` and brain combined `avg_r2=0.9604`.

## Stop Justification
- Final milestone is `majority` rather than `near_complete` because fixed-exponent SI-CANN baseline calibration and some per-mode ablation values remain uneven, although both core GI-CANN claims and most proposed-method secondary checks are reproduced.
