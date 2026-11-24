# Hardening (iteration 11)

`REDTEAM.md` (iteration 10) is the starting point; there was no `REVIEW.md`. Every number below
comes from job `d6917da9-15f`. In that job the reference was retrained from the final code, then
every run in `redteam/hardening/run_checks.sh` went through the hidden test. Each run gets a
sandbox, the gym's `eval/` and `scripts/evaluate_test.sh` are put back, and
`bash scripts/evaluate_test.sh` runs with no arguments. Per-run outputs are in
`redteam/hardening/results/`, and the full table is in `redteam/hardening/SUMMARY.md`.

Two kinds of delta are reported, both against the stored reference `scoring/scores_test.json`.

- **Raw Δ**: Σ over experiments of weight × Σ over metrics of |coef| × the improvement in each
  metric's better direction.
- **Norm Δ**: the same sum, with each metric's difference divided by its standard deviation across
  the honest runs. This stands in for the rank's division by the spread across ranked attempts.
  Raw Δ mixes units (one term in `n_terms` against 0.01 of R²), so Norm Δ is the one to read.

## 1. What changed

1. **Laws are scored, not numbers.** A method writes one law file per method,
   `outputs/laws/<name>.json`. The file holds sums of `id`, `exp` and `ln` terms of generalized
   invariants J_α for every material and exponent family.
   - `eval/train/evaluate.py` and `eval/test/evaluate.py` compute every stress themselves, in
     closed form for ut, uc, bt, ps and ss. They are standard-library-only and share no code.
   - They never read a number the method wrote: R², relative error, term count and α* are all
     recomputed from the law.
   - A law is invalid if it has a negative or non-finite weight, |α| > 60, an exponent of the
     wrong sign for its family, an undefined ln term, or a non-finite stress. It is also invalid
     with zero active terms or more than 12.
   - A missing or invalid law gives the worst value in every metric of that experiment
     (R² 0, relative error 1, n_terms 12).
   - A missing or unreadable `GI-CANN.json` gives worst values in every experiment.
   - The old score file is deleted before scoring. Each file is reported under its own name.
   - The SI-CANN baseline and the ground truth are frozen inside `eval/train` and `eval/test`.
2. **The paper's own data.** The gym used the paper's Table A.5, which is rounded. Stretches are
   given to two decimals, and some stresses to one: the first pure-shear point, 0.066 MPa, is
   printed as 0.10. The gym now reads the authors' spreadsheets, the files their notebooks load:
   `LivingMatterLab/CANN@d3caca0f`, `GICANN/input/*.xlsx`, fetched with sha256 checks by
   `scripts/download.sh`. With the exact data the reproduction matches the paper (§6).
3. **The held-out answers are out of reach.**
   - The same files hold Treloar rubber at 50 °C and three more Budday brain regions, all tested
     in the paper's loading modes.
   - The development side is the paper's two materials plus the corpus callosum. Every curve
     there is visible, and it feeds `scores.json` and `scores_train.json`.
   - The hidden test is Treloar 50 °C, corona radiata and basal ganglia. Their held-out curves (ps
     and ss) exist only in `eval/test/ground_truth.json`; `data/` holds just their stretch values.
   - The test's held-out curves are neither in the paper nor in the workspace. The old test's
     answers were printed in the paper.
4. **The reference follows the authors' notebooks.**
   - The model is the two-layer CANN with non-negative weights, and α is a non-negative variable
     with a fixed sign per branch.
   - The α starts are the notebooks' values (rubber 2 or 1.7/0.1; brain 5 or 2/2), not the old
     hand-set −18.
   - Training uses the loss Σ per-mode MSE + 0.01 Σ|w|, Adam at lr 1e-3, batch 12, 30k/40k
     epochs, early stopping with patience 3000 on the training loss, and the best epoch restored.
     There is no held-out checkpoint selection any more.
   - Two additions make the reference land on the law the paper reports instead of a random stop:
     1. A full-batch convergence phase on the same objective.
     2. The best of 20 random starts, chosen only by the training objective.
   - The reason for the additions: on cortex, 17 of 20 single runs stop in a local optimum near
     α = −15. The objective is also flat along an (α, w) valley, and where early stopping halts
     along it moved corona-radiata shear R² between 0.82 and 0.88.
   - Two numerical guards were added: the ln argument stays ≥ 1e-3, and w21·(J−3) ≤ 10 at the
     most deformed visible input. The notebooks' float32 code returns NaN in both regimes.
5. **Metrics and weights.**
   - The metrics are the paper's quantities:
     - per-mode R² clipped at 0, as the paper's tables report it;
     - `avg_r2`;
     - mean relative error on the held-out mode (the paper's Figs. 4–5);
     - `n_terms`, the compactness constraint (the paper claims accuracy *without* giving up an
       interpretable law, and names PS-CANN's density as its weakness).
   - Removed:
     - `shape_violation_rate`, which was 0 for everything;
     - normalized RMSE and tail MAE, which the paper does not measure.
   - New weights: main experiments 0.30 each, combined family 0.15 each, mismatched family 0.05
     each. The combined family is the paper's general GI-CANN (S = ℝ, claim 4) and deserves more
     than a sanity ablation.

## 2. Do better results score higher? (variants, before → after)

| variant | before (raw Δ / norm) | after: raw Δ | after: norm Δ |
|---|---|---|---|
| noise: reference rerun with seeds 20–39 | rerun −0.0003 (norm about −5.4 to −5.9) | +0.0000 | **+0.001** |
| **better: L1 strength chosen on the development materials** (r = 0.02 for brain, from cortex + corpus-callosum shear) | – | **+0.0128** | **+0.221** |
| two generalized invariants per sign | (old "stronger combined" +0.0015 / +5.97) | −0.0272 | +0.027 |
| mini-batch phase only, no convergence | – | +0.0029 | −0.021 |
| scale-free (variance-weighted) loss | – | −0.0177 | −0.311 |
| single paper runs, seeds 0–4 (no multi-start) | – | −0.018 to −0.062 | −0.18 to −0.78 |
| PS-CANN (dense exponent enumeration) | +0.0185 / +20.35 | −0.4682 | −3.831 |
| brute-force α grid (exact weights) | −0.0117 / −18.97 | +0.0060 | −1.300 |
| weaker: α fixed at ±2 (SI-CANN laws in the GI slot) | −0.2174 / −418 | −0.3568 | −3.352 |
| weaker: 300 epochs, no convergence | −0.7599 (brain unchanged) | −0.3038 | −3.768 |

- **Noise is gone.** Two independent 20-start references give identical laws: Norm Δ +0.001.
  Determinism is shown in `redteam/hardening/logs/determinism.log`: four runs repeated alone are
  bit-identical to the parallel reference runs, and there is no wall-clock limit anywhere in the
  method.
- **The better variant beats the reference by far more than the noise.**
  - Brain generalization: avg R² 0.9544 → 0.9572, shear R² 0.9037 → 0.9114, relative error
    0.201 → 0.171.
  - It was found from the development side alone, a hyperparameter of the paper's own loss.
- **Weaker variants lose clearly.**
  - Short training now hurts brain as well: avg R² 0.954 → 0.599. Before, brain was insensitive,
    because α started at the answer.
  - Fixed α loses everywhere the paper says it should.
- **PS-CANN no longer "wins".** In the old gym it put a +30 exponent into the negative slot, which
  the sign rule now forbids. Dense laws also pay in `n_terms`.
- The variants that add terms or change the loss land close to the reference. That is honest
  headroom, not a loophole.

## 3. Major shortcuts: fix and score (before → after, raw Δ / Norm Δ)

| shortcut | before | fix | after |
|---|---|---|---|
| copy the measured stresses as predictions | +0.1167 | stresses are never accepted, only laws | −4.82 / −175 (unusable file) |
| stresses inside the law object | – | schema requires terms | −4.82 / −175 |
| stale `scores_test.json` left behind | +0.1167 | evaluator deletes it first; missing GI-CANN gets worst values | −4.82 / −175 |
| edit the ground truth in `data/` | +0.0433 | eval reads its own frozen `ground_truth.json` | 0.0000 (no effect) |
| main law relabelled into the mismatched-family slots | +0.0590 | family sign rule; violators fail the experiment | −0.379 / −6.14 |
| drop bad modes / missing slots | +0.0088 | eval computes every mode itself; a missing law gets worst values | −0.379 / −6.14 |
| NaN outputs, missing method, crash, leftover checkpoints | 0 (tie) | worst values written; no checkpoint fallback | −4.82 / −175 |
| empty predictions (error metrics at their best) | best error | worst values | −4.82 / −175 |
| second method file overwrites the GI-CANN slot | reporting bug | every file gets its own entry | GI-CANN unaffected (0.0000) |
| agent's `SI-CANN.json` replaces the baseline | – | baseline fixed in `eval/`, outputs copy reported as `SI-CANN@outputs` | baseline unchanged |
| negative-weight, huge-exponent, zero-exponent, 40-term dense laws | – | validity rules | −4.82 / −175 each |
| train on the held-out mode | +0.0025 (brain +0.0031) | test materials' held-out curves are not in the workspace | not possible from inside the workspace |
| degenerate outputs (constant, copy input) | failed | stresses cannot be submitted at all | n/a |

## 4. Shortcuts left, and why

- **Fitting the test materials' held-out curves obtained from outside the workspace.** The
  authors' spreadsheets are public, and the paper cites the repository. No real-data test can
  hide public data from an agent with internet access; `download.sh` deletes the spreadsheets
  after extracting the visible curves, and `scoring/CONSTRAINTS.md` forbids it.
  - Measured with one-term laws fitted to all three modes (`left_fit_heldout_answers`): raw
    +0.059, Norm −0.09. It gains on brain (+0.43 norm) and loses elsewhere.
  - Richer laws fitted to the answers would gain more.
  - This is left to the grader: the code would have to fetch the files and fit ps/ss curves,
    which is easy to see.
- **The paper's own curves are public in the paper.** Table A.5 prints cortex ss and Treloar
  20 °C ps. Those are development materials now, so this is no longer a test leak.

## 5. Reference score, before → after

| | rubber avg / held-out R² | brain avg / held-out R² |
|---|---|---|
| before: old hidden test (half of the rows of the rounded Table A.5 curves, incl. training rows) | 0.9988 / 0.9994 | 0.9643 / 0.9683 |
| after: hidden test (Treloar 50 °C; corona radiata + basal ganglia) | 0.9939 / 0.9966, rel. err 0.111, 2 terms | 0.9544 / 0.9037, rel. err 0.201, 1 term |
| after: paper's materials (`scores.json`) | 0.9987 / 0.9990 | 0.9669 / 0.9727 |

The new values are not comparable to the old ones: they come from different materials, metrics
and data precision.

## 6. The reproduction shows the paper's claims

`scores.json` covers the paper's materials, GI-CANN vs SI-CANN, with the paper in brackets.

| experiment | GI-CANN (paper) | SI-CANN (paper) |
|---|---|---|
| rubber, positive family | 0.9987 (0.998) | 0.9816 (0.982) |
| rubber, negative family | 0.2611 (0.264) | 0.3169 (0.318) |
| rubber, combined family | 0.9997 (0.999) | 0.9989 (0.977) |
| brain, negative family | 0.9669 (0.966) | 0.6174 (0.610) |
| brain, positive family | 0.5310 (0.526) | 0.5359 (0.527) |
| brain, combined family | 0.9669 (0.966) | 0.6174 (0.617) |

The discovered laws match the paper's reported laws:

| material | ours | paper |
|---|---|---|
| rubber, 20 °C | α* = 1.676, w_id = 0.231, w_exp = (0.109, 0.282) | Eq. 19: 1.675, 0.227, (0.107, 0.297) |
| cortex | α* = −18.11, w = 0.0092 kPa, identity term only | Eq. 20: −18.016, 0.009 |

On the hidden test the paper's direction holds:

- Rubber 50 °C: GI 0.9939 vs SI 0.9751 avg R², and ps R² 0.9966 vs 0.9724.
- Brain: 0.9544 vs 0.5429 avg R², ss R² 0.9037 vs 0.8695, relative error 0.201 vs 0.431.
- The development side shows the same.

## 7. Headroom check

- The reference sits at a metric's bound only for `n_terms` in the brain experiments (1 term).
  That bound is the constraint the paper claims to keep.
- Every other metric has room:
  - brain shear R² 0.90 (corona radiata alone 0.849);
  - brain relative error 0.20;
  - rubber relative error 0.11;
  - rubber avg R² 0.994.
- The added scored experiment is the hidden test itself: new materials on which the paper's
  one-term law falls short. On the development side, corpus-callosum shear is also only 0.779.
- Two variants show improvement is reachable without breaking the constraint:
  - L1 strength tuned on the development side: +0.22 norm.
  - No-convergence local optima: better brain shear, worse rubber.
- No simple variant reaches the bound.
- The simple baseline (SI-CANN) sits clearly below the reference on both main experiments.

## 8. Brute force

The α-grid enumeration (exact weights, training-loss selection) scores Norm Δ −1.30 against the
reference. It ties on rubber, loses on brain because its exponent generalizes worse to shear, and
loses on the combined family with a relative error of 0.367 against 0.078. Its raw Δ is
slightly positive (+0.006), but only because raw sums count one fewer term on rubber combined as
0.0375; normalized by the spreads, it does not outrank the reference. PS-CANN, the dense
enumeration of every exponent, scores −3.83.

## 9. Could not fix, or worth knowing

- **Rubber, combined family.** The SI-CANN baseline (α = ±2, Mooney-Rivlin-like) is much
  stronger than the paper reports: 0.9989 on Treloar 20 °C, against the paper's 0.977. On the
  hidden test it is level with GI-CANN in avg R² (0.9969 vs 0.9974) and slightly ahead in ps R²
  (0.9963 vs 0.9946), though GI-CANN has the lower relative error (0.078 vs 0.111). In this
  experiment the paper's "two learned invariants beat fixed ones" shows only weakly.
- **The mismatched-family ablations stay poor for every method.** Rubber with negative exponents
  scores about 0.33 and brain with positive exponents about 0.48–0.50, as the paper finds.
  - Their weight is 0.05 each.
  - `n_terms` in brain-positive barely varies (spread 0.13).
- **Normalization.** Norm Δ uses the honest runs here as a stand-in for the spread across future
  attempts. With a different spread the weighting between metrics shifts.
- **Departures from the notebooks** (all in `method/gicann.py`):
  - The convergence phase and best-of-20 (§1.4).
  - The two numerical guards (§1.4).
  - Simple shear uses J_α = λ₊^α + 1 + λ₋^α. The paper's Eq. 18 and its notebook use the
    exponent-free λ₊ + λ₋ + 1, which only matters for exp/ln terms in shear.
  - Treloar 50 °C has uniaxial and biaxial curves of different lengths (15 and 20 points), so
    each mode is batched separately, 12 points per batch.
- **Superseded files.** `REDTEAM.md`, `DESIGN_NOTES.md`, `VERIFICATION.md` and the old
  `redteam/` runs describe the pre-hardening gym and are kept as history.
