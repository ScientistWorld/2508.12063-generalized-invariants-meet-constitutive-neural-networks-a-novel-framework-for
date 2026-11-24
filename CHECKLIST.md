# Reproduction Checklist

## Briefing
- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json` — paper's reported numbers
- [x] Workspace validated via `python validate.py`
- [x] `scoring/scores.json` — reproduced numbers
- [x] Score files regenerated from the reference laws in the iteration-11 job (see `HARDENING.md`)
- [x] `scoring/TARGETS.md` — filled in
- [x] `scoring/CONSTRAINTS.md` — filled in
- [x] `scoring/DIRECTION.md` — filled in
- [x] `scoring/EXPERIMENTS.md` — filled in

## Code
- [x] Evaluation code in `eval/`
- [x] Law-based scoring: `eval/train/evaluate.py` and `eval/test/evaluate.py` recompute stresses from declared strain-energy laws
- [x] Method implementation in `method/`
- [x] SI-CANN baseline (`method/train.py --model SI-CANN`), installed as a fixed baseline in `eval/*/baselines/`
- [x] Data loading in `data/`
- [x] Import separation (eval/ is standard-library only)

## Scripts
- [x] `scripts/evaluate.sh`
- [x] `scripts/reproduce.sh`
- [x] `scripts/download.sh` (fetches the authors' data files at a pinned commit, sha256-checked)
- [x] `scripts/baseline.sh`
- [x] `scripts/method.sh`
- [x] `scripts/run.sh`

## Environment
- [x] `environment/container.def` + `environment/setup.sh`

## Data
- [x] Dataset acquired (authors' spreadsheets, LivingMatterLab/CANN@d3caca0f -> `data/curves.json`)
- [x] Pretrained models downloaded if needed (N/A — no external models)

## Milestones
- [x] `method_runs` — executes end-to-end without errors
- [x] `core_claim` — minimum experiment supports central claim (rubber reproduced)
- [x] Brain checkpoint retraining after observed-stress loss fix
- [x] `core_claim_plus` — rubber and brain core settings reproduced
- [x] `secondary_claims` — main combined/positive/negative ablations scored
- [x] `majority` — both core claims and most proposed-method ablations reproduced
- [ ] `near_complete`
- [ ] `full`
