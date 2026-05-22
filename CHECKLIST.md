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
- [x] `scoring/TARGETS.md` — filled in
- [x] `scoring/CONSTRAINTS.md` — filled in
- [x] `scoring/DIRECTION.md` — filled in
- [x] `scoring/EXPERIMENTS.md` — filled in

## Code
- [x] Evaluation code in `eval/`
- [x] Model-agnostic prediction artifact scoring path in `eval/artifacts.py`
- [x] Method implementation in `method/`
- [x] Baseline implementation in `baseline/` (via sicann.py)
- [x] Data loading in `data/`
- [x] Import separation (eval/models.py created for validation)

## Scripts
- [x] `scripts/evaluate.sh`
- [x] `scripts/reproduce.sh`
- [x] `scripts/download.sh` (idempotent, data embedded)
- [x] `scripts/baseline.sh`
- [x] `scripts/method.sh`
- [x] `scripts/run.sh`

## Environment
- [x] `environment/container.def` + `environment/setup.sh`

## Data
- [x] Dataset acquired (embedded in data/load_data.py from Table A.5)
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
