# Reproduction Checklist

## Briefing
- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json` — paper's reported numbers
- [ ] Workspace validated via `python validate.py`
- [ ] `scoring/scores.json` — reproduced numbers
- [x] `scoring/TARGETS.md` — placeholder (do not edit until wrap up)
- [x] `scoring/CONSTRAINTS.md` — placeholder (do not edit until wrap up)
- [x] `scoring/DIRECTION.md` — placeholder (do not edit until wrap up)

## Code
- [x] Evaluation code in `eval/`
- [x] Method implementation in `method/`
- [x] Baseline implementation in `baseline/`
- [x] Data loading in `data/`

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
- [ ] `core_claim` — minimum experiment supports central claim
- [ ] `core_claim_plus` — additional settings
- [ ] `secondary_claims`
- [ ] `majority`
- [ ] `near_complete`
- [ ] `full`
