# Progress

## What Works

- GI-CANN model implementation (method/gicann.py) with all three branch types (identity, exponential, logarithmic)
- SI-CANN baseline implementation (method/sicann.py) with fixed exponents
- Data loading from paper's Table A.5 (data/load_data.py): rubber and brain cortex stress-strain data
- Training scripts for both models with R² evaluation
- Evaluation script producing scoring/scores.json
- All shell scripts for the full pipeline
- Container definition with PyTorch
- Briefing documentation (problem.md, evaluation.md, method.md, overview.md)
- Reference.json with paper's reported numbers

## Results

Not yet run — waiting for first experiment.

Expected results (from paper):
- GI-CANN on rubber (positive, α*≈1.675): avg R² ≈ 0.998
- SI-CANN on rubber (positive, α=2): avg R² ≈ 0.982
- GI-CANN on brain (negative, α*≈-18): avg R² ≈ 0.966
- SI-CANN on brain (negative, α=-2): avg R² ≈ 0.610

## Remaining

- Run the full training pipeline (baseline + GI-CANN)
- Validate scoring/scores.json matches reference.json structure
- Verify R² values match paper's reported numbers
- Optional: expand to additional configurations

## Issues

- Login node may not have PyTorch installed; container will be built with PyTorch for compute node
- Need to verify the stress formulas work correctly for all deformation modes

## Deviations from Paper

- No external downloads needed: data is embedded in code from Table A.5 of the paper
- No model checkpoints to download
- Container uses Python 3.11 slim base (not a specific research ML image)
