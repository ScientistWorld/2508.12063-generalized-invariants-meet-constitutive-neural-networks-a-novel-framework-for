# Constraints

These constraints define what a scientist must hold fixed. Genuine improvement requires methodology advances, not just resource scaling.

## Data Constraints

**Must Use**: The stress-strain data from the paper's Table A.5, located in `data/` and loaded via `data/load_data.py`.

- **Rubber (VHB 4910)**: 
  - Train: Uniaxial tension (UT) + Equibiaxial tension (BT)
  - Test: Pure shear (PS)
  - Range: λ1 ∈ [1.0, 7.25], stresses up to ~500 kPa

- **Brain Cortex**:
  - Train: Uniaxial tension (UT) + Uniaxial compression (UC)
  - Test: Simple shear (SS), γ ∈ [0, 0.2]
  - Range: λ1 ∈ [0.9, 1.1], stresses ∈ [-1.03, 0.54] kPa

**Cannot Modify**: Data values, train/test split, deformation modes.

## Model Constraints

**Must Implement**: Three-branch strain energy function with identity, exponential, and logarithmic components:

```
ψ = ψ_id + ψ_exp + ψ_ln
ψ_id = w_id · J_α
ψ_exp = w_exp1 · exp(w_exp2 · J_α^(-1))
ψ_ln = -w_ln1 · ln(1 - w_ln2 · (J_α - 3))
```

Where J_α = Σ λ_i^α (generalized invariant).

**Fixed Constraints**:
- Must use trainable exponent α (GI-CANN) or fixed α=±2 (SI-CANN baseline)
- Must compute stress via σ = 2·F·∂ψ/∂C for uniaxial/equibiaxial, B-eigenvalue formula for simple shear
- Cannot add architectural complexity (no additional layers, no attention, etc.)

**Variable**: Weight initialization, learning rate, optimizer, training epochs, gradient clipping — these are hyperparameters to tune.

## Evaluation Constraints

**Must Compute**: R² score on test deformation mode.

**Metric**: avg_r2 = mean(R²_train_modes + R²_test_mode)

- Rubber: avg_r2 = mean(R²_ut, R²_bt, R²_ps)
- Brain: avg_r2 = mean(R²_ut, R²_uc, R²_ss)

**Cannot Modify**: Evaluation metric, train/test modes.

## Parameter Count Constraint

**Constraint**: GI-CANN has 12 trainable parameters (per branch: α, w_id, w_exp1, w_exp2, w_ln1, w_ln2). Any proposed method must have similar parameter count (< 100) to demonstrate the invariant formulation's power, not model capacity.

## Training Constraints

**Must**: Train from random initialization (or sensible defaults) — cannot load pretrained weights from external sources.

**Cannot**: Use more than 30 GPU-hours total budget per agent.
