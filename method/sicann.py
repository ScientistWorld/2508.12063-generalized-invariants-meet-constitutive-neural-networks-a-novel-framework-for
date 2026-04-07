#!/usr/bin/env python3
"""SI-CANN: Standard Invariant Constitutive Artificial Neural Network.

This is the baseline model from the paper:
SI-CANN uses fixed classical invariants (α=2 for I₁, α=-2 for I₂).
The exponents are NOT trainable — only the branch weights are learned.

Reference: Linka, K., Kuhl, E. (2023). A new family of Constitutive Artificial
Neural Networks towards automated model discovery.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from method.gicann import GICANN, detect_deformation_mode, get_principal_stretches


class SICANN(nn.Module):
    """SI-CANN model: same architecture as GI-CANN but with FIXED exponents.

    Unlike GI-CANN where α is trainable, here α is fixed at:
    - α = 2 for positive (corresponds to I₁)
    - α = -2 for negative (corresponds to I₂)

    This is the paper's baseline for comparison.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        self.mode = config.get('mode', 'positive')  # 'positive' → α=2, 'negative' → α=-2
        self.r = config.get('r', 0.01)
        self.alpha_fixed = 2.0 if self.mode == 'positive' else -2.0

        # --- Branch 1: Identity ψ_id = w_id * (J_α - 3) ---
        # Scale weights by material stress magnitude:
        # - Rubber: stress 0-32 MPa → w_id ~ 1-5 MPa
        # - Brain: stress 0-0.5 kPa → w_id ~ 0.01-0.1 kPa
        if self.mode in ('positive', 'combined'):
            self.w_id_plus = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
            self.w_id_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

        # --- Branch 2: Exponential ψ_exp = w_exp2 * [exp(w_exp1 * (J_α - 3)) - 1] ---
        # CRITICAL: w_exp1 must be small to avoid exp overflow.
        # For rubber with α=2: J-3 up to ~50 → w_exp1 * 50 must be < ~10
        #   → w_exp1 ≈ 0.001-0.02 for rubber (UT reaches J≈52).
        #   Use 0.001 for rubber negative/combined (α=-2 branch with J up to 52 is unstable).
        #   Use 0.02 for rubber positive (α=2, J up to 52 with clamp prevents overflow).
        # For brain: J-3 up to ~0.03 → w_exp1 ≈ 1-10. Use 0.1.
        if self.mode in ('positive', 'combined'):
            self.w_exp1_plus = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
            self.w_exp2_plus = nn.Parameter(torch.tensor(5.0, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
            # Use material-specific initialization
            # Brain: J-3 small (~0.03) → w_exp1 can be 0.1
            # Rubber: J-3 large (~50) for α=-2 branch → need w_exp1 ≈ 0.001
            # Default to brain-safe; override in training script for rubber
            self.w_exp1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_exp2_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

        # --- Branch 3: Logarithmic ψ_ln = w_ln2 * ln(1 - w_ln1 * (J_α - 3)) ---
        if self.mode in ('positive', 'combined'):
            self.w_ln1_plus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_plus = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
            self.w_ln1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

    def get_strain_energy_weights(self):
        weights = []
        for _, param in self.named_parameters():
            weights.append(param)
        return weights

    def get_parameters_dict(self):
        params = {'alpha_fixed': self.alpha_fixed}
        for name, param in self.named_parameters():
            params[name] = param.item()
        return params

    def dpsi_dJ(self, l1, l2, l3, branch='plus'):
        """Compute dψ/dJ for all samples in a batch."""
        alpha = self.alpha_fixed if branch == 'plus' else self.alpha_fixed
        if branch == 'minus':
            alpha = -2.0 if self.mode == 'negative' else -2.0

        J_minus_3 = l1 ** alpha + l2 ** alpha + l3 ** alpha - 3.0

        if branch == 'plus':
            dpsi = self.w_id_plus * torch.ones_like(J_minus_3)
            exp_arg = torch.clamp(self.w_exp1_plus * J_minus_3, max=10.0)
            dpsi = dpsi + self.w_exp2_plus * self.w_exp1_plus * torch.exp(exp_arg)
            ln_denom = 1.0 - self.w_ln1_plus * J_minus_3
            ln_denom = torch.clamp(ln_denom, min=1e-6)
            dpsi = dpsi - self.w_ln2_plus * self.w_ln1_plus / ln_denom
        else:
            dpsi = self.w_id_minus * torch.ones_like(J_minus_3)
            exp_arg = torch.clamp(self.w_exp1_minus * J_minus_3, max=10.0)
            dpsi = dpsi + self.w_exp2_minus * self.w_exp1_minus * torch.exp(exp_arg)
            ln_denom = 1.0 - self.w_ln1_minus * J_minus_3
            ln_denom = torch.clamp(ln_denom, min=1e-6)
            dpsi = dpsi - self.w_ln2_minus * self.w_ln1_minus / ln_denom

        return dpsi

    def compute_stress(self, F, gamma_override=None):
        """Compute Cauchy stress tensor from deformation gradient."""
        batch = F.shape[0]
        l1, l2, l3 = get_principal_stretches(F)
        modes = detect_deformation_mode(l1, l2, l3, F)

        sigma = torch.zeros((batch, 3, 3), dtype=F.dtype, device=F.device)

        for mode in set(modes):
            indices = [i for i, m in enumerate(modes) if m == mode]
            l1_m = l1[indices]
            l2_m = l2[indices]
            l3_m = l3[indices]
            F_m = F[indices]

            alpha_p = 2.0
            alpha_n = -2.0

            if mode in ('ut', 'uc', 'ps'):
                if self.mode == 'combined':
                    dpsi_p = self.dpsi_dJ(l1_m, l2_m, l3_m, 'plus')
                    dpsi_n = self.dpsi_dJ(l1_m, l2_m, l3_m, 'minus')
                    sigma11 = alpha_p * dpsi_p * (l1_m ** alpha_p - l3_m ** alpha_p)
                    sigma11 = sigma11 + alpha_n * dpsi_n * (l1_m ** alpha_n - l3_m ** alpha_n)
                elif self.mode == 'positive':
                    dpsi = self.dpsi_dJ(l1_m, l2_m, l3_m, 'plus')
                    sigma11 = alpha_p * dpsi * (l1_m ** alpha_p - l3_m ** alpha_p)
                else:
                    dpsi = self.dpsi_dJ(l1_m, l2_m, l3_m, 'minus')
                    sigma11 = alpha_n * dpsi * (l1_m ** alpha_n - l3_m ** alpha_n)

                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx, 0, 0] = sigma11[idx]

            elif mode == 'bt':
                if self.mode == 'combined':
                    dpsi_p = self.dpsi_dJ(l1_m, l2_m, l3_m, 'plus')
                    dpsi_n = self.dpsi_dJ(l1_m, l2_m, l3_m, 'minus')
                    sigma11 = alpha_p * dpsi_p * (l1_m ** alpha_p - l3_m ** alpha_p)
                    sigma11 = sigma11 + alpha_n * dpsi_n * (l1_m ** alpha_n - l3_m ** alpha_n)
                elif self.mode == 'positive':
                    dpsi = self.dpsi_dJ(l1_m, l2_m, l3_m, 'plus')
                    sigma11 = alpha_p * dpsi * (l1_m ** alpha_p - l3_m ** alpha_p)
                else:
                    dpsi = self.dpsi_dJ(l1_m, l2_m, l3_m, 'minus')
                    sigma11 = alpha_n * dpsi * (l1_m ** alpha_n - l3_m ** alpha_n)

                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx, 0, 0] = sigma11[idx]
                    sigma[sample_idx, 1, 1] = sigma11[idx]

            elif mode == 'ss':
                if gamma_override is not None:
                    # Use gamma values from the dataset (stored in gamma_override tensor)
                    # indices contains original sample positions in the full batch
                    gamma = gamma_override[[i for i in indices]]
                else:
                    # Fallback: extract from deformation gradient
                    gamma = F_m[:, 0, 1]

                gamma = gamma.type_as(l1_m)
                sqrt_term = torch.sqrt(gamma ** 2 + 4.0)  # s = √(γ² + 4)

                # B-eigenvalues for SS: l1_B = √(1+γ), l3_B = √(1-γ)
                l1_B = torch.sqrt(1.0 + gamma)
                l3_B = torch.sqrt(torch.clamp(1.0 - gamma, min=1e-8))
                l2_B = torch.ones_like(gamma)

                if self.mode == 'combined':
                    dpsi_p = self.dpsi_dJ(l1_B, l2_B, l3_B, 'plus')
                    # σ_12 = α * dψ/dJ * [(1+γ)^((α-1)/2) - (1-γ)^((α-1)/2)] / √(γ²+4)
                    term_p = (l1_B ** (alpha_p - 1.0) - l3_B ** (alpha_p - 1.0)) / sqrt_term
                    sigma12 = alpha_p * dpsi_p * term_p

                    dpsi_n = self.dpsi_dJ(l1_B, l2_B, l3_B, 'minus')
                    # For negative alpha: σ_12 = |α| * dψ/dJ * [(1+γ)^((|α|-1)/2) + (1-γ)^((|α|-1)/2)] / √(γ²+4)
                    term_n = (l1_B ** (alpha_n - 1.0) + l3_B ** (alpha_n - 1.0)) / sqrt_term
                    # alpha_n is negative, so |α| * term_n uses abs(alpha_n)
                    sigma12 = sigma12 + abs(alpha_n) * dpsi_n * term_n

                elif self.mode == 'positive':
                    dpsi = self.dpsi_dJ(l1_B, l2_B, l3_B, 'plus')
                    term = (l1_B ** (alpha_p - 1.0) - l3_B ** (alpha_p - 1.0)) / sqrt_term
                    sigma12 = alpha_p * dpsi * term
                else:
                    dpsi = self.dpsi_dJ(l1_B, l2_B, l3_B, 'minus')
                    # σ_12 = |α| * dψ/dJ * [(1+γ)^((|α|-1)/2) + (1-γ)^((|α|-1)/2)] / √(γ²+4)
                    term_n = (l1_B ** (alpha_n - 1.0) + l3_B ** (alpha_n - 1.0)) / sqrt_term
                    sigma12 = abs(alpha_n) * dpsi * term_n

                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx, 0, 1] = sigma12[idx]
                    sigma[sample_idx, 1, 0] = sigma12[idx]

        return sigma

    def forward(self, F, gamma_override=None):
        return self.compute_stress(F, gamma_override=gamma_override)

    def loss(self, F, S_target, gamma=None):
        S_pred = self.compute_stress(F, gamma_override=gamma)
        mse = torch.mean((S_pred - S_target) ** 2)
        l1_loss = self.r * sum(torch.sum(torch.abs(w)) for w in self.get_strain_energy_weights())
        return mse + l1_loss, mse
