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
from method.gicann import GICANN, detect_mode, get_principal_stretches_batch


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
        if self.mode in ('positive', 'combined'):
            self.w_id_plus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
            self.w_id_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

        # --- Branch 2: Exponential ψ_exp ---
        if self.mode in ('positive', 'combined'):
            self.w_exp1_plus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_exp2_plus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
            self.w_exp1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_exp2_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

        # --- Branch 3: Logarithmic ψ_ln ---
        if self.mode in ('positive', 'combined'):
            self.w_ln1_plus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_plus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
            self.w_ln1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

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

    def compute_stress(self, F):
        """Compute Cauchy stress tensor from deformation gradient."""
        batch = F.shape[0]
        l1, l2, l3 = get_principal_stretches_batch(F)
        modes = detect_mode(F)

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
                gamma = F_m[:, 0, 1]

                if self.mode == 'combined':
                    sigma12 = torch.zeros(len(indices), dtype=F.dtype, device=F.device)
                    sqrt_term = torch.sqrt(1 + gamma ** 2 / 4)

                    dpsi_p = self.dpsi_dJ(l1_m, l2_m, l3_m, 'plus')
                    term_p = ((gamma + sqrt_term) ** alpha_p - (-gamma + sqrt_term) ** alpha_p) / sqrt_term
                    sigma12 = sigma12 + alpha_p / (2 ** alpha_p) * dpsi_p * term_p

                    dpsi_n = self.dpsi_dJ(l1_m, l2_m, l3_m, 'minus')
                    term_n = ((gamma + sqrt_term) ** alpha_n - (-gamma + sqrt_term) ** alpha_n) / sqrt_term
                    sigma12 = sigma12 + alpha_n / (2 ** alpha_n) * dpsi_n * term_n

                elif self.mode == 'positive':
                    dpsi = self.dpsi_dJ(l1_m, l2_m, l3_m, 'plus')
                    sqrt_term = torch.sqrt(1 + gamma ** 2 / 4)
                    term = ((gamma + sqrt_term) ** alpha_p - (-gamma + sqrt_term) ** alpha_p) / sqrt_term
                    sigma12 = alpha_p / (2 ** alpha_p) * dpsi * term
                else:
                    dpsi = self.dpsi_dJ(l1_m, l2_m, l3_m, 'minus')
                    sqrt_term = torch.sqrt(1 + gamma ** 2 / 4)
                    term = ((gamma + sqrt_term) ** alpha_n - (-gamma + sqrt_term) ** alpha_n) / sqrt_term
                    sigma12 = alpha_n / (2 ** alpha_n) * dpsi * term

                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx, 0, 1] = sigma12[idx]
                    sigma[sample_idx, 1, 0] = sigma12[idx]

        return sigma

    def forward(self, F, gamma=None):
        return self.compute_stress(F, gamma_override=gamma)

    def loss(self, F, S_target, gamma=None):
        S_pred = self.compute_stress(F, gamma_override=gamma)
        mse = torch.mean((S_pred - S_target) ** 2)
        l1_loss = self.r * sum(torch.sum(torch.abs(w)) for w in self.get_strain_energy_weights())
        return mse + l1_loss, mse
