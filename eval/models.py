#!/usr/bin/env python3
"""Model definitions for evaluation — minimal reconstruction from checkpoints.

These are minimal model classes used by eval/evaluate.py to reconstruct
models from checkpoints without importing from method/.
"""
import torch
import torch.nn as nn
import numpy as np


class MinimalCANN(nn.Module):
    """Minimal CANN model for evaluation.

    This class matches the structure of both GI-CANN and SI-CANN for
    checkpoint loading, but is stripped of training-specific code.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        self.mode = config.get('mode', 'positive')
        self.r = config.get('r', 0.01)

        # Trainable exponents (GI-CANN only)
        if self.mode == 'positive':
            self.alpha_plus = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        elif self.mode == 'negative':
            self.alpha_minus = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
        elif self.mode == 'combined':
            self.alpha_plus = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
            self.alpha_minus = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))

        # Identity branch weights
        if self.mode in ('positive', 'combined'):
            self.w_id_plus = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
            self.w_id_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

        # Exponential branch weights
        if self.mode in ('positive', 'combined'):
            self.w_exp1_plus = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
            self.w_exp2_plus = nn.Parameter(torch.tensor(5.0, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
            self.w_exp1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_exp2_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

        # Logarithmic branch weights
        if self.mode in ('positive', 'combined'):
            self.w_ln1_plus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_plus = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
            self.w_ln1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

    def get_parameters_dict(self):
        """Return parameters as a dict."""
        params = {}
        if self.mode == 'positive':
            params['alpha_plus'] = self.alpha_plus.item()
        elif self.mode == 'negative':
            params['alpha_minus'] = self.alpha_minus.item()
        elif self.mode == 'combined':
            params['alpha_plus'] = self.alpha_plus.item()
            params['alpha_minus'] = self.alpha_minus.item()
        for n, p in self.named_parameters():
            if n.startswith('alpha'):
                continue
            params[n] = p.item()
        return params

    # The full forward/compute_stress methods are needed for evaluation
    # These are copied from the original implementation

    def _dpsi_dJ(self, J_minus_3, branch):
        """Compute dψ/dJ for a specific branch."""
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

    def _get_principal_stretches(self, F):
        """Get principal stretches via SVD."""
        batch = F.shape[0]
        l1 = torch.zeros(batch, device=F.device, dtype=F.dtype)
        l2 = torch.zeros(batch, device=F.device, dtype=F.dtype)
        l3 = torch.zeros(batch, device=F.device, dtype=F.dtype)

        for i in range(batch):
            Fi = F[i]
            try:
                _, sv_s, _ = torch.linalg.svd(Fi, full_matrices=False)
                l1[i] = sv_s[0]
                l2[i] = sv_s[1]
                l3[i] = sv_s[2]
            except Exception:
                Ci = Fi.T @ Fi
                eigvals = torch.linalg.eigvalsh(Ci)
                eigvals = torch.clamp(eigvals, min=1e-12)
                l1[i] = torch.sqrt(eigvals[2])
                l2[i] = torch.sqrt(eigvals[1])
                l3[i] = torch.sqrt(eigvals[0])

        l1 = torch.clamp(l1, min=1e-8)
        l2 = torch.clamp(l2, min=1e-8)
        l3 = torch.clamp(l3, min=1e-8)
        return l1, l2, l3

    def _detect_deformation_mode(self, l1, l2, l3, F=None):
        """Detect deformation mode."""
        batch = l1.shape[0]
        modes = []
        for i in range(batch):
            l1_i = l1[i].item()
            l2_i = l2[i].item()
            l3_i = l3[i].item()

            if F is not None:
                Fi = F[i]
                if (abs(Fi[0, 1]) > 1e-6 or abs(Fi[1, 0]) > 1e-6 or
                    abs(Fi[0, 2]) > 1e-6 or abs(Fi[2, 0]) > 1e-6):
                    modes.append('ss')
                    continue

            r23 = abs(l2_i - l3_i) / max(l2_i, l3_i, 1e-8)
            r12 = abs(l1_i - l2_i) / max(l1_i, l2_i, 1e-8)
            r2_1 = abs(l2_i - 1.0)

            if r12 < 0.05:
                modes.append('bt')
            elif r23 < 0.05:
                modes.append('ut')
            elif r2_1 < 0.05:
                modes.append('ps')
            else:
                modes.append('ut')
        return modes

    def _stress_uniaxial(self, l1, l2, l3):
        """Compute stress for uniaxial deformation."""
        batch = l1.shape[0]
        sigma = torch.zeros((batch, 3, 3), dtype=l1.dtype, device=l1.device)

        if self.mode in ('positive', 'combined'):
            J_minus_3_plus = l1 ** self.alpha_plus + l2 ** self.alpha_plus + l3 ** self.alpha_plus - 3.0
            dpsi_plus = self._dpsi_dJ(J_minus_3_plus, 'plus')

        if self.mode in ('negative', 'combined'):
            J_minus_3_minus = (l1 ** self.alpha_minus + l2 ** self.alpha_minus +
                               l3 ** self.alpha_minus - 3.0)
            dpsi_minus = self._dpsi_dJ(J_minus_3_minus, 'minus')

        if self.mode == 'combined':
            s11 = (self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus) +
                   self.alpha_minus * dpsi_minus * (l3 ** self.alpha_minus - l1 ** self.alpha_minus))
        elif self.mode == 'positive':
            s11 = self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus)
        else:
            s11 = self.alpha_minus * dpsi_minus * (l3 ** self.alpha_minus - l1 ** self.alpha_minus)

        sigma[:, 0, 0] = s11
        return sigma

    def _stress_equibiaxial(self, l1, l2, l3):
        """Compute stress for equibiaxial deformation."""
        batch = l1.shape[0]
        sigma = torch.zeros((batch, 3, 3), dtype=l1.dtype, device=l1.device)

        if self.mode in ('positive', 'combined'):
            J_minus_3_plus = l1 ** self.alpha_plus + l2 ** self.alpha_plus + l3 ** self.alpha_plus - 3.0
            dpsi_plus = self._dpsi_dJ(J_minus_3_plus, 'plus')

        if self.mode in ('negative', 'combined'):
            J_minus_3_minus = (l1 ** self.alpha_minus + l2 ** self.alpha_minus +
                               l3 ** self.alpha_minus - 3.0)
            dpsi_minus = self._dpsi_dJ(J_minus_3_minus, 'minus')

        if self.mode == 'combined':
            s11 = (self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus) +
                   self.alpha_minus * dpsi_minus * (l3 ** self.alpha_minus - l1 ** self.alpha_minus))
        elif self.mode == 'positive':
            s11 = self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus)
        else:
            s11 = self.alpha_minus * dpsi_minus * (l3 ** self.alpha_minus - l1 ** self.alpha_minus)

        sigma[:, 0, 0] = s11
        sigma[:, 1, 1] = s11
        return sigma

    def _stress_simple_shear(self, l1, l2, l3, gamma):
        """Compute stress for simple shear deformation."""
        batch = l1.shape[0]
        sigma = torch.zeros((batch, 3, 3), dtype=l1.dtype, device=l1.device)

        gamma = gamma.view(batch)
        sqrt_term = torch.sqrt(gamma ** 2 + 4.0)

        l1_B = torch.sqrt(1.0 + gamma)
        l3_B = torch.sqrt(torch.clamp(1.0 - gamma, min=1e-8))
        l2_B = torch.ones(batch, dtype=l1.dtype, device=l1.device)

        if self.mode in ('positive', 'combined'):
            J_minus_3_plus = l1_B ** self.alpha_plus + l2_B ** self.alpha_plus + l3_B ** self.alpha_plus - 3.0
            dpsi_plus = self._dpsi_dJ(J_minus_3_plus, 'plus')

        if self.mode in ('negative', 'combined'):
            J_minus_3_minus = l1_B ** self.alpha_minus + l2_B ** self.alpha_minus + l3_B ** self.alpha_minus - 3.0
            dpsi_minus = self._dpsi_dJ(J_minus_3_minus, 'minus')

        if self.mode == 'combined':
            s12_pos = (self.alpha_plus * dpsi_plus *
                       (l1_B ** (self.alpha_plus - 1.0) - l3_B ** (self.alpha_plus - 1.0)) / sqrt_term)
            s12_neg = (torch.abs(self.alpha_minus) * dpsi_minus *
                       (l1_B ** (torch.abs(self.alpha_minus) - 1.0) + l3_B ** (torch.abs(self.alpha_minus) - 1.0)) / sqrt_term)
            s12 = s12_pos + s12_neg
        elif self.mode == 'positive':
            s12 = (self.alpha_plus * dpsi_plus *
                   (l1_B ** (self.alpha_plus - 1.0) - l3_B ** (self.alpha_plus - 1.0)) / sqrt_term)
        else:
            s12 = (torch.abs(self.alpha_minus) * dpsi_minus *
                   (l1_B ** (torch.abs(self.alpha_minus) - 1.0) + l3_B ** (torch.abs(self.alpha_minus) - 1.0)) / sqrt_term)

        sigma[:, 0, 1] = s12
        sigma[:, 1, 0] = s12
        return sigma

    def compute_stress(self, F, gamma_override=None):
        """Compute Cauchy stress tensor."""
        batch = F.shape[0]
        l1, l2, l3 = self._get_principal_stretches(F)
        modes = self._detect_deformation_mode(l1, l2, l3, F)

        sigma = torch.zeros((batch, 3, 3), dtype=F.dtype, device=F.device)

        mode_groups = {}
        for i, mode in enumerate(modes):
            if mode not in mode_groups:
                mode_groups[mode] = []
            mode_groups[mode].append(i)

        for mode, indices in mode_groups.items():
            indices_t = torch.tensor(indices, device=F.device, dtype=torch.long)
            l1_m = l1[indices_t]
            l2_m = l2[indices_t]
            l3_m = l3[indices_t]

            if mode == 'ss' and gamma_override is not None:
                gamma_m = gamma_override[indices_t]
            elif mode == 'ss':
                gamma_m = F[indices_t, 0, 1] + F[indices_t, 1, 0]
                gamma_m = gamma_m / 2
            else:
                gamma_m = None

            if mode in ('ut', 'uc', 'ps'):
                sigma_batch = self._stress_uniaxial(l1_m, l2_m, l3_m)
                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx] = sigma_batch[idx]
            elif mode == 'bt':
                sigma_batch = self._stress_equibiaxial(l1_m, l2_m, l3_m)
                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx] = sigma_batch[idx]
            elif mode == 'ss':
                sigma_batch = self._stress_simple_shear(l1_m, l2_m, l3_m, gamma_m)
                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx] = sigma_batch[idx]

        return sigma

    def forward(self, F, gamma_override=None):
        return self.compute_stress(F, gamma_override=gamma_override)
