#!/usr/bin/env python3
"""Evaluation-only model definitions reconstructed from checkpoints.

These classes intentionally mirror the paper implementation in ``method/``
without importing it, so ``eval/`` stays separated from training code while
still scoring the actual constitutive law described by the paper.
"""

import torch
import torch.nn as nn


class _MinimalCANNBase(nn.Module):
    """Shared stress computation for evaluation-time CANN models."""

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        self.mode = config.get("mode", "positive")
        self.r = config.get("r", 0.01)

    def _dpsi_dJ(self, J_minus_3, branch):
        if branch == "plus":
            dpsi = self.w_id_plus * torch.ones_like(J_minus_3)
            exp_arg = torch.clamp(self.w_exp1_plus * J_minus_3, max=10.0)
            dpsi = dpsi + self.w_exp2_plus * self.w_exp1_plus * torch.exp(exp_arg)
            ln_denom = torch.clamp(1.0 - self.w_ln1_plus * J_minus_3, min=1e-6)
            dpsi = dpsi - self.w_ln2_plus * self.w_ln1_plus / ln_denom
        else:
            dpsi = self.w_id_minus * torch.ones_like(J_minus_3)
            exp_arg = torch.clamp(self.w_exp1_minus * J_minus_3, max=10.0)
            dpsi = dpsi + self.w_exp2_minus * self.w_exp1_minus * torch.exp(exp_arg)
            ln_denom = torch.clamp(1.0 - self.w_ln1_minus * J_minus_3, min=1e-6)
            dpsi = dpsi - self.w_ln2_minus * self.w_ln1_minus / ln_denom
        return dpsi

    def _get_principal_stretches(self, F):
        batch = F.shape[0]
        l1 = torch.zeros(batch, device=F.device, dtype=F.dtype)
        l2 = torch.zeros(batch, device=F.device, dtype=F.dtype)
        l3 = torch.zeros(batch, device=F.device, dtype=F.dtype)

        for i in range(batch):
            Fi = F[i]
            off_diag = Fi - torch.diag(torch.diagonal(Fi))
            if torch.all(torch.abs(off_diag) < 1e-10):
                l1[i] = torch.abs(Fi[0, 0])
                l2[i] = torch.abs(Fi[1, 1])
                l3[i] = torch.abs(Fi[2, 2])
                continue
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

        return (
            torch.clamp(l1, min=1e-8),
            torch.clamp(l2, min=1e-8),
            torch.clamp(l3, min=1e-8),
        )

    def _detect_deformation_mode(self, l1, l2, l3, F=None):
        modes = []
        for i in range(l1.shape[0]):
            if F is not None:
                Fi = F[i]
                if (
                    abs(Fi[0, 1]) > 1e-10
                    or abs(Fi[1, 0]) > 1e-10
                    or abs(Fi[0, 2]) > 1e-10
                    or abs(Fi[2, 0]) > 1e-10
                ):
                    modes.append("ss")
                    continue
                if abs(Fi[1, 1] - 1.0) < 1e-10:
                    modes.append("ps")
                    continue
                if abs(Fi[0, 0] - Fi[1, 1]) < 1e-10:
                    modes.append("bt")
                    continue
                if abs(Fi[1, 1] - Fi[2, 2]) < 1e-10:
                    modes.append("ut" if Fi[0, 0] >= 1.0 else "uc")
                    continue

            l1_i = l1[i].item()
            l2_i = l2[i].item()
            l3_i = l3[i].item()
            r23 = abs(l2_i - l3_i) / max(l2_i, l3_i, 1e-8)
            r12 = abs(l1_i - l2_i) / max(l1_i, l2_i, 1e-8)
            r2_1 = abs(l2_i - 1.0)

            if r12 < 0.05:
                modes.append("bt")
            elif r23 < 0.05:
                modes.append("ut" if l1_i >= 1.0 else "uc")
            elif r2_1 < 0.05:
                modes.append("ps")
            else:
                modes.append("ut" if l1_i >= 1.0 else "uc")
        return modes

    def _stress_uniaxial(self, l1, l2, l3):
        sigma = torch.zeros((l1.shape[0], 3, 3), dtype=l1.dtype, device=l1.device)

        if self.mode in ("positive", "combined"):
            J_minus_3_plus = l1 ** self.alpha_plus + l2 ** self.alpha_plus + l3 ** self.alpha_plus - 3.0
            dpsi_plus = self._dpsi_dJ(J_minus_3_plus, "plus")

        if self.mode in ("negative", "combined"):
            J_minus_3_minus = l1 ** self.alpha_minus + l2 ** self.alpha_minus + l3 ** self.alpha_minus - 3.0
            dpsi_minus = self._dpsi_dJ(J_minus_3_minus, "minus")

        if self.mode == "combined":
            s11 = (
                self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus)
                + self.alpha_minus * dpsi_minus * (l1 ** self.alpha_minus - l3 ** self.alpha_minus)
            )
        elif self.mode == "positive":
            s11 = self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus)
        else:
            s11 = self.alpha_minus * dpsi_minus * (l1 ** self.alpha_minus - l3 ** self.alpha_minus)

        sigma[:, 0, 0] = s11
        return sigma

    def _stress_equibiaxial(self, l1, l2, l3):
        sigma = torch.zeros((l1.shape[0], 3, 3), dtype=l1.dtype, device=l1.device)

        if self.mode in ("positive", "combined"):
            J_minus_3_plus = l1 ** self.alpha_plus + l2 ** self.alpha_plus + l3 ** self.alpha_plus - 3.0
            dpsi_plus = self._dpsi_dJ(J_minus_3_plus, "plus")

        if self.mode in ("negative", "combined"):
            J_minus_3_minus = l1 ** self.alpha_minus + l2 ** self.alpha_minus + l3 ** self.alpha_minus - 3.0
            dpsi_minus = self._dpsi_dJ(J_minus_3_minus, "minus")

        if self.mode == "combined":
            s11 = (
                self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus)
                + self.alpha_minus * dpsi_minus * (l1 ** self.alpha_minus - l3 ** self.alpha_minus)
            )
        elif self.mode == "positive":
            s11 = self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus)
        else:
            s11 = self.alpha_minus * dpsi_minus * (l1 ** self.alpha_minus - l3 ** self.alpha_minus)

        sigma[:, 0, 0] = s11
        sigma[:, 1, 1] = s11
        return sigma

    def _stress_simple_shear(self, gamma):
        sigma = torch.zeros((gamma.shape[0], 3, 3), dtype=gamma.dtype, device=gamma.device)

        sqrt_term = torch.sqrt(gamma ** 2 + 4.0)
        lambda_plus = 0.5 * (sqrt_term + gamma)
        lambda_minus = 0.5 * (sqrt_term - gamma)
        ones = torch.ones_like(gamma)

        if self.mode in ("positive", "combined"):
            J_minus_3_plus = lambda_plus ** self.alpha_plus + ones + lambda_minus ** self.alpha_plus - 3.0
            dpsi_plus = self._dpsi_dJ(J_minus_3_plus, "plus")

        if self.mode in ("negative", "combined"):
            J_minus_3_minus = lambda_plus ** self.alpha_minus + ones + lambda_minus ** self.alpha_minus - 3.0
            dpsi_minus = self._dpsi_dJ(J_minus_3_minus, "minus")

        if self.mode == "combined":
            s12 = (
                (self.alpha_plus / (2.0 ** self.alpha_plus))
                * dpsi_plus
                * ((gamma + sqrt_term) ** self.alpha_plus - (-gamma + sqrt_term) ** self.alpha_plus)
                / sqrt_term
            )
            s12 = s12 + (
                (self.alpha_minus / (2.0 ** self.alpha_minus))
                * dpsi_minus
                * ((gamma + sqrt_term) ** self.alpha_minus - (-gamma + sqrt_term) ** self.alpha_minus)
                / sqrt_term
            )
        elif self.mode == "positive":
            s12 = (
                (self.alpha_plus / (2.0 ** self.alpha_plus))
                * dpsi_plus
                * ((gamma + sqrt_term) ** self.alpha_plus - (-gamma + sqrt_term) ** self.alpha_plus)
                / sqrt_term
            )
        else:
            s12 = (
                (self.alpha_minus / (2.0 ** self.alpha_minus))
                * dpsi_minus
                * ((gamma + sqrt_term) ** self.alpha_minus - (-gamma + sqrt_term) ** self.alpha_minus)
                / sqrt_term
            )

        sigma[:, 0, 1] = s12
        sigma[:, 1, 0] = s12
        return sigma

    def compute_stress(self, F, gamma_override=None):
        l1, l2, l3 = self._get_principal_stretches(F)
        modes = self._detect_deformation_mode(l1, l2, l3, F)
        sigma = torch.zeros((F.shape[0], 3, 3), dtype=F.dtype, device=F.device)

        mode_groups = {}
        for i, mode in enumerate(modes):
            mode_groups.setdefault(mode, []).append(i)

        for mode, indices in mode_groups.items():
            indices_t = torch.tensor(indices, device=F.device, dtype=torch.long)
            F_m = F[indices_t]
            l1_m = F_m[:, 0, 0]
            l2_m = F_m[:, 1, 1]
            l3_m = F_m[:, 2, 2]

            if mode == "ss" and gamma_override is not None:
                gamma_m = gamma_override[indices_t]
            elif mode == "ss":
                gamma_m = F_m[:, 0, 1] + F_m[:, 1, 0]
                gamma_m = gamma_m / 2.0
            else:
                gamma_m = None

            if mode in ("ut", "uc", "ps"):
                sigma_batch = self._stress_uniaxial(l1_m, l2_m, l3_m)
            elif mode == "bt":
                sigma_batch = self._stress_equibiaxial(l1_m, l2_m, l3_m)
            else:
                sigma_batch = self._stress_simple_shear(gamma_m)

            for idx, sample_idx in enumerate(indices):
                sigma[sample_idx] = sigma_batch[idx]

        return sigma

    def forward(self, F, gamma_override=None):
        return self.compute_stress(F, gamma_override=gamma_override)


class MinimalGICANN(_MinimalCANNBase):
    """Evaluation-time reconstruction of GI-CANN checkpoints."""

    def __init__(self, config=None):
        super().__init__(config)

        if self.mode == "positive":
            self.alpha_plus = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        elif self.mode == "negative":
            self.alpha_minus = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
        elif self.mode == "combined":
            self.alpha_plus = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
            self.alpha_minus = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))

        if self.mode in ("positive", "combined"):
            self.w_id_plus = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
            self.w_exp1_plus = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
            self.w_exp2_plus = nn.Parameter(torch.tensor(5.0, dtype=torch.float64))
            self.w_ln1_plus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_plus = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        if self.mode in ("negative", "combined"):
            self.w_id_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
            self.w_exp1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_exp2_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
            self.w_ln1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

    def get_parameters_dict(self):
        params = {}
        if self.mode == "positive":
            params["alpha_plus"] = self.alpha_plus.item()
        elif self.mode == "negative":
            params["alpha_minus"] = self.alpha_minus.item()
        elif self.mode == "combined":
            params["alpha_plus"] = self.alpha_plus.item()
            params["alpha_minus"] = self.alpha_minus.item()

        for n, p in self.named_parameters():
            if not n.startswith("alpha"):
                params[n] = p.item()
        return params


class MinimalSICANN(_MinimalCANNBase):
    """Evaluation-time reconstruction of SI-CANN checkpoints."""

    def __init__(self, config=None):
        super().__init__(config)
        self.alpha_fixed = 2.0 if self.mode == "positive" else -2.0

        if self.mode in ("positive", "combined"):
            self.register_buffer("alpha_plus", torch.tensor(2.0, dtype=torch.float64))
            self.w_id_plus = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
            self.w_exp1_plus = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
            self.w_exp2_plus = nn.Parameter(torch.tensor(5.0, dtype=torch.float64))
            self.w_ln1_plus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_plus = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        if self.mode in ("negative", "combined"):
            self.register_buffer("alpha_minus", torch.tensor(-2.0, dtype=torch.float64))
            self.w_id_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
            self.w_exp1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_exp2_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
            self.w_ln1_minus = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.w_ln2_minus = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))

    def get_parameters_dict(self):
        params = {"alpha_fixed": self.alpha_fixed}
        for n, p in self.named_parameters():
            params[n] = p.item()
        return params

    def _dpsi_fixed(self, l1, l2, l3, branch):
        alpha = 2.0 if branch == "plus" else -2.0
        J_minus_3 = l1**alpha + l2**alpha + l3**alpha - 3.0
        return self._dpsi_dJ(J_minus_3, branch)

    def compute_stress(self, F, gamma_override=None):
        l1, l2, l3 = self._get_principal_stretches(F)
        modes = self._detect_deformation_mode(l1, l2, l3, F)
        sigma = torch.zeros((F.shape[0], 3, 3), dtype=F.dtype, device=F.device)

        for mode in set(modes):
            indices = [i for i, detected_mode in enumerate(modes) if detected_mode == mode]
            indices_t = torch.tensor(indices, device=F.device, dtype=torch.long)
            F_m = F[indices_t]
            l1_m = F_m[:, 0, 0]
            l2_m = F_m[:, 1, 1]
            l3_m = F_m[:, 2, 2]
            alpha_p = 2.0
            alpha_n = -2.0

            if mode in ("ut", "uc", "ps"):
                if self.mode == "combined":
                    dpsi_p = self._dpsi_fixed(l1_m, l2_m, l3_m, "plus")
                    dpsi_n = self._dpsi_fixed(l1_m, l2_m, l3_m, "minus")
                    sigma11 = alpha_p * dpsi_p * (l1_m**alpha_p - l3_m**alpha_p)
                    sigma11 = sigma11 + alpha_n * dpsi_n * (l1_m**alpha_n - l3_m**alpha_n)
                elif self.mode == "positive":
                    dpsi = self._dpsi_fixed(l1_m, l2_m, l3_m, "plus")
                    sigma11 = alpha_p * dpsi * (l1_m**alpha_p - l3_m**alpha_p)
                else:
                    dpsi = self._dpsi_fixed(l1_m, l2_m, l3_m, "minus")
                    sigma11 = alpha_n * dpsi * (l1_m**alpha_n - l3_m**alpha_n)

                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx, 0, 0] = sigma11[idx]

            elif mode == "bt":
                if self.mode == "combined":
                    dpsi_p = self._dpsi_fixed(l1_m, l2_m, l3_m, "plus")
                    dpsi_n = self._dpsi_fixed(l1_m, l2_m, l3_m, "minus")
                    sigma11 = alpha_p * dpsi_p * (l1_m**alpha_p - l3_m**alpha_p)
                    sigma11 = sigma11 + alpha_n * dpsi_n * (l1_m**alpha_n - l3_m**alpha_n)
                elif self.mode == "positive":
                    dpsi = self._dpsi_fixed(l1_m, l2_m, l3_m, "plus")
                    sigma11 = alpha_p * dpsi * (l1_m**alpha_p - l3_m**alpha_p)
                else:
                    dpsi = self._dpsi_fixed(l1_m, l2_m, l3_m, "minus")
                    sigma11 = alpha_n * dpsi * (l1_m**alpha_n - l3_m**alpha_n)

                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx, 0, 0] = sigma11[idx]
                    sigma[sample_idx, 1, 1] = sigma11[idx]

            elif mode == "ss":
                if gamma_override is not None:
                    gamma = gamma_override[indices_t]
                else:
                    gamma = F_m[:, 0, 1]

                gamma = gamma.type_as(l1_m)
                sqrt_term = torch.sqrt(gamma**2 + 4.0)
                lambda_plus = 0.5 * (sqrt_term + gamma)
                lambda_minus = 0.5 * (sqrt_term - gamma)
                ones = torch.ones_like(gamma)

                if self.mode == "combined":
                    dpsi_p = self._dpsi_fixed(lambda_plus, ones, lambda_minus, "plus")
                    sigma12 = (
                        (alpha_p / (2.0**alpha_p))
                        * dpsi_p
                        * ((gamma + sqrt_term) ** alpha_p - (-gamma + sqrt_term) ** alpha_p)
                        / sqrt_term
                    )
                    dpsi_n = self._dpsi_fixed(lambda_plus, ones, lambda_minus, "minus")
                    sigma12 = sigma12 + (
                        (alpha_n / (2.0**alpha_n))
                        * dpsi_n
                        * ((gamma + sqrt_term) ** alpha_n - (-gamma + sqrt_term) ** alpha_n)
                        / sqrt_term
                    )
                elif self.mode == "positive":
                    dpsi = self._dpsi_fixed(lambda_plus, ones, lambda_minus, "plus")
                    sigma12 = (
                        (alpha_p / (2.0**alpha_p))
                        * dpsi
                        * ((gamma + sqrt_term) ** alpha_p - (-gamma + sqrt_term) ** alpha_p)
                        / sqrt_term
                    )
                else:
                    dpsi = self._dpsi_fixed(lambda_plus, ones, lambda_minus, "minus")
                    sigma12 = (
                        (alpha_n / (2.0**alpha_n))
                        * dpsi
                        * ((gamma + sqrt_term) ** alpha_n - (-gamma + sqrt_term) ** alpha_n)
                        / sqrt_term
                    )

                for idx, sample_idx in enumerate(indices):
                    sigma[sample_idx, 0, 1] = sigma12[idx]
                    sigma[sample_idx, 1, 0] = sigma12[idx]

        return sigma
