#!/usr/bin/env python3
"""GI-CANN: Generalized Invariant Constitutive Artificial Neural Network.

This module implements the GI-CANN model from:
"Generalized invariants meet constitutive neural networks: A novel framework
for hyperelastic materials" (Martonova, Goriely, Kuhl, 2025).

The model discovers both the optimal exponent α for generalized invariants
J_α = Σ λ_i^α and the strain energy function ψ(J_α) from stress-strain data.
"""

import torch
import torch.nn as nn
import numpy as np


def get_principal_stretches(F):
    """Get principal stretches (eigenvalues of F) for a batch of deformation gradients.

    Uses SVD: F = USV^T, singular values = principal stretches.

    For general 3D deformations (UT, UC, BT, PS), SVD = B-eigenvalues.
    For SS (simple shear), SVD differs from B-eigenvalues; see get_ss_b_eigenvalues.

    Args:
        F: (batch, 3, 3) deformation gradients

    Returns:
        l1, l2, l3: (batch,) principal stretch tensors
    """
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

    # Ensure positive
    l1 = torch.clamp(l1, min=1e-8)
    l2 = torch.clamp(l2, min=1e-8)
    l3 = torch.clamp(l3, min=1e-8)
    return l1, l2, l3


def get_ss_b_eigenvalues(F, gamma):
    """Get B-eigenvalues for simple shear.

    For SS, F = [[1, γ, 0], [0, 1, 0], [0, 0, 1]].
    B = FF^T = [[1+γ², γ, 0], [γ, 1, 0], [0, 0, 1]].

    The 2×2 block [[1+γ², γ], [γ, 1]] has eigenvalues (1+γ² ± γ√(γ²+4))/2.
    Taking square roots: l1 = √((1+γ²+γ√(γ²+4))/2), l3 = √((1+γ²-γ√(γ²+4))/2).
    These simplify to l1 = √(1+γ), l3 = √(1-γ) (eigenvalues of B^½, not B itself).

    For stress computation, we need B-eigenvalues = eigenvalues of B^½:
    l1_B = √(1+γ), l2_B = 1, l3_B = √(1-γ).

    For the J_α computation in SS, we use l1 = √(1+γ), l3 = √(1-γ).

    Args:
        F: (batch, 3, 3) deformation gradients (SS mode)
        gamma: (batch,) shear strains

    Returns:
        l1_B, l3_B: (batch,) B-eigenvalues for SS
    """
    batch = F.shape[0]
    gamma = gamma.view(batch)
    l1_B = torch.sqrt(1.0 + gamma)
    l3_B = torch.sqrt(torch.clamp(1.0 - gamma, min=1e-8))
    return l1_B, l3_B


def detect_deformation_mode(l1, l2, l3, F=None):
    """Detect deformation mode from principal stretches.

    For incompressible isotropic materials:
    - uniaxial (ut/uc): l2 ≈ l3 (within 5%)
    - equibiaxial (bt): l1 ≈ l2 (within 5%)
    - pure shear (ps): l2 ≈ 1 (within 5%)
    - simple shear (ss): determined by off-diagonal components of F

    Args:
        l1, l2, l3: (batch,) principal stretches
        F: (batch, 3, 3) optional deformation gradients for SS detection

    Returns:
        modes: list of mode strings
    """
    batch = l1.shape[0]
    modes = []

    for i in range(batch):
        l1_i = l1[i].item()
        l2_i = l2[i].item()
        l3_i = l3[i].item()

        # Check simple shear first (off-diagonal components)
        if F is not None:
            Fi = F[i]
            if (abs(Fi[0, 1]) > 1e-6 or abs(Fi[1, 0]) > 1e-6 or
                abs(Fi[0, 2]) > 1e-6 or abs(Fi[2, 0]) > 1e-6):
                modes.append('ss')
                continue

        # Ratios
        r23 = abs(l2_i - l3_i) / max(l2_i, l3_i, 1e-8)
        r12 = abs(l1_i - l2_i) / max(l1_i, l2_i, 1e-8)
        r2_1 = abs(l2_i - 1.0)

        if r12 < 0.05:
            # l1 ≈ l2: equibiaxial
            modes.append('bt')
        elif r23 < 0.05:
            # l2 ≈ l3: uniaxial
            modes.append('ut')
        elif r2_1 < 0.05:
            # l2 ≈ 1: pure shear
            modes.append('ps')
        else:
            # Default
            modes.append('ut')

    return modes


class GICANN(nn.Module):
    """GI-CANN model for hyperelastic constitutive modeling.

    The network has three parallel branches computing different forms
    of the strain energy function ψ(J_α), plus a trainable exponent α.

    Strain energy function branches (Eq. 13 in paper):
    1. Identity:   ψ_id  = w_id  * (J_α - 3)
    2. Exponential: ψ_exp = w_exp2 * [exp(w_exp1 * (J_α - 3)) - 1]
    3. Logarithmic: ψ_ln  = w_ln2  * ln(1 - w_ln1 * (J_α - 3))

    Total: ψ = ψ_id + ψ_exp + ψ_ln
    Cauchy stress is derived by differentiating ψ with respect to F.

    Args:
        config: Configuration dict with keys:
            - mode: 'positive', 'negative', or 'combined'
            - r: L1 regularization coefficient
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        self.mode = config.get('mode', 'positive')
        self.r = config.get('r', 0.01)

        # --- Trainable exponents α ---
        # Initialize near expected values to avoid numerical issues.
        # For rubber: α* ≈ 1.675 (stretch-dominated). For brain: α* ≈ -18 (shear-dominated).
        if self.mode == 'positive':
            self.alpha_plus = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        elif self.mode == 'negative':
            self.alpha_minus = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
        elif self.mode == 'combined':
            self.alpha_plus = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
            self.alpha_minus = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))

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
        # For rubber with α≈2: J-3 up to ~50 → w_exp1 * 50 must be < ~10
        #   → w_exp1 ≈ 0.01-0.05. Use 0.02 as a safe default.
        # For brain: J-3 up to ~0.03 → w_exp1 ≈ 1-10. Use 0.1.
        if self.mode in ('positive', 'combined'):
            self.w_exp1_plus = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
            self.w_exp2_plus = nn.Parameter(torch.tensor(5.0, dtype=torch.float64))
        if self.mode in ('negative', 'combined'):
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
        """Return the weights w (excluding α) for L1 regularization."""
        return [p for n, p in self.named_parameters() if not n.startswith('alpha')]

    def get_parameters_dict(self):
        """Return discovered parameters as a dict."""
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

    def _dpsi_dJ(self, J_minus_3, branch):
        """Compute dψ/dJ for a specific branch.

        Args:
            J_minus_3: tensor of (J - 3) values
            branch: 'plus' or 'minus'

        Returns:
            dpsi_dJ: tensor of same shape
        """
        if branch == 'plus':
            dpsi = self.w_id_plus * torch.ones_like(J_minus_3)
            # Clamp exp argument to prevent overflow. The rubber data has J-3 up to ~50
            # for α=2, but α*≈1.675 keeps it to ~4-7. Clamp to 10 for safety.
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

    def _dJ_dF(self, l1, l2, l3, mode, gamma=None, batch_idx=None):
        """Compute dJ_α/dF_{ij} for each deformation mode.

        For the stress formula: σ_ij = α * dψ/dJ * dJ_α/dF_{ij} * F_{ji}
        But since we're computing individual components, we return the factor
        that multiplies (α * dψ/dJ).

        Actually, for the specific stress components we need:
        - Uniaxial: σ_11 = α * dψ/dJ * α * (l1^(α-1) * dl1/dλ1) * (l1 - l3^α/l1^(α-1))
                   = α * dψ/dJ * (l1^α - l3^α)
        - Equibiaxial: σ_11 = σ_22 = α * dψ/dJ * (l1^α - l3^α)
        - Simple shear: σ_12 = α/2^α * dψ/dJ * [(γ+√)/2]^α-1 * [(γ+√) - (-γ+√)] / √
                      = α/2^α * dψ/dJ * [(γ+√)/2]^(α-1) * (2γ) / √
        where γ is the engineering shear strain.
        """
        return None  # We compute stress directly

    def compute_stress(self, F, gamma_override=None):
        """Compute Cauchy stress tensor from deformation gradient.

        Args:
            F: (batch, 3, 3) deformation gradient tensors
            gamma_override: optional (batch,) tensor of shear strains for SS mode

        Returns:
            sigma: (batch, 3, 3) Cauchy stress tensors
        """
        batch = F.shape[0]
        l1, l2, l3 = get_principal_stretches(F)
        modes = detect_deformation_mode(l1, l2, l3, F)

        sigma = torch.zeros((batch, 3, 3), dtype=F.dtype, device=F.device)

        # Group samples by mode
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
            F_m = F[indices_t]

            if mode == 'ss' and gamma_override is not None:
                gamma_m = gamma_override[indices_t]
            elif mode == 'ss':
                # For simple shear, extract gamma from the off-diagonal of F
                # For proper simple shear: F = [1 γ 0; 0 1 0; 0 0 1]
                # But we may have a general F with off-diagonal components
                gamma_m = F_m[:, 0, 1] + F_m[:, 1, 0]
                gamma_m = gamma_m / 2  # engineering strain
            else:
                gamma_m = None

            if mode in ('ut', 'uc', 'ps'):
                sigma_batch = self._stress_uniaxial(l1_m, l2_m, l3_m, mode)
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

    def _stress_uniaxial(self, l1, l2, l3, mode):
        """Compute Cauchy stress for uniaxial deformation.

        Paper Eq. (15):
        σ_11 = α (∂ψ/∂J_α) (λ_1^α - λ_1^(-α/2))
        """
        batch = l1.shape[0]
        sigma = torch.zeros((batch, 3, 3), dtype=l1.dtype, device=l1.device)

        # Compute J - 3 and dψ/dJ for each branch (fully tensor-based)
        if self.mode in ('positive', 'combined'):
            J_minus_3_plus = l1 ** self.alpha_plus + l2 ** self.alpha_plus + l3 ** self.alpha_plus - 3.0
            dpsi_plus = self._dpsi_dJ(J_minus_3_plus, 'plus')

        if self.mode in ('negative', 'combined'):
            J_minus_3_minus = (l1 ** self.alpha_minus + l2 ** self.alpha_minus +
                               l3 ** self.alpha_minus - 3.0)
            dpsi_minus = self._dpsi_dJ(J_minus_3_minus, 'minus')

        if self.mode == 'combined':
            s11 = (self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus) +
                   self.alpha_minus * dpsi_minus * (l1 ** self.alpha_minus - l3 ** self.alpha_minus))
        elif self.mode == 'positive':
            s11 = self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus)
        else:  # negative
            s11 = self.alpha_minus * dpsi_minus * (l1 ** self.alpha_minus - l3 ** self.alpha_minus)

        sigma[:, 0, 0] = s11
        return sigma

    def _stress_equibiaxial(self, l1, l2, l3):
        """Compute Cauchy stress for equibiaxial deformation.

        Paper Eq. (16):
        σ_11 = α (∂ψ/∂J_α) (λ_1^α - λ_1^(-2α))
        """
        batch = l1.shape[0]
        sigma = torch.zeros((batch, 3, 3), dtype=l1.dtype, device=l1.device)

        # Compute J - 3 and dψ/dJ for each branch (fully tensor-based)
        if self.mode in ('positive', 'combined'):
            J_minus_3_plus = l1 ** self.alpha_plus + l2 ** self.alpha_plus + l3 ** self.alpha_plus - 3.0
            dpsi_plus = self._dpsi_dJ(J_minus_3_plus, 'plus')

        if self.mode in ('negative', 'combined'):
            J_minus_3_minus = (l1 ** self.alpha_minus + l2 ** self.alpha_minus +
                               l3 ** self.alpha_minus - 3.0)
            dpsi_minus = self._dpsi_dJ(J_minus_3_minus, 'minus')

        if self.mode == 'combined':
            s11 = (self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus) +
                   self.alpha_minus * dpsi_minus * (l1 ** self.alpha_minus - l3 ** self.alpha_minus))
        elif self.mode == 'positive':
            s11 = self.alpha_plus * dpsi_plus * (l1 ** self.alpha_plus - l3 ** self.alpha_plus)
        else:  # negative
            s11 = self.alpha_minus * dpsi_minus * (l1 ** self.alpha_minus - l3 ** self.alpha_minus)

        sigma[:, 0, 0] = s11
        sigma[:, 1, 1] = s11
        return sigma

    def _stress_simple_shear(self, l1, l2, l3, gamma):
        """Compute Cauchy stress for simple shear deformation.

        Paper Eq. (18):
        λ_± = (sqrt(γ² + 4) ± γ) / 2
        J_α = λ_+^α + 1 + λ_-^α
        σ_12 = (α / 2^α) (∂ψ/∂J_α)
              * [ (γ + sqrt(γ²+4))^α - (-γ + sqrt(γ²+4))^α ] / sqrt(γ²+4)
        """
        batch = l1.shape[0]
        sigma = torch.zeros((batch, 3, 3), dtype=l1.dtype, device=l1.device)

        gamma = gamma.view(batch)
        sqrt_term = torch.sqrt(gamma ** 2 + 4.0)
        lambda_plus = 0.5 * (sqrt_term + gamma)
        lambda_minus = 0.5 * (sqrt_term - gamma)
        ones = torch.ones(batch, dtype=l1.dtype, device=l1.device)

        if self.mode in ('positive', 'combined'):
            J_minus_3_plus = lambda_plus ** self.alpha_plus + ones + lambda_minus ** self.alpha_plus - 3.0
            dpsi_plus = self._dpsi_dJ(J_minus_3_plus, 'plus')

        if self.mode in ('negative', 'combined'):
            J_minus_3_minus = lambda_plus ** self.alpha_minus + ones + lambda_minus ** self.alpha_minus - 3.0
            dpsi_minus = self._dpsi_dJ(J_minus_3_minus, 'minus')

        if self.mode == 'combined':
            s12_pos = (
                (self.alpha_plus / (2.0 ** self.alpha_plus)) * dpsi_plus *
                ((gamma + sqrt_term) ** self.alpha_plus - (-gamma + sqrt_term) ** self.alpha_plus) / sqrt_term
            )
            s12_neg = (
                (self.alpha_minus / (2.0 ** self.alpha_minus)) * dpsi_minus *
                ((gamma + sqrt_term) ** self.alpha_minus - (-gamma + sqrt_term) ** self.alpha_minus) / sqrt_term
            )
            s12 = s12_pos + s12_neg
        elif self.mode == 'positive':
            s12 = (
                (self.alpha_plus / (2.0 ** self.alpha_plus)) * dpsi_plus *
                ((gamma + sqrt_term) ** self.alpha_plus - (-gamma + sqrt_term) ** self.alpha_plus) / sqrt_term
            )
        else:  # negative
            s12 = (
                (self.alpha_minus / (2.0 ** self.alpha_minus)) * dpsi_minus *
                ((gamma + sqrt_term) ** self.alpha_minus - (-gamma + sqrt_term) ** self.alpha_minus) / sqrt_term
            )

        sigma[:, 0, 1] = s12
        sigma[:, 1, 0] = s12
        return sigma

    def forward(self, F, gamma_override=None):
        """Forward pass: predict Cauchy stress from deformation gradient.

        Args:
            F: (batch, 3, 3) deformation gradient tensors
            gamma_override: optional (batch,) shear strains for SS mode

        Returns:
            sigma: (batch, 3, 3) predicted Cauchy stress tensors
        """
        return self.compute_stress(F, gamma_override=gamma_override)

    def loss(self, F, S_target, gamma=None):
        """Compute loss = MSE(stress) + L1 regularization."""
        S_pred = self.compute_stress(F, gamma_override=gamma)
        mse = torch.mean((S_pred - S_target) ** 2)
        l1_loss = self.r * sum(torch.sum(torch.abs(w)) for w in self.get_strain_energy_weights())
        return mse + l1_loss, mse

    def predict_stress(self, F, gamma=None):
        """Predict Cauchy stress as numpy array."""
        with torch.no_grad():
            sigma = self.compute_stress(F, gamma_override=gamma)
        return sigma.detach().numpy()
