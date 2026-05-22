#!/usr/bin/env python3
"""Load rubber and brain tissue stress-strain data from the paper (Table A.5).

Rubber (Treloar 1944): uniaxial tension, equibiaxial tension, pure shear. Units: MPa.
Brain cortex (Budday et al. 2017): uniaxial tension, uniaxial compression, simple shear. Units: kPa.

Deformation modes:
- ut (uniaxial tension): F = diag(λ1, λ1^(-1/2), λ1^(-1/2))
- uc (uniaxial compression): same as ut with λ1 < 1
- bt (equibiaxial tension): F = diag(λ1, λ1, λ1^(-2))
- ps (pure shear): F = diag(λ1, 1, λ1^(-1))
- ss (simple shear): F = [1 γ; 0 1] with det=1 (2D), extended to 3D
"""

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Placeholder so target arrays can be imported without PyTorch."""

        pass


# Rubber data from Treloar (1944) — Table A.5, units: MPa
RUBBER_UT = np.array([
    [1.00, 0.00],
    [1.13, 0.15],
    [1.41, 0.47],
    [1.89, 0.98],
    [2.45, 1.67],
    [3.06, 2.65],
    [3.62, 3.83],
    [4.06, 5.03],
    [4.82, 7.71],
    [5.41, 10.50],
    [5.79, 13.30],
    [6.23, 16.70],
    [6.96, 26.40],
    [7.25, 32.60],
], dtype=np.float64)

RUBBER_BT = np.array([
    [1.00, 0.00],
    [1.08, 0.17],
    [1.15, 0.26],
    [1.21, 0.40],
    [1.32, 0.58],
    [1.43, 0.74],
    [1.70, 1.12],
    [1.95, 1.51],
    [2.50, 2.42],
    [3.04, 3.82],
    [3.44, 5.07],
    [4.03, 7.95],
    [4.26, 9.50],
    [4.45, 10.90],
], dtype=np.float64)

RUBBER_PS = np.array([
    [1.00, 0.00],
    [1.05, 0.10],
    [1.13, 0.20],
    [1.20, 0.30],
    [1.33, 0.40],
    [1.45, 0.60],
    [1.86, 1.10],
    [2.40, 1.80],
    [2.99, 2.80],
    [3.50, 4.00],
    [3.98, 5.20],
    [4.39, 6.50],
    [4.72, 7.80],
    [4.99, 9.10],
], dtype=np.float64)

# Brain cortex data from Budday et al. (2017) — Table A.5, units: kPa
BRAIN_UT = np.array([
    [1.00, 0.00],
    [1.01, 0.03],
    [1.01, 0.05],
    [1.02, 0.07],
    [1.03, 0.08],
    [1.03, 0.10],
    [1.04, 0.12],
    [1.05, 0.16],
    [1.06, 0.19],
    [1.06, 0.21],
    [1.07, 0.24],
    [1.08, 0.27],
    [1.09, 0.33],
    [1.09, 0.37],
    [1.10, 0.45],
], dtype=np.float64)

BRAIN_UC = np.array([
    [1.00, 0.00],
    [0.99, -0.03],
    [0.98, -0.08],
    [0.97, -0.13],
    [0.97, -0.18],
    [0.96, -0.23],
    [0.96, -0.28],
    [0.95, -0.33],
    [0.94, -0.40],
    [0.94, -0.47],
    [0.93, -0.53],
    [0.92, -0.61],
    [0.91, -0.70],
    [0.91, -0.80],
    [0.90, -1.03],
], dtype=np.float64)

BRAIN_SS = np.array([
    [0.00, 0.00],
    [0.01, 0.01],
    [0.03, 0.03],
    [0.04, 0.05],
    [0.05, 0.06],
    [0.06, 0.08],
    [0.08, 0.13],
    [0.10, 0.15],
    [0.11, 0.18],
    [0.13, 0.22],
    [0.14, 0.26],
    [0.15, 0.31],
    [0.16, 0.37],
    [0.18, 0.45],
    [0.20, 0.54],
], dtype=np.float64)


def build_F(mode, l1, l2=None, l3=None, gamma=None):
    """Build a deformation gradient tensor.

    Args:
        mode: 'ut', 'uc', 'bt', 'ps', or 'ss'
        l1, l2, l3: principal stretches (λ values)
        gamma: shear strain for simple shear

    Returns:
        F: (3, 3) numpy array
    """
    F = np.eye(3)

    if mode in ('ut', 'uc'):
        # Uniaxial: F = diag(λ1, λ1^(-1/2), λ1^(-1/2))
        F[0, 0] = l1
        F[1, 1] = l1 ** (-0.5)
        F[2, 2] = l1 ** (-0.5)

    elif mode == 'bt':
        # Equibiaxial: F = diag(λ1, λ1, λ1^(-2))
        F[0, 0] = l1
        F[1, 1] = l1
        F[2, 2] = l1 ** (-2)

    elif mode == 'ps':
        # Pure shear: F = diag(λ1, 1, λ1^(-1))
        F[0, 0] = l1
        F[1, 1] = 1.0
        F[2, 2] = l1 ** (-1)

    elif mode == 'ss':
        # Simple shear: F = [1 γ 0; 0 1 0; 0 0 1]
        # The shear is in the 1-2 plane
        F[0, 0] = 1.0
        F[0, 1] = gamma
        F[1, 0] = 0.0
        F[1, 1] = 1.0
        F[2, 2] = 1.0

    return F


def build_stress(mode, s11=None, s22=None, s12=None):
    """Build a Cauchy stress tensor.

    Args:
        mode: 'ut', 'uc', 'bt', 'ps', or 'ss'
        s11, s22: normal stress components
        s12: shear stress component

    Returns:
        S: (3, 3) numpy array
    """
    S = np.zeros((3, 3))

    if mode in ('ut', 'uc', 'ps'):
        S[0, 0] = s11

    elif mode == 'bt':
        S[0, 0] = s11
        S[1, 1] = s11  # σ22 = σ11 for equibiaxial

    elif mode == 'ss':
        S[0, 1] = s12
        S[1, 0] = s12

    return S


class HyperelasticDataset(Dataset):
    """Dataset for hyperelastic constitutive modeling.

    Args:
        deformation_modes: list of (name, data_array, mode) tuples
            For ss mode: data_array has columns [gamma, stress]
            For other modes: data_array has columns [lambda, stress]
    """

    def __init__(self, deformation_modes):
        if torch is None:
            raise RuntimeError("PyTorch is required to instantiate HyperelasticDataset")
        self.F_list = []
        self.S_list = []
        self.gamma_list = []  # shear strain for SS mode
        self.mode_list = []   # deformation mode name
        self.mode_indices = {}  # map mode name to indices
        self.all_gamma = []  # gamma values per sample (0 for non-SS)

        idx = 0
        for name, data, mode in deformation_modes:
            n = len(data)
            F_batch = []
            S_batch = []
            gamma_batch = []

            for i in range(n):
                if mode in ('ut', 'uc'):
                    l1 = data[i, 0]
                    s11 = data[i, 1]
                    F_i = build_F(mode, l1)
                    S_i = build_stress(mode, s11)
                    gamma_batch.append(0.0)

                elif mode == 'bt':
                    l1 = data[i, 0]
                    s11 = data[i, 1]
                    F_i = build_F(mode, l1)
                    S_i = build_stress(mode, s11)
                    gamma_batch.append(0.0)

                elif mode == 'ps':
                    l1 = data[i, 0]
                    s11 = data[i, 1]
                    F_i = build_F(mode, l1)
                    S_i = build_stress(mode, s11)
                    gamma_batch.append(0.0)

                elif mode == 'ss':
                    gamma_val = data[i, 0]  # shear strain
                    s12 = data[i, 1]       # shear stress
                    F_i = build_F(mode, None, gamma=gamma_val)
                    S_i = build_stress(mode, s12=s12)
                    gamma_batch.append(gamma_val)

                F_batch.append(F_i)
                S_batch.append(S_i)

            self.mode_indices[name] = list(range(idx, idx + n))
            self.F_list.extend(F_batch)
            self.S_list.extend(S_batch)
            self.gamma_list.extend(gamma_batch)
            self.mode_list.extend([name] * n)
            idx += n

        self.F = torch.tensor(np.stack(self.F_list), dtype=torch.float64)
        self.S = torch.tensor(np.stack(self.S_list), dtype=torch.float64)
        self.gamma = torch.tensor(self.gamma_list, dtype=torch.float64)
        self.n_samples = len(self.F)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.F[idx], self.S[idx], self.gamma[idx]


def get_rubber_dataset():
    """Rubber dataset: train on ut+bt, test on ps."""
    train = [
        ('ut', RUBBER_UT, 'ut'),
        ('bt', RUBBER_BT, 'bt'),
    ]
    test = [
        ('ps', RUBBER_PS, 'ps'),
    ]
    return HyperelasticDataset(train), HyperelasticDataset(test)


def get_brain_dataset():
    """Brain tissue dataset: train on ut+uc, test on ss."""
    train = [
        ('ut', BRAIN_UT, 'ut'),
        ('uc', BRAIN_UC, 'uc'),
    ]
    test = [
        ('ss', BRAIN_SS, 'ss'),
    ]
    return HyperelasticDataset(train), HyperelasticDataset(test)


def compute_r2(pred, target):
    """Compute R² coefficient of determination.

    Args:
        pred: (n,) or (n, ...) predicted values
        target: (n,) or (n, ...) target values

    Returns:
        float R² value
    """
    pred = pred.flatten()
    target = target.flatten()
    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if np.abs(ss_res) < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def extract_observed_stress(stress, mode_names):
    """Extract the measured stress component for each sample.

    Table A.5 provides one observed stress component per deformation state:
    - ut / uc / bt / ps: sigma_11
    - ss: sigma_12

    Args:
        stress: (n, 3, 3) numpy array or torch tensor
        mode_names: list of deformation mode strings with length n

    Returns:
        (n,) array/tensor of observed stress components
    """
    if len(mode_names) == 0:
        return stress[..., 0, 0][..., :0]

    if torch is not None and torch.is_tensor(stress):
        observed = torch.empty(len(mode_names), dtype=stress.dtype, device=stress.device)
        for i, mode_name in enumerate(mode_names):
            if mode_name == 'ss':
                observed[i] = stress[i, 0, 1]
            else:
                observed[i] = stress[i, 0, 0]
        return observed

    observed = np.empty(len(mode_names), dtype=stress.dtype)
    for i, mode_name in enumerate(mode_names):
        if mode_name == 'ss':
            observed[i] = stress[i, 0, 1]
        else:
            observed[i] = stress[i, 0, 0]
    return observed


if __name__ == '__main__':
    print("=== Rubber Dataset ===")
    train_ds, test_ds = get_rubber_dataset()
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Mode indices (train): {train_ds.mode_indices}")
    print(f"Mode indices (test): {test_ds.mode_indices}")
    F, S, g = train_ds[0]
    print(f"Sample F shape: {F.shape}, S shape: {S.shape}, gamma: {g.item():.2f}")

    print("\n=== Brain Dataset ===")
    train_ds, test_ds = get_brain_dataset()
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Mode indices (train): {train_ds.mode_indices}")
    print(f"Mode indices (test): {test_ds.mode_indices}")

    # Test SS mode
    ss_idx = test_ds.mode_indices['ss'][0]
    F_ss, S_ss, gamma_ss = test_ds[ss_idx]
    print(f"SS sample F:\n{F_ss}")
    print(f"SS sample S:\n{S_ss}")
    print(f"SS gamma: {gamma_ss.item():.3f}")
