#!/usr/bin/env python3
"""Stronger-model variant: principal-stretch CANN (paper Table 1 / Table 4, "PS-CANN").

Identity-branch strain energy on a dense set of generalized invariants with fixed
integer exponents k in {-30..-1, 1..30}:  psi = sum_k w_k (J_k - 3),  w_k >= 0,
trained with the paper's loss (MSE over the training curves + r * sum|w|, r = 0.01).
The problem is convex; it is solved with accelerated projected gradient followed by
an exact active-set polish.  Only the brain slots are written (rubber stretches up
to 7.25 make k = +-30 numerically meaningless; the paper only reports PS-CANN on brain).
Selection never touches the held-out mode.  Run with cwd = sandbox root.  NumPy only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SANDBOX = Path.cwd()
sys.path.insert(0, str(SANDBOX))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from brute_alpha import CURVES, TRAIN, kin  # noqa: E402

R = 0.01


def design(material, modes, ks):
    cols, ys = [], []
    for m in modes:
        arr = CURVES[material][m]
        _, g = kin(m, arr[:, 0], ks[:, None])
        cols.append((ks[:, None] * g).T)
        ys.append(arr[:, 1])
    return np.concatenate(cols, 0), np.concatenate(ys)


def nn_lasso(A, y, iters=200000):
    n = len(y)
    G, b = A.T @ A, A.T @ y
    L = 2.0 / n * np.linalg.eigvalsh(G).max()
    w = np.zeros(A.shape[1]); v = w.copy(); t = 1.0
    for _ in range(iters):
        grad = 2.0 / n * (G @ v - b) + R
        w_new = np.maximum(v - grad / L, 0.0)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        v = w_new + (t - 1) / t_new * (w_new - w)
        w, t = w_new, t_new
    # exact polish on the active set (KKT: G_SS w_S = b_S - n r / 2, w_S > 0)
    for _ in range(50):
        S = w > 1e-14
        if not S.any():
            break
        wS = np.linalg.lstsq(G[np.ix_(S, S)], b[S] - n * R / 2, rcond=None)[0]
        if np.all(wS > 0):
            w2 = np.zeros_like(w); w2[S] = wS
            grad = 2.0 / n * (G @ w2 - b) + R
            if np.all(grad[~S] >= -1e-10):
                w = w2
            break
        w[np.where(S)[0][wS <= 0]] = 0.0
    obj = float(np.mean((A @ w - y) ** 2) + R * w.sum())
    return w, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="brain_negative")
    args = ap.parse_args()
    ks = np.array([k for k in range(-30, 31) if k != 0], dtype=float)
    for slot in [s.strip() for s in args.slots.split(",") if s.strip()]:
        material, mode = slot.split("_")
        A, y = design(material, TRAIN[material], ks)
        w, obj = nn_lasso(A, y)
        preds = {}
        for m in CURVES[material]:
            Am, _ = design(material, [m], ks)
            preds[m] = [float(v) for v in Am @ w]
        active = {int(k): float(v) for k, v in zip(ks, w) if v > 1e-12}
        dominant = max(active, key=lambda k: active[k] * abs(k)) if active else 0
        payload = {
            "schema": "hyperelastic_observed_stress_predictions_v1",
            "paper": "2508.12063", "material": material, "mode": mode, "model": "GI-CANN",
            "method_note": "PS-CANN: identity branch on J_k, k in {-30..30}\\{0}, w>=0, L1 r=0.01, trained on training curves only",
            "alpha_star": float(dominant), "params": {"active_terms": active, "train_loss": obj},
            "predictions": preds,
        }
        path = SANDBOX / "outputs" / "predictions" / f"{material}_{mode}_gicann.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{path.name}: train_loss={obj:.6g} active={active}", flush=True)


if __name__ == "__main__":
    main()
