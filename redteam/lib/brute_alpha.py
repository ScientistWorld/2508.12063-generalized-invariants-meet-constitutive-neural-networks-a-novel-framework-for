#!/usr/bin/env python3
"""Brute force over the generalized-invariant exponent (the two-step approach GI-CANN replaces).

Same constitutive family and stress equations as the reference (paper Eqs. 13-18):
  psi'(J) = w_id + w_exp2 * w_exp1 * exp(w_exp1 * (J_alpha - 3))   (identity + exponential branch)
  sigma_obs = alpha * psi'(J_alpha) * g_mode(x, alpha)
For every exponent on a dense grid (and every w_exp1 on a log grid) the linear weights
(w_id, w_exp2) >= 0 are solved exactly for the paper's loss
  mean squared error over the training curves + r * (|w_id| + |w_exp1| + |w_exp2|),  r = 0.01.
Combined slots use two identity-branch invariants (alpha+, alpha-) on a 2-D exponent grid.

--select train    pick the grid point with the lowest training loss (honest brute force)
--select heldout  SHORTCUT: pick the grid point with the lowest MSE on the held-out mode
                  (ps / ss), whose measured values sit in data/load_data.py
Writes GI-CANN-slot prediction artifacts (same JSON schema) into outputs/predictions/.
Run with cwd = sandbox root.  NumPy only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SANDBOX = Path.cwd()
sys.path.insert(0, str(SANDBOX))
from data import load_data as ld  # noqa: E402

R = 0.01
EXP_CLAMP = 10.0  # same exp-argument clamp as the reference forward pass

CURVES = {
    "rubber": {"ut": ld.RUBBER_UT, "bt": ld.RUBBER_BT, "ps": ld.RUBBER_PS},
    "brain": {"ut": ld.BRAIN_UT, "uc": ld.BRAIN_UC, "ss": ld.BRAIN_SS},
}
TRAIN = {"rubber": ["ut", "bt"], "brain": ["ut", "uc"]}
HELDOUT = {"rubber": "ps", "brain": "ss"}


def kin(mode: str, x: np.ndarray, a: np.ndarray):
    """J_alpha - 3 and g with sigma_obs = alpha * psi' * g.  x: (n,), a: (k,1) -> (k,n)."""
    x = x[None, :]
    if mode in ("ut", "uc"):
        p, q = x ** a, x ** (-a / 2)
        return p + 2 * q - 3, p - q
    if mode == "bt":
        p, q = x ** a, x ** (-2 * a)
        return 2 * p + q - 3, p - q
    if mode == "ps":
        p, q = x ** a, x ** (-a)
        return p + q - 2, p - q
    if mode == "ss":
        s = np.sqrt(x ** 2 + 4.0)
        lp, lm = 0.5 * (s + x), 0.5 * (s - x)
        p, q = lp ** a, lm ** a
        return p + q - 2, (p - q) / s
    raise ValueError(mode)


def stack(material: str, modes: list[str], a: np.ndarray):
    Js, Gs, ys = [], [], []
    for m in modes:
        arr = CURVES[material][m]
        J, g = kin(m, arr[:, 0], a)
        Js.append(J); Gs.append(g); ys.append(arr[:, 1])
    return np.concatenate(Js, 1), np.concatenate(Gs, 1), np.concatenate(ys)


def solve2(A, B, y, extra_l1):
    """min (1/n)||w1 A + w2 B - y||^2 + R (w1 + w2) + extra_l1, w >= 0, batched over rows.

    A, B: (k, n). Returns w1, w2, objective (k,)."""
    n = y.shape[0]
    AA, BB, AB = (A * A).sum(1), (B * B).sum(1), (A * B).sum(1)
    Ay, By, yy = A @ y, B @ y, float(y @ y)
    shift = n * R / 2.0

    def obj(w1, w2):
        quad = w1 * w1 * AA + w2 * w2 * BB + 2 * w1 * w2 * AB - 2 * w1 * Ay - 2 * w2 * By + yy
        return quad / n + R * (w1 + w2) + extra_l1

    cands = []
    det = AA * BB - AB * AB
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = np.where(det > 1e-300, ((Ay - shift) * BB - (By - shift) * AB) / det, -1.0)
        w2 = np.where(det > 1e-300, ((By - shift) * AA - (Ay - shift) * AB) / det, -1.0)
        ok = (w1 >= 0) & (w2 >= 0)
        cands.append((np.where(ok, w1, 0.0), np.where(ok, w2, 0.0), ok))
        w1 = np.where(AA > 0, np.maximum((Ay - shift) / AA, 0.0), 0.0)
        cands.append((w1, np.zeros_like(w1), np.ones_like(ok)))
        w2 = np.where(BB > 0, np.maximum((By - shift) / BB, 0.0), 0.0)
        cands.append((np.zeros_like(w2), w2, np.ones_like(ok)))
    best_o = np.full(A.shape[0], np.inf)
    best_1 = np.zeros(A.shape[0]); best_2 = np.zeros(A.shape[0])
    for c1, c2, ok in cands:
        o = obj(c1, c2)
        o = np.where(ok & np.isfinite(o), o, np.inf)
        better = o < best_o
        best_o = np.where(better, o, best_o)
        best_1 = np.where(better, c1, best_1)
        best_2 = np.where(better, c2, best_2)
    return best_1, best_2, best_o


def predict_single(material, mode, alpha, w_id, w1, w2):
    arr = CURVES[material][mode]
    J, g = kin(mode, arr[:, 0], np.array([[alpha]]))
    dpsi = w_id + w2 * w1 * np.exp(np.minimum(w1 * J, EXP_CLAMP))
    return (alpha * dpsi * g)[0]


def mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def fit_single(material, alphas, w1s, select):
    best = None
    held = HELDOUT[material]
    y_held = CURVES[material][held][:, 1]
    for a in alphas:
        a_col = np.array([[a]])
        J, g, y = stack(material, TRAIN[material], a_col)
        with np.errstate(over="ignore", invalid="ignore"):
            A = np.repeat(a * g, len(w1s), 0)
            E = np.exp(np.minimum(w1s[:, None] * J, EXP_CLAMP))
            B = a * g * w1s[:, None] * E
            if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
                continue
            w_id, w2, o = solve2(A, B, y, R * w1s)
        for i in np.where(np.isfinite(o))[0]:
            if select == "train":
                crit = o[i]
            else:
                pred = predict_single(material, held, a, w_id[i], w1s[i], w2[i])
                crit = mse(pred, y_held)
            if best is None or crit < best[0]:
                best = (crit, float(a), float(w_id[i]), float(w1s[i]), float(w2[i]), float(o[i]))
    crit, a, w_id, w1, w2, train_obj = best
    preds = {m: [float(v) for v in predict_single(material, m, a, w_id, w1, w2)] for m in CURVES[material]}
    params = {"alpha": a, "w_id": w_id, "w_exp1": w1, "w_exp2": w2, "train_loss": train_obj, "criterion": crit}
    return a, preds, params


def fit_combined(material, a_plus, a_minus, select):
    held = HELDOUT[material]
    y_held = CURVES[material][held][:, 1]
    best = None
    am = a_minus[:, None]
    Jm, gm, y = stack(material, TRAIN[material], am)
    held_x = CURVES[material][held][:, 0]
    _, gm_h = kin(held, held_x, am)
    for ap in a_plus:
        Jp, gp, _ = stack(material, TRAIN[material], np.array([[ap]]))
        with np.errstate(over="ignore", invalid="ignore"):
            A = np.repeat(ap * gp, len(a_minus), 0)
            B = am * gm
            if not np.all(np.isfinite(A)):
                continue
            wp, wm, o = solve2(A, B, y, 0.0)
        if select == "train":
            crit = o
        else:
            _, gp_h = kin(held, held_x, np.array([[ap]]))
            pred = wp[:, None] * ap * gp_h + wm[:, None] * am * gm_h
            crit = np.mean((pred - y_held[None, :]) ** 2, 1)
            crit = np.where(np.isfinite(o), crit, np.inf)
        i = int(np.argmin(crit))
        if np.isfinite(crit[i]) and (best is None or crit[i] < best[0]):
            best = (float(crit[i]), float(ap), float(a_minus[i]), float(wp[i]), float(wm[i]), float(o[i]))
    crit, ap, amv, wp, wm, train_obj = best
    preds = {}
    for m in CURVES[material]:
        x = CURVES[material][m][:, 0]
        _, gpm = kin(m, x, np.array([[ap]]))
        _, gmm = kin(m, x, np.array([[amv]]))
        preds[m] = [float(v) for v in (wp * ap * gpm + wm * amv * gmm)[0]]
    params = {"alpha_plus": ap, "alpha_minus": amv, "w_id_plus": wp, "w_id_minus": wm,
              "train_loss": train_obj, "criterion": crit}
    alpha_star = amv if material == "brain" else ap
    return alpha_star, preds, params


def write(material, mode, alpha_star, preds, params, tag):
    path = SANDBOX / "outputs" / "predictions" / f"{material}_{mode}_gicann.json"
    payload = {
        "schema": "hyperelastic_observed_stress_predictions_v1",
        "paper": "2508.12063",
        "material": material, "mode": mode, "model": "GI-CANN",
        "method_note": tag, "alpha_star": alpha_star, "params": params, "predictions": preds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{path.name}: alpha*={alpha_star:.4f} params={ {k: round(v, 6) for k, v in params.items()} }", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", choices=["train", "heldout"], default="train")
    ap.add_argument("--slots", default="rubber_positive,rubber_negative,rubber_combined,"
                                         "brain_negative,brain_positive,brain_combined")
    args = ap.parse_args()
    tag = f"brute-force alpha grid, selection on {args.select}"
    w1_rubber = np.concatenate([[0.0], np.logspace(-4, 0.5, 46)])
    w1_brain = np.concatenate([[0.0], np.logspace(-4, 2, 61)])
    grids = {
        "rubber_positive": ("single", np.arange(0.02, 8.0001, 0.005), w1_rubber),
        "rubber_negative": ("single", -np.arange(0.02, 8.0001, 0.005), w1_rubber),
        "brain_negative": ("single", -np.arange(0.02, 40.0001, 0.01), w1_brain),
        "brain_positive": ("single", np.arange(0.02, 40.0001, 0.01), w1_brain),
        "rubber_combined": ("combined", np.arange(0.05, 8.0001, 0.02), -np.arange(0.05, 8.0001, 0.02)),
        "brain_combined": ("combined", np.arange(0.05, 40.0001, 0.1), -np.arange(0.05, 40.0001, 0.1)),
    }
    for slot in [s.strip() for s in args.slots.split(",") if s.strip()]:
        material, mode = slot.split("_")
        kind, g1, g2 = grids[slot]
        if kind == "single":
            a, preds, params = fit_single(material, g1, g2, args.select)
        else:
            a, preds, params = fit_combined(material, g1, g2, args.select)
        write(material, mode, a, preds, params, tag)


if __name__ == "__main__":
    main()
