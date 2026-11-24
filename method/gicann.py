#!/usr/bin/env python3
"""GI-CANN: generalized-invariant constitutive artificial neural network (paper 2508.12063).

Pure-Python implementation that follows the authors' own notebooks
(LivingMatterLab/CANN @ d3caca0f, GICANN/*.ipynb):

  * Generalized invariants J_alpha = l1^alpha + l2^alpha + l3^alpha, one per exponent in S. Each
    exponent is a trainable variable a_k >= 0 (Keras NonNeg) with a fixed sign, alpha_k = s_k a_k
    (positive family: s = +1, negative family: s = -1, combined: one of each).
  * First layer: J_alpha - 3. Second layer, per invariant: identity, exp(.) - 1 and -ln(1 - .)
    with inner weights w11, w21, w31; output layer wx2 (3 weights per invariant) - Eq. (13):
        psi = sum_k wx2_k1 w11_k (J_k - 3) + wx2_k2 [exp(w21_k (J_k - 3)) - 1]
                    + wx2_k3 [-ln(1 - w31_k (J_k - 3))]
    Every weight is constrained non-negative (Keras NonNeg).
  * Cauchy stress from psi for uniaxial tension/compression, equibiaxial tension, pure shear and
    simple shear (Eqs. 15-18; for simple shear J_alpha is evaluated with the stretches raised to
    alpha, l+^alpha + 1 + l-^alpha).
  * Loss (Eq. 14 as implemented in the notebooks): sum over training modes of the mean squared
    stress error, plus r * sum |w| over all weights (r = 0.01, exponents not regularized).
  * Adam (lr 1e-3), mini-batches of 12 points per mode, reshuffled every epoch; 30,000 epochs
    (40,000 for the brain two-invariant model); early stopping with patience 3,000 on the epoch's
    training loss and restoration of the best epoch's weights (Keras EarlyStopping +
    ModelCheckpoint on "loss").
  * Initialization: glorot-normal (truncated) for w11, w31 and wx2, uniform(0, 0.1) for w21, and
    the notebooks' exponent starts (rubber 2.0, or 1.7 / 0.1 for two invariants; brain 5.0, or
    2.0 / 2.0 for two invariants).
  * SI-CANN is the same network with every exponent fixed at 2 (S = {2}, {-2} or {2, -2}).

Convergence: after the paper's mini-batch phase the same objective is minimized further with
full-batch Adam and a learning rate decaying from 1e-3 to 1e-7 (converge(); 20,000 steps), so every
run ends at a stationary point of the paper's loss instead of wherever early stopping happened to
halt on a noisy epoch loss. The objective is flat along an (exponent, weight) valley, and where a
run stops along that valley moves the held-out prediction; finishing the optimization removes that
randomness. (On the cortex data this lands at alpha* = -18.11, w = 0.0092 kPa; the paper reports
-18.016 and 0.009 kPa.)

Two numerical guards keep the strain energy defined where the law is evaluated (the Keras code
has none and would return NaN): after every step w31 is capped so that 1 - w31 (J - 3) >= 1e-3
and w21 so that w21 (J - 3) <= 10 at the most deformed visible input of each mode (without it, the
exponential branch of a rubber law with alpha = -2 starts near exp(50) in equibiaxial tension,
which overflows in the notebooks' float32 and leaves Adam's moment estimates unusable).

The trained network is exported as a declared strain-energy law (terms with exponent, function
and weights) in the schema the evaluator reads.
"""

from __future__ import annotations

import math
import random

R_L1 = 0.01
LR = 1e-3
BETA1, BETA2, EPS = 0.9, 0.999, 1e-7
BATCH = 12
PATIENCE = 3000
A_MAX = 59.0
EXP_ARG_MAX = 10.0  # cap on w21 * (J - 3) at the most deformed visible input
TN_CORR = 0.87962566103423978  # Keras truncated-normal std correction

# Paper notebook settings by material class and exponent family.
PAPER_CONFIG = {
    ("rubber", "positive"): {"signs": [1], "a0": [2.0], "epochs": 30000},
    ("rubber", "negative"): {"signs": [-1], "a0": [2.0], "epochs": 30000},
    ("rubber", "combined"): {"signs": [1, -1], "a0": [1.7, 0.1], "epochs": 30000},
    ("brain", "positive"): {"signs": [1], "a0": [5.0], "epochs": 30000},
    ("brain", "negative"): {"signs": [-1], "a0": [5.0], "epochs": 30000},
    ("brain", "combined"): {"signs": [1, -1], "a0": [2.0, 2.0], "epochs": 40000},
}


class Point:
    __slots__ = ("mode", "x", "y", "L", "s")

    def __init__(self, mode: str, x: float, y: float = 0.0):
        self.mode, self.x, self.y = mode, float(x), float(y)
        if mode == "ss":
            self.s = math.sqrt(self.x * self.x + 4.0)
            self.L = math.log(0.5 * (self.s + self.x))  # ln(l+), l- = 1 / l+
        else:
            self.s = 1.0
            self.L = math.log(self.x)


def kinematics(pt: Point, alpha: float):
    """(J - 3, g, dJ/dalpha, dg/dalpha) with observed stress = alpha * dpsi/dJ * g."""
    L, m = pt.L, pt.mode
    if m == "ut" or m == "uc":
        p, q = math.exp(alpha * L), math.exp(-0.5 * alpha * L)
        g = p - q
        return p + 2.0 * q - 3.0, g, L * g, L * p + 0.5 * L * q
    if m == "bt":
        p, q = math.exp(alpha * L), math.exp(-2.0 * alpha * L)
        g = p - q
        return 2.0 * p + q - 3.0, g, 2.0 * L * g, L * p + 2.0 * L * q
    p, q = math.exp(alpha * L), math.exp(-alpha * L)
    if m == "ps":
        return p + q - 2.0, p - q, L * (p - q), L * (p + q)
    if m == "ss":
        s = pt.s
        return p + q - 2.0, (p - q) / s, L * (p - q), L * (p + q) / s
    raise ValueError(m)


def _tn(rng: random.Random, std: float) -> float:
    while True:
        z = rng.gauss(0.0, 1.0)
        if abs(z) <= 2.0:
            return z * std


class CANN:
    """Parameters theta = [a_0..a_K-1, w11_k, w21_k, w31_k, wx2_k1..3]."""

    def __init__(self, signs, a0, train_alpha=True, seed=0, extremes=None):
        self.K = K = len(signs)
        self.signs = [float(s) for s in signs]
        self.train_alpha = train_alpha
        self.extremes = extremes or []  # most deformed visible input per mode
        rng = random.Random(seed)
        std1 = math.sqrt(2.0 / 2.0) / TN_CORR
        std2 = math.sqrt(2.0 / (3 * K + 1)) / TN_CORR
        th = [float(a) for a in a0]
        th += [_tn(rng, std1) for _ in range(K)]          # w11
        th += [rng.uniform(0.0, 0.1) for _ in range(K)]   # w21
        th += [_tn(rng, std1) for _ in range(K)]          # w31
        th += [_tn(rng, std2) for _ in range(3 * K)]      # wx2
        self.theta = th
        self.project()

    # index helpers
    def ia(self, k): return k
    def i11(self, k): return self.K + k
    def i21(self, k): return 2 * self.K + k
    def i31(self, k): return 3 * self.K + k
    def iv(self, k, j): return 4 * self.K + 3 * k + j

    def alphas(self):
        return [self.signs[k] * self.theta[k] for k in range(self.K)]

    def project(self):
        th = self.theta
        for i in range(len(th)):
            if not th[i] > 0.0:
                th[i] = 0.0
        for k in range(self.K):
            th[k] = min(th[k], A_MAX)
            alpha = self.signs[k] * th[k]
            xmax = 0.0
            for pt in self.extremes:
                X = kinematics(pt, alpha)[0]
                xmax = max(xmax, X)
            if xmax > 0.0:
                th[self.i31(k)] = min(th[self.i31(k)], 0.999 / xmax)
                th[self.i21(k)] = min(th[self.i21(k)], EXP_ARG_MAX / xmax)

    def stress(self, pt: Point) -> float:
        th, s = self.theta, 0.0
        for k in range(self.K):
            alpha = self.signs[k] * th[k]
            X, g, _, _ = kinematics(pt, alpha)
            w11, w21, w31 = th[self.i11(k)], th[self.i21(k)], th[self.i31(k)]
            v0, v1, v2 = th[self.iv(k, 0)], th[self.iv(k, 1)], th[self.iv(k, 2)]
            P = v0 * w11 + v1 * w21 * math.exp(w21 * X) + v2 * w31 / (1.0 - w31 * X)
            s += alpha * P * g
        return s

    def l1(self) -> float:
        return R_L1 * sum(abs(v) for v in self.theta[self.K:])

    def loss_and_grad(self, chunks):
        """chunks: list of point lists (one per training mode). Returns (loss, grad)."""
        th, K = self.theta, self.K
        grad = [0.0] * len(th)
        loss = 0.0
        for pts in chunks:
            n = len(pts)
            if n == 0:
                continue
            for pt in pts:
                cache = []
                sig = 0.0
                for k in range(K):
                    alpha = self.signs[k] * th[k]
                    X, g, dX, dg = kinematics(pt, alpha)
                    w21, w31 = th[2 * K + k], th[3 * K + k]
                    E = math.exp(w21 * X)
                    D = 1.0 - w31 * X
                    if D <= 0.0:
                        return math.inf, grad
                    P = th[4 * K + 3 * k] * th[K + k] + th[4 * K + 3 * k + 1] * w21 * E + th[4 * K + 3 * k + 2] * w31 / D
                    sig += alpha * P * g
                    cache.append((alpha, X, g, dX, dg, E, D, P))
                e = sig - pt.y
                loss += e * e / n
                c = 2.0 * e / n
                for k in range(K):
                    alpha, X, g, dX, dg, E, D, P = cache[k]
                    ag = c * alpha * g
                    i11, i21, i31 = K + k, 2 * K + k, 3 * K + k
                    i0 = 4 * K + 3 * k
                    w11, w21, w31 = th[i11], th[i21], th[i31]
                    v0, v1, v2 = th[i0], th[i0 + 1], th[i0 + 2]
                    grad[i11] += ag * v0
                    grad[i0] += ag * w11
                    grad[i0 + 1] += ag * w21 * E
                    grad[i21] += ag * v1 * E * (1.0 + w21 * X)
                    grad[i0 + 2] += ag * w31 / D
                    grad[i31] += ag * v2 / (D * D)
                    if self.train_alpha:
                        dPdX = v1 * w21 * w21 * E + v2 * w31 * w31 / (D * D)
                        grad[k] += c * self.signs[k] * (P * g + alpha * (dPdX * dX * g + P * dg))
        for i in range(K, len(th)):
            v = th[i]
            loss += R_L1 * abs(v)
            if v > 0.0:
                grad[i] += R_L1
            elif v < 0.0:
                grad[i] -= R_L1
        return loss, grad

    def objective(self, train_pts: dict) -> float:
        """Paper loss on the full training curves (sum of per-mode MSE + L1)."""
        total = 0.0
        for pts in train_pts.values():
            total += sum((self.stress(p) - p.y) ** 2 for p in pts) / len(pts)
        return total + self.l1()

    def export_law(self) -> dict:
        th, K, terms = self.theta, self.K, []
        for k in range(K):
            alpha = self.signs[k] * th[k]
            if abs(alpha) <= 1e-12:
                continue
            w11, w21, w31 = th[K + k], th[2 * K + k], th[3 * K + k]
            v0, v1, v2 = th[4 * K + 3 * k], th[4 * K + 3 * k + 1], th[4 * K + 3 * k + 2]
            if v0 * w11 > 0.0:
                terms.append({"fn": "id", "alpha": alpha, "w": v0 * w11})
            if v1 > 0.0 and w21 > 0.0:
                terms.append({"fn": "exp", "alpha": alpha, "w1": w21, "w2": v1})
            if v2 > 0.0 and w31 > 0.0:
                terms.append({"fn": "ln", "alpha": alpha, "w1": w31, "w2": v2})
        return {"terms": terms}


def train(model: CANN, train_pts: dict, epochs: int, seed: int = 0, patience: int = PATIENCE,
          lr: float = LR, batch: int = BATCH, log_every: int = 0, log=print) -> dict:
    """Keras-style training: shuffled mini-batches per mode, Adam, NonNeg projection,
    early stopping on the epoch's training loss with restoration of the best epoch."""
    rng = random.Random(10_000 + seed)
    modes = list(train_pts)
    n_par = len(model.theta)
    m1, m2 = [0.0] * n_par, [0.0] * n_par
    t = 0
    best_loss, best_theta, best_epoch, wait = math.inf, list(model.theta), -1, 0
    stop_reason = "epochs"
    epoch = -1
    for epoch in range(epochs):
        perms = {}
        for mode in modes:
            idx = list(range(len(train_pts[mode])))
            rng.shuffle(idx)
            perms[mode] = idx
        nb = max(-(-len(train_pts[m]) // batch) for m in modes)
        ep_sum, ep_cnt, bad = 0.0, 0, False
        for b in range(nb):
            chunks = [[train_pts[m][i] for i in perms[m][b * batch:(b + 1) * batch]] for m in modes]
            loss, grad = model.loss_and_grad(chunks)
            if not math.isfinite(loss) or not all(math.isfinite(gv) for gv in grad):
                bad = True
                break
            size = max(len(c) for c in chunks)
            ep_sum += loss * size
            ep_cnt += size
            t += 1
            c1, c2 = 1.0 - BETA1 ** t, 1.0 - BETA2 ** t
            th = model.theta
            for i in range(n_par):
                gi = grad[i]
                if i < model.K and not model.train_alpha:
                    continue
                m1[i] = BETA1 * m1[i] + (1.0 - BETA1) * gi
                m2[i] = BETA2 * m2[i] + (1.0 - BETA2) * gi * gi
                th[i] -= lr * (m1[i] / c1) / (math.sqrt(m2[i] / c2) + EPS)
            model.project()
        if bad:
            stop_reason = "non-finite loss"
            break
        ep_loss = ep_sum / ep_cnt
        if ep_loss < best_loss:
            best_loss, best_theta, best_epoch, wait = ep_loss, list(model.theta), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                stop_reason = "early stopping"
                break
        if log_every and (epoch + 1) % log_every == 0:
            log(f"    epoch {epoch + 1:6d} loss {ep_loss:.6g} alphas {[round(a, 4) for a in model.alphas()]}")
    model.theta = best_theta
    return {"epochs_run": epoch + 1, "best_epoch": best_epoch + 1, "best_epoch_loss": best_loss,
            "stop_reason": stop_reason}


def converge(model: CANN, train_pts: dict, steps: int = 20000, lr0: float = 1e-3, lr1: float = 1e-7) -> dict:
    """Finish minimizing the paper's objective: full-batch Adam, learning rate decaying
    geometrically from lr0 to lr1, same constraints; keeps the best objective seen."""
    chunks = [list(train_pts[m]) for m in train_pts]
    n_par = len(model.theta)
    m1, m2 = [0.0] * n_par, [0.0] * n_par
    start = model.objective(train_pts)
    best, best_theta = start, list(model.theta)
    ratio = (lr1 / lr0) ** (1.0 / max(1, steps - 1))
    lr = lr0
    for t in range(1, steps + 1):
        loss, grad = model.loss_and_grad(chunks)
        if not math.isfinite(loss) or not all(math.isfinite(g) for g in grad):
            break
        if loss < best:
            best, best_theta = loss, list(model.theta)
        c1, c2 = 1.0 - BETA1 ** t, 1.0 - BETA2 ** t
        th = model.theta
        for i in range(n_par):
            if i < model.K and not model.train_alpha:
                continue
            gi = grad[i]
            m1[i] = BETA1 * m1[i] + (1.0 - BETA1) * gi
            m2[i] = BETA2 * m2[i] + (1.0 - BETA2) * gi * gi
            th[i] -= lr * (m1[i] / c1) / (math.sqrt(m2[i] / c2) + EPS)
        model.project()
        lr *= ratio
    final = model.objective(train_pts)
    if final < best:
        best, best_theta = final, list(model.theta)
    model.theta = best_theta
    return {"objective_before_converge": start, "objective_after_converge": best}
