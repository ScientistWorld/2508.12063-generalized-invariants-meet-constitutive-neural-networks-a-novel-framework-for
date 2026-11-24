#!/usr/bin/env python3
"""PS-CANN (paper Table 1 / Table 4): identity-branch strain energy on a dense, fixed exponent grid,
    psi = sum_k w_k (J_k - 3),  w_k >= 0,
fitted with the paper's loss (sum of per-mode MSE on the training curves + r sum w, r = 0.01).
The problem is convex; it is solved exactly by coordinate descent. This is the enumeration of
every candidate exponent that GI-CANN replaces with a learned exponent.
Exponents: brain k in {-30..30}\\{0}; rubber k in {-8, -7.5, ..., 8}\\{0}; restricted to the
family's sign. Usage: pscann.py OUT.json   (run from the workspace root)
"""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from data.load_data import load_curves  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from brute_alpha import kin_g  # noqa: E402

R = 0.01


def nn_lasso(cols, y, w, sweeps=20000):
    k, n = len(cols), len(y)
    coef = [0.0] * k
    res = list(y)
    cc = [sum(wi * c * c for c, wi in zip(col, w)) for col in cols]
    for _ in range(sweeps):
        mx = 0.0
        for j in range(k):
            if cc[j] <= 0:
                continue
            col = cols[j]
            rho = sum(wi * c * r for c, r, wi in zip(col, res, w)) + cc[j] * coef[j]
            new = max((rho - R / 2) / cc[j], 0.0)
            d = new - coef[j]
            if d:
                for i in range(n):
                    res[i] -= d * col[i]
                coef[j] = new
                mx = max(mx, abs(d) * math.sqrt(cc[j]))
        if mx < 1e-13:
            break
    return coef


def main():
    out = Path(sys.argv[1])
    laws = {}
    for mat, info in load_curves().items():
        cls = info["class"]
        grid = [k for k in range(-30, 31) if k] if cls == "brain" else [k / 2 for k in range(-16, 17) if k]
        train = {m: info["curves"][m] for m in info["train_modes"]}
        laws[mat] = {}
        for fam in ("positive", "negative", "combined"):
            ks = [k for k in grid if fam == "combined" or (k > 0) == (fam == "positive")]
            cols = [[] for _ in ks]; y = []; w = []
            for m, pts in train.items():
                for x, s in pts:
                    for j, a in enumerate(ks):
                        cols[j].append(a * kin_g(m, x, a)[1])
                    y.append(s); w.append(1.0 / len(pts))
            coef = nn_lasso(cols, y, w)
            terms = [{"fn": "id", "alpha": float(a), "w": c} for a, c in zip(ks, coef) if c > 0]
            laws[mat][fam] = {"terms": terms}
            print(mat, fam, {a: round(c, 6) for a, c in zip(ks, coef) if c > 0}, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "generalized_invariant_laws_v1", "method": "PS-CANN", "laws": laws}, indent=1))


if __name__ == "__main__":
    main()
