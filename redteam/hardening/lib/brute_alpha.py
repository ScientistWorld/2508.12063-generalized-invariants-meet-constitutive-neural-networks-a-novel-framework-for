#!/usr/bin/env python3
"""Brute force over the generalized-invariant exponent: the enumeration GI-CANN replaces.

Single-family slots: one invariant with identity + exponential branch,
    dpsi/dJ = w_id + w2 * w1 * exp(w1 * (J - 3)),
enumerated over a dense exponent grid and a log grid of w1; for every grid point the linear
weights (w_id, w2) >= 0 are solved exactly for the paper's loss
    sum over training modes of the mean squared stress error + r (w_id + w1 + w2),  r = 0.01.
Combined slots: two identity-branch invariants (alpha+, alpha-) on a 2-D exponent grid, weights
solved exactly. The grid point with the lowest training loss wins (training curves only).
Writes a law file in the gym schema. Pure Python. Run from the workspace root.
Usage: brute_alpha.py OUT.json [--extra-gt GROUND_TRUTH.json]
  --extra-gt adds every mode found in the given ground-truth file (including the held-out mode) to
  the fitted curves: this simulates the shortcut of fitting held-out answers obtained elsewhere.
"""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from data.load_data import load_curves  # noqa: E402

R = 0.01


def kin_g(mode, x, a):
    if mode in ("ut", "uc"):
        p, q = x ** a, x ** (-a / 2); return p + 2 * q - 3, p - q
    if mode == "bt":
        p, q = x ** a, x ** (-2 * a); return 2 * p + q - 3, p - q
    if mode == "ps":
        p, q = x ** a, x ** (-a); return p + q - 2, p - q
    s = math.sqrt(x * x + 4); lp, lm = 0.5 * (s + x), 0.5 * (s - x)
    p, q = lp ** a, lm ** a; return p + q - 2, (p - q) / s


def solve2(rows, extra=0.0):
    AA = BB = AB = Ay = By = yy = 0.0
    for A, B, y, w in rows:
        AA += w * A * A; BB += w * B * B; AB += w * A * B; Ay += w * A * y; By += w * B * y; yy += w * y * y
    def obj(a, b):
        return a * a * AA + b * b * BB + 2 * a * b * AB - 2 * a * Ay - 2 * b * By + yy + R * (a + b) + extra
    cands = [(0.0, 0.0)]
    det = AA * BB - AB * AB
    if det > 1e-300:
        a = ((Ay - R / 2) * BB - (By - R / 2) * AB) / det
        b = ((By - R / 2) * AA - (Ay - R / 2) * AB) / det
        if a >= 0 and b >= 0:
            cands.append((a, b))
    if AA > 0: cands.append((max((Ay - R / 2) / AA, 0.0), 0.0))
    if BB > 0: cands.append((0.0, max((By - R / 2) / BB, 0.0)))
    return min((obj(a, b), a, b) for a, b in cands)


def fit_single(train, alphas, w1s):
    best = None
    for a in alphas:
        base = []
        try:
            for m, pts in train.items():
                n = len(pts)
                for x, y in pts:
                    J, g = kin_g(m, x, a)
                    base.append((a * g, J, y, 1.0 / n))
        except OverflowError:
            continue
        for w1 in w1s:
            rows = []
            ok = True
            for A, J, y, w in base:
                if w1 * J > 60:
                    ok = False; break
                rows.append((A, A * w1 * math.exp(w1 * J), y, w))
            if not ok:
                continue
            o, wid, w2 = solve2(rows, R * w1)
            if best is None or o < best[0]:
                best = (o, a, w1, wid, w2)
    o, a, w1, wid, w2 = best
    terms = []
    if wid > 0: terms.append({"fn": "id", "alpha": a, "w": wid})
    if w2 > 0 and w1 > 0: terms.append({"fn": "exp", "alpha": a, "w1": w1, "w2": w2})
    return {"terms": terms, "train_objective": o}


def fit_combined(train, aps, ams):
    best = None
    cols = {}
    for a in list(aps) + list(ams):
        try:
            cols[a] = [a * kin_g(m, x, a)[1] for m, pts in train.items() for x, _ in pts]
        except OverflowError:
            pass
    ys = [(y, 1.0 / len(pts)) for m, pts in train.items() for _, y in pts]
    for ap in aps:
        if ap not in cols: continue
        for am in ams:
            if am not in cols: continue
            rows = [(A, B, y, w) for A, B, (y, w) in zip(cols[ap], cols[am], ys)]
            o, wp, wm = solve2(rows)
            if best is None or o < best[0]:
                best = (o, ap, am, wp, wm)
    o, ap, am, wp, wm = best
    terms = []
    if wp > 0: terms.append({"fn": "id", "alpha": ap, "w": wp})
    if wm > 0: terms.append({"fn": "id", "alpha": am, "w": wm})
    return {"terms": terms, "train_objective": o}


def main():
    out = Path(sys.argv[1])
    extra = {}
    if "--extra-gt" in sys.argv:
        extra = json.loads(Path(sys.argv[sys.argv.index("--extra-gt") + 1]).read_text())["materials"]
    laws = {}
    for mat, info in load_curves().items():
        cls = info["class"]
        train = {m: [tuple(p) for p in info["curves"][m]] for m in info["train_modes"]}
        if mat in extra:
            train = {m: [tuple(p) for p in pts] for m, pts in extra[mat]["curves"].items()}
        if cls == "rubber":
            pos = [0.02 * i for i in range(10, 400)]
            w1s = [0.0] + [10 ** (-4 + 0.1 * i) for i in range(46)]
            comb_p = [0.1 * i for i in range(1, 81)]
        else:
            pos = [0.1 * i for i in range(1, 401)]
            w1s = [0.0] + [10 ** (-3 + 0.1 * i) for i in range(56)]
            comb_p = [0.5 * i for i in range(1, 81)]
        neg = [-a for a in pos]
        laws[mat] = {
            "positive": fit_single(train, pos, w1s),
            "negative": fit_single(train, neg, w1s),
            "combined": fit_combined(train, comb_p, [-a for a in comb_p]),
        }
        print(mat, {f: [(t["fn"], round(t["alpha"], 3)) for t in v["terms"]] for f, v in laws[mat].items()}, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "generalized_invariant_laws_v1", "method": "brute-alpha-grid" + ("+heldout-answers" if extra else ""), "laws": laws}, indent=1))


if __name__ == "__main__":
    main()
