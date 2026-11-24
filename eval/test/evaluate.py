#!/usr/bin/env python3
"""Hidden-test evaluator: scores declared strain-energy laws on the test materials.

What is scored
--------------
A method submits constitutive laws, never stresses. Each file outputs/laws/<METHOD>.json holds,
for every material and exponent family, a strain energy built from generalized invariants
J_a = l1^a + l2^a + l3^a (incompressible, l1 l2 l3 = 1):

    psi = sum over terms of
          id : w  * x                     x = (J_alpha - 3)^power
          exp: w2 * (exp(w1 * x) - 1)
          ln : -w2 * ln(1 - w1 * x)

with every weight >= 0, power in [1, 3] (default 1) and the ln argument positive wherever the law
is evaluated. This evaluator recomputes the Cauchy stress of every loading mode from the law
itself (closed-form stress for uniaxial, equibiaxial, pure shear and simple shear) and compares it
with the frozen ground truth next to this file. Numbers a method writes are never read.

Families: "positive" (every exponent > 0), "negative" (every exponent < 0), "combined" (any sign).
A law that breaks its family's sign rule, uses a negative or non-finite weight, leaves the ln
domain, produces a non-finite stress, has no active term or more than MAX_ACTIVE_TERMS active
terms is invalid. An experiment whose law is missing or invalid for any of its materials is
scored as failed: every scored metric gets its worst value.

Metrics per experiment (averaged over the experiment's materials on this side):
  r2_<mode>          coefficient of determination per loading mode, clipped to [0, 1]
                     (the paper reports negative R2 as 0)
  avg_r2             mean of the three clipped per-mode R2 (training modes + held-out mode)
  heldout_rel_error  mean relative stress error on the held-out mode, |s - s_hat| / |s_hat| per
                     loaded point, each point capped at 1
  n_terms            number of active terms (a term is active if its stress contribution reaches
                     1% of the measured stress range of some loading mode)
  alpha_star         exponent of the dominant term (report only)

Usage: python eval/test/evaluate.py [--laws-dir outputs/laws] [--scores-json scoring/scores_test.json]
       [--materials m1,m2]   (default: every material in ground_truth.json)
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

SIDE = "test"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_SCORES = ROOT / "scoring" / ("scores_%s.json" % SIDE)
GROUND_TRUTH = HERE / "ground_truth.json"
BASELINES_DIR = HERE / "baselines"

EXPERIMENTS = {
    "rubber_generalization": ("rubber", "positive"),
    "rubber_negative_ablation": ("rubber", "negative"),
    "rubber_combined_ablation": ("rubber", "combined"),
    "brain_generalization": ("brain", "negative"),
    "brain_positive_ablation": ("brain", "positive"),
    "brain_combined_ablation": ("brain", "combined"),
}
MODES = {"rubber": ("ut", "bt", "ps"), "brain": ("ut", "uc", "ss")}
HELDOUT = {"rubber": "ps", "brain": "ss"}
PRIMARY_SLOT = "GI-CANN"  # always reported; worst values when its law file is missing

MAX_FILE_BYTES = 5_000_000
MAX_DECLARED_TERMS = 64
MAX_ACTIVE_TERMS = 12
ALPHA_MAX = 60.0
ALPHA_ZERO = 1e-12
POWER_MIN, POWER_MAX = 1.0, 3.0
ACTIVE_FRACTION = 0.01
REL_ERR_CAP = 1.0
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+\-]{0,63}$")


class InvalidLaw(Exception):
    pass


# --------------------------------------------------------------------------- kinematics
def kinematics(mode: str, x: float, alpha: float) -> tuple[float, float]:
    """Return (J_alpha - 3, g) with observed stress = alpha * dpsi/dJ * g.

    ut/uc: F = diag(l, l^-1/2, l^-1/2)  -> sigma_11
    bt   : F = diag(l, l, l^-2)         -> sigma_11 = sigma_22
    ps   : F = diag(l, 1, l^-1)         -> sigma_11
    ss   : F = [[1, g, 0], [0, 1, 0], [0, 0, 1]], principal stretches l+- = (sqrt(g^2+4) +- g)/2, 1
           -> sigma_12
    """
    if mode in ("ut", "uc"):
        p, q = x ** alpha, x ** (-alpha / 2.0)
        return p + 2.0 * q - 3.0, p - q
    if mode == "bt":
        p, q = x ** alpha, x ** (-2.0 * alpha)
        return 2.0 * p + q - 3.0, p - q
    if mode == "ps":
        p, q = x ** alpha, x ** (-alpha)
        return p + q - 2.0, p - q
    if mode == "ss":
        s = math.sqrt(x * x + 4.0)
        lp, lm = 0.5 * (s + x), 0.5 * (s - x)
        p, q = lp ** alpha, lm ** alpha
        return p + q - 2.0, (p - q) / s
    raise ValueError(mode)


# --------------------------------------------------------------------------- laws
def _finite_nonneg(term: dict, key: str) -> float:
    v = term.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InvalidLaw(f"term weight {key!r} missing or not a number")
    v = float(v)
    if not math.isfinite(v) or v < 0.0:
        raise InvalidLaw(f"term weight {key!r}={v!r} must be finite and >= 0")
    return v


def parse_law(law, family: str) -> list[dict]:
    if not isinstance(law, dict) or not isinstance(law.get("terms"), list):
        raise InvalidLaw("law must be an object with a 'terms' list")
    terms = law["terms"]
    if not terms:
        raise InvalidLaw("law has no terms")
    if len(terms) > MAX_DECLARED_TERMS:
        raise InvalidLaw(f"law declares {len(terms)} terms (limit {MAX_DECLARED_TERMS})")
    parsed = []
    for t in terms:
        if not isinstance(t, dict):
            raise InvalidLaw("term must be an object")
        fn = t.get("fn")
        if fn not in ("id", "exp", "ln"):
            raise InvalidLaw(f"unknown term function {fn!r}")
        a = t.get("alpha")
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(float(a)):
            raise InvalidLaw("term exponent 'alpha' missing or not finite")
        a = float(a)
        if abs(a) > ALPHA_MAX:
            raise InvalidLaw(f"|alpha|={abs(a):.3g} exceeds {ALPHA_MAX}")
        power = t.get("power", 1.0)
        if isinstance(power, bool) or not isinstance(power, (int, float)) or not (POWER_MIN <= float(power) <= POWER_MAX):
            raise InvalidLaw(f"term power {power!r} outside [{POWER_MIN}, {POWER_MAX}]")
        if fn == "id":
            weights = {"w": _finite_nonneg(t, "w")}
        else:
            weights = {"w1": _finite_nonneg(t, "w1"), "w2": _finite_nonneg(t, "w2")}
        if abs(a) <= ALPHA_ZERO:
            continue  # J_0 = 3: the term carries no energy
        if family == "positive" and a <= 0.0:
            raise InvalidLaw(f"exponent {a:.4g} breaks the positive family")
        if family == "negative" and a >= 0.0:
            raise InvalidLaw(f"exponent {a:.4g} breaks the negative family")
        parsed.append({"fn": fn, "alpha": a, "power": float(power), **weights})
    return parsed


def term_dpsi(term: dict, X: float) -> float:
    X = max(X, 0.0)  # J_alpha >= 3 for incompressible stretches; guard round-off
    pw = term["power"]
    if pw == 1.0:
        base, xp = 1.0, X
    else:
        base, xp = pw * X ** (pw - 1.0), X ** pw
    fn = term["fn"]
    if fn == "id":
        return term["w"] * base
    w1, w2 = term["w1"], term["w2"]
    if fn == "exp":
        return w2 * w1 * math.exp(w1 * xp) * base
    denom = 1.0 - w1 * xp
    if denom <= 0.0:
        raise InvalidLaw("ln term evaluated outside its domain (1 - w1 * x <= 0)")
    return w2 * w1 / denom * base


def term_stress(term: dict, mode: str, x: float) -> float:
    X, g = kinematics(mode, x, term["alpha"])
    s = term["alpha"] * term_dpsi(term, X) * g
    if not math.isfinite(s):
        raise InvalidLaw("non-finite stress")
    return s


# --------------------------------------------------------------------------- metrics
def r2_clipped(pred, y):
    mean = sum(y) / len(y)
    ss_tot = sum((v - mean) ** 2 for v in y)
    ss_res = sum((p - v) ** 2 for p, v in zip(pred, y))
    if not math.isfinite(ss_res):
        raise InvalidLaw("non-finite residual")
    r2 = 1.0 - ss_res / ss_tot
    return min(1.0, max(0.0, r2))


def rel_error(pred, y):
    errs = [min(abs(p - v) / abs(v), REL_ERR_CAP) for p, v in zip(pred, y) if abs(v) > 1e-12]
    return sum(errs) / len(errs)


def score_material(terms: list[dict], material: dict) -> dict:
    cls = material["class"]
    curves = material["curves"]
    preds = {}
    contrib_max = [0.0] * len(terms)
    active_flags = [False] * len(terms)
    try:
        for mode in MODES[cls]:
            pts = curves[mode]
            scale = max(abs(y) for _, y in pts)
            total = [0.0] * len(pts)
            for k, term in enumerate(terms):
                mx = 0.0
                for i, (x, _) in enumerate(pts):
                    s = term_stress(term, mode, x)
                    total[i] += s
                    mx = max(mx, abs(s))
                contrib_max[k] = max(contrib_max[k], mx)
                if mx >= ACTIVE_FRACTION * scale:
                    active_flags[k] = True
            preds[mode] = total
    except (OverflowError, ZeroDivisionError, ValueError) as exc:
        raise InvalidLaw(f"stress evaluation failed: {exc}") from None
    n_active = sum(active_flags)
    if n_active == 0:
        raise InvalidLaw("no active term (the law carries no stress)")
    if n_active > MAX_ACTIVE_TERMS:
        raise InvalidLaw(f"{n_active} active terms (limit {MAX_ACTIVE_TERMS})")
    out = {}
    for mode in MODES[cls]:
        out["r2_" + mode] = r2_clipped(preds[mode], [y for _, y in curves[mode]])
    out["avg_r2"] = sum(out["r2_" + m] for m in MODES[cls]) / len(MODES[cls])
    held = HELDOUT[cls]
    out["heldout_rel_error"] = rel_error(preds[held], [y for _, y in curves[held]])
    out["n_terms"] = float(n_active)
    out["alpha_star"] = terms[max(range(len(terms)), key=lambda k: contrib_max[k])]["alpha"]
    return out


def worst_metrics(cls: str) -> dict:
    out = {"r2_" + m: 0.0 for m in MODES[cls]}
    out.update({"avg_r2": 0.0, "heldout_rel_error": REL_ERR_CAP, "n_terms": float(MAX_ACTIVE_TERMS)})
    return out


def score_method(laws, gt_materials: dict, experiments: dict) -> tuple[dict, list[str]]:
    results, notes = {}, []
    for exp, (cls, family) in experiments.items():
        mats = [m for m, info in gt_materials.items() if info["class"] == cls]
        per_mat, failed = [], None
        for mat in mats:
            try:
                fam_laws = laws.get(mat) if isinstance(laws, dict) else None
                if not isinstance(fam_laws, dict) or family not in fam_laws:
                    raise InvalidLaw(f"no '{family}' law for {mat}")
                terms = parse_law(fam_laws[family], family)
                if not terms:
                    raise InvalidLaw("law has only zero-exponent terms")
                per_mat.append(score_material(terms, gt_materials[mat]))
            except InvalidLaw as exc:
                failed = f"{mat}/{family}: {exc}"
                break
        if failed or not per_mat:
            notes.append(f"{exp}: FAILED ({failed or 'no materials'}) -> worst values")
            results[exp] = worst_metrics(cls)
            continue
        keys = per_mat[0].keys()
        results[exp] = {k: sum(r[k] for r in per_mat) / len(per_mat) for k in keys}
    return results, notes


# --------------------------------------------------------------------------- I/O
def load_law_file(path: Path):
    if path.stat().st_size > MAX_FILE_BYTES:
        raise InvalidLaw(f"file larger than {MAX_FILE_BYTES} bytes")
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise InvalidLaw(f"unreadable JSON: {exc}") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("laws"), dict):
        raise InvalidLaw("file has no 'laws' object")
    return payload["laws"]


def collect_methods(laws_dir: Path) -> dict:
    """Map method name -> laws (or the InvalidLaw explaining why the file is unusable)."""
    methods = {}
    reserved = {p.stem for p in BASELINES_DIR.glob("*.json")} if BASELINES_DIR.is_dir() else set()
    if laws_dir.is_dir():
        for path in sorted(laws_dir.glob("*.json")):
            name = path.stem
            if not NAME_RE.match(name):
                print(f"skipping {path.name}: method names use [A-Za-z0-9_.+-]")
                continue
            if name in reserved:
                name = name + "@outputs"  # eval/ fixes the baselines; an outputs copy is reported separately
            try:
                methods[name] = load_law_file(path)
            except InvalidLaw as exc:
                methods[name] = exc
    if PRIMARY_SLOT not in methods:
        methods[PRIMARY_SLOT] = InvalidLaw(f"{laws_dir}/{PRIMARY_SLOT}.json not found")
    return methods


def round_metrics(d: dict) -> dict:
    return {k: round(float(v), 6) for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{SIDE}-side evaluator (declared strain-energy laws)")
    ap.add_argument("--laws-dir", default=str(ROOT / "outputs" / "laws"))
    ap.add_argument("--scores-json", default=str(DEFAULT_SCORES))
    ap.add_argument("--materials", default=None, help="comma-separated subset of the ground-truth materials")
    args = ap.parse_args()

    out_path = Path(args.scores_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # never leave a stale score file behind

    gt = json.loads(GROUND_TRUTH.read_text())["materials"]
    if args.materials:
        keep = [m.strip() for m in args.materials.split(",") if m.strip()]
        unknown = [m for m in keep if m not in gt]
        if unknown:
            sys.exit(f"unknown materials: {unknown}")
        gt = {m: gt[m] for m in keep}
    classes = {info["class"] for info in gt.values()}
    experiments = {e: v for e, v in EXPERIMENTS.items() if v[0] in classes}

    methods = {}
    if BASELINES_DIR.is_dir():
        for path in sorted(BASELINES_DIR.glob("*.json")):
            methods[path.stem] = load_law_file(path)
    methods.update(collect_methods(Path(args.laws_dir)))

    scores = {e: {"results": {}} for e in experiments}
    for name in sorted(methods):
        laws = methods[name]
        if isinstance(laws, InvalidLaw):
            print(f"[{name}] unusable law file ({laws}) -> worst values in every experiment")
            for exp, (cls, _) in experiments.items():
                scores[exp]["results"][name] = round_metrics(worst_metrics(cls))
            continue
        res, notes = score_method(laws, gt, experiments)
        for note in notes:
            print(f"[{name}] {note}")
        for exp, metrics in res.items():
            scores[exp]["results"][name] = round_metrics(metrics)

    payload = {"experiments": scores}
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, out_path)
    print(f"Scores for {len(methods)} methods x {len(experiments)} experiments written to {out_path}")
    for exp in experiments:
        for name, m in scores[exp]["results"].items():
            held = "r2_" + HELDOUT[EXPERIMENTS[exp][0]]
            print(f"  {exp:26s} {name:22s} avg_r2={m['avg_r2']:.4f} {held}={m[held]:.4f} "
                  f"rel_err={m['heldout_rel_error']:.4f} n_terms={m['n_terms']:.2f}")


if __name__ == "__main__":
    main()
