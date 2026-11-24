#!/usr/bin/env python3
"""Train GI-CANN variants for the hardening checks (imports the reference implementation).

Variants (all train on the training curves only and select seeds by training objective):
  short       paper mini-batch phase capped at --epochs, no convergence phase (weaker: under-trained)
  noconverge  paper mini-batch phase only (the notebooks' procedure without the convergence phase)
  twoinv      two invariants per sign (positive: [+,+], negative: [-,-], combined: [+,+,-,-]),
              paper procedure: a richer but still compact law
  rtuned      the reference procedure with the L1 strength chosen on the development materials:
              on brain_CX and brain_CC (visible shear) r = 0.02 gives better held-out shear R2 and
              relative error than the paper's r = 0.01 at equal avg R2, so brain laws use r = 0.02;
              rubber keeps r = 0.01. A better method found from the development side only.
  scalefree   paper procedure, but each training mode's squared error is divided by that mode's
              stress variance (the loss becomes a sum of 1 - R2 over training modes, the quantity
              the paper reports) and the L1 strength is scaled by the same factor on average
Usage: train_variant.py VARIANT OUT.json [--seeds 0-19] [--epochs N] [--jobs J]
       [--runs-dir DIR]
Run from the workspace root (the sandbox).
"""
import argparse, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from data.load_data import load_curves, training_data  # noqa: E402
from method.gicann import PAPER_CONFIG, CANN, Point, converge, train  # noqa: E402
from method.train import extreme_points  # noqa: E402

TWOINV = {
    ("rubber", "positive"): ([1, 1], [2.0, 4.0]),
    ("rubber", "negative"): ([-1, -1], [2.0, 0.5]),
    ("rubber", "combined"): ([1, 1, -1, -1], [1.7, 4.0, 0.1, 2.0]),
    ("brain", "positive"): ([1, 1], [5.0, 10.0]),
    ("brain", "negative"): ([-1, -1], [5.0, 10.0]),
    ("brain", "combined"): ([1, 1, -1, -1], [2.0, 10.0, 2.0, 10.0]),
}


def train_weighted(net, data, wmode, epochs, seed, phase="train"):
    """Paper training loop with per-mode loss weights (wraps CANN.loss_and_grad)."""
    orig = net.loss_and_grad
    modes = list(data)

    def weighted(chunks):
        loss, grad = 0.0, [0.0] * len(net.theta)
        l1 = net.l1()
        for m, ch in zip(modes, chunks):
            if not ch:
                continue
            l, g = orig([ch])
            l -= l1
            w = wmode[m]
            loss += w * l
            for i, gv in enumerate(g):
                grad[i] += w * gv
        # remove the L1 gradient added by each call, add it once
        n_calls = sum(1 for ch in chunks if ch)
        th = net.theta
        for i in range(net.K, len(th)):
            sgn = (th[i] > 0) - (th[i] < 0)
            grad[i] -= sgn * 0.01 * sum(wmode[m] for m, ch in zip(modes, chunks) if ch)
            grad[i] += sgn * 0.01
        return loss + l1, grad

    net.loss_and_grad = weighted
    objective = net.objective
    net.objective = lambda pts: weighted([list(pts[m]) for m in modes])[0]
    try:
        if phase == "converge":
            return converge(net, data)
        return train(net, data, epochs, seed=seed)
    finally:
        net.objective = objective
        net.loss_and_grad = orig


def run(args):
    variant, mat, fam, seed, epochs, runs_dir = args
    tag = f"{variant}_{mat}_{fam}_s{seed}_e{epochs}_v3"
    path = Path(runs_dir) / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    cls = load_curves()[mat]["class"]
    cfg = PAPER_CONFIG[(cls, fam)]
    import method.gicann as G
    G.R_L1 = (0.02 if cls == "brain" else 0.01) if variant == "rtuned" else 0.01
    signs, a0 = (TWOINV[(cls, fam)] if variant == "twoinv" else (cfg["signs"], cfg["a0"]))
    net = CANN(signs, a0, True, seed=seed, extremes=extreme_points(mat))
    data = {m: [Point(m, x, y) for x, y in pts] for m, pts in training_data(mat).items()}
    t0 = time.time()
    if variant == "scalefree":
        # weight each training mode's squared error by 1 / var(stress of that mode), normalized so
        # the weights average to 1 (the L1 strength keeps its meaning)
        import statistics
        var = {m: statistics.pvariance([p.y for p in pts]) for m, pts in data.items()}
        inv = {m: 1.0 / v for m, v in var.items()}
        mean_inv = sum(inv.values()) / len(inv)
        wmode = {m: inv[m] / mean_inv for m in data}
        info = train_weighted(net, data, wmode, cfg["epochs"], seed)
        info.update(train_weighted(net, data, wmode, None, seed, phase="converge"))
        obj = sum(wmode[m] * sum((net.stress(p) - p.y) ** 2 for p in pts) / len(pts) for m, pts in data.items()) + net.l1()
    else:
        n_ep = epochs or cfg["epochs"]
        info = train(net, data, n_ep, seed=seed)
        if variant in ("twoinv", "rtuned"):
            info.update(converge(net, data))
    if variant != "scalefree":
        obj = net.objective(data)
    rec = {"tag": tag, "material": mat, "family": fam, "seed": seed, "alphas": net.alphas(),
           "train_objective": obj, "law": net.export_law(), **info,
           "cpu_seconds": round(time.time() - t0, 1)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["short", "noconverge", "twoinv", "scalefree", "rtuned"])
    ap.add_argument("out")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--runs-dir", default="outputs/variant_runs")
    a = ap.parse_args()
    from method.train import parse_seeds
    seeds = parse_seeds(a.seeds)
    mats = list(load_curves())
    jobs = [(a.variant, m, f, s, a.epochs, a.runs_dir)
            for m in mats for f in ("positive", "negative", "combined") for s in seeds]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        recs = list(ex.map(run, jobs))
    laws = {}
    for m in mats:
        for f in ("positive", "negative", "combined"):
            best = min((r for r in recs if r["material"] == m and r["family"] == f), key=lambda r: r["train_objective"])
            laws.setdefault(m, {})[f] = {**best["law"], "seed": best["seed"], "train_objective": best["train_objective"]}
            print(f"{m} {f}: alphas={[round(x, 3) for x in best['alphas']]} obj={best['train_objective']:.6g}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"schema": "generalized_invariant_laws_v1", "method": a.variant, "laws": laws}, indent=1))


if __name__ == "__main__":
    main()
