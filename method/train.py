#!/usr/bin/env python3
"""Train GI-CANN (or the SI-CANN baseline) on every material/family and write a law file.

Reference procedure: for every (material, family) the paper's training run (method/gicann.py:
mini-batch Adam with early stopping, then full-batch convergence of the same objective) is
repeated from 20 random initializations (seeds 0-19) and the run with the lowest value of the
paper's own training objective (loss on the training curves) is kept. Single runs from a random
start often stop in a local optimum; on the paper's cortex data 3 of 20 starts reach the global
optimum alpha* = -18.02 (identity term only), which is the law the paper reports (alpha* = -18.016,
w = 0.009 kPa). No held-out data is used anywhere.

Each (model, material, family, seed, epochs) run is cached as outputs/runs/<tag>.json, so an
interrupted job resumes where it stopped. Runs are independent and deterministic (pure Python,
seeded), so running them in parallel processes gives the same numbers as running them alone; no
wall-clock limit enters the method.

Examples
  python method/train.py --model GI-CANN --out outputs/laws/GI-CANN.json            # the reference
  python method/train.py --model SI-CANN --out outputs/baselines/SI-CANN.json       # the baseline
  python method/train.py --model GI-CANN --seeds 0 --select first-seed --out ...    # one paper run
  python method/train.py --model GI-CANN --epochs 300 --out outputs/laws/short.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.load_data import FAMILIES, evaluation_inputs, load_curves, training_data  # noqa: E402
from method.gicann import PAPER_CONFIG, CANN, Point, converge, train  # noqa: E402

PROCEDURE = "v3"  # paper mini-batch training + full-batch convergence (cache tag)

MODEL_NAMES = ("GI-CANN", "SI-CANN")


def extreme_points(material: str) -> list[Point]:
    pts = []
    for mode, xs in evaluation_inputs(material).items():
        x = min(xs) if mode == "uc" else max(xs)
        pts.append(Point(mode, x))
    return pts


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def run_one(model: str, material: str, family: str, seed: int, epochs: int | None, runs_dir: str) -> dict:
    cls = load_curves()[material]["class"]
    cfg = PAPER_CONFIG[(cls, family)]
    n_epochs = cfg["epochs"] if epochs is None else epochs
    tag = f"{model}_{material}_{family}_s{seed}_e{n_epochs}_{PROCEDURE}"
    path = Path(runs_dir) / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    gi = model == "GI-CANN"
    a0 = cfg["a0"] if gi else [2.0] * len(cfg["signs"])
    net = CANN(cfg["signs"], a0, train_alpha=gi, seed=seed, extremes=extreme_points(material))
    data = {m: [Point(m, x, y) for x, y in pts] for m, pts in training_data(material).items()}
    t0 = time.time()
    info = train(net, data, n_epochs, seed=seed)
    info.update(converge(net, data))
    record = {
        "tag": tag, "model": model, "material": material, "family": family, "seed": seed,
        "epochs": n_epochs, "alpha_init": [s * a for s, a in zip(cfg["signs"], a0)],
        "alphas": net.alphas(), "train_objective": net.objective(data), "law": net.export_law(),
        "theta": net.theta, **info, "cpu_seconds": round(time.time() - t0, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, path)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=MODEL_NAMES, default="GI-CANN")
    ap.add_argument("--materials", default="all", help="all | dev | test | comma-separated names")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--seeds", default="0-19", help="comma list and/or ranges, e.g. 0-19 or 0,3,5-9")
    ap.add_argument("--epochs", type=int, default=None, help="override the paper's epoch count")
    ap.add_argument("--select", default="best-objective", choices=["first-seed", "best-objective"],
                    help="which seed's law goes into the output file")
    ap.add_argument("--runs-dir", default=str(ROOT / "outputs" / "runs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    curves = load_curves()
    if args.materials in ("all", "dev", "test"):
        mats = [m for m, i in curves.items() if args.materials == "all" or i["split"] == args.materials]
    else:
        mats = [m.strip() for m in args.materials.split(",") if m.strip()]
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    seeds = parse_seeds(args.seeds)
    jobs = [(args.model, m, f, s, args.epochs, args.runs_dir) for m in mats for f in fams for s in seeds]

    records = []
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, *j): j for j in jobs}
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                print(f"  {rec['tag']}: alphas={[round(a, 3) for a in rec['alphas']]} "
                      f"obj={rec['train_objective']:.6g} epochs={rec['epochs_run']} ({rec['stop_reason']}) "
                      f"{rec['cpu_seconds']}s", flush=True)
    else:
        for j in jobs:
            rec = run_one(*j)
            records.append(rec)
            print(f"  {rec['tag']}: alphas={[round(a, 3) for a in rec['alphas']]} obj={rec['train_objective']:.6g}",
                  flush=True)

    laws: dict = {}
    for m in mats:
        for f in fams:
            cands = [r for r in records if r["material"] == m and r["family"] == f]
            if args.select == "first-seed":
                chosen = min(cands, key=lambda r: seeds.index(r["seed"]))
            else:
                chosen = min(cands, key=lambda r: r["train_objective"])
            laws.setdefault(m, {})[f] = {**chosen["law"], "seed": chosen["seed"],
                                         "train_objective": chosen["train_objective"]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "generalized_invariant_laws_v1",
        "method": args.model,
        "settings": {"seeds": seeds, "select": args.select, "epochs": args.epochs or "paper"},
        "laws": laws,
    }
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out} ({len(mats)} materials x {len(fams)} families)")


if __name__ == "__main__":
    main()
