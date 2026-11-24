#!/usr/bin/env python3
"""Split a baseline law file by material and install it as the evaluators' fixed baseline."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1])
payload = json.loads(src.read_text())
name = src.stem
for side in ("train", "test"):
    gt = json.loads((ROOT / "eval" / side / "ground_truth.json").read_text())["materials"]
    out = {"schema": payload["schema"], "method": name,
           "note": "fixed baseline installed by scripts/baseline.sh (SI-CANN: exponents fixed at +-2)",
           "laws": {m: payload["laws"][m] for m in gt}}
    dst = ROOT / "eval" / side / "baselines" / f"{name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(f"installed {dst}")
