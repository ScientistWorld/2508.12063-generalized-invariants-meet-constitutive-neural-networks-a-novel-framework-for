#!/usr/bin/env python3
"""Maintainer tool: rebuild eval/train/ground_truth.json from data/curves.json.

The development materials (rubber_T20, brain_CX, brain_CC) are fully visible, so their ground
truth is a frozen copy of their curves in data/curves.json. The train evaluator reads only this
frozen copy, so editing data/ cannot change what it scores against.

Usage: python eval/train/build_ground_truth.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "data" / "curves.json"
OUT = HERE / "ground_truth.json"
DEV = ("rubber_T20", "brain_CX", "brain_CC")


def main():
    data = json.loads(SRC.read_text())
    materials = {}
    for mat in DEV:
        info = data["materials"][mat]
        cls = info["class"]
        modes = ("ut", "bt", "ps") if cls == "rubber" else ("ut", "uc", "ss")
        materials[mat] = {"class": cls, "curves": {m: info["curves"][m] for m in modes}}
    payload = {
        "schema": "hyperelastic_ground_truth_v1",
        "side": "train",
        "source": data["source"]["repository"] + "@" + data["source"]["commit"],
        "materials": materials,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
