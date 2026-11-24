#!/usr/bin/env python3
"""Load the visible stress-stretch curves (data/curves.json).

Six materials, each tested in three loading modes (see data/prepare_data.py for the source):

  material     class   training modes   held-out mode   split
  rubber_T20   rubber  ut, bt           ps              dev   (paper's rubber, Treloar 20 C)
  rubber_T50   rubber  ut, bt           ps              test  (Treloar 50 C)
  brain_CX     brain   ut, uc           ss              dev   (paper's brain, cortex)
  brain_CC     brain   ut, uc           ss              dev   (corpus callosum)
  brain_CR     brain   ut, uc           ss              test  (corona radiata)
  brain_BG     brain   ut, uc           ss              test  (basal ganglia)

Loading modes (incompressible):
  ut / uc  uniaxial tension / compression  F = diag(l, l^-1/2, l^-1/2)   observed sigma_11
  bt       equibiaxial tension             F = diag(l, l, l^-2)           observed sigma_11 (= sigma_22)
  ps       pure shear                      F = diag(l, 1, l^-1)           observed sigma_11
  ss       simple shear                    F = [[1, g, 0], [0, 1, 0], [0, 0, 1]]  observed sigma_12
Units: rubber MPa, brain kPa. Each curve is a list of [stretch or shear, Cauchy stress] rows and
starts at the undeformed state.

The held-out mode's stresses are present only for the development materials. For the test
materials, `heldout_inputs` lists where the held-out mode is evaluated (stretches / shears only).
Standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path

CURVES_PATH = Path(__file__).resolve().parent / "curves.json"
MODES = {"rubber": ("ut", "bt", "ps"), "brain": ("ut", "uc", "ss")}
TRAIN_MODES = {"rubber": ("ut", "bt"), "brain": ("ut", "uc")}
HELDOUT_MODE = {"rubber": "ps", "brain": "ss"}
FAMILIES = ("positive", "negative", "combined")


def load_curves(path: str | Path | None = None) -> dict:
    """Return {material: info} with keys class, split, train_modes, heldout_mode, curves, heldout_inputs."""
    return json.loads(Path(path or CURVES_PATH).read_text())["materials"]


def materials(split: str | None = None, path=None) -> list[str]:
    mats = load_curves(path)
    return [m for m, info in mats.items() if split is None or info["split"] == split]


def training_data(material: str, path=None) -> dict[str, list[tuple[float, float]]]:
    """{mode: [(x, sigma), ...]} for the material's training modes."""
    info = load_curves(path)[material]
    return {m: [tuple(p) for p in info["curves"][m]] for m in info["train_modes"]}


def evaluation_inputs(material: str, path=None) -> dict[str, list[float]]:
    """{mode: [x, ...]} for all three modes, including the held-out mode's inputs."""
    info = load_curves(path)[material]
    out = {m: [p[0] for p in pts] for m, pts in info["curves"].items()}
    out[info["heldout_mode"]] = list(info["heldout_inputs"])
    return out


if __name__ == "__main__":
    for mat, info in load_curves().items():
        sizes = {m: len(v) for m, v in info["curves"].items()}
        print(f"{mat:11s} {info['class']:6s} {info['split']:4s} visible={sizes} "
              f"heldout={info['heldout_mode']} ({len(info['heldout_inputs'])} inputs)")
