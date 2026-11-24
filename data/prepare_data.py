#!/usr/bin/env python3
"""Build data/curves.json (the curves a method may see) from the authors' public data files.

Source: the paper's own data files, LivingMatterLab/CANN at commit
d3caca0fafa5eaf8cbd3cefb377598f1f3e9b481, GICANN/input/CANNsRUBBERdata.xlsx and
GICANN/input/CANNsBRAINdata.xlsx. These are the files the paper's notebooks read; Table A.5 of
the paper is a rounded transcription of the same curves (two decimals on stretch, one or two on
stress), so the exact spreadsheet values are used here.

Conversions follow the authors' notebooks:
  * uniaxial tension / compression, equibiaxial tension, pure shear: Cauchy sigma_11 = lambda * P_11
  * simple shear: sigma_12 = P_12

Materials (six), each tested in three loading modes:
  rubber_T20  Treloar (1944) rubber, 20 C   ut, bt, ps   (the paper's rubber data)
  rubber_T50  Treloar (1944) rubber, 50 C   ut, bt, ps
  brain_CX    Budday et al. (2017) cortex           ut, uc, ss   (the paper's brain data)
  brain_CC    Budday et al. (2017) corpus callosum  ut, uc, ss
  brain_CR    Budday et al. (2017) corona radiata   ut, uc, ss
  brain_BG    Budday et al. (2017) basal ganglia    ut, uc, ss

Every material's training modes (rubber: ut + bt, brain: ut + uc) are written in full. The
held-out mode (rubber: ps, brain: ss) is written in full only for the development materials
(rubber_T20, brain_CX, brain_CC). For the test materials (rubber_T50, brain_CR, brain_BG) only the
held-out mode's stretch / shear values are written, never its stresses: those are the hidden test
answers and live only in eval/test/.

Standard library only. Usage:
  python data/prepare_data.py --rubber-xlsx R.xlsx --brain-xlsx B.xlsx --out data/curves.json
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

SOURCE = {
    "repository": "https://github.com/LivingMatterLab/CANN",
    "commit": "d3caca0fafa5eaf8cbd3cefb377598f1f3e9b481",
    "files": {
        "GICANN/input/CANNsRUBBERdata.xlsx": "7d2411b03f9f7939f67c0729289000eba3704b87be3a5ca4f88813a18ec87750",
        "GICANN/input/CANNsBRAINdata.xlsx": "701f9c7d4495bc55e78f69189719a448b1de81cdf1941e9c556995fa30d4c2bb",
    },
}

DEV = ("rubber_T20", "brain_CX", "brain_CC")
TEST = ("rubber_T50", "brain_CR", "brain_BG")
DESCRIPTIONS = {
    "rubber_T20": "Treloar (1944) vulcanised rubber at 20 C (the paper's rubber data)",
    "rubber_T50": "Treloar (1944) vulcanised rubber at 50 C",
    "brain_CX": "Budday et al. (2017) human brain cortex (the paper's brain data)",
    "brain_CC": "Budday et al. (2017) human brain corpus callosum",
    "brain_CR": "Budday et al. (2017) human brain corona radiata",
    "brain_BG": "Budday et al. (2017) human brain basal ganglia",
}
TRAIN_MODES = {"rubber": ["ut", "bt"], "brain": ["ut", "uc"]}
HELDOUT_MODE = {"rubber": "ps", "brain": "ss"}
UNITS = {"rubber": "MPa", "brain": "kPa"}


def _col(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_sheet(path: Path, name: str) -> dict[tuple[int, int], str]:
    """Return {(row, col): text} for one worksheet (0-based indices)."""
    z = zipfile.ZipFile(path)
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    relmap = {r.get("Id"): r.get("Target") for r in rels}
    for sh in wb.find("m:sheets", NS):
        if sh.get("name") != name:
            continue
        target = relmap[sh.get("{%s}id" % REL_NS)]
        target = target if target.startswith("xl/") else "xl/" + target.lstrip("/")
        root = ET.fromstring(z.read(target))
        grid: dict[tuple[int, int], str] = {}
        for row in root.iter("{%s}row" % NS["m"]):
            r = int(row.get("r")) - 1
            for c in row.findall("m:c", NS):
                col = _col(re.match(r"[A-Z]+", c.get("r")).group())
                v = c.find("m:v", NS)
                if v is None:
                    continue
                grid[(r, col)] = shared[int(v.text)] if c.get("t") == "s" else v.text
        return grid
    raise KeyError(f"sheet {name!r} not found in {path}")


def _num(grid, r, c):
    try:
        return float(grid.get((r, c)))
    except (TypeError, ValueError):
        return None


def _pairs(grid, cx, cy, r0, r1):
    out = []
    for r in range(r0, r1):
        x, y = _num(grid, r, cx), _num(grid, r, cy)
        if x is None or y is None:
            continue
        out.append((x, y))
    return out


def _brain_columns(grid):
    """Sheet1 of the brain file: per region (lambda, P11, gamma, P12) in columns 5k..5k+3."""
    out = {}
    for k, region in enumerate(("CX", "CR", "BG", "CC")):
        c = 5 * k
        label = grid.get((1, c), "")
        if not label.startswith(region):
            raise ValueError(f"unexpected brain sheet layout: column {c} is {label!r}, expected {region}")
        out[f"brain_{region}"] = c
    return out


def extract_visible(rubber_xlsx: Path, brain_xlsx: Path) -> dict:
    materials: dict[str, dict] = {}

    g = read_sheet(brain_xlsx, "Sheet1")
    for mat, c in _brain_columns(g).items():
        # compression rows: lambda 0.9 -> 1.0 ; tension rows: lambda 1.0 -> 1.1 ; shear rows: gamma 0 -> 0.2
        uc = sorted(_pairs(g, c, c + 1, 4, 21), key=lambda t: -t[0])
        ut = _pairs(g, c, c + 1, 20, 37)
        ss_x = [x for x, _ in _pairs(g, c + 2, c + 3, 20, 37)]
        curves = {
            "ut": [[x, x * p] for x, p in ut],
            "uc": [[x, x * p] for x, p in uc],
        }
        if mat in DEV:
            curves["ss"] = [[x, p] for x, p in _pairs(g, c + 2, c + 3, 20, 37)]
        materials[mat] = {"class": "brain", "curves": curves, "heldout_inputs": ss_x}

    g = read_sheet(rubber_xlsx, "Sheet1")
    for mat, c0 in (("rubber_T20", 0), ("rubber_T50", 10)):
        if not str(g.get((0, c0), "")).startswith("Treloar44"):
            raise ValueError("unexpected rubber sheet layout")
        ut = _pairs(g, c0, c0 + 2, 5, 40)
        bt = _pairs(g, c0 + 3, c0 + 5, 5, 40)
        ps_x = [x for x, _ in _pairs(g, c0 + 6, c0 + 8, 5, 40)]
        curves = {"ut": [[x, x * p] for x, p in ut], "bt": [[x, x * p] for x, p in bt]}
        if mat in DEV:
            curves["ps"] = [[x, x * p] for x, p in _pairs(g, c0 + 6, c0 + 8, 5, 40)]
        materials[mat] = {"class": "rubber", "curves": curves, "heldout_inputs": ps_x}

    ordered = {}
    for mat in DEV + TEST:
        info = materials[mat]
        cls = info["class"]
        ordered[mat] = {
            "class": cls,
            "split": "dev" if mat in DEV else "test",
            "description": DESCRIPTIONS[mat],
            "units": UNITS[cls],
            "train_modes": TRAIN_MODES[cls],
            "heldout_mode": HELDOUT_MODE[cls],
            "curves": info["curves"],
            "heldout_inputs": info["heldout_inputs"],
        }
    return {
        "schema": "hyperelastic_curves_v1",
        "source": SOURCE,
        "notes": (
            "Columns are [stretch or shear, Cauchy stress]. Each curve starts at the undeformed state. "
            "Held-out-mode stresses of the test materials are not included (hidden test)."
        ),
        "materials": ordered,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rubber-xlsx", required=True, type=Path)
    ap.add_argument("--brain-xlsx", required=True, type=Path)
    ap.add_argument("--out", default=Path(__file__).resolve().parent / "curves.json", type=Path)
    args = ap.parse_args()
    payload = extract_visible(args.rubber_xlsx, args.brain_xlsx)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    n = sum(len(c) for m in payload["materials"].values() for c in m["curves"].values())
    print(f"wrote {args.out} ({len(payload['materials'])} materials, {n} visible points)")


if __name__ == "__main__":
    main()
