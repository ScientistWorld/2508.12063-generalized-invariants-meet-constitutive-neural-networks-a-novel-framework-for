#!/usr/bin/env python3
"""Maintainer tool: rebuild eval/test/ground_truth.json (the hidden test answers).

Reads the paper's own data files (LivingMatterLab/CANN @ d3caca0f, GICANN/input/*.xlsx) and
writes every loading mode of the three test materials:
  rubber_T50 (Treloar 1944, 50 C): ut, bt, ps      -> ps is the held-out mode
  brain_CR   (Budday 2017, corona radiata): ut, uc, ss -> ss is the held-out mode
  brain_BG   (Budday 2017, basal ganglia):  ut, uc, ss -> ss is the held-out mode
Cauchy stress conversion as in the authors' notebooks (sigma = lambda * P for normal modes,
sigma_12 = P_12 for simple shear). Self-contained (standard library only); the hidden test never
runs this file, it only reads the JSON it produced.

Usage: python eval/test/build_ground_truth.py RUBBER.xlsx BRAIN.xlsx
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
OUT = Path(__file__).resolve().parent / "ground_truth.json"


def _col(letters):
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_sheet(path, name):
    z = zipfile.ZipFile(path)
    shared = []
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
        grid = {}
        for row in root.iter("{%s}row" % NS["m"]):
            r = int(row.get("r")) - 1
            for c in row.findall("m:c", NS):
                col = _col(re.match(r"[A-Z]+", c.get("r")).group())
                v = c.find("m:v", NS)
                if v is not None:
                    grid[(r, col)] = shared[int(v.text)] if c.get("t") == "s" else v.text
        return grid
    raise KeyError(name)


def _pairs(grid, cx, cy, r0, r1):
    out = []
    for r in range(r0, r1):
        try:
            x, y = float(grid.get((r, cx))), float(grid.get((r, cy)))
        except (TypeError, ValueError):
            continue
        out.append((x, y))
    return out


def main():
    rubber_xlsx, brain_xlsx = Path(sys.argv[1]), Path(sys.argv[2])
    materials = {}
    g = read_sheet(rubber_xlsx, "Sheet1")
    c0 = 10  # Treloar44 50 block
    assert str(g.get((0, c0), "")).startswith("Treloar44")
    materials["rubber_T50"] = {
        "class": "rubber",
        "curves": {
            "ut": [[x, x * p] for x, p in _pairs(g, c0, c0 + 2, 5, 40)],
            "bt": [[x, x * p] for x, p in _pairs(g, c0 + 3, c0 + 5, 5, 40)],
            "ps": [[x, x * p] for x, p in _pairs(g, c0 + 6, c0 + 8, 5, 40)],
        },
    }
    g = read_sheet(brain_xlsx, "Sheet1")
    for region, c in (("CR", 5), ("BG", 10)):
        assert g.get((1, c), "").startswith(region)
        uc = sorted(_pairs(g, c, c + 1, 4, 21), key=lambda t: -t[0])
        materials[f"brain_{region}"] = {
            "class": "brain",
            "curves": {
                "ut": [[x, x * p] for x, p in _pairs(g, c, c + 1, 20, 37)],
                "uc": [[x, x * p] for x, p in uc],
                "ss": [[x, p] for x, p in _pairs(g, c + 2, c + 3, 20, 37)],
            },
        }
    payload = {
        "schema": "hyperelastic_ground_truth_v1",
        "side": "test",
        "source": "LivingMatterLab/CANN@d3caca0fafa5eaf8cbd3cefb377598f1f3e9b481 GICANN/input/*.xlsx",
        "materials": materials,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
