#!/usr/bin/env python3
"""Raw rank delta of a hidden-test scores file against the gym reference.

For each experiment e and scored metric m (|coefficient| > 0 in scoring/reference.json):

    delta = sum_e weight_e * sum_m |coef_m| * sign_m * (run_m - ref_m)

with sign_m = +1 if higher_is_better else -1, run/ref taken from each experiment's
proposed-method entry (type "proposed" in reference.json, e.g. "GI-CANN",
"GI-CANN_Ja-").  The reference values are the gym's own scoring/scores_test.json
(frozen in redteam/lib/reference_scores_test.json).

The real rank additionally divides each metric's improvement by its spread across
ranked attempts; that spread is unknown here, so this reports RAW units.  Missing
entries are listed and skipped (as the rank does); NaN values are listed.
Pure standard library so it runs anywhere.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent
RUBRIC = json.loads((LIB / "reference_rubric.json").read_text())
REF = json.loads((LIB / "reference_scores_test.json").read_text())
MAIN = ("rubber_generalization", "brain_generalization")


def proposed_name(exp: dict) -> str:
    for name, info in exp["results"].items():
        if info.get("type") == "proposed":
            return name
    raise KeyError("no proposed method")


def rank(scores: dict | None) -> dict:
    out = {"total": 0.0, "main": 0.0, "per_experiment": {}, "missing": [], "nan": [], "values": {}}
    for exp_name, exp in RUBRIC["experiments"].items():
        method = proposed_name(exp)
        ref_entry = REF["experiments"][exp_name]["results"][method]
        run_entry = None
        if scores is not None:
            run_entry = scores.get("experiments", {}).get(exp_name, {}).get("results", {}).get(method)
        exp_delta = 0.0
        out["values"][exp_name] = {}
        if run_entry is None:
            out["missing"].append(f"{exp_name}/{method}")
            out["per_experiment"][exp_name] = None
            continue
        for metric, info in exp["metrics"].items():
            coef = abs(float(info.get("coefficient", 0.0)))
            if coef == 0.0:
                continue
            val = run_entry.get(metric)
            out["values"][exp_name][metric] = val
            if val is None:
                out["missing"].append(f"{exp_name}/{method}/{metric}")
                continue
            val = float(val)
            if math.isnan(val):
                out["nan"].append(f"{exp_name}/{method}/{metric}")
                continue
            sign = 1.0 if info.get("higher_is_better", True) else -1.0
            exp_delta += coef * sign * (val - float(ref_entry[metric]))
        weighted = float(exp["weight"]) * exp_delta
        out["per_experiment"][exp_name] = round(weighted, 6)
        out["total"] += weighted
        if exp_name in MAIN:
            out["main"] += weighted
    out["total"] = round(out["total"], 6)
    out["main"] = round(out["main"], 6)
    return out


def load(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "scoring/scores_test.json"
    result = rank(load(path))
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
