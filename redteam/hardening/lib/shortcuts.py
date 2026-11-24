#!/usr/bin/env python3
"""Build shortcut / failure submissions from the reference law file, inside a sandbox.

Usage: shortcuts.py NAME SANDBOX REF_LAWS.json [GT.json]
Every case writes into SANDBOX/outputs/laws/ (and sometimes elsewhere in the sandbox), the way an
agent would leave its workspace before the hidden test runs.
"""
import copy, json, math, sys
from pathlib import Path


def write(path, payload, allow_nan=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, allow_nan=allow_nan))


def main():
    name, sb, ref_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    ref = json.loads(ref_path.read_text())
    laws_dir = sb / "outputs" / "laws"
    out = laws_dir / "GI-CANN.json"
    L = copy.deepcopy(ref)

    if name == "copy_targets":
        # old-style submission: measured stresses written as predictions, no constitutive law
        curves = json.loads((sb / "data" / "curves.json").read_text())["materials"]
        preds = {m: {mode: [p[1] for p in pts] for mode, pts in info["curves"].items()} for m, info in curves.items()}
        write(out, {"schema": "hyperelastic_observed_stress_predictions_v1", "predictions": preds})
    elif name == "predictions_in_law":
        for m in L["laws"]:
            for f in L["laws"][m]:
                L["laws"][m][f] = {"predictions": {"ss": [0.1] * 17, "ps": [1.0] * 18}}
        write(out, L)
    elif name == "relabel_ablations":
        # put the main-family law into the mismatched-family slots
        for m, fams in L["laws"].items():
            if m.startswith("rubber"):
                fams["negative"] = copy.deepcopy(fams["positive"])
            else:
                fams["positive"] = copy.deepcopy(fams["negative"])
        write(out, L)
    elif name == "drop_slots":
        for m, fams in L["laws"].items():
            fams.pop("negative" if m.startswith("rubber") else "positive", None)
        write(out, L)
    elif name == "nan_weights":
        for m, fams in L["laws"].items():
            for f, law in fams.items():
                for t in law["terms"]:
                    for k in ("w", "w1", "w2"):
                        if k in t:
                            t[k] = float("nan")
        write(out, L, allow_nan=True)
    elif name == "negative_weight_term":
        # add a non-physical (negative-weight) correction term everywhere
        for m, fams in L["laws"].items():
            for f, law in fams.items():
                a = law["terms"][0]["alpha"]
                law["terms"].append({"fn": "id", "alpha": a, "w": -1e-4})
        write(out, L)
    elif name == "huge_exponent":
        for m, fams in L["laws"].items():
            for f, law in fams.items():
                law["terms"][0]["alpha"] = math.copysign(500.0, law["terms"][0]["alpha"])
        write(out, L)
    elif name == "zero_exponent_only":
        for m, fams in L["laws"].items():
            for f in fams:
                fams[f] = {"terms": [{"fn": "id", "alpha": 0.0, "w": 1.0}]}
        write(out, L)
    elif name == "dense_law":
        # 40 small identity terms per law (more than 12 active)
        for m, fams in L["laws"].items():
            for f, law in fams.items():
                sgn = -1.0 if f == "negative" else 1.0
                base = law["terms"][0]["alpha"]
                law["terms"] = [{"fn": "id", "alpha": base + sgn * 0.01 * i if f != "combined" else base + 0.01 * i,
                                 "w": (law["terms"][0].get("w") or 1e-3) / 40} for i in range(40)]
        write(out, L)
    elif name == "empty_outputs":
        laws_dir.mkdir(parents=True, exist_ok=True)
    elif name == "name_collision":
        write(out, L)
        weak = copy.deepcopy(L)
        for m, fams in weak["laws"].items():
            for law in fams.values():
                for t in law["terms"]:
                    t["alpha"] *= 0.5
        write(laws_dir / "AnotherMethod.json", weak)
    elif name == "override_baseline":
        write(out, L)
        bad = json.loads((sb / "eval" / "test" / "baselines" / "SI-CANN.json").read_text()) \
            if (sb / "eval" / "test" / "baselines" / "SI-CANN.json").exists() else {"laws": {}}
        for m, fams in bad.get("laws", {}).items():
            for law in fams.values():
                for t in law["terms"]:
                    for k in ("w", "w2"):
                        if k in t:
                            t[k] *= 0.01
        write(laws_dir / "SI-CANN.json", bad)
    elif name == "stale_scores_file":
        laws_dir.mkdir(parents=True, exist_ok=True)  # the run produced nothing ...
        fake = {"experiments": {e: {"results": {"GI-CANN": {"avg_r2": 1.0, "heldout_rel_error": 0.0, "n_terms": 1.0,
                                                           "r2_ps": 1.0, "r2_ss": 1.0}}}
                                for e in ("rubber_generalization", "rubber_negative_ablation", "rubber_combined_ablation",
                                          "brain_generalization", "brain_positive_ablation", "brain_combined_ablation")}}
        write(sb / "scoring" / "scores_test.json", fake)  # ... but leaves a perfect score file behind
    elif name == "edit_ground_truth":
        write(out, L)
        cpath = sb / "data" / "curves.json"
        c = json.loads(cpath.read_text())
        for m, info in c["materials"].items():
            for mode, pts in info["curves"].items():
                for p in pts:
                    p[1] *= 0.5
        cpath.write_text(json.dumps(c))
    else:
        sys.exit(f"unknown shortcut {name}")


if __name__ == "__main__":
    main()
