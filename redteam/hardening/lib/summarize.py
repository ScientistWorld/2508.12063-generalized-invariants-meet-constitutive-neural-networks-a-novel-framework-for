#!/usr/bin/env python3
"""Summarize redteam/hardening/results/* against the stored reference (scoring/scores_test.json).

raw delta  = sum_e weight_e * sum_m |coef_m| * (s_m - ref_m) * direction     (GI-CANN entries)
norm delta = the same with each metric's difference divided by its standard deviation across the
             honest runs (reference noise set, single paper runs, variants): a stand-in for the
             rank's division by the spread across ranked attempts.
"""
import json, math
from pathlib import Path

WS = Path(__file__).resolve().parents[3]
RES = WS / "redteam" / "hardening" / "results"
rub = json.loads((WS / "scoring" / "reference.json").read_text())["experiments"]
ref = json.loads((WS / "scoring" / "scores_test.json").read_text())["experiments"]
EXPS = list(rub)
HELD = {e: ("r2_ps" if e.startswith("rubber") else "r2_ss") for e in EXPS}


def gi(scores, e, method="GI-CANN"):
    return scores.get(e, {}).get("results", {}).get(method)


runs = {}
for d in sorted(RES.iterdir()):
    p = d / "scores_test.json"
    runs[d.name] = json.loads(p.read_text())["experiments"] if p.exists() else None

honest = [n for n in runs if n.startswith(("noise_", "single_", "variant_", "brute_", "weak_"))] + ["reference_rescore"]
spread = {}
for e in EXPS:
    for m, md in rub[e]["metrics"].items():
        if md["coefficient"] == 0:
            continue
        vals = [gi(runs[n], e)[m] for n in honest if runs.get(n) and gi(runs[n], e) and m in gi(runs[n], e)]
        mu = sum(vals) / len(vals) if vals else 0.0
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) if len(vals) > 1 else 0.0
        spread[(e, m)] = sd


def delta(scores, normalized):
    tot, per = 0.0, {}
    for e in EXPS:
        s, r = gi(scores, e), gi(ref, e)
        if s is None or r is None:
            per[e] = None
            continue
        t = 0.0
        for m, md in rub[e]["metrics"].items():
            c = abs(md["coefficient"])
            if c == 0 or m not in s or m not in r:
                continue
            d = (s[m] - r[m]) * (1 if md["higher_is_better"] else -1)
            if normalized:
                d = d / spread[(e, m)] if spread[(e, m)] > 1e-12 else 0.0
            t += c * d
        per[e] = rub[e]["weight"] * t
        tot += per[e]
    return tot, per


lines = ["# Hardening checks: hidden-test results", "",
         "GI-CANN entry of each run's `scores_test.json` (produced by `bash scripts/evaluate_test.sh` in a sandbox with the gym's eval/ restored).",
         "Columns: avg_r2 / held-out R2 / held-out relative error / n_terms. Raw and normalized rank deltas are against the stored reference (`scoring/scores_test.json`).",
         "Normalized: each metric's difference divided by its std across the honest runs (reference noise set, single paper runs, variants).", "",
         "| run | rubber_generalization | brain_generalization | rubber_combined | brain_combined | rubber_negative | brain_positive | raw delta | norm delta | failed experiments |",
         "|---|---|---|---|---|---|---|---|---|---|"]
order = ["rubber_generalization", "brain_generalization", "rubber_combined_ablation", "brain_combined_ablation",
         "rubber_negative_ablation", "brain_positive_ablation"]
for name, sc in runs.items():
    if sc is None:
        lines.append(f"| {name} | no scores_test.json | | | | | | | | |")
        continue
    cells = []
    for e in order:
        g = gi(sc, e)
        cells.append("missing" if g is None else f"{g['avg_r2']:.4f} / {g[HELD[e]]:.4f} / {g['heldout_rel_error']:.3f} / {g['n_terms']:.1f}")
    raw, _ = delta(sc, False)
    nrm, _ = delta(sc, True)
    log = (RES / name / "evaluate_test.log").read_text() if (RES / name / "evaluate_test.log").exists() else ""
    failed = sorted({ln.split("]")[1].split(":")[0].strip() for ln in log.splitlines()
                     if ln.startswith("[GI-CANN]") and "FAILED" in ln})
    if "[GI-CANN] unusable law file" in log:
        failed = ["all (unusable file)"]
    lines.append(f"| {name} | " + " | ".join(cells) + f" | {raw:+.4f} | {nrm:+.3f} | {', '.join(failed) or '-'} |")
lines += ["", "## Per-experiment normalized deltas", "",
          "| run | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
for name, sc in runs.items():
    if sc is None:
        continue
    _, per = delta(sc, True)
    lines.append(f"| {name} | " + " | ".join("missing" if per[e] is None else f"{per[e]:+.3f}" for e in order) + " |")
lines += ["", "## Spread used for normalization (std across honest runs)", "",
          "| experiment | " + " | ".join(["avg_r2", "held-out R2", "rel_error", "n_terms"]) + " |", "|---|---|---|---|---|"]
for e in order:
    lines.append(f"| {e} | {spread[(e, 'avg_r2')]:.4f} | {spread[(e, HELD[e])]:.4f} | {spread[(e, 'heldout_rel_error')]:.4f} | {spread[(e, 'n_terms')]:.3f} |")
# other entries worth showing
extra = []
for name in ("shortcut_name_collision", "shortcut_override_baseline"):
    sc = runs.get(name)
    if sc:
        for meth in sorted(sc["rubber_generalization"]["results"]):
            g = sc["rubber_generalization"]["results"][meth]
            b = sc["brain_generalization"]["results"][meth]
            extra.append(f"| {name} | {meth} | {g['avg_r2']:.4f} | {b['avg_r2']:.4f} |")
if extra:
    lines += ["", "## Other entries in multi-method runs", "", "| run | method | rubber_generalization avg_r2 | brain_generalization avg_r2 |", "|---|---|---|---|"] + extra
print("\n".join(lines))
