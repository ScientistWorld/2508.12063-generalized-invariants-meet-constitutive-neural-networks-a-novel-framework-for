#!/usr/bin/env python3
"""Collect every red-team hidden-test result into redteam/SUMMARY.md and redteam/summary.json."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent
RT = LIB.parent
sys.path.insert(0, str(LIB))
from rank import RUBRIC, REF, load, proposed_name, rank  # noqa: E402

KEYS = [("rubber_generalization", "avg_r2"), ("rubber_generalization", "r2_ps"),
        ("brain_generalization", "avg_r2"), ("brain_generalization", "r2_ss")]
SEED_RUNS = ["check_rerun_reference", "seed_jitter_1", "seed_jitter_2", "seed_jitter_3"]


def run_dirs():
    for d in sorted(RT.iterdir()):
        if not d.is_dir() or d.name == "lib":
            continue
        if (d / "evaluate_test.rc").exists():
            yield d.name, d
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and (sub / "evaluate_test.rc").exists():
                yield f"{d.name}/{sub.name}", sub


def entry(scores, exp, metric):
    if scores is None:
        return None
    method = proposed_name(RUBRIC["experiments"][exp])
    return scores.get("experiments", {}).get(exp, {}).get("results", {}).get(method, {}).get(metric)


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    return f"{v:.4f}"


def main():
    rows, summary = [], {}
    ref_vals = {f"{e}.{m}": entry(REF, e, m) for e, m in KEYS}
    summary["reference"] = {"values": ref_vals, "rank": rank(REF)}
    for name, d in run_dirs():
        scores = load(d / "scores_test.json")
        r = rank(scores)
        rc = (d / "evaluate_test.rc").read_text().strip()
        vals = {f"{e}.{m}": entry(scores, e, m) for e, m in KEYS}
        summary[name] = {"values": vals, "rank": r, "evaluate_test_rc": rc,
                         "scores_file": scores is not None}
        flags = []
        if scores is None:
            flags.append("no scores file")
        if r["missing"]:
            flags.append(f"{len(r['missing'])} missing")
        if r["nan"]:
            flags.append(f"{len(r['nan'])} NaN")
        rows.append(f"| {name} | " + " | ".join(fmt(vals[f'{e}.{m}']) for e, m in KEYS)
                    + f" | {r['main']:+.4f} | {r['total']:+.4f} | {rc} | {', '.join(flags)} |")
    lines = ["# Red-team hidden-test results", "",
             "Raw rank delta = sum over experiments of weight x sum over scored metrics of |coef| x "
             "improvement over the reference's own hidden-test score (GI-CANN entries), before the "
             "rank's division by the cross-attempt spread. 0 = tie with the reference.", "",
             "| run | rubber avg_r2 | rubber r2_ps | brain avg_r2 | brain r2_ss | raw Δ main | raw Δ total | test rc | flags |",
             "|---|---|---|---|---|---|---|---|---|",
             "| **reference** | " + " | ".join(fmt(ref_vals[f'{e}.{m}']) for e, m in KEYS) + " | +0.0000 | +0.0000 | 0 | |"]
    lines += rows
    seeds = [summary[n] for n in SEED_RUNS if n in summary and summary[n]["scores_file"]]
    if len(seeds) >= 2:
        lines += ["", "## Seed spread (reference trainer: rerun + 3 init-jitter seeds)", "",
                  "| quantity | mean | sd | min | max | max-min |", "|---|---|---|---|---|---|"]
        spread = {}
        for key in [f"{e}.{m}" for e, m in KEYS] + ["rank.main", "rank.total"]:
            if key.startswith("rank."):
                xs = [s["rank"][key.split(".")[1]] for s in seeds]
            else:
                xs = [s["values"][key] for s in seeds if s["values"][key] is not None]
            if len(xs) < 2:
                continue
            sd = statistics.stdev(xs)
            spread[key] = {"mean": statistics.mean(xs), "sd": sd, "min": min(xs), "max": max(xs)}
            lines.append(f"| {key} | {statistics.mean(xs):.4f} | {sd:.4f} | {min(xs):.4f} | {max(xs):.4f} | {max(xs)-min(xs):.4f} |")
        summary["seed_spread"] = spread
    lines += ["", "## Per-experiment raw deltas", "",
              "| run | " + " | ".join(RUBRIC["experiments"]) + " |",
              "|---|" + "---|" * len(RUBRIC["experiments"])]
    for name, s in summary.items():
        if name in ("reference", "seed_spread"):
            continue
        pe = s["rank"]["per_experiment"]
        lines.append(f"| {name} | " + " | ".join("missing" if pe[e] is None else f"{pe[e]:+.4f}" for e in RUBRIC["experiments"]) + " |")
    (RT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (RT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
