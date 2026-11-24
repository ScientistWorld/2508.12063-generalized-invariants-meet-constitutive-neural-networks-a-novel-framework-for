#!/usr/bin/env python3
"""Raw rank delta of a scores file against the reference scores, using scoring/reference.json.

delta = sum_e weight_e * sum_m |coef_m| * (s_m - ref_m) * (+1 if higher_is_better else -1)
for the GI-CANN entry (or --method). Missing values count as 0 (the real rank skips them).
Usage: rank.py SCORES.json [--ref REF_SCORES.json] [--method GI-CANN] [--json]
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]

def load(p):
    d = json.loads(Path(p).read_text())
    return d.get("experiments", d)

def deltas(scores, ref, rubric, method="GI-CANN", ref_method="GI-CANN"):
    out = {}
    for exp, info in rubric["experiments"].items():
        w = info["weight"]
        s = scores.get(exp, {}).get("results", {}).get(method)
        r = ref.get(exp, {}).get("results", {}).get(ref_method)
        if s is None or r is None:
            out[exp] = None
            continue
        tot = 0.0
        for m, md in info["metrics"].items():
            c = abs(md["coefficient"])
            if c == 0 or m not in s or m not in r:
                continue
            sign = 1.0 if md["higher_is_better"] else -1.0
            tot += c * sign * (s[m] - r[m])
        out[exp] = w * tot
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--ref", default=str(ROOT / "scoring" / "scores_test.json"))
    ap.add_argument("--rubric", default=str(ROOT / "scoring" / "reference.json"))
    ap.add_argument("--method", default="GI-CANN")
    ap.add_argument("--ref-method", default="GI-CANN")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = deltas(load(a.scores), load(a.ref), json.loads(Path(a.rubric).read_text()), a.method, a.ref_method)
    total = sum(v for v in d.values() if v is not None)
    if a.json:
        print(json.dumps({"per_experiment": d, "total": total}))
    else:
        for k, v in d.items():
            print(f"{k:28s} {'missing' if v is None else f'{v:+.4f}'}")
        print(f"{'TOTAL':28s} {total:+.4f}")

if __name__ == "__main__":
    main()
