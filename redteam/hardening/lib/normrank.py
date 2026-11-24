#!/usr/bin/env python3
"""Normalized rank estimate: each metric's improvement over the reference is divided by that
metric's standard deviation across a set of attempts (the real rank divides by the spread across
the attempts later ranked; this uses the runs at hand as a stand-in).
Usage: normrank.py RUBRIC REF_SCORES NAME=SCORES [NAME=SCORES ...]   (GI-CANN entries)
"""
import json, math, sys

def load(p):
    d = json.load(open(p)); return d.get("experiments", d)

def main():
    rub = json.load(open(sys.argv[1]))["experiments"]
    ref = load(sys.argv[2])
    runs = {}
    for a in sys.argv[3:]:
        name, p = a.split("=", 1); runs[name] = load(p)
    pool = list(runs.values()) + [ref]
    spread = {}
    for e, info in rub.items():
        for m, md in info["metrics"].items():
            if md["coefficient"] == 0: continue
            vals = [r[e]["results"]["GI-CANN"][m] for r in pool if m in r.get(e, {}).get("results", {}).get("GI-CANN", {})]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1))
            spread[(e, m)] = sd if sd > 1e-9 else None
    print("name".ljust(26) + "".join(e[:14].rjust(15) for e in rub) + "     TOTAL")
    for name, s in runs.items():
        row, tot = [], 0.0
        for e, info in rub.items():
            t = 0.0
            for m, md in info["metrics"].items():
                c = abs(md["coefficient"])
                if c == 0 or spread[(e, m)] is None: continue
                a = s[e]["results"]["GI-CANN"].get(m); b = ref[e]["results"]["GI-CANN"].get(m)
                if a is None or b is None: continue
                t += c * (1 if md["higher_is_better"] else -1) * (a - b) / spread[(e, m)]
            t *= info["weight"]; tot += t; row.append(f"{t:+15.3f}")
        print(name.ljust(26) + "".join(row) + f"  {tot:+8.3f}")

if __name__ == "__main__":
    main()
