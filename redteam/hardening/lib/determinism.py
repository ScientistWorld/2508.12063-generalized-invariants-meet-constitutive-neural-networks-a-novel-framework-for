#!/usr/bin/env python3
"""Rerun a few reference training runs alone (one process, fresh cache) and compare them with the
cached runs of the reference job (run in parallel). Identical laws show the stored reference does
not depend on machine load or on running in parallel: the method has no wall-clock limit and is
pure, seeded Python arithmetic."""
import json, sys, tempfile
from pathlib import Path

WS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WS))
from method.train import run_one  # noqa: E402

CASES = [("GI-CANN", "brain_CX", "negative", 5), ("GI-CANN", "rubber_T50", "positive", 0),
         ("GI-CANN", "brain_CR", "combined", 3), ("SI-CANN", "brain_BG", "negative", 7)]
ok = True
with tempfile.TemporaryDirectory() as tmp:
    for model, mat, fam, seed in CASES:
        fresh = run_one(model, mat, fam, seed, None, tmp)
        cached_path = WS / "outputs" / "runs" / f"{fresh['tag']}.json"
        if not cached_path.exists():
            print(f"{fresh['tag']}: no cached run to compare"); ok = False; continue
        cached = json.loads(cached_path.read_text())
        same = fresh["law"] == cached["law"] and fresh["theta"] == cached["theta"]
        ok &= same
        print(f"{fresh['tag']}: {'identical' if same else 'DIFFERENT'} alphas={fresh['alphas']} obj={fresh['train_objective']!r}")
print("determinism:", "OK" if ok else "FAILED")
