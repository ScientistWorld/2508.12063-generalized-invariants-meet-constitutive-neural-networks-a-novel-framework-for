#!/usr/bin/env python3
"""Run the reference GI-CANN trainer (method/training_runner.py) with red-team patches.

Run with cwd = a sandbox root (see common.sh).  The sandbox's own method/ and data/
are imported, checkpoints and prediction artifacts are written inside the sandbox.

Patches (all optional, all applied around training_runner.train):
  --jitter-seed S        multiply every trainable parameter by (1 + X * N(0,1)) right before
                         training starts (after the runner's own warm starts), X = --jitter-scale.
                         The reference has no random component at all (constant init,
                         full-batch Adam), so this is how "seeds" are defined here.
  --select-on-train      choose the kept checkpoint by training-mode MSE instead of by MSE on the
                         held-out mode (ps / ss), i.e. remove the held-out-mode model selection.
  --brain-generic-alpha  brain negative/combined: start alpha_minus at the class default -2.0
                         instead of the paper's answer -18, and train alpha at the base lr
                         (the 1e-4 warm-start alpha lr cannot move alpha by more than ~3).
  --train-on-heldout     SHORTCUT: add the held-out mode (rubber ps / brain ss) to the training set.
Everything else is forwarded to training_runner.main() (e.g. --materials, --modes, --families).
"""

import argparse
import sys
import zlib
from pathlib import Path

SANDBOX = Path.cwd()
sys.path.insert(0, str(SANDBOX))

import torch  # noqa: E402

torch.set_num_threads(1)

from method import training_runner as tr  # noqa: E402
from data.load_data import (  # noqa: E402
    BRAIN_SS, BRAIN_UC, BRAIN_UT, RUBBER_BT, RUBBER_PS, RUBBER_UT, HyperelasticDataset,
)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--jitter-seed", type=int, default=None)
parser.add_argument("--jitter-scale", type=float, default=0.05)
parser.add_argument("--select-on-train", action="store_true")
parser.add_argument("--brain-generic-alpha", action="store_true")
parser.add_argument("--train-on-heldout", action="store_true")
OURS, REST = parser.parse_known_args()

_orig_train = tr.train


def patched_train(model, train_ds, test_ds, epochs, lr, seed, train_modes, test_modes, device,
                  grad_clip=0.5, separate_alpha_lr=None, clamp_rules=None, patience=3000):
    is_brain = "uc" in train_modes
    material = "brain" if is_brain else "rubber"
    notes = []
    if OURS.brain_generic_alpha and is_brain and hasattr(model, "alpha_minus"):
        with torch.no_grad():
            model.alpha_minus.fill_(-2.0)
        separate_alpha_lr = None
        notes.append("alpha_minus init -2.0, alpha lr = base lr")
    if OURS.jitter_seed is not None:
        key = f"{material}_{model.mode}".encode()
        gen = torch.Generator().manual_seed(OURS.jitter_seed * 1000 + zlib.crc32(key) % 1000)
        with torch.no_grad():
            for _, p in model.named_parameters():
                noise = torch.randn(p.shape, generator=gen, dtype=p.dtype)
                p.mul_(1.0 + OURS.jitter_scale * noise)
        seed = OURS.jitter_seed
        init = {k: round(v, 5) for k, v in model.get_parameters_dict().items()}
        notes.append(f"jitter seed {OURS.jitter_seed} scale {OURS.jitter_scale}: init {init}")
    if OURS.select_on_train:
        test_ds, test_modes = train_ds, train_modes
        notes.append("checkpoint selection on training-mode MSE")
    for n in notes:
        print(f"  [redteam patch] {n}", flush=True)
    return _orig_train(model, train_ds, test_ds, epochs, lr, seed, train_modes, test_modes, device,
                       grad_clip=grad_clip, separate_alpha_lr=separate_alpha_lr,
                       clamp_rules=clamp_rules, patience=patience)


tr.train = patched_train

if OURS.train_on_heldout:
    def rubber_with_heldout():
        train = HyperelasticDataset([("ut", RUBBER_UT, "ut"), ("bt", RUBBER_BT, "bt"), ("ps", RUBBER_PS, "ps")])
        test = HyperelasticDataset([("ps", RUBBER_PS, "ps")])
        return train, test

    def brain_with_heldout():
        train = HyperelasticDataset([("ut", BRAIN_UT, "ut"), ("uc", BRAIN_UC, "uc"), ("ss", BRAIN_SS, "ss")])
        test = HyperelasticDataset([("ss", BRAIN_SS, "ss")])
        return train, test

    tr.get_rubber_dataset = rubber_with_heldout
    tr.get_brain_dataset = brain_with_heldout
    print("  [redteam patch] SHORTCUT: held-out mode added to the training set", flush=True)

sys.argv = ["training_runner.py"] + REST
tr.main()
