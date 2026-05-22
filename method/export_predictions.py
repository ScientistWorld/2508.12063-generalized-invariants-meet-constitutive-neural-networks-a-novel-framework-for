#!/usr/bin/env python3
"""Export neutral prediction artifacts from trained CANN checkpoints."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from method.gicann import GICANN  # noqa: E402
from method.sicann import SICANN  # noqa: E402
from method.training_runner import load_checkpoint, save_prediction_artifact  # noqa: E402


def checkpoint_metadata(path: Path, config: dict) -> tuple[str, str, str]:
    stem = path.stem.lower()
    material = "rubber" if "rubber" in stem else "brain"
    mode = config.get("mode", "unknown")
    model_name = "SI-CANN" if "sicann" in stem else "GI-CANN"
    return material, mode, model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Export observed-stress prediction artifacts")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoints_dir)
    for checkpoint_path in sorted(checkpoint_dir.glob("*.pt")):
        if checkpoint_path.name.startswith("_"):
            continue
        state = load_checkpoint(checkpoint_path)
        config = state.get("config", {})
        material, mode, model_name = checkpoint_metadata(checkpoint_path, config)
        model = SICANN(config) if model_name == "SI-CANN" else GICANN(config)
        model.load_state_dict(state["model_state"], strict=model_name != "SI-CANN")
        model = model.to(device)
        params = state.get("results", {}).get("params", {})
        if not params and hasattr(model, "get_parameters_dict"):
            params = model.get_parameters_dict()
        save_prediction_artifact(model, material, mode, model_name, params, args.predictions_dir, device)


if __name__ == "__main__":
    main()
