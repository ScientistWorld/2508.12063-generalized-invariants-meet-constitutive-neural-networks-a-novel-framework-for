#!/usr/bin/env python3
"""Train-slice evaluation for the GI-CANN benchmark.

This development slice evaluates the same checkpoint interface as the original
``scripts/evaluate.sh`` entry point, but recomputes predictions on a fixed
alternating subset of each deformation curve. The neutral zero-stress point is
excluded so the slice is not dominated by the trivial undeformed state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Older checkpoints can contain pickled NumPy objects under the internal
# ``numpy._core`` module path, while the container ships NumPy exposing the
# same objects via ``numpy.core``. Alias the old path so torch.load can
# deserialize the trusted local checkpoints.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

from data.load_data import (  # noqa: E402
    BRAIN_SS,
    BRAIN_UC,
    BRAIN_UT,
    RUBBER_BT,
    RUBBER_PS,
    RUBBER_UT,
    HyperelasticDataset,
    compute_r2,
    extract_observed_stress,
)
from eval.artifacts import build_scores_from_artifacts, load_prediction_artifacts  # noqa: E402

torch = None
MinimalGICANN = None
MinimalSICANN = None


def ensure_checkpoint_dependencies():
    """Import PyTorch-only dependencies for legacy checkpoint scoring."""
    global torch, MinimalGICANN, MinimalSICANN
    if torch is None:
        import torch as torch_module
        from eval.models import MinimalGICANN as gicann_cls, MinimalSICANN as sicann_cls

        torch = torch_module
        MinimalGICANN = gicann_cls
        MinimalSICANN = sicann_cls


def split_rows(data: np.ndarray) -> np.ndarray:
    """Return this slice's nonzero alternating rows for one deformation curve."""
    indices = np.arange(1, len(data), dtype=int)
    return data[indices[::2]]


def get_slice_dataset(material: str) -> tuple[HyperelasticDataset, list[str]]:
    if material == "rubber":
        modes = [
            ("ut", split_rows(RUBBER_UT), "ut"),
            ("bt", split_rows(RUBBER_BT), "bt"),
            ("ps", split_rows(RUBBER_PS), "ps"),
        ]
        return HyperelasticDataset(modes), ["ut", "bt", "ps"]

    modes = [
        ("ut", split_rows(BRAIN_UT), "ut"),
        ("uc", split_rows(BRAIN_UC), "uc"),
        ("ss", split_rows(BRAIN_SS), "ss"),
    ]
    return HyperelasticDataset(modes), ["ut", "uc", "ss"]


def extract_params_from_state(model_state: dict, config: dict, is_sicann: bool) -> dict:
    params = {}
    for key, value in model_state.items():
        if torch.is_tensor(value) and value.numel() == 1:
            params[key] = round(float(value.item()), 4)

    if is_sicann:
        params["alpha_fixed"] = 2.0 if config.get("mode", "positive") == "positive" else -2.0
    return params


def evaluate_checkpoint(checkpoint_path: Path, device: str = "cpu") -> dict:
    # These benchmark checkpoints are trusted local reproduction artifacts.
    # PyTorch 2.6 defaults to weights_only=True, which rejects their NumPy
    # scalar config metadata before the model state can be read.
    ensure_checkpoint_dependencies()
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    config = state.get("config", {})
    stem = checkpoint_path.stem
    is_sicann = "sicann" in stem.lower()
    material = "rubber" if "rubber" in stem else "brain"
    model_type = "SI-CANN" if is_sicann else "GI-CANN"

    if is_sicann:
        model = MinimalSICANN(config)
    else:
        model = MinimalGICANN(config)

    model.load_state_dict(state["model_state"], strict=not is_sicann)
    model.eval()

    device_obj = torch.device(device)
    model = model.to(device_obj)
    dataset, mode_names = get_slice_dataset(material)
    F = dataset.F.to(device_obj)
    S = dataset.S.to(device_obj)
    gamma = dataset.gamma.to(device_obj)

    r2_by_mode = {}
    with torch.no_grad():
        for mode_name in mode_names:
            indices = dataset.mode_indices[mode_name]
            idx_t = torch.tensor(indices, dtype=torch.long, device=device_obj)
            if mode_name == "ss":
                pred = model(F[idx_t], gamma_override=gamma[idx_t])
            else:
                pred = model(F[idx_t])
            target = S[idx_t]
            pred_obs = extract_observed_stress(pred, [mode_name] * len(indices)).cpu().numpy()
            target_obs = extract_observed_stress(target, [mode_name] * len(indices)).cpu().numpy()
            r2_by_mode[mode_name] = float(compute_r2(pred_obs, target_obs))

    return {
        "material": material,
        "mode": config.get("mode", "unknown"),
        "model_type": model_type,
        "r2_by_mode": r2_by_mode,
        "params": extract_params_from_state(state["model_state"], config, is_sicann),
        "checkpoint": str(checkpoint_path),
    }


def experiment_and_method(result: dict) -> tuple[str, str] | tuple[None, None]:
    material = result["material"]
    mode = result["mode"]
    model_type = result["model_type"]

    if material == "rubber":
        if mode == "positive":
            return "rubber_generalization", "SI-CANN" if model_type == "SI-CANN" else "GI-CANN"
        if mode == "negative":
            return "rubber_negative_ablation", "SI-CANN_Jα-" if model_type == "SI-CANN" else "GI-CANN_Jα-"
        if mode == "combined":
            return "rubber_combined_ablation", "SI-CANN_Jα+Jα-" if model_type == "SI-CANN" else "GI-CANN_Jα+Jα-"
        return None, None

    if mode == "negative":
        return "brain_generalization", "SI-CANN" if model_type == "SI-CANN" else "GI-CANN"
    if mode == "positive":
        return "brain_positive_ablation", "SI-CANN_Jα+" if model_type == "SI-CANN" else "GI-CANN_Jα+"
    if mode == "combined":
        return "brain_combined_ablation", "SI-CANN_Jα+Jα-" if model_type == "SI-CANN" else "GI-CANN_Jα+Jα-"
    return None, None


def build_scores_json(results: list[dict], output_path: str, experiments_filter: set[str] | None = None) -> dict:
    exp_data: dict[str, dict] = {}

    for result in results:
        exp_name, method_name = experiment_and_method(result)
        if exp_name is None or method_name is None:
            continue
        if experiments_filter and exp_name not in experiments_filter:
            continue

        r2 = result["r2_by_mode"]
        params = result["params"]
        mode = result["mode"]
        model_type = result["model_type"]
        exp_data.setdefault(exp_name, {"results": {}})

        if result["material"] == "rubber":
            r2_ut = r2.get("ut", 0.0)
            r2_bt = r2.get("bt", 0.0)
            r2_ps = r2.get("ps", 0.0)
            if model_type == "SI-CANN":
                alpha_star = params.get("alpha_fixed", 2.0 if mode == "positive" else -2.0)
            elif mode == "negative":
                alpha_star = params.get("alpha_minus", 0.0)
            else:
                alpha_star = params.get("alpha_plus", params.get("alpha_minus", 0.0))
            exp_data[exp_name]["results"][method_name] = {
                "r2_ut": round(r2_ut, 4),
                "r2_bt": round(r2_bt, 4),
                "r2_ps": round(r2_ps, 4),
                "avg_r2": round(float(np.mean([r2_ut, r2_bt, r2_ps])), 4),
                "alpha_star": round(float(alpha_star), 4),
            }
        else:
            r2_ut = r2.get("ut", 0.0)
            r2_uc = r2.get("uc", 0.0)
            r2_ss = r2.get("ss", 0.0)
            if model_type == "SI-CANN":
                alpha_star = params.get("alpha_fixed", 2.0 if mode == "positive" else -2.0)
            elif mode == "positive":
                alpha_star = params.get("alpha_plus", 0.0)
            else:
                alpha_star = params.get("alpha_minus", params.get("alpha_plus", 0.0))
            exp_data[exp_name]["results"][method_name] = {
                "r2_ut": round(r2_ut, 4),
                "r2_uc": round(r2_uc, 4),
                "r2_ss": round(r2_ss, 4),
                "avg_r2": round(float(np.mean([r2_ut, r2_uc, r2_ss])), 4),
                "alpha_star": round(float(alpha_star), 4),
            }

    scores = {"experiments": dict(sorted(exp_data.items()))}
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")
    print(f"Scores written to {output_path}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GI-CANN checkpoints on the train slice")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--predictions-dir", type=str, default="outputs/predictions")
    parser.add_argument("--scores-json", type=str, default="scoring/scores_train.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--prefer-checkpoints",
        action="store_true",
        help="Use legacy checkpoint reconstruction even when neutral prediction artifacts exist.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional experiment name or comma-separated list of experiment names to keep in scores.json",
    )
    args = parser.parse_args()

    experiments_filter = None
    if args.experiment:
        experiments_filter = {item.strip() for item in args.experiment.split(",") if item.strip()}

    if not args.checkpoint and not args.prefer_checkpoints:
        artifacts = load_prediction_artifacts(args.predictions_dir)
        if artifacts:
            print(f"Evaluating {len(artifacts)} prediction artifacts from {args.predictions_dir}")
            build_scores_from_artifacts(
                artifacts,
                args.scores_json,
                split="train",
                experiments_filter=experiments_filter,
            )
            return

    results = []
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            print(f"Evaluating {ckpt_path}")
            results.append(evaluate_checkpoint(ckpt_path, args.device))
        else:
            print(f"Checkpoint not found: {ckpt_path}")

    if args.checkpoints_dir and not args.checkpoint:
        ckpt_dir = Path(args.checkpoints_dir)
        if ckpt_dir.exists():
            for ckpt_path in sorted(ckpt_dir.glob("*.pt")):
                if ckpt_path.name.startswith("_"):
                    continue
                print(f"Evaluating {ckpt_path.name}")
                try:
                    result = evaluate_checkpoint(ckpt_path, args.device)
                    results.append(result)
                    print(
                        f"  {result['model_type']} | {result['material']} | "
                        f"{result['mode']} | avg_r2={np.mean(list(result['r2_by_mode'].values())):.4f}"
                    )
                except Exception as exc:
                    print(f"  ERROR: {exc}")

    if results:
        build_scores_json(results, args.scores_json, experiments_filter)


if __name__ == "__main__":
    main()
