#!/usr/bin/env python3
"""Held-out test-slice evaluation for the GI-CANN benchmark."""

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
from eval.artifacts import curve_quality_metrics  # noqa: E402

torch = None
MinimalGICANN = None
MinimalSICANN = None


def score_entries(scores: dict, experiments_filter: set[str] | None = None) -> set[tuple[str, str]]:
    entries = set()
    for exp_name, exp_info in scores.get("experiments", {}).items():
        if experiments_filter and exp_name not in experiments_filter:
            continue
        for method_name in exp_info.get("results", {}):
            entries.add((exp_name, method_name))
    return entries


def validate_train_entries_present(scores: dict, train_scores_path: str, experiments_filter: set[str] | None = None) -> None:
    """Ensure held-out scores cover every method entry exposed in train scores."""
    path = Path(train_scores_path)
    if not path.exists():
        print(f"Train score file not found at {path}; skipping train/test entry validation.")
        return

    with open(path) as f:
        train_scores = json.load(f)

    missing = sorted(score_entries(train_scores, experiments_filter) - score_entries(scores, experiments_filter))
    if missing:
        formatted = ", ".join(f"{exp}/{method}" for exp, method in missing)
        raise SystemExit(f"Held-out test scores are missing train-scored method entries: {formatted}")


def ensure_checkpoint_dependencies():
    """Import PyTorch-only dependencies for legacy checkpoint scoring."""
    global torch, MinimalGICANN, MinimalSICANN
    if torch is None:
        import torch as torch_module
        from eval.models import MinimalGICANN as gicann_cls, MinimalSICANN as sicann_cls

        torch = torch_module
        MinimalGICANN = gicann_cls
        MinimalSICANN = sicann_cls


def balanced_holdout_indices(data: np.ndarray) -> np.ndarray:
    """Return a target-balanced nonzero holdout subset for one curve.

    The curves are small enough that every-other rows can make R2 unstable for
    weak ablations. This target-only partition keeps mean, spread, and endpoint
    stress statistics similar across the complementary slices.
    """
    indices = np.arange(1, len(data), dtype=int)
    if len(indices) <= 2:
        return indices[1::2]

    targets = data[indices, 1].astype(float)
    target_scale = float(np.ptp(targets))
    if target_scale <= 1e-12:
        target_scale = max(float(np.max(np.abs(targets))), 1.0)

    holdout_size = len(indices) // 2
    best_score = None
    best_subset = None
    from itertools import combinations

    def summary(rows: np.ndarray) -> np.ndarray:
        values = data[rows, 1].astype(float)
        return np.array(
            [
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
            ],
            dtype=float,
        ) / target_scale

    all_set = set(int(i) for i in indices)
    for subset_tuple in combinations((int(i) for i in indices), holdout_size):
        subset = np.asarray(subset_tuple, dtype=int)
        complement = np.asarray(sorted(all_set - set(subset_tuple)), dtype=int)
        score = float(np.linalg.norm(summary(subset) - summary(complement)))
        if best_score is None or score < best_score:
            best_score = score
            best_subset = subset

    return np.asarray(best_subset, dtype=int)


def slice_indices(data: np.ndarray) -> np.ndarray:
    """Return the held-out target-balanced rows for one full deformation curve."""
    return balanced_holdout_indices(data)


def split_rows(data: np.ndarray) -> np.ndarray:
    """Return this slice's held-out target-balanced nonzero rows."""
    return data[slice_indices(data)]


def rows_by_material() -> dict[str, dict[str, np.ndarray]]:
    return {
        "rubber": {
            "ut": slice_indices(RUBBER_UT),
            "bt": slice_indices(RUBBER_BT),
            "ps": slice_indices(RUBBER_PS),
        },
        "brain": {
            "ut": slice_indices(BRAIN_UT),
            "uc": slice_indices(BRAIN_UC),
            "ss": slice_indices(BRAIN_SS),
        },
    }


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

    predictions_by_mode = {}
    targets_by_mode = {}
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
            predictions_by_mode[mode_name] = pred_obs
            targets_by_mode[mode_name] = target_obs

    return {
        "material": material,
        "mode": config.get("mode", "unknown"),
        "model_type": model_type,
        "metrics": curve_quality_metrics(material, predictions_by_mode, targets_by_mode),
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

        metrics = result["metrics"]
        params = result["params"]
        mode = result["mode"]
        model_type = result["model_type"]
        exp_data.setdefault(exp_name, {"results": {}})

        if result["material"] == "rubber":
            if model_type == "SI-CANN":
                alpha_star = params.get("alpha_fixed", 2.0 if mode == "positive" else -2.0)
            elif mode == "negative":
                alpha_star = params.get("alpha_minus", 0.0)
            else:
                alpha_star = params.get("alpha_plus", params.get("alpha_minus", 0.0))
            exp_data[exp_name]["results"][method_name] = {
                "r2_ut": round(metrics["r2_ut"], 4),
                "r2_bt": round(metrics["r2_bt"], 4),
                "r2_ps": round(metrics["r2_ps"], 4),
                "avg_r2": round(metrics["avg_r2"], 4),
                "avg_normalized_rmse": round(metrics["avg_normalized_rmse"], 4),
                "tail_normalized_mae": round(metrics["tail_normalized_mae"], 4),
                "shape_violation_rate": round(metrics["shape_violation_rate"], 4),
                "alpha_star": round(float(alpha_star), 4),
            }
        else:
            if model_type == "SI-CANN":
                alpha_star = params.get("alpha_fixed", 2.0 if mode == "positive" else -2.0)
            elif mode == "positive":
                alpha_star = params.get("alpha_plus", 0.0)
            else:
                alpha_star = params.get("alpha_minus", params.get("alpha_plus", 0.0))
            exp_data[exp_name]["results"][method_name] = {
                "r2_ut": round(metrics["r2_ut"], 4),
                "r2_uc": round(metrics["r2_uc"], 4),
                "r2_ss": round(metrics["r2_ss"], 4),
                "avg_r2": round(metrics["avg_r2"], 4),
                "avg_normalized_rmse": round(metrics["avg_normalized_rmse"], 4),
                "tail_normalized_mae": round(metrics["tail_normalized_mae"], 4),
                "shape_violation_rate": round(metrics["shape_violation_rate"], 4),
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
    parser = argparse.ArgumentParser(description="Evaluate GI-CANN checkpoints on the held-out test slice")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--predictions-dir", type=str, default="outputs/predictions")
    parser.add_argument("--scores-json", type=str, default="scoring/scores_test.json")
    parser.add_argument("--train-scores-json", type=str, default="scoring/scores_train.json")
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
            scores = build_scores_from_artifacts(
                artifacts,
                args.scores_json,
                rows_by_material=rows_by_material(),
                experiments_filter=experiments_filter,
            )
            validate_train_entries_present(scores, args.train_scores_json, experiments_filter)
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
                        f"{result['mode']} | avg_r2={result['metrics']['avg_r2']:.4f}"
                    )
                except Exception as exc:
                    print(f"  ERROR: {exc}")

    if results:
        scores = build_scores_json(results, args.scores_json, experiments_filter)
        validate_train_entries_present(scores, args.train_scores_json, experiments_filter)


if __name__ == "__main__":
    main()
