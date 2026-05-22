#!/usr/bin/env python3
"""Model-agnostic scoring helpers for prediction artifacts.

The standard gym interface is a directory of JSON artifacts, one per
material/model/mode run. Each artifact contains observed-stress predictions for
the canonical deformation curves. Evaluation code computes metrics from those
predictions and the tracked benchmark data, without loading method checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from data.load_data import (
    BRAIN_SS,
    BRAIN_UC,
    BRAIN_UT,
    RUBBER_BT,
    RUBBER_PS,
    RUBBER_UT,
    compute_r2,
)


RUBBER_MODES = {"ut": RUBBER_UT, "bt": RUBBER_BT, "ps": RUBBER_PS}
BRAIN_MODES = {"ut": BRAIN_UT, "uc": BRAIN_UC, "ss": BRAIN_SS}
RUBBER_REQUIRED_MODES = ["ut", "bt", "ps"]
BRAIN_REQUIRED_MODES = ["ut", "uc", "ss"]


def experiment_and_method(material: str, mode: str, model_name: str) -> tuple[str | None, str | None]:
    """Map an artifact's metadata to scoring experiment and method names."""
    is_si = model_name == "SI-CANN"

    if material == "rubber":
        if mode == "positive":
            return "rubber_generalization", "SI-CANN" if is_si else "GI-CANN"
        if mode == "negative":
            return "rubber_negative_ablation", "SI-CANN_J\u03b1-" if is_si else "GI-CANN_J\u03b1-"
        if mode == "combined":
            return "rubber_combined_ablation", "SI-CANN_J\u03b1+J\u03b1-" if is_si else "GI-CANN_J\u03b1+J\u03b1-"
        return None, None

    if mode == "negative":
        return "brain_generalization", "SI-CANN" if is_si else "GI-CANN"
    if mode == "positive":
        return "brain_positive_ablation", "SI-CANN_J\u03b1+" if is_si else "GI-CANN_J\u03b1+"
    if mode == "combined":
        return "brain_combined_ablation", "SI-CANN_J\u03b1+J\u03b1-" if is_si else "GI-CANN_J\u03b1+J\u03b1-"
    return None, None


def mode_targets(material: str) -> dict[str, np.ndarray]:
    """Return observed stress targets by deformation mode."""
    source = RUBBER_MODES if material == "rubber" else BRAIN_MODES
    return {mode: values[:, 1].astype(float) for mode, values in source.items()}


def load_prediction_artifacts(predictions_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load all prediction artifacts from a directory."""
    root = Path(predictions_dir)
    if not root.exists():
        return []
    artifacts = []
    for path in sorted(root.glob("*.json")):
        with open(path) as f:
            payload = json.load(f)
        payload["_path"] = str(path)
        artifacts.append(payload)
    return artifacts


def alpha_star_from_artifact(artifact: dict[str, Any]) -> float:
    """Extract the branch alpha reported in the paper tables."""
    if "alpha_star" in artifact:
        return float(artifact["alpha_star"])

    model_name = artifact.get("model", "")
    mode = artifact.get("mode", "")
    params = artifact.get("params", {})
    if model_name == "SI-CANN":
        return float(params.get("alpha_fixed", 2.0 if mode == "positive" else -2.0))
    if mode == "negative":
        return float(params.get("alpha_minus", 0.0))
    if mode == "positive":
        return float(params.get("alpha_plus", 0.0))
    if mode == "combined":
        if artifact.get("material") == "brain":
            return float(params.get("alpha_minus", params.get("alpha_plus", 0.0)))
        return float(params.get("alpha_plus", params.get("alpha_minus", 0.0)))
    return 0.0


def curve_quality_metrics(
    material: str,
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute fit, magnitude-error, and shape metrics for selected rows."""
    required = RUBBER_REQUIRED_MODES if material == "rubber" else BRAIN_REQUIRED_MODES

    r2 = {}
    normalized_rmse = []
    tail_normalized_abs_error = []
    shape_violation_rates = []
    for mode_name in required:
        if mode_name not in predictions:
            continue

        pred = np.asarray(predictions[mode_name], dtype=float).flatten()
        target = np.asarray(targets[mode_name], dtype=float).flatten()
        if len(pred) != len(target):
            raise ValueError(
                f"{mode_name}: selected prediction length {len(pred)} != target length {len(target)}"
            )
        if len(target) == 0:
            continue

        scale = float(np.ptp(target))
        if scale <= 1e-12:
            scale = max(float(np.max(np.abs(target))), 1.0)

        residual = pred - target
        r2[mode_name] = float(compute_r2(pred, target))
        normalized_rmse.append(float(np.sqrt(np.mean(residual**2)) / scale))
        tail_normalized_abs_error.append(float(abs(residual[-1]) / scale))

        if len(target) > 1:
            target_delta = np.diff(target)
            pred_delta = np.diff(pred)
            meaningful = np.abs(target_delta) > 1e-12
            if np.any(meaningful):
                violations = pred_delta[meaningful] * target_delta[meaningful] < -1e-12
                shape_violation_rates.append(float(np.mean(violations)))

    metrics = {
        "avg_r2": float(np.mean([r2.get(mode, 0.0) for mode in required])),
        "avg_normalized_rmse": float(np.mean(normalized_rmse)) if normalized_rmse else 0.0,
        "tail_normalized_mae": (
            float(np.mean(tail_normalized_abs_error)) if tail_normalized_abs_error else 0.0
        ),
        "shape_violation_rate": float(np.mean(shape_violation_rates)) if shape_violation_rates else 0.0,
    }
    if material == "rubber":
        metrics.update(
            {
                "r2_ut": r2.get("ut", 0.0),
                "r2_bt": r2.get("bt", 0.0),
                "r2_ps": r2.get("ps", 0.0),
            }
        )
    else:
        metrics.update(
            {
                "r2_ut": r2.get("ut", 0.0),
                "r2_uc": r2.get("uc", 0.0),
                "r2_ss": r2.get("ss", 0.0),
            }
        )
    return metrics


def score_artifact_rows(artifact: dict[str, Any], rows_by_mode: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute metrics for one prediction artifact on evaluator-selected rows."""
    material = artifact["material"]
    artifact_predictions = artifact.get("predictions", {})
    full_targets = mode_targets(material)

    predictions = {}
    targets = {}
    for mode_name, target in full_targets.items():
        if mode_name not in artifact_predictions:
            continue
        pred = np.asarray(artifact_predictions[mode_name], dtype=float)
        if len(pred) != len(target):
            raise ValueError(
                f"{artifact.get('_path', '<artifact>')} mode {mode_name}: "
                f"prediction length {len(pred)} != target length {len(target)}"
            )
        idx = np.asarray(rows_by_mode[mode_name], dtype=int)
        predictions[mode_name] = pred[idx]
        targets[mode_name] = target[idx]

    metrics = curve_quality_metrics(material, predictions, targets)
    metrics["alpha_star"] = alpha_star_from_artifact(artifact)
    return metrics


def build_scores_from_artifacts(
    artifacts: list[dict[str, Any]],
    output_path: str,
    rows_by_material: dict[str, dict[str, np.ndarray]],
    experiments_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Build a scores.json-compatible payload from prediction artifacts."""
    exp_data: dict[str, dict[str, Any]] = {}

    for artifact in artifacts:
        exp_name, method_name = experiment_and_method(
            artifact.get("material", ""),
            artifact.get("mode", ""),
            artifact.get("model", ""),
        )
        if exp_name is None or method_name is None:
            continue
        if experiments_filter and exp_name not in experiments_filter:
            continue
        material = artifact.get("material", "")
        if material not in rows_by_material:
            continue
        metrics = {
            key: round(float(value), 4)
            for key, value in score_artifact_rows(
                artifact,
                rows_by_material[material],
            ).items()
        }
        exp_data.setdefault(exp_name, {"results": {}})
        exp_data[exp_name]["results"][method_name] = metrics

    scores = {"experiments": dict(sorted(exp_data.items()))}
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")
    print(f"Scores written to {output_path}")
    return scores


def all_rows_by_material() -> dict[str, dict[str, np.ndarray]]:
    """Rows for the original full-data evaluator."""
    return {
        "rubber": {mode: np.arange(len(values), dtype=int) for mode, values in RUBBER_MODES.items()},
        "brain": {mode: np.arange(len(values), dtype=int) for mode, values in BRAIN_MODES.items()},
    }
