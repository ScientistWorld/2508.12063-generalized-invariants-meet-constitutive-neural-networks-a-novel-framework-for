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


def split_indices(n_rows: int, split: str) -> np.ndarray:
    """Rows used by the full, train-slice, or held-out test-slice evaluator."""
    if split == "full":
        return np.arange(n_rows, dtype=int)
    nonzero = np.arange(1, n_rows, dtype=int)
    if split == "train":
        return nonzero[::2]
    if split == "test":
        return nonzero[1::2]
    raise ValueError(f"Unknown split: {split}")


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


def score_artifact(artifact: dict[str, Any], split: str = "full") -> dict[str, float]:
    """Compute R2 metrics for one prediction artifact."""
    material = artifact["material"]
    predictions = artifact.get("predictions", {})
    targets = mode_targets(material)

    r2 = {}
    for mode_name, target in targets.items():
        if mode_name not in predictions:
            continue
        pred = np.asarray(predictions[mode_name], dtype=float)
        if len(pred) != len(target):
            raise ValueError(
                f"{artifact.get('_path', '<artifact>')} mode {mode_name}: "
                f"prediction length {len(pred)} != target length {len(target)}"
            )
        idx = split_indices(len(target), split)
        r2[mode_name] = float(compute_r2(pred[idx], target[idx]))

    if material == "rubber":
        required = ["ut", "bt", "ps"]
        return {
            "r2_ut": r2.get("ut", 0.0),
            "r2_bt": r2.get("bt", 0.0),
            "r2_ps": r2.get("ps", 0.0),
            "avg_r2": float(np.mean([r2.get("ut", 0.0), r2.get("bt", 0.0), r2.get("ps", 0.0)])),
            "alpha_star": alpha_star_from_artifact(artifact),
        }

    required = ["ut", "uc", "ss"]
    return {
        "r2_ut": r2.get("ut", 0.0),
        "r2_uc": r2.get("uc", 0.0),
        "r2_ss": r2.get("ss", 0.0),
        "avg_r2": float(np.mean([r2.get("ut", 0.0), r2.get("uc", 0.0), r2.get("ss", 0.0)])),
        "alpha_star": alpha_star_from_artifact(artifact),
    }


def build_scores_from_artifacts(
    artifacts: list[dict[str, Any]],
    output_path: str,
    split: str = "full",
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
        metrics = {
            key: round(float(value), 4)
            for key, value in score_artifact(artifact, split=split).items()
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
