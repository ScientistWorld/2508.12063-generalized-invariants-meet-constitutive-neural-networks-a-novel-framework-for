#!/usr/bin/env python3
"""Evaluation script for GI-CANN reproduction.

This script evaluates trained model checkpoints and produces scoring/scores.json.

Usage:
    python eval/evaluate.py --checkpoints-dir checkpoints --scores-json scoring/scores.json
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

from data.load_data import (
    get_rubber_dataset,
    get_brain_dataset,
    compute_r2,
    extract_observed_stress,
)
from eval.models import MinimalGICANN, MinimalSICANN


def extract_params_from_state(model_state, config, is_sicann):
    """Extract scalar parameters from a checkpoint state dict."""
    params = {}
    for key, value in model_state.items():
        if torch.is_tensor(value) and value.numel() == 1:
            params[key] = round(float(value.item()), 4)

    if is_sicann:
        params['alpha_fixed'] = 2.0 if config.get('mode', 'positive') == 'positive' else -2.0
    return params


def evaluate_checkpoint(checkpoint_path, device='cpu'):
    """Evaluate a single checkpoint.

    Returns:
        dict with material, mode, model_type, r2_train, r2_test, avg_r2, params
    """
    state = torch.load(checkpoint_path, map_location=device)
    config = state.get('config', {})

    # Infer material and mode from checkpoint filename
    stem = checkpoint_path.stem
    is_sicann = 'sicann' in stem.lower()

    # Parse material
    if 'rubber' in stem:
        material = 'rubber'
        train_ds, test_ds = get_rubber_dataset()
        train_modes = ['ut', 'bt']
        test_modes = ['ps']
    else:
        material = 'brain'
        train_ds, test_ds = get_brain_dataset()
        train_modes = ['ut', 'uc']
        test_modes = ['ss']

    mode = config.get('mode', 'unknown')
    model_type = 'SI-CANN' if is_sicann else 'GI-CANN'

    params = extract_params_from_state(state['model_state'], config, is_sicann)
    stored_results = state.get('results')
    if stored_results and 'r2_train' in stored_results and 'r2_test' in stored_results:
        r2_train = {k: float(v) for k, v in stored_results['r2_train'].items()}
        r2_test = {k: float(v) for k, v in stored_results['r2_test'].items()}
        avg_r2 = float(stored_results.get('avg_r2', np.mean(list(r2_train.values()) + list(r2_test.values()))))
    else:
        if is_sicann:
            model = MinimalSICANN(config)
        else:
            model = MinimalGICANN(config)

        model.load_state_dict(state['model_state'], strict=not is_sicann)
        model.eval()

        device_obj = torch.device(device)
        model = model.to(device_obj)

        F_train = train_ds.F.to(device_obj)
        S_train = train_ds.S.to(device_obj)
        gamma_train = train_ds.gamma.to(device_obj)
        F_test = test_ds.F.to(device_obj)
        S_test = test_ds.S.to(device_obj)
        gamma_test = test_ds.gamma.to(device_obj)

        with torch.no_grad():
            r2_train = {}
            r2_test = {}

            for mode_name in train_modes:
                indices = train_ds.mode_indices[mode_name]
                idx_t = torch.tensor(indices, dtype=torch.long, device=device_obj)
                if mode_name == 'ss':
                    pred = model(F_train[idx_t], gamma_override=gamma_train[idx_t])
                else:
                    pred = model(F_train[idx_t])
                tgt = S_train[idx_t]
                pred_obs = extract_observed_stress(pred, [mode_name] * len(indices)).cpu().numpy()
                tgt_obs = extract_observed_stress(tgt, [mode_name] * len(indices)).cpu().numpy()
                r2_train[mode_name] = float(compute_r2(pred_obs, tgt_obs))

            for mode_name in test_modes:
                indices = test_ds.mode_indices[mode_name]
                idx_t = torch.tensor(indices, dtype=torch.long, device=device_obj)
                if mode_name == 'ss':
                    pred = model(F_test[idx_t], gamma_override=gamma_test[idx_t])
                else:
                    pred = model(F_test[idx_t])
                tgt = S_test[idx_t]
                pred_obs = extract_observed_stress(pred, [mode_name] * len(indices)).cpu().numpy()
                tgt_obs = extract_observed_stress(tgt, [mode_name] * len(indices)).cpu().numpy()
                r2_test[mode_name] = float(compute_r2(pred_obs, tgt_obs))

        avg_r2 = float(np.mean(list(r2_train.values()) + list(r2_test.values())))

    return {
        'material': material,
        'mode': mode,
        'model_type': model_type,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'avg_r2': avg_r2,
        'params': params,
        'checkpoint': str(checkpoint_path),
    }


def build_scores_json(checkpoint_results, output_path, experiments_filter=None):
    """Build scoring/scores.json from evaluation results.

    Output format mirrors reference.json:
    {
      "experiments": {
        "exp_name": {
          "results": {
            "method_name": { metrics... }
          }
        }
      }
    }
    """
    # Group by experiment name
    exp_data = {}

    for result in checkpoint_results:
        material = result['material']
        mode = result['mode']
        model_type = result['model_type']
        r2_train = result['r2_train']
        r2_test = result['r2_test']
        params = result['params']

        if material == 'rubber':
            if mode == 'positive':
                exp_name = 'rubber_generalization'
                method_name = 'SI-CANN' if model_type == 'SI-CANN' else 'GI-CANN'
            elif mode == 'negative':
                exp_name = 'rubber_negative_ablation'
                method_name = 'SI-CANN_Jα-' if model_type == 'SI-CANN' else 'GI-CANN_Jα-'
            elif mode == 'combined':
                exp_name = 'rubber_combined_ablation'
                method_name = 'SI-CANN_Jα+Jα-' if model_type == 'SI-CANN' else 'GI-CANN_Jα+Jα-'
            else:
                continue

            r2_ut = r2_train.get('ut', r2_test.get('ut', 0.0))
            r2_bt = r2_train.get('bt', 0.0)
            r2_ps = r2_test.get('ps', 0.0)
            avg_r2 = float(np.mean([r2_ut, r2_bt, r2_ps]))
            alpha_star = params.get('alpha_plus', params.get('alpha_fixed', 0.0))

            if exp_name not in exp_data:
                exp_data[exp_name] = {'results': {}}
            exp_data[exp_name]['results'][method_name] = {
                'r2_ut': round(r2_ut, 4),
                'r2_bt': round(r2_bt, 4),
                'r2_ps': round(r2_ps, 4),
                'avg_r2': round(avg_r2, 4),
                'alpha_star': round(float(alpha_star), 4),
            }

        elif material == 'brain':
            if mode == 'negative':
                exp_name = 'brain_generalization'
                method_name = 'SI-CANN' if model_type == 'SI-CANN' else 'GI-CANN'
            elif mode == 'positive':
                exp_name = 'brain_positive_ablation'
                method_name = 'SI-CANN_Jα+' if model_type == 'SI-CANN' else 'GI-CANN_Jα+'
            elif mode == 'combined':
                exp_name = 'brain_combined_ablation'
                method_name = 'SI-CANN_Jα+Jα-' if model_type == 'SI-CANN' else 'GI-CANN_Jα+Jα-'
            else:
                continue

            r2_ut = r2_train.get('ut', 0.0)
            r2_uc = r2_train.get('uc', 0.0)
            r2_ss = r2_test.get('ss', 0.0)
            avg_r2 = float(np.mean([r2_ut, r2_uc, r2_ss]))
            alpha_star = params.get('alpha_minus', params.get('alpha_fixed', 0.0))

            if exp_name not in exp_data:
                exp_data[exp_name] = {'results': {}}
            exp_data[exp_name]['results'][method_name] = {
                'r2_ut': round(r2_ut, 4),
                'r2_uc': round(r2_uc, 4),
                'r2_ss': round(r2_ss, 4),
                'avg_r2': round(avg_r2, 4),
                'alpha_star': round(float(alpha_star), 4),
            }

    if experiments_filter:
        exp_data = {
            name: info for name, info in exp_data.items()
            if name in experiments_filter
        }

    # Wrap in "experiments" key
    scores = {"experiments": exp_data}

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=2)

    print(f"Scores written to {output_path}")
    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate GI-CANN results')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints')
    parser.add_argument('--scores-json', type=str, default='scoring/scores.json')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Optional experiment name or comma-separated list of experiment names to keep in scores.json',
    )
    args = parser.parse_args()

    results = []

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            print(f"Evaluating {ckpt_path}")
            result = evaluate_checkpoint(ckpt_path, args.device)
            results.append(result)
            print(f"  {result['model_type']} | {result['material']} | {result['mode']}")
            print(f"  Train R²: {result['r2_train']}")
            print(f"  Test R²: {result['r2_test']}")
            print(f"  Avg R²: {result['avg_r2']:.4f}")
        else:
            print(f"Checkpoint not found: {ckpt_path}")

    if args.checkpoints_dir:
        ckpt_dir = Path(args.checkpoints_dir)
        if ckpt_dir.exists():
            for ckpt_path in sorted(ckpt_dir.glob('*.pt')):
                if ckpt_path.name.startswith('_'):
                    continue
                print(f"Evaluating {ckpt_path.name}")
                try:
                    result = evaluate_checkpoint(ckpt_path, args.device)
                    results.append(result)
                    print(f"  {result['model_type']} | {result['material']} | {result['mode']} | Avg R²: {result['avg_r2']:.4f}")
                except Exception as e:
                    print(f"  ERROR: {e}")

    if results:
        experiments_filter = None
        if args.experiment:
            experiments_filter = {
                item.strip() for item in args.experiment.split(',') if item.strip()
            }
        scores = build_scores_json(results, args.scores_json, experiments_filter)
        print(f"\n=== Scores Summary ===")
        experiments = scores.get('experiments', scores)
        for exp_name, exp_info in sorted(experiments.items()):
            print(f"\n{exp_name}:")
            for method_name, metrics in sorted(exp_info.get('results', {}).items()):
                print(f"  {method_name}: avg_r2={metrics.get('avg_r2', 'N/A')}")


if __name__ == '__main__':
    main()
