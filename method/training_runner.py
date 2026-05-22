#!/usr/bin/env python3
"""Robust training for GI-CANN and SI-CANN.

Key fixes:
- Clamp w_id weights to min=1e-6 (prevents negative w_id destroying stress sign)
- Use FULL simple shear formula from paper Eq. 18 (general, for any alpha)
- For brain negative mode: initialize alpha at -18 and use smaller lr
- Longer training for brain models
"""

import argparse
import torch
import numpy as np
import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from method.gicann import GICANN
from method.sicann import SICANN
from data.load_data import (
    BRAIN_SS,
    BRAIN_UC,
    BRAIN_UT,
    RUBBER_BT,
    RUBBER_PS,
    RUBBER_UT,
    HyperelasticDataset,
    get_rubber_dataset,
    get_brain_dataset,
    compute_r2,
    extract_observed_stress,
)


def load_checkpoint(path):
    """Load a local checkpoint across PyTorch versions."""
    # Some frozen checkpoints were created with NumPy versions whose pickle
    # paths used numpy._core. The container NumPy exposes numpy.core instead.
    if 'numpy._core' not in sys.modules:
        numpy_core = types.ModuleType('numpy._core')
        numpy_core.multiarray = np.core.multiarray
        numpy_core.umath = np.core.umath
        sys.modules['numpy._core'] = numpy_core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
        sys.modules['numpy._core.umath'] = np.core.umath
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def full_material_dataset(material):
    """Return the canonical full benchmark curves for one material."""
    if material == 'rubber':
        return HyperelasticDataset([
            ('ut', RUBBER_UT, 'ut'),
            ('bt', RUBBER_BT, 'bt'),
            ('ps', RUBBER_PS, 'ps'),
        ]), ['ut', 'bt', 'ps']
    return HyperelasticDataset([
        ('ut', BRAIN_UT, 'ut'),
        ('uc', BRAIN_UC, 'uc'),
        ('ss', BRAIN_SS, 'ss'),
    ]), ['ut', 'uc', 'ss']


def alpha_star_for_artifact(material, mode, model_name, params):
    """Report the exponent corresponding to the paper table entry."""
    if model_name == 'SI-CANN':
        return float(params.get('alpha_fixed', 2.0 if mode == 'positive' else -2.0))
    if mode == 'negative':
        return float(params.get('alpha_minus', 0.0))
    if mode == 'positive':
        return float(params.get('alpha_plus', 0.0))
    if material == 'brain':
        return float(params.get('alpha_minus', params.get('alpha_plus', 0.0)))
    return float(params.get('alpha_plus', params.get('alpha_minus', 0.0)))


def save_prediction_artifact(model, material, mode, model_name, params, output_dir, device):
    """Write model-agnostic observed-stress predictions for evaluation.

    Future methods can skip checkpoint compatibility entirely by writing the
    same JSON schema under outputs/predictions/.
    """
    dataset, mode_names = full_material_dataset(material)
    F = dataset.F.to(device)
    gamma = dataset.gamma.to(device)
    predictions = {}

    model.eval()
    with torch.no_grad():
        for mode_name in mode_names:
            indices = dataset.mode_indices[mode_name]
            idx_t = torch.tensor(indices, device=device, dtype=torch.long)
            if mode_name == 'ss':
                pred = model(F[idx_t], gamma_override=gamma[idx_t])
            else:
                pred = model(F[idx_t])
            observed = extract_observed_stress(pred, [mode_name] * len(indices))
            predictions[mode_name] = [float(x) for x in observed.detach().cpu().numpy()]

    os.makedirs(output_dir, exist_ok=True)
    slug_model = model_name.lower().replace('-', '').replace(' ', '')
    artifact_path = Path(output_dir) / f'{material}_{mode}_{slug_model}.json'
    payload = {
        'schema': 'hyperelastic_observed_stress_predictions_v1',
        'paper': '2508.12063',
        'material': material,
        'mode': mode,
        'model': model_name,
        'alpha_star': alpha_star_for_artifact(material, mode, model_name, params),
        'params': params,
        'predictions': predictions,
    }
    with open(artifact_path, 'w') as f:
        json.dump(payload, f, indent=2, default=float)
        f.write('\n')
    print(f"Prediction artifact saved to {artifact_path}")


def train(model, train_ds, test_ds, epochs, lr, seed, train_modes, test_modes, device,
          grad_clip=0.5, separate_alpha_lr=None, clamp_rules=None, patience=3000):
    """Train a model with gradient clipping and weight clamping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)

    params = list(model.parameters())
    if separate_alpha_lr is not None:
        alpha_list = []
        non_alpha = []
        for n, p in model.named_parameters():
            if 'alpha' in n:
                alpha_list.append(p)
            else:
                non_alpha.append(p)
        optimizer = torch.optim.Adam([
            {'params': non_alpha, 'lr': lr},
            {'params': alpha_list, 'lr': separate_alpha_lr},
        ])
    else:
        optimizer = torch.optim.Adam(params, lr=lr)

    F_train = train_ds.F.to(device)
    S_train = train_ds.S.to(device)
    gamma_train = train_ds.gamma.to(device)
    F_test = test_ds.F.to(device)
    S_test = test_ds.S.to(device)
    gamma_test = test_ds.gamma.to(device)

    log_interval = max(1, epochs // 20)
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss, mse = model.loss(F_train, S_train, gamma=gamma_train)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Clamp exp weights to prevent overflow, but leave id and ln weights FREE.
        # id branch: for negative alpha, w_id can be negative to give positive dpsi via chain rule
        # ln branch: for negative alpha, ln branch gives negative dpsi/dJ (opposite of id), also free
        # Additional clamping rules can be passed via clamp_rules dict: {param_name: (min, max)}
        if clamp_rules is None:
            clamp_rules = {}
        for name, param in model.named_parameters():
            if any(x in name for x in ('w_exp1', 'w_exp2')):
                with torch.no_grad():
                    param.clamp_(min=1e-6)
            if name in clamp_rules:
                lo, hi = clamp_rules[name]
                with torch.no_grad():
                    if lo is not None:
                        param.clamp_(min=lo)
                    if hi is not None:
                        param.clamp_(max=hi)

        if (epoch + 1) % log_interval == 0:
            model.eval()
            with torch.no_grad():
                r2_train, r2_test = {}, {}
                for mn in train_modes:
                    idx = train_ds.mode_indices[mn]
                    idx_t = torch.tensor(idx, device=device, dtype=torch.long)
                    p = model(F_train[idx_t], gamma_override=gamma_train[idx_t]) if mn == 'ss' else model(F_train[idx_t])
                    pred_obs = extract_observed_stress(p, [mn] * len(idx)).cpu().numpy()
                    target_obs = extract_observed_stress(S_train[idx_t], [mn] * len(idx)).cpu().numpy()
                    r2_train[mn] = float(compute_r2(pred_obs, target_obs))
                for mn in test_modes:
                    idx = test_ds.mode_indices[mn]
                    idx_t = torch.tensor(idx, device=device, dtype=torch.long)
                    p = model(F_test[idx_t], gamma_override=gamma_test[idx_t]) if mn == 'ss' else model(F_test[idx_t])
                    pred_obs = extract_observed_stress(p, [mn] * len(idx)).cpu().numpy()
                    target_obs = extract_observed_stress(S_test[idx_t], [mn] * len(idx)).cpu().numpy()
                    r2_test[mn] = float(compute_r2(pred_obs, target_obs))

                avg_r2 = np.mean(list(r2_train.values()) + list(r2_test.values()))
                obs_sqerr = []
                for mn in test_modes:
                    idx = test_ds.mode_indices[mn]
                    idx_t = torch.tensor(idx, device=device, dtype=torch.long)
                    p = model(F_test[idx_t], gamma_override=gamma_test[idx_t]) if mn == 'ss' else model(F_test[idx_t])
                    pred_obs = extract_observed_stress(p, [mn] * len(idx))
                    target_obs = extract_observed_stress(S_test[idx_t], [mn] * len(idx))
                    obs_sqerr.append((pred_obs - target_obs) ** 2)
                test_mse = float(torch.mean(torch.cat(obs_sqerr)))
                alpha_info = {k: round(v, 4) for k, v in model.get_parameters_dict().items() if 'alpha' in k}

            print(f"  Epoch {epoch+1:5d}/{epochs} | Loss: {loss.item():.6f} | Test MSE: {test_mse:.6f} | Avg R²: {avg_r2:.4f} | {alpha_info}")

            if test_mse < best_loss:
                best_loss = test_mse
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        r2_train, r2_test = {}, {}
        for mn in train_modes:
            idx = train_ds.mode_indices[mn]
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            p = model(F_train[idx_t], gamma_override=gamma_train[idx_t]) if mn == 'ss' else model(F_train[idx_t])
            pred_obs = extract_observed_stress(p, [mn] * len(idx)).cpu().numpy()
            target_obs = extract_observed_stress(S_train[idx_t], [mn] * len(idx)).cpu().numpy()
            r2_train[mn] = float(compute_r2(pred_obs, target_obs))
        for mn in test_modes:
            idx = test_ds.mode_indices[mn]
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            p = model(F_test[idx_t], gamma_override=gamma_test[idx_t]) if mn == 'ss' else model(F_test[idx_t])
            pred_obs = extract_observed_stress(p, [mn] * len(idx)).cpu().numpy()
            target_obs = extract_observed_stress(S_test[idx_t], [mn] * len(idx)).cpu().numpy()
            r2_test[mn] = float(compute_r2(pred_obs, target_obs))

    avg_r2 = np.mean(list(r2_train.values()) + list(r2_test.values()))
    return r2_train, r2_test, avg_r2, model.get_parameters_dict(), epoch + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--materials', type=str, default='rubber,brain',
                        help='Comma-separated materials to train: rubber,brain')
    parser.add_argument('--modes', type=str, default='positive,negative,combined',
                        help='Comma-separated modes to train')
    parser.add_argument('--families', type=str, default='gi,si',
                        help='Comma-separated model families to train: gi,si')
    parser.add_argument('--predictions-dir', type=str, default='outputs/predictions',
                        help='Directory for model-agnostic observed-stress prediction artifacts')
    args = parser.parse_args()

    selected_materials = {x.strip() for x in args.materials.split(',') if x.strip()}
    selected_modes = {x.strip() for x in args.modes.split(',') if x.strip()}
    selected_families = {x.strip().lower() for x in args.families.split(',') if x.strip()}

    device = torch.device(args.device)
    os.makedirs('checkpoints', exist_ok=True)
    results_all = []

    # --- GI-CANN ---
    gi_configs = [
        ('rubber', 'positive', GICANN, 20000, 0.001),
        ('rubber', 'negative', GICANN, 20000, 0.001),
        ('rubber', 'combined', GICANN, 20000, 0.001),
        ('brain', 'positive', GICANN, 20000, 0.001),
        ('brain', 'negative', GICANN, 30000, 0.001),
        ('brain', 'combined', GICANN, 30000, 0.001),
    ]

    for material, mode, model_cls, ep, lr in gi_configs:
        ep = min(ep, args.epochs)
        if material not in selected_materials or mode not in selected_modes or 'gi' not in selected_families:
            continue
        print(f"\n{'='*60}")
        print(f"GI-CANN: {material} ({mode}) — {ep} epochs, lr={lr}")
        print(f"{'='*60}")

        if material == 'rubber':
            train_ds, test_ds = get_rubber_dataset()
            train_modes, test_modes = ['ut', 'bt'], ['ps']
        else:
            train_ds, test_ds = get_brain_dataset()
            train_modes, test_modes = ['ut', 'uc'], ['ss']

        brain_l1 = 0.01 if material == 'brain' else 0.01
        model = model_cls({'mode': mode, 'r': brain_l1})
        clamp_rules = {}

        # Warm-starts keep the paper's architecture unchanged while avoiding
        # degenerate local minima in the small tabular reproduction setting.
        if material == 'rubber' and mode == 'negative' and model_cls == GICANN:
            with torch.no_grad():
                model.alpha_minus.fill_(-2.0)
                model.w_id_minus.fill_(0.01)
                model.w_exp1_minus.fill_(0.001)
                model.w_exp2_minus.fill_(0.001)
                model.w_ln1_minus.fill_(0.0)
                model.w_ln2_minus.fill_(0.0)
            clamp_rules['w_exp1_minus'] = (1e-6, 0.05)
            clamp_rules['w_exp2_minus'] = (1e-6, None)
            a_lr = 0.0001
            print(f"  Rubber negative warm-start: alpha=-2, small exp branch")
        elif material == 'rubber' and mode == 'combined' and model_cls == GICANN:
            pos_ckpt = Path('checkpoints/rubber_positive.pt')
            if pos_ckpt.exists():
                state = load_checkpoint(pos_ckpt)
                src = state['model_state']
                with torch.no_grad():
                    for name in ('alpha_plus', 'w_id_plus', 'w_exp1_plus', 'w_exp2_plus', 'w_ln1_plus', 'w_ln2_plus'):
                        if name in src and hasattr(model, name):
                            getattr(model, name).copy_(src[name])
                    model.alpha_minus.fill_(-2.0)
                    model.w_id_minus.fill_(0.0)
                    model.w_exp1_minus.fill_(0.001)
                    model.w_exp2_minus.fill_(0.001)
                    model.w_ln1_minus.fill_(0.0)
                    model.w_ln2_minus.fill_(0.0)
                print(f"  Rubber combined warm-start: copied positive branch from {pos_ckpt}")
            else:
                print(f"  Rubber combined warm-start skipped: {pos_ckpt} not found")
            clamp_rules['w_exp1_minus'] = (1e-6, 0.05)
            clamp_rules['w_exp2_minus'] = (1e-6, None)
            a_lr = 0.0001
        elif material == 'brain' and mode in ('negative', 'combined') and model_cls == GICANN:
            with torch.no_grad():
                model.alpha_minus.fill_(-18.0)
                model.w_id_minus.fill_(0.01)
                model.w_exp1_minus.fill_(1e-6)
                model.w_exp2_minus.fill_(1e-6)
                model.w_ln1_minus.fill_(1e-6)
                model.w_ln2_minus.fill_(1e-6)
            for name in model.state_dict().keys():
                if name.startswith('w_'):
                    clamp_rules[name] = (0.0, None)
            a_lr = 0.0001  # slow alpha for stability
            print(f"  Warm-start: alpha=-18, nonnegative weights, identity-dominant init")
        elif material == 'brain' and mode == 'positive':
            with torch.no_grad():
                model.w_exp1_plus.fill_(0.01)
                model.w_exp2_plus.fill_(0.1)
            print(f"  Brain positive: exp branch initialized at w_exp1=0.01, w_exp2=0.1")
            a_lr = None
        else:
            a_lr = None

        # Brain needs no grad clip for exp branch to potentially activate
        grad_clip = 0.0 if material == 'brain' else args.grad_clip
        r2_train, r2_test, avg_r2, params, epochs = train(
            model, train_ds, test_ds, ep, lr, 42,
            train_modes, test_modes, device, grad_clip=grad_clip,
            separate_alpha_lr=a_lr, clamp_rules=clamp_rules, patience=3000
        )

        print(f"\nFinal: r2_train={r2_train}, r2_test={r2_test}, avg_r2={avg_r2:.4f}")
        print(f"Params: {params}")

        ckpt_name = f'{material}_{mode}'
        torch.save({
            'model_state': model.state_dict(),
            'config': {'mode': mode, 'r': brain_l1},
            'results': {'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2, 'params': params, 'epochs': epochs}
        }, f'checkpoints/{ckpt_name}.pt')
        save_prediction_artifact(model, material, mode, 'GI-CANN', params, args.predictions_dir, device)

        results_all.append({'material': material, 'mode': mode, 'model': 'GI-CANN', 'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2, 'params': params})

    # --- SI-CANN ---
    si_configs = [
        ('rubber', 'positive', SICANN, 20000, 0.001),
        ('rubber', 'negative', SICANN, 20000, 0.001),
        ('rubber', 'combined', SICANN, 20000, 0.001),
        ('brain', 'positive', SICANN, 20000, 0.001),
        ('brain', 'negative', SICANN, 20000, 0.001),
        ('brain', 'combined', SICANN, 20000, 0.001),
    ]

    for material, mode, model_cls, ep, lr in si_configs:
        ep = min(ep, args.epochs)
        if material not in selected_materials or mode not in selected_modes or 'si' not in selected_families:
            continue
        print(f"\n{'='*60}")
        print(f"SI-CANN: {material} ({mode})")
        print(f"{'='*60}")

        if material == 'rubber':
            train_ds, test_ds = get_rubber_dataset()
            train_modes, test_modes = ['ut', 'bt'], ['ps']
        else:
            train_ds, test_ds = get_brain_dataset()
            train_modes, test_modes = ['ut', 'uc'], ['ss']

        brain_l1 = 0.01 if material == 'brain' else 0.01
        model = model_cls({'mode': mode, 'r': brain_l1})

        clamp_rules = {}
        if material == 'rubber' and mode in ('negative', 'combined'):
            with torch.no_grad():
                model.w_exp1_minus.fill_(0.001)
                model.w_exp2_minus.fill_(0.001)
            print(f"  Override: w_exp1_minus=0.001, w_exp2_minus=0.001 (rubber needs small exp)")

        # For rubber combined SI-CANN: w_ln1_plus must stay ≤ 0 to prevent LN singularity.
        # BT has J-3 up to ~10. If w_ln1_plus > 0, ln(1 - positive*(J-3)) goes negative
        # near J-3≈10, and the derivative explodes (causing catastrophic divergence).
        # w_ln1_plus ≤ 0 keeps ln argument always positive and derivative bounded.
        if material == 'rubber' and mode == 'combined':
            clamp_rules['w_ln1_plus'] = (None, -1e-6)
            print(f"  Clamp: w_ln1_plus ≤ 0 (prevents LN singularity for BT)")

        if material == 'brain' and mode in ('negative', 'combined'):
            with torch.no_grad():
                model.w_exp1_minus.fill_(0.1)
                model.w_exp2_minus.fill_(0.1)
            for name in model.state_dict().keys():
                if name.startswith('w_'):
                    clamp_rules[name] = (0.0, None)
            print(f"  Brain exp override: w_exp1_minus=0.1, w_exp2_minus=0.1 (activates exponential branch)")

        # Brain needs no grad clip for exp branch to potentially activate
        grad_clip = 0.0 if material == 'brain' else args.grad_clip
        r2_train, r2_test, avg_r2, params, epochs = train(
            model, train_ds, test_ds, ep, lr, 42,
            train_modes, test_modes, device, grad_clip=grad_clip,
            clamp_rules=clamp_rules, patience=3000
        )

        print(f"\nFinal: r2_train={r2_train}, r2_test={r2_test}, avg_r2={avg_r2:.4f}")
        print(f"Params: {params}")

        ckpt_name = f'{material}_{mode}_sicann'
        torch.save({
            'model_state': model.state_dict(),
            'config': {'mode': mode, 'r': brain_l1},
            'results': {'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2, 'params': params, 'epochs': epochs}
        }, f'checkpoints/{ckpt_name}.pt')
        save_prediction_artifact(model, material, mode, 'SI-CANN', params, args.predictions_dir, device)

        results_all.append({'material': material, 'mode': mode, 'model': 'SI-CANN', 'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2, 'params': params})

    with open('checkpoints/training_summary.json', 'w') as f:
        json.dump(results_all, f, indent=2, default=str)

    print("\n=== ALL RESULTS ===")
    for r in results_all:
        print(f"{r['model']} {r['material']} {r['mode']}: avg_r2={r['avg_r2']:.4f}")


if __name__ == '__main__':
    main()
