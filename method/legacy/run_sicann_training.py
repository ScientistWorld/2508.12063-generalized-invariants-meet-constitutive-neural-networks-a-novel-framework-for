#!/usr/bin/env python3
"""Retrain SI-CANN models with fixes.

Key fixes:
- Rubber combined SI-CANN: clamp w_ln1_plus ≤ 0 (prevents LN branch divergence)
- Brain SI-CANN: no grad clip, r=0.001 (reduced L1)
"""

import torch
import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from method.sicann import SICANN
from data.load_data import get_rubber_dataset, get_brain_dataset, compute_r2


def train(model, train_ds, test_ds, epochs, lr, seed, train_modes, test_modes, device,
          grad_clip=0.5, clamp_rules=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        # Clamp exp weights to prevent overflow
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
                S_test_pred = model(F_test, gamma_override=gamma_test)
                test_mse = float(torch.mean((S_test_pred - S_test) ** 2))
                r2_train, r2_test = {}, {}
                for mn in train_modes:
                    idx = train_ds.mode_indices[mn]
                    idx_t = torch.tensor(idx, device=device, dtype=torch.long)
                    p = model(F_train[idx_t], gamma_override=gamma_train[idx_t]) if mn == 'ss' else model(F_train[idx_t])
                    r2_train[mn] = float(compute_r2(p.cpu().numpy().flatten(), S_train[idx_t].cpu().numpy().flatten()))
                for mn in test_modes:
                    idx = test_ds.mode_indices[mn]
                    idx_t = torch.tensor(idx, device=device, dtype=torch.long)
                    p = model(F_test[idx_t], gamma_override=gamma_test[idx_t]) if mn == 'ss' else model(F_test[idx_t])
                    r2_test[mn] = float(compute_r2(p.cpu().numpy().flatten(), S_test[idx_t].cpu().numpy().flatten()))
                avg_r2 = np.mean(list(r2_train.values()) + list(r2_test.values()))
            alpha_info = {k: round(v, 4) for k, v in model.get_parameters_dict().items() if 'alpha' in k}
            print(f"  Ep {epoch+1:5d}/{epochs} | MSE: {test_mse:.6f} | R²: {avg_r2:.4f} | {alpha_info}")
            if test_mse < best_loss:
                best_loss = test_mse
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 500:
                    print(f"  Early stop at epoch {epoch+1}")
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
            r2_train[mn] = float(compute_r2(p.cpu().numpy().flatten(), S_train[idx_t].cpu().numpy().flatten()))
        for mn in test_modes:
            idx = test_ds.mode_indices[mn]
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            p = model(F_test[idx_t], gamma_override=gamma_test[idx_t]) if mn == 'ss' else model(F_test[idx_t])
            r2_test[mn] = float(compute_r2(p.cpu().numpy().flatten(), S_test[idx_t].cpu().numpy().flatten()))
    avg_r2 = np.mean(list(r2_train.values()) + list(r2_test.values()))
    return r2_train, r2_test, avg_r2, model.get_parameters_dict(), epoch + 1


def main():
    device = torch.device('cpu')
    results = []

    configs = [
        ('rubber', 'positive', SICANN, 20000, 0.001),
        ('rubber', 'negative', SICANN, 20000, 0.001),
        ('rubber', 'combined', SICANN, 20000, 0.001),
        ('brain', 'positive', SICANN, 20000, 0.001),
        ('brain', 'negative', SICANN, 20000, 0.001),
        ('brain', 'combined', SICANN, 20000, 0.001),
    ]

    for material, mode, model_cls, ep, lr in configs:
        print(f"\n{'='*50}")
        print(f"SI-CANN: {material} ({mode})")
        print(f"{'='*50}")

        if material == 'rubber':
            train_ds, test_ds = get_rubber_dataset()
            train_modes, test_modes = ['ut', 'bt'], ['ps']
        else:
            train_ds, test_ds = get_brain_dataset()
            train_modes, test_modes = ['ut', 'uc'], ['ss']

        brain_l1 = 0.001 if material == 'brain' else 0.01
        model = model_cls({'mode': mode, 'r': brain_l1})

        clamp_rules = {}
        if material == 'rubber' and mode in ('negative', 'combined'):
            with torch.no_grad():
                model.w_exp1_minus.fill_(0.001)
                model.w_exp2_minus.fill_(0.001)
            print(f"  Exp override: w_exp1_minus=0.001, w_exp2_minus=0.001")

        # CRITICAL FIX: w_ln1_plus must stay ≤ 0 for rubber combined.
        # Positive w_ln1_plus causes ln(1 - positive*(J-3)) to become negative near J-3=10
        # (BT reaches J-3 up to ~10), causing the LN branch derivative to explode.
        if material == 'rubber' and mode == 'combined':
            clamp_rules['w_ln1_plus'] = (None, -1e-6)
            print(f"  LN clamp: w_ln1_plus ≤ 0 (prevents BT singularity)")

        # Brain: no grad clip, reduced L1
        grad_clip = 0.0 if material == 'brain' else 0.5
        if material == 'brain' and mode in ('negative', 'combined'):
            with torch.no_grad():
                model.w_exp1_minus.fill_(0.1)
                model.w_exp2_minus.fill_(0.1)
            print(f"  Brain exp: w_exp1_minus=0.1, w_exp2_minus=0.1")

        r2_train, r2_test, avg_r2, params, epochs = train(
            model, train_ds, test_ds, ep, lr, 42,
            train_modes, test_modes, device, grad_clip=grad_clip,
            clamp_rules=clamp_rules
        )

        print(f"\nFinal: r2_train={r2_train}, r2_test={r2_test}, avg_r2={avg_r2:.4f}")
        print(f"Params: {params}")

        ckpt_name = f'{material}_{mode}_sicann'
        torch.save({
            'model_state': model.state_dict(),
            'config': {'mode': mode, 'r': brain_l1},
            'results': {'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2, 'params': params, 'epochs': epochs}
        }, f'checkpoints/{ckpt_name}.pt')

        results.append({'material': material, 'mode': mode, 'model': 'SI-CANN',
                       'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2, 'params': params})

    # Save summary
    with open('checkpoints/sicann_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== SI-CANN RESULTS ===")
    for r in results:
        print(f"  {r['material']} {r['mode']}: avg_r2={r['avg_r2']:.4f}")


if __name__ == '__main__':
    main()
