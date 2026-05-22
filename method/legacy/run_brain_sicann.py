#!/usr/bin/env python3
"""Quick retrain brain SI-CANN models with correct settings."""

import torch
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from method.sicann import SICANN
from data.load_data import get_brain_dataset, compute_r2


def train(model, train_ds, test_ds, epochs, lr, seed, train_modes, test_modes, grad_clip=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    F_train, S_train, gamma_train = train_ds.F, train_ds.S, train_ds.gamma
    F_test, S_test, gamma_test = test_ds.F, test_ds.S, test_ds.gamma

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
        for name, param in model.named_parameters():
            if 'w_exp' in name:
                with torch.no_grad():
                    param.clamp_(min=1e-6)

        if (epoch + 1) % log_interval == 0:
            model.eval()
            with torch.no_grad():
                S_pred = model(F_test, gamma_override=gamma_test)
                test_mse = float(torch.mean((S_pred - S_test) ** 2))
                r2_train, r2_test = {}, {}
                for mn in train_modes:
                    idx = train_ds.mode_indices[mn]
                    p = model(F_train[idx], gamma_override=gamma_train[idx]) if mn == 'ss' else model(F_train[idx])
                    r2_train[mn] = float(compute_r2(p.cpu().numpy().flatten(), S_train[idx].cpu().numpy().flatten()))
                for mn in test_modes:
                    idx = test_ds.mode_indices[mn]
                    p = model(F_test[idx], gamma_override=gamma_test[idx]) if mn == 'ss' else model(F_test[idx])
                    r2_test[mn] = float(compute_r2(p.cpu().numpy().flatten(), S_test[idx].cpu().numpy().flatten()))
                avg_r2 = np.mean(list(r2_train.values()) + list(r2_test.values()))
            print(f"  Ep {epoch+1:5d}/{epochs} | MSE: {test_mse:.6f} | R²: {avg_r2:.4f}")
            if test_mse < best_loss:
                best_loss = test_mse
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 500:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        r2_train, r2_test = {}, {}
        for mn in train_modes:
            idx = train_ds.mode_indices[mn]
            p = model(F_train[idx], gamma_override=gamma_train[idx]) if mn == 'ss' else model(F_train[idx])
            r2_train[mn] = float(compute_r2(p.cpu().numpy().flatten(), S_train[idx].cpu().numpy().flatten()))
        for mn in test_modes:
            idx = test_ds.mode_indices[mn]
            p = model(F_test[idx], gamma_override=gamma_test[idx]) if mn == 'ss' else model(F_test[idx])
            r2_test[mn] = float(compute_r2(p.cpu().numpy().flatten(), S_test[idx].cpu().numpy().flatten()))
    avg_r2 = np.mean(list(r2_train.values()) + list(r2_test.values()))
    return r2_train, r2_test, avg_r2, model.get_parameters_dict(), epoch + 1


train_ds, test_ds = get_brain_dataset()
train_modes, test_modes = ['ut', 'uc'], ['ss']

# Brain positive SI-CANN
print("\n=== Brain Positive SI-CANN ===")
model = SICANN({'mode': 'positive', 'r': 0.01})
r2_train, r2_test, avg_r2, params, epochs = train(model, train_ds, test_ds, 20000, 0.001, 42, train_modes, test_modes, grad_clip=0.5)
print(f"Final: avg_r2={avg_r2:.4f}, params={params}")
torch.save({'model_state': model.state_dict(), 'config': {'mode': 'positive', 'r': 0.01},
             'results': {'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2, 'params': params, 'epochs': epochs}},
            'checkpoints/brain_positive_sicann.pt')

# Brain negative SI-CANN
print("\n=== Brain Negative SI-CANN ===")
model = SICANN({'mode': 'negative', 'r': 0.01})
r2_train, r2_test, avg_r2, params, epochs = train(model, train_ds, test_ds, 20000, 0.001, 42, train_modes, test_modes, grad_clip=0.5)
print(f"Final: avg_r2={avg_r2:.4f}, params={params}")
torch.save({'model_state': model.state_dict(), 'config': {'mode': 'negative', 'r': 0.01},
             'results': {'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2, 'params': params, 'epochs': epochs}},
            'checkpoints/brain_negative_sicann.pt')

# Brain combined SI-CANN
print("\n=== Brain Combined SI-CANN ===")
model = SICANN({'mode': 'combined', 'r': 0.01})
r2_train, r2_test, avg_r2, params, epochs = train(model, train_ds, test_ds, 20000, 0.001, 42, train_modes, test_modes, grad_clip=0.5)
print(f"Final: avg_r2={avg_r2:.4f}, params={params}")
torch.save({'model_state': model.state_dict(), 'config': {'mode': 'combined', 'r': 0.01},
             'results': {'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2, 'params': params, 'epochs': epochs}},
            'checkpoints/brain_combined_sicann.pt')

print("\n=== Done ===")
