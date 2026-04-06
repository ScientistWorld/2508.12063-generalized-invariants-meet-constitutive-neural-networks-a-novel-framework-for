#!/usr/bin/env python3
"""Train GI-CANN model on rubber or brain tissue data.

Usage:
    python method/train.py --material rubber --mode positive --epochs 5000
    python method/train.py --material brain --mode negative --epochs 5000
"""

import argparse
import torch
import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from method.gicann import GICANN
from data.load_data import (
    get_rubber_dataset, get_brain_dataset,
    compute_r2
)


def train_model(model, train_ds, test_ds, epochs=5000, lr=0.001,
                weight_decay=0.0, patience=500, seed=42,
                train_mode_names=None, test_mode_names=None):
    """Train GI-CANN model.

    Args:
        model: GICANN instance
        train_ds: training HyperelasticDataset (returns F, S, gamma)
        test_ds: test HyperelasticDataset
        epochs: max training epochs
        lr: learning rate
        weight_decay: weight decay (L2)
        patience: early stopping patience
        seed: random seed
        train_mode_names: list of training mode names
        test_mode_names: list of test mode names

    Returns:
        results dict with R² per mode and parameters
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare data
    F_train = train_ds.F.to(device)
    S_train = train_ds.S.to(device)
    gamma_train = train_ds.gamma.to(device)

    F_test = test_ds.F.to(device)
    S_test = test_ds.S.to(device)
    gamma_test = test_ds.gamma.to(device)

    best_loss = float('inf')
    patience_counter = 0
    log_interval = max(1, epochs // 20)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        loss, mse = model.loss(F_train, S_train, gamma_override=gamma_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % log_interval == 0 or epoch < 5:
            model.eval()
            with torch.no_grad():
                S_test_pred = model(F_test, gamma_override=gamma_test)
                test_mse = float(torch.mean((S_test_pred - S_test) ** 2))

                r2_train = {}
                r2_test = {}

                for mode_name in train_mode_names:
                    indices = train_ds.mode_indices[mode_name]
                    idx_t = torch.tensor(indices, device=device, dtype=torch.long)
                    if mode_name == 'ss':
                        pred = model(F_train[idx_t], gamma_override=gamma_train[idx_t])
                    else:
                        pred = model(F_train[idx_t])
                    tgt = S_train[idx_t]
                    pred_flat = pred.detach().cpu().numpy().flatten()
                    tgt_flat = tgt.cpu().numpy().flatten()
                    r2_train[mode_name] = float(compute_r2(pred_flat, tgt_flat))

                for mode_name in test_mode_names:
                    indices = test_ds.mode_indices[mode_name]
                    idx_t = torch.tensor(indices, device=device, dtype=torch.long)
                    if mode_name == 'ss':
                        pred = model(F_test[idx_t], gamma_override=gamma_test[idx_t])
                    else:
                        pred = model(F_test[idx_t])
                    tgt = S_test[idx_t]
                    pred_flat = pred.detach().cpu().numpy().flatten()
                    tgt_flat = tgt.cpu().numpy().flatten()
                    r2_test[mode_name] = float(compute_r2(pred_flat, tgt_flat))

                avg_r2 = np.mean(list(r2_train.values()) + list(r2_test.values()))
                alpha_info = {k: round(v, 4)
                              for k, v in model.get_parameters_dict().items()
                              if 'alpha' in k}

            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss.item():.6f} | "
                  f"MSE: {mse.item():.6f} | Test MSE: {test_mse:.6f} | "
                  f"Avg R²: {avg_r2:.4f} | α: {alpha_info}")

            if test_mse < best_loss:
                best_loss = test_mse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        r2_train_final = {}
        r2_test_final = {}

        for mode_name in train_mode_names:
            indices = train_ds.mode_indices[mode_name]
            idx_t = torch.tensor(indices, device=device, dtype=torch.long)
            if mode_name == 'ss':
                pred = model(F_train[idx_t], gamma_override=gamma_train[idx_t])
            else:
                pred = model(F_train[idx_t])
            tgt = S_train[idx_t]
            pred_flat = pred.cpu().numpy().flatten()
            tgt_flat = tgt.cpu().numpy().flatten()
            r2_train_final[mode_name] = float(compute_r2(pred_flat, tgt_flat))

        for mode_name in test_mode_names:
            indices = test_ds.mode_indices[mode_name]
            idx_t = torch.tensor(indices, device=device, dtype=torch.long)
            if mode_name == 'ss':
                pred = model(F_test[idx_t], gamma_override=gamma_test[idx_t])
            else:
                pred = model(F_test[idx_t])
            tgt = S_test[idx_t]
            pred_flat = pred.cpu().numpy().flatten()
            tgt_flat = tgt.cpu().numpy().flatten()
            r2_test_final[mode_name] = float(compute_r2(pred_flat, tgt_flat))

    avg_r2 = np.mean(list(r2_train_final.values()) + list(r2_test_final.values()))
    final_params = model.get_parameters_dict()

    return {
        'r2_train': r2_train_final,
        'r2_test': r2_test_final,
        'avg_r2': avg_r2,
        'params': final_params,
        'epochs_trained': epoch + 1,
    }


def main():
    parser = argparse.ArgumentParser(description='Train GI-CANN model')
    parser.add_argument('--material', type=str, default='rubber',
                        choices=['rubber', 'brain'])
    parser.add_argument('--mode', type=str, default='positive',
                        choices=['positive', 'negative', 'combined'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--r', type=float, default=0.01,
                        help='L1 regularization coefficient')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results JSON')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path to load')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    # Get datasets
    if args.material == 'rubber':
        train_ds, test_ds = get_rubber_dataset()
        train_modes = ['ut', 'bt']
        test_modes = ['ps']
    else:
        train_ds, test_ds = get_brain_dataset()
        train_modes = ['ut', 'uc']
        test_modes = ['ss']

    print(f"=== Training GI-CANN on {args.material} ({args.mode} mode) ===")
    print(f"Train modes: {train_modes} ({len(train_ds)} samples)")
    print(f"Test modes: {test_modes} ({len(test_ds)} samples)")

    config = {'mode': args.mode, 'r': args.r}
    model = GICANN(config)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state['model_state'])
        print(f"Loaded checkpoint from {args.checkpoint}")

    results = train_model(
        model, train_ds, test_ds,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        train_mode_names=train_modes,
        test_mode_names=test_modes,
    )

    print("\n=== Final Results ===")
    print(f"Train R²: {results['r2_train']}")
    print(f"Test R²: {results['r2_test']}")
    print(f"Average R²: {results['avg_r2']:.4f}")
    print(f"Discovered parameters: {results['params']}")
    print(f"Epochs trained: {results['epochs_trained']}")

    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{args.material}_{args.mode}_seed{args.seed}.pt')
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'results': results,
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    if args.output:
        output_data = {
            'material': args.material,
            'mode': args.mode,
            'r2_train': results['r2_train'],
            'r2_test': results['r2_test'],
            'avg_r2': results['avg_r2'],
            'params': results['params'],
            'epochs_trained': results['epochs_trained'],
            'seed': args.seed,
        }
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")

    return results


if __name__ == '__main__':
    main()
