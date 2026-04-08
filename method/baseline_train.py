#!/usr/bin/env python3
"""Train SI-CANN baseline model on rubber or brain tissue data.

SI-CANN uses fixed classical invariants (α=2 for I₁, α=-2 for I₂).
The exponents are NOT trainable — only the branch weights are learned.

Usage:
    python method/baseline_train.py --material rubber --mode positive
    python method/baseline_train.py --material brain --mode negative
"""

import argparse
import torch
import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from method.sicann import SICANN
from data.load_data import (
    get_rubber_dataset, get_brain_dataset,
    compute_r2
)


def main():
    parser = argparse.ArgumentParser(description='Train SI-CANN baseline')
    parser.add_argument('--material', type=str, default='rubber',
                        choices=['rubber', 'brain'])
    parser.add_argument('--mode', type=str, default='positive',
                        choices=['positive', 'negative', 'combined'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--r', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    args = parser.parse_args()

    if args.material == 'rubber':
        train_ds, test_ds = get_rubber_dataset()
        train_modes = ['ut', 'bt']
        test_modes = ['ps']
    else:
        train_ds, test_ds = get_brain_dataset()
        train_modes = ['ut', 'uc']
        test_modes = ['ss']

    print(f"=== Training SI-CANN ({args.mode}) on {args.material} ===")

    torch.manual_seed(args.seed)
    model = SICANN({'mode': args.mode, 'r': args.r}).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    F_train, S_train = train_ds.F, train_ds.S
    gamma_train = train_ds.gamma
    F_test, S_test = test_ds.F, test_ds.S
    gamma_test = test_ds.gamma

    device = torch.device('cpu')
    model = model.to(device)
    F_train = F_train.to(device)
    S_train = S_train.to(device)
    gamma_train = gamma_train.to(device)
    F_test = F_test.to(device)
    S_test = S_test.to(device)
    gamma_test = gamma_test.to(device)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss, mse = model.loss(F_train, S_train, gamma=gamma_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                S_test_pred = model(F_test, gamma=gamma_test)
                test_mse = float(torch.mean((S_test_pred - S_test) ** 2))
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}, test_mse={test_mse:.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        r2_train = {}
        r2_test = {}

        for mode_name in train_modes:
            indices = train_ds.mode_indices[mode_name]
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            if mode_name == 'ss':
                pred = model(F_train[idx_t], gamma_override=gamma_train[idx_t])
            else:
                pred = model(F_train[idx_t])
            tgt = S_train[idx_t]
            r2_train[mode_name] = float(compute_r2(
                pred.cpu().numpy().flatten(), tgt.cpu().numpy().flatten()))

        for mode_name in test_modes:
            indices = test_ds.mode_indices[mode_name]
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            if mode_name == 'ss':
                pred = model(F_test[idx_t], gamma_override=gamma_test[idx_t])
            else:
                pred = model(F_test[idx_t])
            tgt = S_test[idx_t]
            r2_test[mode_name] = float(compute_r2(
                pred.cpu().numpy().flatten(), tgt.cpu().numpy().flatten()))

        all_r2 = list(r2_train.values()) + list(r2_test.values())
        avg_r2 = float(np.mean(all_r2))

    print(f"\nFinal: r2_train={r2_train}, r2_test={r2_test}, avg_r2={avg_r2:.4f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_name = f"{args.material}_{args.mode}_sicann"
    ckpt_path = os.path.join(args.checkpoint_dir, f"{ckpt_name}.pt")
    torch.save({
        'model_state': model.state_dict(),
        'config': {'mode': args.mode, 'r': args.r},
        'results': {'r2_train': r2_train, 'r2_test': r2_test, 'avg_r2': avg_r2}
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == '__main__':
    main()
