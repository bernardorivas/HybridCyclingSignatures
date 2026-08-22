"""NON-RUNNABLE legacy entry point.

This script predates the Section-4 two-phase loss API and also imports stale
visualization names.  Use ``scripts/train_rimless.py``.  It is retained only
to make the historical split explicit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    raise SystemExit(
        "NON-RUNNABLE: scripts/train.py is a legacy entry point; "
        "use scripts/train_rimless.py"
    )

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from config import config
from system import RimlessWheelHybridSystem, generate_suspension_dataset
from networks import SuspensionNetworks
from losses import calculate_composite_losses
from visualize import compute_deep_crossing_validation, plot_hybrid_suspension


def train_hybrid_suspension():
    print(f"[{'='*60}]")
    print(f"[{'RIMLESS WHEEL HYBRID SUSPENSION SEMIFLOW':^58}]")
    print(f"[{'='*60}]")

    if config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Hardware Compute Device: {device}\n")

    # 1. Generate Dataset
    print(f"1. Instantiating Rimless Wheel Dynamics and Mapping Cylinder...")
    sys = RimlessWheelHybridSystem(config)
    print(f"2. Continuously integrating {config.num_train_samples} initial conditions (Tau={config.integration_tau}s)...")

    X_train, Y_train = generate_suspension_dataset(sys, config.integration_tau, config.num_train_samples, points_per_orbit=config.points_per_orbit, cfg=config)
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 2. Networks
    print(f"3. Initializing Base-Preserving Residual Encoders...")
    model = SuspensionNetworks(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 3. Training Loop
    print(f"\n[{'TRAINING INITIALIZED':^58}]\n")

    history_glue = []
    history_comm = []

    for epoch in range(config.epochs):
        model.train()
        epoch_metrics = {'comm': [], 'glue': [], 'recon': [], 'total': []}

        for x_i, x_next in dataloader:
            x_i, x_next = x_i.to(device), x_next.to(device)
            optimizer.zero_grad()

            total_loss, metrics = calculate_composite_losses(model, sys, x_i, x_next, config)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k].append(v)

        scheduler.step()

        # Aggregate logs
        avg_comm = np.mean(epoch_metrics['comm'])
        avg_glue = np.mean(epoch_metrics['glue'])
        avg_recon = np.mean(epoch_metrics['recon'])
        avg_total = np.mean(epoch_metrics['total'])

        history_glue.append(avg_glue)
        history_comm.append(avg_comm)

        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {(epoch+1):04d}/{config.epochs} | Total Loss: {avg_total:8.4f}")
            print(f"  |- Commutativity:   {avg_comm:8.4f}")
            print(f"  |- Topo Gluing:     {avg_glue:8.4f}")
            print(f"  \\- Reconstruction:  {avg_recon:8.4f}\n")

    print("\nTraining Complete! Running Deep Crossing Diagnostics...")
    compute_deep_crossing_validation(model, sys, config)
    plot_hybrid_suspension(model, sys, config, history_glue)

    out_path = ROOT / "runs" / "rimless_wheel" / "model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Model weights saved to {out_path}")


if __name__ == '__main__':
    train_hybrid_suspension()
