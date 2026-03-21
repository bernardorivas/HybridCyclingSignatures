import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
ROOT = Path(__file__).parent.parent

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from config import config, SystemType
from system import CompassGaitHybridSystem, generate_suspension_dataset
from networks import SuspensionNetworks
from losses import calculate_composite_losses
from visualize import OKABE_ITO, PUB_STYLE, FIGURES_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
config.system_type = SystemType.COMPASS_GAIT
config.epochs = 100
config.num_train_samples = 3000
config.embed_extra = 0          # 0 -> 5D, 1 -> 6D, 2 -> 7D, etc.


def train_compass_gait():
    matplotlib.rcParams.update(PUB_STYLE)
    print(f"Compass-Gait Hybrid Suspension")
    print(f"  state_dim={config.state_dim}, embed_dim={config.embed_dim} "
          f"(extra={config.embed_extra})")
    print(f"  phi={np.degrees(config.phi):.1f} deg, mu={config.mu}, beta={config.beta}")

    if config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  device={device}\n")

    # 1. Generate Dataset
    print("Generating dataset...")
    hds = CompassGaitHybridSystem(config)
    X_train, Y_train = generate_suspension_dataset(
        hds, config.integration_tau, config.num_train_samples,
        points_per_orbit=config.points_per_orbit, cfg=config)
    print(f"  {X_train.shape[0]} training pairs")
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 2. Networks
    model = SuspensionNetworks(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} parameters\n")
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 3. Training Loop
    history = {'comm': [], 'glue': [], 'recon': [], 'total': []}

    for epoch in range(config.epochs):
        model.train()
        epoch_metrics = {k: [] for k in history}

        for x_i, x_next in dataloader:
            x_i, x_next = x_i.to(device), x_next.to(device)
            optimizer.zero_grad()

            total_loss, metrics = calculate_composite_losses(
                model, hds, x_i, x_next, config)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k].append(v)

        scheduler.step()

        for k in history:
            history[k].append(np.mean(epoch_metrics[k]))

        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            h = {k: history[k][-1] for k in history}
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"total={h['total']:.4f}  comm={h['comm']:.4f}  "
                  f"glue={h['glue']:.4f}  recon={h['recon']:.4f}")

    # 4. Save model
    out_dir = ROOT / "runs" / "compass_gait"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nModel saved to {out_path}")

    # 5. Figures
    fig_dir = FIGURES_DIR / "compass_gait"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Fig B: Optimization losses
    fig, ax = plt.subplots(figsize=(5, 3.5))
    epochs_arr = np.arange(1, config.epochs + 1)
    ax.semilogy(epochs_arr, history['total'], color=OKABE_ITO[7],
                linewidth=1.8, label='Total')
    ax.semilogy(epochs_arr, history['comm'], color=OKABE_ITO[0],
                linewidth=1.2, label='Commutativity')
    ax.semilogy(epochs_arr, history['glue'], color=OKABE_ITO[1],
                linewidth=1.2, label='Gluing')
    ax.semilogy(epochs_arr, history['recon'], color=OKABE_ITO[2],
                linewidth=1.2, label='Reconstruction')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Optimization Convergence')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_compass_optimization_losses.png", dpi=200)
    plt.close(fig)
    print(f"Loss figure saved to {fig_dir / 'fig_compass_optimization_losses.png'}")


if __name__ == '__main__':
    train_compass_gait()
