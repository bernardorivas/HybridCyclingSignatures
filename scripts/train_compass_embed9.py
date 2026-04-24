"""
Experiment `embed9`: identical to baseline training except embedding dim
d = n + 1 + embed_extra = 9 (Menger-Nöbeling bound 2n+1 for n=4).

Writes to new paths; does NOT overwrite the baseline artifacts:
    runs/compass_gait/model_embed9.pt
    figures/compass_gait/fig_compass_optimization_losses_embed9.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
ROOT = Path(__file__).parent.parent

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import config, SystemType
from system import CompassGaitHybridSystem, generate_suspension_dataset
from networks import SuspensionNetworks
from losses import calculate_composite_losses
from visualize import OKABE_ITO, PUB_STYLE, FIGURES_DIR


config.system_type = SystemType.COMPASS_GAIT
config.embed_extra = 4        # <-- the only change vs baseline: d = 5 + 4 = 9
config.num_train_samples = 3500
config.points_per_orbit = 20
config.phase1_epochs = 120
config.phase2_epochs = 150
config.weight_conf = 0.0001

EXP_TAG = "embed9"


def _resolve_device():
    if config.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train():
    matplotlib.rcParams.update(PUB_STYLE)
    print(f"Compass-gait experiment [{EXP_TAG}]: embed_dim={config.embed_dim}")
    print(f"  state_dim={config.state_dim}, embed_extra={config.embed_extra}")
    device = _resolve_device()
    print(f"  device={device}\n")

    print(f"Generating dataset ({config.num_train_samples} orbits x "
          f"{config.points_per_orbit} steps)...", flush=True)
    import time
    t0 = time.time()
    hds = CompassGaitHybridSystem(config)
    X_train, Y_train = generate_suspension_dataset(
        hds, config.integration_tau, config.num_train_samples,
        points_per_orbit=config.points_per_orbit, cfg=config)
    print(f"  generated {X_train.shape[0]} pairs in {time.time() - t0:.1f}s",
          flush=True)
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = SuspensionNetworks(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} parameters\n", flush=True)

    # ----- Phase I -----
    print(f"Phase I: E + F on (dyn, glue, conf, coll, utb), "
          f"{config.phase1_epochs} epochs, lr={config.phase1_lr}", flush=True)
    opt_p1 = optim.AdamW(
        list(model.E.parameters()) + list(model.F.parameters()),
        lr=config.phase1_lr, weight_decay=config.weight_decay,
    )
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt_p1, T_max=config.phase1_epochs)

    p1_hist = {k: [] for k in ('dyn', 'glue', 'conf', 'coll', 'utb', 'total')}

    for epoch in range(config.phase1_epochs):
        model.train()
        epoch_metrics = {k: [] for k in p1_hist}
        for x_i, x_next in dataloader:
            x_i = x_i.to(device); x_next = x_next.to(device)
            opt_p1.zero_grad()
            total_loss, metrics = calculate_composite_losses(
                model, hds, x_i, x_next, config, phase=1)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.E.parameters()) + list(model.F.parameters()),
                max_norm=2.0)
            opt_p1.step()
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
        sched1.step()
        for k in p1_hist:
            p1_hist[k].append(float(np.mean(epoch_metrics[k])))
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            h = {k: p1_hist[k][-1] for k in p1_hist}
            print(f"  [P1] Epoch {epoch+1:3d}/{config.phase1_epochs}  "
                  f"total={h['total']:.4f}  dyn={h['dyn']:.4f}  glue={h['glue']:.4f}  "
                  f"conf={h['conf']:.2e}  coll={h['coll']:.4f}  utb={h['utb']:.4f}",
                  flush=True)

    for p in model.E.parameters(): p.requires_grad = False
    for p in model.F.parameters(): p.requires_grad = False

    # ----- Phase II -----
    print(f"\nPhase II: D alone on masked L_recon, "
          f"{config.phase2_epochs} epochs, lr={config.phase2_lr}", flush=True)
    opt_p2 = optim.AdamW(
        model.D.parameters(),
        lr=config.phase2_lr, weight_decay=config.weight_decay,
    )
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=config.phase2_epochs)

    p2_hist = {'recon': [], 'total': []}

    for epoch in range(config.phase2_epochs):
        model.train()
        epoch_metrics = {k: [] for k in p2_hist}
        for x_i, _ in dataloader:
            x_i = x_i.to(device)
            opt_p2.zero_grad()
            total_loss, metrics = calculate_composite_losses(
                model, hds, x_i, x_i, config, phase=2)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.D.parameters(), max_norm=2.0)
            opt_p2.step()
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
        sched2.step()
        for k in p2_hist:
            p2_hist[k].append(float(np.mean(epoch_metrics[k])))
        if (epoch + 1) == 1 or (epoch + 1) % 5 == 0:
            print(f"  [P2] Epoch {epoch+1:3d}/{config.phase2_epochs}  "
                  f"recon={p2_hist['recon'][-1]:.4f}", flush=True)

    # Save to experimental paths only
    out_dir = ROOT / "runs" / "compass_gait"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model_{EXP_TAG}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nModel saved to {out_path}", flush=True)

    fig_dir = FIGURES_DIR / "compass_gait"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    ep1 = np.arange(1, config.phase1_epochs + 1)
    ax1.semilogy(ep1, p1_hist['total'], color=OKABE_ITO[7], lw=1.8, label='total')
    ax1.semilogy(ep1, p1_hist['dyn'],   color=OKABE_ITO[0], lw=1.2, label=r'$\mathcal{L}_{\mathrm{dyn}}$')
    ax1.semilogy(ep1, p1_hist['glue'],  color=OKABE_ITO[1], lw=1.2, label=r'$\mathcal{L}_{\mathrm{glue}}$')
    ax1.semilogy(ep1, p1_hist['conf'],  color=OKABE_ITO[2], lw=1.2, label=r'$\mathcal{L}_{\mathrm{conf}}$')
    ax1.semilogy(ep1, p1_hist['coll'],  color=OKABE_ITO[3], lw=1.2, label=r'$\mathcal{L}_{\mathrm{coll}}$')
    ax1.semilogy(ep1, p1_hist['utb'],   color=OKABE_ITO[4], lw=1.2, label=r'$\mathcal{L}_{\mathrm{utb}}$')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (MSE)')
    ax1.set_title(f'Phase I: encoder + semiflow ({EXP_TAG})')
    ax1.legend(fontsize=8, ncol=2)
    ep2 = np.arange(1, config.phase2_epochs + 1)
    ax2.semilogy(ep2, p2_hist['recon'], color=OKABE_ITO[5], lw=1.8,
                 label=r'$\mathcal{L}_{\mathrm{recon}}$ (masked)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss (MSE)')
    ax2.set_title(f'Phase II: decoder ({EXP_TAG})')
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig_path = fig_dir / f"fig_compass_optimization_losses_{EXP_TAG}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Loss figure saved to {fig_path}", flush=True)


if __name__ == '__main__':
    train()
