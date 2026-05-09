"""
Rimless-wheel training, faithful to Section 4 / Algorithm 1 of the manuscript.

Mirrors scripts/train_compass.py but targets the 2-D rimless system.

Phase I   — train E and F jointly on five terms (dyn, glue, conf, coll, utb),
            no reconstruction.
Phase II  — freeze E and F, train D alone on masked L_recon with larger LR.

Saves model weights to runs/rimless_wheel/model.pt and per-term loss history
plot to figures/rimless_wheel/fig_rimless_optimization_losses.png.
"""
import argparse
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
from system import RimlessWheelHybridSystem, generate_suspension_dataset
from networks import SuspensionNetworks
from losses import calculate_composite_losses
from visualize import OKABE_ITO, PUB_STYLE, FIGURES_DIR


config.system_type = SystemType.RIMLESS_WHEEL
config.embed_extra = 0
# Defaults from compass tuning carried over as a starting point. Rimless is
# 2-D so the same schedule is likely overkill but cheap; tune later.
config.num_train_samples = 3500
config.points_per_orbit = 20
config.phase1_epochs = 120
config.phase2_epochs = 150
config.weight_conf = 0.0001


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weight-conf", type=float, default=config.weight_conf,
                   help="Weight on the conformal Jacobian penalty in Phase I.")
    p.add_argument("--weight-utb", type=float, default=config.weight_utb,
                   help="Weight on the L_utb tangent-anchoring penalty.")
    p.add_argument("--embed-extra", type=int, default=config.embed_extra,
                   help="Extra latent dims beyond n+1; sets embed_dim = n+1+embed_extra.")
    p.add_argument("--out-suffix", default="",
                   help="Suffix for output files (e.g. '_embed7'); produces "
                        "model{suffix}.pt and fig_rimless_optimization_losses{suffix}.png.")
    return p.parse_args()


def _resolve_device():
    if config.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_rimless_wheel(out_suffix=""):
    matplotlib.rcParams.update(PUB_STYLE)
    print("Rimless-wheel relaxed-space training (Section 4 pipeline)")
    print(f"  state_dim={config.state_dim}, embed_dim={config.embed_dim}")
    print(f"  alpha={config.alpha}, gamma={config.gamma}, "
          f"theta_guard={config.theta_guard:.3f}, theta_reset={config.theta_reset:.3f}")
    print(f"  weight_conf={config.weight_conf}, weight_utb={config.weight_utb}, "
          f"embed_extra={config.embed_extra}, out_suffix={out_suffix!r}")

    device = _resolve_device()
    print(f"  device={device}\n")

    # 1. Dataset
    print(f"Generating dataset ({config.num_train_samples} orbits x "
          f"{config.points_per_orbit} steps)...", flush=True)
    import time
    t0 = time.time()
    hds = RimlessWheelHybridSystem(config)
    X_train, Y_train = generate_suspension_dataset(
        hds, config.integration_tau, config.num_train_samples,
        points_per_orbit=config.points_per_orbit, cfg=config)
    print(f"  generated {X_train.shape[0]} pairs in {time.time() - t0:.1f}s",
          flush=True)
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # 2. Model
    model = SuspensionNetworks(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} parameters\n")

    # ----- Phase I -----
    print(f"Phase I: E + F on (dyn, glue, conf, coll, utb), "
          f"{config.phase1_epochs} epochs, lr={config.phase1_lr}")
    opt_phase1 = optim.AdamW(
        list(model.E.parameters()) + list(model.F.parameters()),
        lr=config.phase1_lr, weight_decay=config.weight_decay,
    )
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt_phase1, T_max=config.phase1_epochs)

    phase1_history = {k: [] for k in ('dyn', 'glue', 'conf', 'coll', 'utb', 'total')}

    for epoch in range(config.phase1_epochs):
        model.train()
        epoch_metrics = {k: [] for k in phase1_history}

        for x_i, x_next in dataloader:
            x_i = x_i.to(device); x_next = x_next.to(device)
            opt_phase1.zero_grad()
            total_loss, metrics = calculate_composite_losses(
                model, hds, x_i, x_next, config, phase=1)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.E.parameters()) + list(model.F.parameters()),
                max_norm=2.0)
            opt_phase1.step()
            for k, v in metrics.items():
                epoch_metrics[k].append(v)

        sched1.step()
        for k in phase1_history:
            phase1_history[k].append(float(np.mean(epoch_metrics[k])))

        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            h = {k: phase1_history[k][-1] for k in phase1_history}
            print(f"  [P1] Epoch {epoch+1:3d}/{config.phase1_epochs}  "
                  f"total={h['total']:.4f}  dyn={h['dyn']:.4f}  glue={h['glue']:.4f}  "
                  f"conf={h['conf']:.2e}  coll={h['coll']:.4f}  utb={h['utb']:.4f}")

    # Freeze E and F
    for p in model.E.parameters(): p.requires_grad = False
    for p in model.F.parameters(): p.requires_grad = False

    # ----- Phase II -----
    print(f"\nPhase II: D alone on masked L_recon, "
          f"{config.phase2_epochs} epochs, lr={config.phase2_lr}")
    opt_phase2 = optim.AdamW(
        model.D.parameters(),
        lr=config.phase2_lr, weight_decay=config.weight_decay,
    )
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt_phase2, T_max=config.phase2_epochs)

    phase2_history = {'recon': [], 'total': []}

    for epoch in range(config.phase2_epochs):
        model.train()
        epoch_metrics = {k: [] for k in phase2_history}

        for x_i, _ in dataloader:
            x_i = x_i.to(device)
            opt_phase2.zero_grad()
            total_loss, metrics = calculate_composite_losses(
                model, hds, x_i, x_i, config, phase=2)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.D.parameters(), max_norm=2.0)
            opt_phase2.step()
            for k, v in metrics.items():
                epoch_metrics[k].append(v)

        sched2.step()
        for k in phase2_history:
            phase2_history[k].append(float(np.mean(epoch_metrics[k])))

        if (epoch + 1) == 1 or (epoch + 1) % 5 == 0:
            print(f"  [P2] Epoch {epoch+1:3d}/{config.phase2_epochs}  "
                  f"recon={phase2_history['recon'][-1]:.4f}")

    # 3. Save model
    out_dir = ROOT / "runs" / "rimless_wheel"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model{out_suffix}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nModel saved to {out_path}")

    # 4. Loss figure
    fig_dir = FIGURES_DIR / "rimless_wheel"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    ep1 = np.arange(1, config.phase1_epochs + 1)
    ax1.semilogy(ep1, phase1_history['total'], color=OKABE_ITO[7], lw=1.8, label='total')
    ax1.semilogy(ep1, phase1_history['dyn'],   color=OKABE_ITO[0], lw=1.2, label=r'$\mathcal{L}_{\mathrm{dyn}}$')
    ax1.semilogy(ep1, phase1_history['glue'],  color=OKABE_ITO[1], lw=1.2, label=r'$\mathcal{L}_{\mathrm{glue}}$')
    ax1.semilogy(ep1, phase1_history['conf'],  color=OKABE_ITO[2], lw=1.2, label=r'$\mathcal{L}_{\mathrm{conf}}$')
    ax1.semilogy(ep1, phase1_history['coll'],  color=OKABE_ITO[3], lw=1.2, label=r'$\mathcal{L}_{\mathrm{coll}}$')
    ax1.semilogy(ep1, phase1_history['utb'],   color=OKABE_ITO[4], lw=1.2, label=r'$\mathcal{L}_{\mathrm{utb}}$')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Phase I: encoder + semiflow')
    ax1.legend(fontsize=8, ncol=2)

    ep2 = np.arange(1, config.phase2_epochs + 1)
    ax2.semilogy(ep2, phase2_history['recon'], color=OKABE_ITO[5], lw=1.8,
                 label=r'$\mathcal{L}_{\mathrm{recon}}$ (masked)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Phase II: decoder')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig_path = fig_dir / f"fig_rimless_optimization_losses{out_suffix}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Loss figure saved to {fig_path}")


if __name__ == '__main__':
    args = _parse_args()
    config.weight_conf = args.weight_conf
    config.weight_utb = args.weight_utb
    config.embed_extra = args.embed_extra
    train_rimless_wheel(out_suffix=args.out_suffix)
