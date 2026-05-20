"""Basic diagnostic plots for the CHyLL v2 baseline (mapping-cylinder data)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CHyLLv2Config
from .networks import CHyLLv2Networks
from .ode import make_ode_func, rollout
from .systems.base import Trajectory


def plot_loss_history(jsonl_path: str | Path, out_path: str | Path) -> None:
    import json
    steps, total, lx, lz, lg, lv, lc = [], [], [], [], [], [], []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            steps.append(r["step"])
            total.append(r["total"])
            lx.append(r["L_x"])
            lz.append(r["L_z"])
            lg.append(r["L_g"])
            lv.append(r["L_v"])
            lc.append(r["L_c"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, total, label="total", color="black", lw=1.5)
    ax.plot(steps, lx, label=r"$\mathcal{L}_x$", alpha=0.8)
    ax.plot(steps, lz, label=r"$\mathcal{L}_z$", alpha=0.8)
    ax.plot(steps, lg, label=r"$\mathcal{L}_g$", alpha=0.8)
    ax.plot(steps, lv, label=r"$\mathcal{L}_v$", alpha=0.8)
    ax.plot(steps, lc, label=r"$\mathcal{L}_c$", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rollout_vs_truth(
    cfg: CHyLLv2Config,
    nets: CHyLLv2Networks,
    trajectory: Trajectory,
    out_path: str | Path,
    rollout_horizon: int | None = None,
) -> None:
    """Plot ground-truth vs decoded latent rollout for one trajectory.

    Plots all ``state_dim`` (augmented) coordinates, including ``s``.
    """
    device = next(nets.parameters()).device
    if rollout_horizon is None:
        rollout_horizon = len(trajectory) - 1

    x0 = torch.tensor(
        trajectory.states[0], dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, state_dim)
    times = torch.linspace(
        0.0, rollout_horizon * cfg.tau, rollout_horizon + 1, device=device
    )

    nets.eval()
    ode_func = make_ode_func(nets.vfield).to(device)
    with torch.no_grad():
        z0 = nets.encoder(x0)
        z_roll = rollout(ode_func=ode_func, z0=z0, times=times, cfg=cfg)
        x_pred = nets.decoder(z_roll).squeeze(1).cpu().numpy()  # (T+1, state_dim)
    nets.train()

    x_true = trajectory.states[: rollout_horizon + 1]
    # X-axis is the discrete sample index k (= step number in the
    # tau-semiflow). Physical time is not the right coordinate because
    # the mapping-cylinder traversal is an artificial parametrisation,
    # not a clock.
    idx = np.arange(rollout_horizon + 1)
    labels = [f"$x_{i}$" for i in range(cfg.base_dim)] + [r"$s$"]

    fig, axes = plt.subplots(cfg.state_dim, 1, figsize=(7, 2.0 * cfg.state_dim), sharex=True)
    if cfg.state_dim == 1:
        axes = [axes]
    for i in range(cfg.state_dim):
        axes[i].plot(idx, x_true[:, i], label="truth", color="black", lw=1.2)
        axes[i].plot(idx, x_pred[:, i], label="rollout", color="C1", lw=1.2, ls="--")
        axes[i].set_ylabel(labels[i])
    axes[-1].set_xlabel("sample index $k$")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phase_portrait_rimless(
    trajectories: List[Trajectory],
    out_path: str | Path,
    max_trajectories: int = 25,
    drop_initial_steps: int = 0,
) -> None:
    """``(theta, omega)`` phase portrait of a sample of trajectories.

    Each base-flow arc (between consecutive resets) is plotted as its own
    line segment so the reset *jumps* don't show up as spurious horizontal
    streaks across the portrait. Cylinder samples (``s > 0``) are drawn
    as red dots.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    cylinder_label_used = False
    for traj in trajectories[:max_trajectories]:
        states = traj.states[drop_initial_steps:]
        s = states[:, -1]
        on_base = s == 0
        # Split base-flow indices into runs of consecutive base samples;
        # each run is one stride from theta_reset to theta_guard.
        idx = np.where(on_base)[0]
        if idx.size > 0:
            # Group consecutive indices into segments.
            splits = np.where(np.diff(idx) > 1)[0] + 1
            for seg in np.split(idx, splits):
                if seg.size >= 2:
                    ax.plot(
                        states[seg, 0], states[seg, 1],
                        lw=0.6, alpha=0.7, color="C0",
                    )
        ax.scatter(
            states[~on_base, 0], states[~on_base, 1],
            s=4, alpha=0.5, color="C3",
            label="cylinder" if not cylinder_label_used else None,
        )
        cylinder_label_used = True
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot\theta$")
    ax.set_title("Rimless wheel: ground-truth trajectories (red = cylinder)")
    if cylinder_label_used:
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
