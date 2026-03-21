"""
Interactive 3D viewer for the trained hybrid suspension embedding.
Loads model.pt and opens a rotatable matplotlib window.

Usage:
    python scripts/explore.py
    python scripts/explore.py --azim 30 --elev 45
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
ROOT = Path(__file__).parent.parent

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import config
from networks import SuspensionNetworks
from system import RimlessWheelHybridSystem
from visualize import generate_massive_background_cloud


def load_model(path=None):
    if path is None:
        path = ROOT / "runs" / "model.pt"
    model = SuspensionNetworks(config)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def build_scene(model):
    sys = RimlessWheelHybridSystem(config)
    cfg = config

    X_bg = generate_massive_background_cloud(cfg, root_resolution=250)
    X_bg_t = torch.tensor(X_bg, dtype=torch.float32)

    Y_bg_list = []
    with torch.no_grad():
        for i in range(0, len(X_bg_t), 10000):
            Y_bg_list.append(model.E(X_bg_t[i:i+10000]).numpy())
    Y_bg = np.vstack(Y_bg_list)

    base_mask = X_bg[:, 2] == 0.0
    cyl_mask = X_bg[:, 2] > 0.0

    initial_conditions = [
        [cfg.theta_reset, 0.5, 0.0],
        [0.0, 1.2, 0.0],
        [cfg.theta_reset, 2.0, 0.0],
    ]
    colors = ["cyan", "magenta", "lime"]
    viz_tau = cfg.viz_tau
    n_steps = int(cfg.viz_orbit_duration / viz_tau)

    orbits_X_true = []
    orbits_Y_true = []
    orbits_Y_pred = []
    orbits_X_pred = []

    for x0 in initial_conditions:
        true_X = sys.generate_tau_timeseries(x0, viz_tau, n_steps)
        orbits_X_true.append(true_X)
        true_X_t = torch.tensor(true_X, dtype=torch.float32)
        with torch.no_grad():
            orbits_Y_true.append(model.E(true_X_t).numpy())
            y_curr = model.E(true_X_t[0].unsqueeze(0))
            pred_Y = [y_curr.squeeze().numpy()]
            for _ in range(len(true_X) - 1):
                y_curr = model.F(y_curr)
                pred_Y.append(y_curr.squeeze().numpy())
            pred_Y_t = torch.tensor(np.array(pred_Y), dtype=torch.float32)
            orbits_Y_pred.append(pred_Y_t.numpy())
            orbits_X_pred.append(model.D(pred_Y_t).numpy())

    return Y_bg, X_bg, base_mask, cyl_mask, orbits_X_true, orbits_Y_true, orbits_Y_pred, orbits_X_pred, colors


def plot_interactive(azim=-50, elev=20):
    print("Loading model...")
    model = load_model()

    print("Building scene (this may take a moment)...")
    Y_bg, X_bg, base_mask, cyl_mask, orbits_X_true, orbits_Y_true, orbits_Y_pred, orbits_X_pred, colors = build_scene(model)

    # --- Figure 1: 3D embedded suspension (interactive) ---
    fig1 = plt.figure(figsize=(14, 12))
    ax1 = fig1.add_subplot(111, projection="3d")

    ax1.scatter(Y_bg[base_mask, 0], Y_bg[base_mask, 1], Y_bg[base_mask, 2],
                color="#1f77b4", s=1, alpha=0.1, label="Embedded Base Space")

    s_col = ax1.scatter(Y_bg[cyl_mask, 0], Y_bg[cyl_mask, 1], Y_bg[cyl_mask, 2],
                        c=X_bg[cyl_mask, 2], cmap="autumn", s=2, alpha=0.4,
                        label="Spoke Impact Cylinder Mapping")

    for i in range(len(orbits_Y_true)):
        ax1.plot(orbits_Y_true[i][:, 0], orbits_Y_true[i][:, 1], orbits_Y_true[i][:, 2],
                 "o-", color="black", mfc=colors[i], markersize=6, alpha=0.8,
                 linewidth=1.5, label=f"True Y Suspended Cycle {i+1}")
        ax1.plot(orbits_Y_pred[i][:, 0], orbits_Y_pred[i][:, 1], orbits_Y_pred[i][:, 2],
                 "--", color=colors[i], linewidth=3,
                 label=f"Predicted Continuous F^n {i+1}")

    ax1.set_title(r"Continuous Rimless Wheel Hybrid Suspension mapped into $\mathbb{R}^3$")
    ax1.set_xlabel("$y_1$")
    ax1.set_ylabel("$y_2$")
    ax1.set_zlabel("$y_3$")
    fig1.colorbar(s_col, ax=ax1, label=r"Cylinder Transit $s \in [0,1]$")
    ax1.legend(fontsize=7)
    ax1.view_init(elev=elev, azim=azim)
    fig1.tight_layout()

    # --- Figure 2: 2D phase portrait with decoded orbits ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    cfg = config
    X_bg_base = X_bg[base_mask]
    E_bg = 0.5 * X_bg_base[:, 1]**2 + np.cos(X_bg_base[:, 0])
    sc = ax2.scatter(X_bg_base[:, 0], X_bg_base[:, 1], c=E_bg, cmap="viridis", s=2, alpha=0.4)
    ax2.axvline(x=cfg.theta_guard, color="red", linewidth=2, label=r"Guard $G$")
    for i in range(len(orbits_X_true)):
        true_base = orbits_X_true[i][orbits_X_true[i][:, 2] < 1e-3]
        if len(true_base):
            ax2.plot(true_base[:, 0], true_base[:, 1], "o-", color=colors[i],
                     markersize=4, linewidth=2, label=f"True Cycle {i+1}")
        dec = orbits_X_pred[i][orbits_X_pred[i][:, 2] < 1e-3]
        if len(dec):
            ax2.plot(dec[:, 0], dec[:, 1], "--", color=colors[i],
                     linewidth=1.5, alpha=0.8, label=f"Decoded $D \circ F^n$ {i+1}")
    ax2.set_title(r"Phase Portrait: True vs Decoded Orbits")
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\omega$")
    ax2.legend(fontsize=7)
    fig2.colorbar(sc, ax=ax2, label="Energy")
    fig2.tight_layout()

    print("Interactive windows open — click and drag to rotate the 3D plot.")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--azim", type=float, default=-50)
    parser.add_argument("--elev", type=float, default=20)
    args = parser.parse_args()
    plot_interactive(azim=args.azim, elev=args.elev)
