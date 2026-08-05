#!/usr/bin/env python
"""
Generate publication-quality phase-plane atlases for Roessler and compass gait.

Produces three figures:
  - roessler_atlas.png: 2x3 grid phase-plane projections (x-y)
  - roessler_data_colored.png: 1x5 grid 3D scatter plots colored by time
  - compass_atlas_check.png: 2x3 grid dual-leg phase planes

All figures are 200 dpi, tight bbox, saved as PNG.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

# Import export_lifts module from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_lifts import load_npz


# ============================================================================
# Roessler Atlas
# ============================================================================

def plot_roessler_atlas(data_root, fig_dir):
    """Generate Roessler phase-plane atlas (2x3 grid).

    Parameters
    ----------
    data_root : Path
        Directory containing roessler_*.npz files.
    fig_dir : Path
        Output directory for figures.
    """
    regimes = ["period1", "period2", "period4", "period8", "chaos"]
    regime_params = {
        "period1": 4.0,
        "period2": 6.0,
        "period4": 8.5,
        "period8": 8.7,
        "chaos": 9.0,
    }

    # Load data and check availability
    data = {}
    roessler_dir = data_root / "roessler"
    for regime in regimes:
        npz_path = roessler_dir / f"roessler_{regime}.npz"
        if npz_path.exists():
            data[regime] = load_npz(npz_path)
        else:
            print(f"Warning: {npz_path} not found, skipping {regime}")

    if not data:
        print("No Roessler data found, skipping roessler_atlas.png")
        return

    # Create figure: 2 rows, 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=200)
    axes = axes.flatten()

    # Plot each regime (a-e)
    for idx, regime in enumerate(regimes):
        if regime not in data:
            # Hide unused axes
            axes[idx].axis("off")
            continue

        ax = axes[idx]
        ts = data[regime]

        # Phase plane: x-y projection
        x_data = ts.x[:, 0]  # x column
        y_data = ts.x[:, 1]  # y column

        ax.plot(x_data, y_data, color="tab:blue", linewidth=0.8, alpha=0.85)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)

        c_val = regime_params[regime]
        panel_label = "Chaotic" if regime == "chaos" else f"Period {regime.replace('period', '')}"
        ax.set_title(f"({chr(97 + idx)}) {panel_label}  |  c = {c_val:.1f}",
                     fontsize=11, fontweight="normal")

    # Sixth panel: text block
    ax_text = axes[5]
    ax_text.axis("off")

    text_str = "Roessler period-doubling cascade\n\n"
    text_str += "a = 0.1, b = 0.1, c varies\n"

    # Extract parameters from first available timeseries
    first_ts = next(iter(data.values()))
    t_span = first_ts.meta.get("t_span", "N/A")
    dt = first_ts.meta.get("dt", "N/A")
    text_str += f"\nFixed time span T = {t_span:.0f}\n"
    text_str += f"dt = {dt:.4f}\n"
    text_str += "\nData feed cycling-signature\n"
    text_str += "comparison via Julia interface."

    ax_text.text(0.1, 0.5, text_str, fontsize=10, verticalalignment="center",
                 family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Rössler System: Phase-Plane Period Doubling", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = fig_dir / "roessler_atlas.png"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ============================================================================
# Roessler Data Colored
# ============================================================================

def plot_roessler_data_colored(data_root, fig_dir):
    """Generate Roessler 3D time-colored scatter plots (1x5 grid).

    Parameters
    ----------
    data_root : Path
        Directory containing roessler_*.npz files.
    fig_dir : Path
        Output directory for figures.
    """
    regimes = ["period1", "period2", "period4", "period8", "chaos"]
    regime_params = {
        "period1": 4.0,
        "period2": 6.0,
        "period4": 8.5,
        "period8": 8.7,
        "chaos": 9.0,
    }

    # Load data
    data = {}
    roessler_dir = data_root / "roessler"
    for regime in regimes:
        npz_path = roessler_dir / f"roessler_{regime}.npz"
        if npz_path.exists():
            data[regime] = load_npz(npz_path)
        else:
            print(f"Warning: {npz_path} not found, skipping {regime}")

    if not data:
        print("No Roessler data found, skipping roessler_data_colored.png")
        return

    # Create figure: 1 row, 5 cols
    fig = plt.figure(figsize=(20, 4), dpi=200)

    cmap = get_cmap("viridis")

    for idx, regime in enumerate(regimes):
        if regime not in data:
            continue

        ax = fig.add_subplot(1, 5, idx + 1, projection="3d")
        ts = data[regime]

        x_data = ts.x[:, 0]
        y_data = ts.x[:, 1]
        z_data = ts.x[:, 2]
        t_data = ts.t

        # Normalize time for colormap
        t_norm = (t_data - t_data.min()) / (t_data.max() - t_data.min() + 1e-10)
        colors = cmap(t_norm)

        # Scatter plot colored by time
        ax.scatter(x_data, y_data, z_data, c=t_norm, cmap="viridis", s=0.5, alpha=0.7)

        c_val = regime_params[regime]
        ax.set_title(f"c = {c_val:.1f}", fontsize=10)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        ax.set_zlabel("z", fontsize=9)

    # Add colorbar
    sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("t (normalized)", fontsize=10)

    plt.suptitle("Rössler System: Time-Colored Trajectories", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    out_path = fig_dir / "roessler_data_colored.png"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ============================================================================
# Compass Gait Atlas Check
# ============================================================================

def plot_compass_atlas_check(data_root, fig_dir):
    """Generate compass gait phase-plane atlas (2x3 grid, dual legs).

    Parameters
    ----------
    data_root : Path
        Directory containing compass_gait_*.npz files (must be under compass_gait/).
    fig_dir : Path
        Output directory for figures.
    """
    regimes = ["period1", "period2", "period4", "period8", "chaos"]
    regime_params = {
        "period1": 4.00,
        "period2": 4.75,
        "period4": 5.00,
        "period8": 5.02,
        "chaos": 5.20,
    }

    # Check for compass_gait subdirectory
    compass_dir = data_root / "compass_gait"
    if not compass_dir.exists():
        print(f"Warning: {compass_dir} not found, skipping compass_atlas_check.png")
        return

    # Load data
    data = {}
    for regime in regimes:
        npz_path = compass_dir / f"compass_{regime}.npz"
        if npz_path.exists():
            data[regime] = load_npz(npz_path)
        else:
            print(f"Warning: {npz_path} not found, skipping {regime}")

    if not data:
        print("No Compass data found, skipping compass_atlas_check.png")
        return

    # Create figure: 2 rows, 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=200)
    axes = axes.flatten()

    # Plot each regime (a-e)
    for idx, regime in enumerate(regimes):
        if regime not in data:
            axes[idx].axis("off")
            continue

        ax = axes[idx]
        ts = data[regime]

        # Physical-leg phase planes, following each leg across the
        # stance/nonstance label swap at every impact: on even strides leg A
        # occupies the (ns) slot, on odd strides the (s) slot. Line breaks
        # (NaN) at impacts keep the jump discontinuities from being drawn as
        # spurious connectors.
        stride_idx = np.searchsorted(ts.impact_times, ts.t, side="right")
        parity = stride_idx % 2
        boundary = np.zeros(len(ts.t), dtype=bool)
        boundary[1:] = stride_idx[1:] != stride_idx[:-1]

        for leg_parity, color, label in (
            (0, "tab:blue", "physical leg A"),
            (1, "tab:orange", "physical leg B"),
        ):
            ns_slot = parity == leg_parity  # this leg currently labeled ns
            theta = np.where(ns_slot, ts.x[:, 0], ts.x[:, 1])
            dtheta = np.where(ns_slot, ts.x[:, 2], ts.x[:, 3])
            theta = theta.copy()
            theta[boundary] = np.nan  # break the polyline across each impact
            ax.plot(theta, dtheta, color=color, linewidth=0.6, label=label,
                    alpha=0.85)

        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_xlabel("angular position $\\theta$ (rad)", fontsize=10)
        ax.set_ylabel("angular velocity $\\dot{\\theta}$ (rad/s)", fontsize=10)

        phi_deg = regime_params[regime]
        panel_label = "Chaotic gait" if regime == "chaos" else f"Period {regime.replace('period', '')}"
        ax.set_title(f"({chr(97 + idx)}) {panel_label}  |  $\\phi$ = {phi_deg:.2f}$^\\circ$",
                     fontsize=11, fontweight="normal")

    # Sixth panel: text block
    ax_text = axes[5]
    ax_text.axis("off")

    text_str = "Compass-gait period-doubling cascade\n\n"
    text_str += "Nominal model: $\\mu = 2$, $\\beta = 1$, $l = 1$ m\n"

    # Extract parameters from first available timeseries
    first_ts = next(iter(data.values()))
    t_span = first_ts.meta.get("t_span", "N/A")
    dt = first_ts.meta.get("dt", "N/A")
    text_str += f"\nFixed time span T = {t_span:.0f}\n"
    text_str += f"dt = {dt:.4f}\n"
    text_str += "\nSolid curves: continuous swing\n"
    text_str += "dynamics between impacts."

    ax_text.text(0.1, 0.5, text_str, fontsize=10, verticalalignment="center",
                 family="monospace", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    fig.suptitle("Passive Compass Gait: Regenerated Fixed-Span Series", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = fig_dir / "compass_atlas_check.png"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality phase-plane atlases for period-doubling systems."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root data directory (default: {package}/data). "
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Output figures directory (default: {package}/figures)."
    )

    args = parser.parse_args()

    # Determine package root (directory containing this script)
    package_root = Path(__file__).resolve().parent

    # Set defaults
    data_root = args.data_root if args.data_root else package_root / "data"
    fig_dir = args.fig_dir if args.fig_dir else package_root / "figures"

    print(f"Data root: {data_root}")
    print(f"Figure directory: {fig_dir}")

    # Create figure directory if needed
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Generate all three figures
    plot_roessler_atlas(data_root, fig_dir)
    plot_roessler_data_colored(data_root, fig_dir)
    plot_compass_atlas_check(data_root, fig_dir)

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
