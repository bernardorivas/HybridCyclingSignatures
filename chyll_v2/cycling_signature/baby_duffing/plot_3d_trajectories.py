"""3D phase-space figures of the forced Duffing orbit at each cascade gamma.

Reads the saved ``*_positions.npy`` from ``run_cascade.py`` and produces
one figure per gamma showing the orbit projected onto ``(x, x_dot, sin(omega t))``
(a natural 3D embedding of the 4D state ``(x, x_dot, cos(omega t), sin(omega t))``).
The drive phase appears as a wrap around the third axis, so a period-n
orbit visibly winds n times around it.

Also produces a single grid figure showing all gammas side-by-side.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "baby_duffing"
FIG_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "figures" / "baby_duffing"

GAMMA_VALUES = [1.0, 1.6, 2.6, 3.0, 10.0]
REGIME_LABELS = {
    1.0: "period-1",
    1.6: "period-2",
    2.6: "period-4",
    3.0: "chaotic band",
    10.0: "chaos",
}


def base_name(gamma: float) -> str:
    return f"duffing_g{gamma:.2f}".replace(".", "p")


def plot_single(positions: np.ndarray, gamma: float, out_path: Path) -> None:
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    x, xdot, _, sin_phi = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]
    ax.plot(x, xdot, sin_phi, lw=0.45, alpha=0.85, color="C0")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\dot x$")
    ax.set_zlabel("$\\sin(\\omega t)$")
    ax.set_title(
        f"Forced Duffing, $\\gamma = {gamma:.2f}$  ({REGIME_LABELS.get(gamma, '')})\n"
        f"$x'' + 0.1 x' + x + x^3 = \\gamma \\cos t$",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_grid(positions_by_g: dict[float, np.ndarray], out_path: Path) -> None:
    n = len(positions_by_g)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4.2 * cols, 3.8 * rows))
    for i, (g, pos) in enumerate(sorted(positions_by_g.items())):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        x, xdot, _, sin_phi = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
        ax.plot(x, xdot, sin_phi, lw=0.4, alpha=0.85, color="C0")
        ax.set_xlabel("$x$", labelpad=-2)
        ax.set_ylabel("$\\dot x$", labelpad=-2)
        ax.set_zlabel("$\\sin(\\omega t)$", labelpad=-2)
        ax.set_title(f"$\\gamma = {g:.2f}$  ({REGIME_LABELS.get(g, '')})", fontsize=10)
        ax.tick_params(labelsize=7, pad=-2)
    fig.suptitle(
        "Forced Duffing cascade  $(x, \\dot x, \\sin\\omega t)$  embedding",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    positions_by_g: dict[float, np.ndarray] = {}
    for g in GAMMA_VALUES:
        base = base_name(g)
        npy_path = DATA_DIR / f"{base}_positions.npy"
        if not npy_path.is_file():
            print(f"missing: {npy_path}; skipping")
            continue
        pos = np.load(npy_path)
        positions_by_g[g] = pos
        plot_single(pos, g, FIG_DIR / f"fig_duffing_3d_{base}.png")
    plot_grid(positions_by_g, FIG_DIR / "fig_duffing_3d_grid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
