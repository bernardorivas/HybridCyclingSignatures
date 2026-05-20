"""Plot the four verified Ueda orbits to sanity-check what we're feeding the homology pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "baby_ueda"
FIG_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "figures" / "baby_ueda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGIMES = [
    ("period_1", 9.000),
    ("period_2", 5.250),
    ("period_4", 5.375),
    ("chaos",    7.500),
]


def base(regime: str, gamma: float, w: int) -> str:
    return f"ueda_{regime}_g{gamma:.3f}_win{w}T".replace(".", "p")


def main() -> int:
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    for row, (regime, gamma) in enumerate(REGIMES):
        # window = 8T is the most-data slice; load that.
        pos = np.load(DATA_DIR / f"{base(regime, gamma, 8)}_positions.npy")
        ax = axes[row, 0]
        ax.plot(pos[:, 0], pos[:, 1], lw=0.4, alpha=0.85, color="C0")
        # Stroboscopic points at integer drive periods.
        strob = pos[::200, :2]
        ax.plot(strob[:, 0], strob[:, 1], "o", ms=6, color="C3")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$\\dot x$")
        ax.set_title(f"{regime}  $\\gamma = {gamma:.3f}$  (8 drive periods)")
        ax.set_aspect("equal", adjustable="datalim")

        # Drive-phase projection.
        ax2 = axes[row, 1]
        ax2.plot(pos[:, 0], pos[:, 2], lw=0.4, alpha=0.85, color="C0")
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$\\cos t$")
        ax2.set_title(f"$(x, \\cos t)$ projection")
    fig.tight_layout()
    out = FIG_DIR / "fig_ueda_orbits_sanity.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
