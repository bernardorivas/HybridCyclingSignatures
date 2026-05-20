"""Compare compass-gait trajectories under uniform-box vs limit-cycle-perturbed IC sampling.

Generates 6 trajectories each, plots ``theta_ns`` over sample index. The
uniform-box panel should show wildly different drift patterns (the
diagnosis), the limit-cycle-perturbed panel should show bounded
periodic walking (the target after the fix).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from chyll_v2.chyll_v2.systems.compass_gait import CompassGait

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "chyll_v2" / "figures" / "compass_gait_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

PHI = 0.07
LIMIT_CYCLE_IC = np.array([-(PHI + 0.27), -(PHI - 0.27), -0.38, -1.09])

TAU = 0.05
N_STEPS = 300
N_PER_PANEL = 6
SEED = 0


def gen_uniform(system: CompassGait, rng: np.random.Generator):
    return system.sample_initial_condition(rng)


def gen_limit_cycle(rng: np.random.Generator):
    pert = rng.normal(0.0, 0.03, size=4)
    return np.concatenate([LIMIT_CYCLE_IC + pert, [0.0]])


def main() -> int:
    system = CompassGait()
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for which, ax, sampler in (
        ("uniform (current chyll_v2)", axes[0], lambda r: gen_uniform(system, r)),
        ("limit-cycle + N(0, 0.03)", axes[1], gen_limit_cycle),
    ):
        for i in range(N_PER_PANEL):
            ic = sampler(rng)
            traj = system.generate_trajectory(ic, tau=TAU, n_steps=N_STEPS)
            theta_ns = traj.states[:, 0]
            ax.plot(theta_ns, lw=0.8, alpha=0.8, label=f"traj {i}")
        ax.set_ylabel(r"$\theta_{ns}$")
        ax.set_title(which, fontsize=10)
        ax.axhline(0, color="C7", lw=0.3, alpha=0.5)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("sample index $k$")
    axes[0].legend(ncol=3, fontsize=7, loc="upper right")

    out_path = OUT / "compass_ic_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
