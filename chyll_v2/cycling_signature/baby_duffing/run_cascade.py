"""Run the forced Duffing period-doubling cascade through the cycling-signature pipeline.

Hardening Duffing oscillator
    x'' + 0.1 x' + x + x^3 = gamma cos(t)

with the drive phase promoted to ``(cos t, sin t)`` so the state is 4D
autonomous. We hit the cascade by varying ``gamma``. Pilot sweep at the
top of this run identified the following cascade points
(stroboscopic-cluster counts in parens):

    gamma = 1.0   period-1   (1 cluster)
    gamma = 1.6   period-2   (2 clusters)
    gamma = 2.6   period-4   (4 clusters)
    gamma = 3.0   chaotic    (~7 clusters, band)
    gamma = 10.0  chaos      (~12 clusters)

These are picked specifically so the period-doubled orbits are
geometrically stacked-separated -- the structure that the
cycling-signature comparison space ought to resolve.

Usage:
    python chyll_v2/cycling_signature/baby_duffing/run_cascade.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from chyll_v2.cycling_signature.baby_duffing.simulate import (  # noqa: E402
    DuffingParams,
    finite_difference_tangents,
    simulate,
    write_lift,
)


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

DELTA = 0.1
ALPHA = 1.0
BETA = 1.0
OMEGA = 1.0

TRANSIENT_TIME = 500.0
RECORD_TIME = 400.0
SAMPLES_PER_DRIVE_PERIOD = 40
# Boxsize selected so the period-2/4 stacked-loop separations are
# resolvable. The orbit extent is roughly (-3.3, 3.3) in x and (-5.5, 5.5)
# in xdot at the chaotic end; (cos t, sin t) is bounded in [-1, 1].
BOXSIZE = 0.2
SB_RADIUS = 1
R_MAX = 1.0
EVAL_RADIUS = 0.3
SEGMENT_LENGTHS = "50:25:1500"
N_RUNS = 200


def base_name(gamma: float) -> str:
    return f"duffing_g{gamma:.2f}".replace(".", "p")


def run_julia(base: str) -> None:
    cmd = [
        "julia",
        f"--project=time series/cycling_signature",
        "chyll_v2/cycling_signature/julia/run_subsegments.jl",
        "--data-dir", str(DATA_DIR.relative_to(REPO_ROOT)),
        "--base", base,
        "--boxsize", str(BOXSIZE),
        "--sb-radius", str(SB_RADIUS),
        "--r-max", str(R_MAX),
        "--eval-radius", str(EVAL_RADIUS),
        "--segment-lengths", SEGMENT_LENGTHS,
        "--n-runs", str(N_RUNS),
        "--out-prefix", f"subsegments_{base}",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def read_metadata_beta1(base: str) -> int:
    meta_path = DATA_DIR / f"subsegments_{base}_metadata.txt"
    for line in meta_path.read_text().splitlines():
        if line.startswith("beta1_Y="):
            return int(line.split("=", 1)[1])
    raise RuntimeError(f"beta1_Y not found in {meta_path}")


def plot_phase_portraits(positions_by_g: dict[float, np.ndarray]) -> None:
    n = len(positions_by_g)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (g, pos) in zip(axes, sorted(positions_by_g.items())):
        ax.plot(pos[:, 0], pos[:, 1], lw=0.4, alpha=0.8, color="C0")
        # Stroboscopic points overlaid in red.
        strob = pos[::SAMPLES_PER_DRIVE_PERIOD, :2]
        ax.plot(strob[:, 0], strob[:, 1], "o", ms=3.0, color="C3", alpha=0.8)
        label = REGIME_LABELS.get(g, "")
        ax.set_title(f"$\\gamma = {g:.2f}$  {label}", fontsize=10)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$\\dot x$")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Forced Duffing $(x, \\dot x)$ phase portrait with stroboscopic points")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_duffing_cascade_phase_portraits.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def plot_beta1_vs_gamma(beta1_by_g: dict[float, int]) -> None:
    gs = sorted(beta1_by_g.keys())
    bs = [beta1_by_g[g] for g in gs]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(gs, bs, marker="o", lw=1.4, color="C0")
    for g, b in zip(gs, bs):
        ax.annotate(
            REGIME_LABELS.get(g, ""), (g, b),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
    ax.set_xlabel("drive amplitude $\\gamma$")
    ax.set_ylabel(r"$\beta_1(Y)$")
    ax.set_title(
        f"$\\beta_1(Y)$ across the Duffing cascade  "
        f"(boxsize={BOXSIZE}, sb_radius={SB_RADIUS})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_duffing_cascade_beta1.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    positions_by_g: dict[float, np.ndarray] = {}
    beta1_by_g: dict[float, int] = {}

    for gamma in GAMMA_VALUES:
        base = base_name(gamma)
        print(f"\n=== gamma = {gamma}  base = {base}  ({REGIME_LABELS.get(gamma, '')}) ===")
        p = DuffingParams(
            delta=DELTA, alpha=ALPHA, beta=BETA, gamma=gamma, omega=OMEGA,
        )
        positions = simulate(
            p,
            transient_time=TRANSIENT_TIME,
            record_time=RECORD_TIME,
            samples_per_drive_period=SAMPLES_PER_DRIVE_PERIOD,
        )
        tangents = finite_difference_tangents(positions)
        write_lift(
            DATA_DIR, base, positions, tangents,
            meta={
                "delta": DELTA, "alpha": ALPHA, "beta": BETA,
                "gamma": gamma, "omega": OMEGA,
                "regime": REGIME_LABELS.get(gamma, ""),
                "samples_per_drive_period": SAMPLES_PER_DRIVE_PERIOD,
                "transient_time": TRANSIENT_TIME,
                "record_time": RECORD_TIME,
            },
        )
        positions_by_g[gamma] = positions

        run_julia(base)
        beta1_by_g[gamma] = read_metadata_beta1(base)
        print(f"gamma = {gamma}  beta1(Y) = {beta1_by_g[gamma]}")

    plot_phase_portraits(positions_by_g)
    plot_beta1_vs_gamma(beta1_by_g)

    summary = {
        f"gamma={g:.2f}": {
            "regime": REGIME_LABELS.get(g, ""),
            "beta1_Y": beta1_by_g[g],
        }
        for g in GAMMA_VALUES
    }
    (FIG_DIR / "cascade_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {FIG_DIR / 'cascade_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
