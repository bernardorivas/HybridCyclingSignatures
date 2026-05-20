"""Run the Rossler period-doubling cascade through the cycling-signature pipeline.

Generates one trajectory + lift CSVs for each value of ``c`` in a
hand-picked cascade list, then drives the Julia subsegment analysis
and the beta_1(Y) sweep on each lift. Finally produces:

  - a phase-portrait grid for visual sanity
  - one rank-distribution + heatmap figure per c (via existing plot script)
  - one summary plot: beta_1(Y) vs c at fixed (boxsize, sb_radius)

Usage:
    python chyll_v2/cycling_signature/baby_rossler/run_cascade.py
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

from chyll_v2.cycling_signature.baby_rossler.simulate import (  # noqa: E402
    RosslerParams,
    finite_difference_tangents,
    simulate,
    write_lift,
)


DATA_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "baby_rossler"
FIG_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "figures" / "baby_rossler"

# Cascade: stable -> period-doubling -> chaos. Values selected so they
# hit distinct regimes; if the signatures don't separate these, the
# compass cascade won't either.
C_VALUES = [2.5, 3.5, 4.0, 4.15, 4.5, 5.7]

TAU = 0.05
TRANSIENT_TIME = 300.0
RECORD_TIME = 500.0     # 10000 samples per c at tau=0.05; ~25-50 periods
BOXSIZE = 0.5            # Rossler attractor extent ~ 10, so boxsize 0.5 is fine
SB_RADIUS = 1
R_MAX = 1.0
EVAL_RADIUS = 0.3
SEGMENT_LENGTHS = "50:25:1500"
N_RUNS = 200


def base_name(c: float) -> str:
    return f"rossler_c{c:.2f}".replace(".", "p")


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


def run_beta1_sweep(base: str) -> None:
    cmd = [
        "julia",
        f"--project=time series/cycling_signature",
        "chyll_v2/cycling_signature/julia/sweep_beta1_comparison_space.jl",
        "--data-dir", str(DATA_DIR.relative_to(REPO_ROOT)),
        "--base", base,
        "--boxsizes", "0.10,0.20,0.30,0.50,0.75,1.00,1.50,2.00",
        "--sb-radii", "1,2,3",
        "--r-max", str(R_MAX),
        "--out", f"subsegments_{base}_beta1_sweep.csv",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def read_metadata_beta1(base: str) -> int:
    meta_path = DATA_DIR / f"subsegments_{base}_metadata.txt"
    for line in meta_path.read_text().splitlines():
        if line.startswith("beta1_Y="):
            return int(line.split("=", 1)[1])
    raise RuntimeError(f"beta1_Y not found in {meta_path}")


def plot_phase_portraits(positions_by_c: dict[float, np.ndarray]) -> None:
    n = len(positions_by_c)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows))
    axes = axes.flatten()
    for ax, (c, pos) in zip(axes, sorted(positions_by_c.items())):
        ax.plot(pos[:, 0], pos[:, 1], lw=0.4, alpha=0.8, color="C0")
        ax.set_title(f"$c = {c:.2f}$", fontsize=10)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Rossler $(x, y)$ phase portrait across the cascade", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_rossler_cascade_phase_portraits.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def plot_beta1_vs_c(beta1_by_c: dict[float, int]) -> None:
    cs = sorted(beta1_by_c.keys())
    bs = [beta1_by_c[c] for c in cs]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(cs, bs, marker="o", lw=1.4, color="C0")
    ax.set_xlabel("Rossler parameter $c$")
    ax.set_ylabel(r"$\beta_1(Y)$")
    ax.set_title(
        f"$\\beta_1(Y)$ across the Rossler cascade  "
        f"(boxsize={BOXSIZE}, sb_radius={SB_RADIUS})"
    )
    ax.set_yticks(range(0, max(bs) + 2))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_rossler_cascade_beta1.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    positions_by_c: dict[float, np.ndarray] = {}
    beta1_by_c: dict[float, int] = {}

    for c in C_VALUES:
        base = base_name(c)
        print(f"\n=== c = {c}  base = {base} ===")
        p = RosslerParams(c=c)
        positions = simulate(
            p,
            transient_time=TRANSIENT_TIME,
            record_time=RECORD_TIME,
            tau=TAU,
        )
        tangents = finite_difference_tangents(positions)
        write_lift(
            DATA_DIR, base, positions, tangents,
            meta={
                "rossler_a": p.a,
                "rossler_b": p.b,
                "rossler_c": p.c,
                "tau": TAU,
                "transient_time": TRANSIENT_TIME,
                "record_time": RECORD_TIME,
            },
        )
        positions_by_c[c] = positions

        run_julia(base)
        beta1_by_c[c] = read_metadata_beta1(base)
        print(f"c = {c}  beta1(Y) = {beta1_by_c[c]}")

    # Phase portraits and beta_1 summary.
    plot_phase_portraits(positions_by_c)
    plot_beta1_vs_c(beta1_by_c)

    summary_path = FIG_DIR / "cascade_summary.json"
    summary_path.write_text(json.dumps(
        {f"c={c:.2f}": beta1_by_c[c] for c in C_VALUES}, indent=2
    ))
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
