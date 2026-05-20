"""Re-run the chaotic gamma values of the Duffing cascade with coarser CS params.

The original ``run_cascade.py`` used boxsize=0.2 + 200 subsegments, which
takes ~1 min per periodic gamma but blows up to ~hours per chaotic gamma.
For the chaos endpoints we instead use boxsize=0.5 + n_runs=50 + a coarser
segment-length grid. The periodic gamma results from ``run_cascade.py``
are kept as-is (don't rerun them).

Usage:
    python chyll_v2/cycling_signature/baby_duffing/run_chaos_rerun.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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

GAMMA_VALUES = [3.0, 10.0]
REGIME_LABELS = {3.0: "chaotic band", 10.0: "chaos"}

DELTA = 0.1
ALPHA = 1.0
BETA = 1.0
OMEGA = 1.0

TRANSIENT_TIME = 500.0
RECORD_TIME = 400.0
SAMPLES_PER_DRIVE_PERIOD = 40

# Chaos-appropriate parameters: coarser boxsize so the cubical comparison
# space has fewer boxes, fewer subsegment runs, sparser segment-length
# grid. r_max bumped because chaos orbits have larger extent.
BOXSIZE = 0.5
SB_RADIUS = 1
R_MAX = 2.0
EVAL_RADIUS = 0.5
SEGMENT_LENGTHS = "50:50:1000"
N_RUNS = 50


def base_name(gamma: float) -> str:
    return f"duffing_g{gamma:.2f}".replace(".", "p")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for gamma in GAMMA_VALUES:
        base = base_name(gamma)
        print(f"\n=== gamma = {gamma}  base = {base}  ({REGIME_LABELS[gamma]}) ===")
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
                "regime": REGIME_LABELS[gamma],
                "samples_per_drive_period": SAMPLES_PER_DRIVE_PERIOD,
                "transient_time": TRANSIENT_TIME,
                "record_time": RECORD_TIME,
                "boxsize_used": BOXSIZE,
                "n_runs_used": N_RUNS,
            },
        )

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
        print(f"gamma = {gamma} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
