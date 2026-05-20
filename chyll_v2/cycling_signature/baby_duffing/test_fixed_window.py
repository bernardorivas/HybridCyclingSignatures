"""Bernardo's fixed-integration-window experiments on the Duffing cascade.

Drive period T = 2 pi / omega = 2 pi at omega = 1. At 40 samples per drive
period that's T = 40 samples.

EXP 1: integration window = T (40 samples).
   Period-1 orbit closes once -> tube around it has beta_1 = 1.
   Period-2+ orbits do not close in [0, T] -> open curve -> beta_1 = 0.
   Expected outcome: beta_1(Y) = (1, 0, 0, 0, 0) across (1.0, 1.6, 2.6, 3.0, 10.0).

EXP 2: integration window = 4 T (160 samples), max_rank = 4.
   Period-1: 4 closures, length-T subsegment achieves rank-1, length-2T rank-2, etc.
   Period-2: 2 closures, length-T half-closure, length-2T rank-1, length-4T rank-2.
   Period-4: 1 closure, length-4T rank-1.
   Period-8+: no closure -> rank-0 everywhere.

For both experiments we slice the existing long trajectories so we don't
re-simulate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from chyll_v2.cycling_signature.baby_duffing.simulate import (  # noqa: E402
    finite_difference_tangents,
)

DATA_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "baby_duffing"

GAMMA_VALUES = [1.0, 1.6, 2.6, 3.0, 10.0]
SAMPLES_PER_T = 40
T_SAMPLES = SAMPLES_PER_T  # one drive period

# Match the boxsize / radius used for the full-trajectory cascade so the
# only thing that changes is integration window length.
BOXSIZE = 0.2
SB_RADIUS = 1


def base_4d(gamma: float) -> str:
    return f"duffing_g{gamma:.2f}".replace(".", "p")


def slice_lift(gamma: float, n_samples: int, suffix: str) -> str:
    base4d = base_4d(gamma)
    positions = np.load(DATA_DIR / f"{base4d}_positions.npy")[:n_samples]
    tangents = finite_difference_tangents(positions)
    new_base = f"{base4d}_{suffix}"
    np.savetxt(DATA_DIR / f"{new_base}_positions.csv", positions, delimiter=" ")
    np.savetxt(DATA_DIR / f"{new_base}_tangents.csv", tangents, delimiter=" ")
    np.save(DATA_DIR / f"{new_base}_positions.npy", positions)
    np.save(DATA_DIR / f"{new_base}_tangents.npy", tangents)
    return new_base


def run_julia(base: str, max_rank: int, segment_lengths: str, n_runs: int,
              r_max: float, eval_radius: float, out_prefix: str) -> int:
    cmd = [
        "julia",
        f"--project=time series/cycling_signature",
        "chyll_v2/cycling_signature/julia/run_subsegments.jl",
        "--data-dir", str(DATA_DIR.relative_to(REPO_ROOT)),
        "--base", base,
        "--boxsize", str(BOXSIZE),
        "--sb-radius", str(SB_RADIUS),
        "--r-max", str(r_max),
        "--eval-radius", str(eval_radius),
        "--segment-lengths", segment_lengths,
        "--n-runs", str(n_runs),
        "--max-rank", str(max_rank),
        "--out-prefix", out_prefix,
    ]
    print(f"  julia subsegments on {base} (max_rank={max_rank})")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    # Parse beta_1(Y) from stdout if successful.
    if res.returncode != 0:
        print(f"  julia FAILED for {base}:")
        print(res.stderr[-1500:])
        return -1
    for line in res.stdout.splitlines():
        if "comparison beta_1(Y)=" in line:
            return int(line.split("=")[-1].strip())
    return -1


def main() -> int:
    # Include the closure point by taking +1 extra sample. Period-1 closes
    # at sample index T_SAMPLES, so slicing [:T_SAMPLES + 1] captures the
    # closed loop. Use larger boxsize so the cubical complex around the
    # ~40-200-sample trajectory has dense-enough box coverage to detect
    # the homology.

    print("\n=== EXPERIMENT 1: integration window = T (single drive period, closure included) ===")
    print(f"trajectory length: {T_SAMPLES + 1} samples per gamma\n")
    for gamma in GAMMA_VALUES:
        base = slice_lift(gamma, T_SAMPLES + 1, f"win1Tclose")
        seg_lengths = f"10:5:{T_SAMPLES - 5}"
        beta1 = run_julia(
            base, max_rank=1, segment_lengths=seg_lengths, n_runs=50,
            r_max=1.0, eval_radius=0.3,
            out_prefix=f"subsegments_{base}",
        )
        print(f"  gamma = {gamma}:  beta_1(Y) = {beta1}\n")

    print("\n=== EXPERIMENT 2: integration window = 4T (closure included, max_rank=4) ===")
    print(f"trajectory length: {4 * T_SAMPLES + 1} samples per gamma\n")
    for gamma in GAMMA_VALUES:
        base = slice_lift(gamma, 4 * T_SAMPLES + 1, f"win4Tclose")
        seg_lengths = f"20:10:{4 * T_SAMPLES - 10}"
        beta1 = run_julia(
            base, max_rank=4, segment_lengths=seg_lengths, n_runs=100,
            r_max=1.0, eval_radius=0.3,
            out_prefix=f"subsegments_{base}",
        )
        print(f"  gamma = {gamma}:  beta_1(Y) = {beta1}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
