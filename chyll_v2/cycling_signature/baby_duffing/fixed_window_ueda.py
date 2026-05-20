"""Bernardo's fixed-integration-window cascade detector on verified Ueda cascade.

Ueda equation x'' + 0.05 x' + x^3 = gamma cos(t).

Verified cascade points (via closure-ratio diagnostic in verify_ueda_cascade.py):

    gamma = 9.000   period-1   (r_1 = 0.002, all r_n linear in n)
    gamma = 5.250   period-2   (r_1 = 0.16, r_2 = 0.0015)
    gamma = 5.375   period-4   (r_1 = 0.50, r_2 = 0.41, r_4 = 0.02)
    gamma = 7.500   chaotic    (no closure at any n)

For each gamma we simulate >=8 drive periods at high sample density,
then slice the first nT samples (with closure included) and run cycling-
signature to read beta_1(Y). Expected table:

                   window=T   window=2T   window=4T   window=8T
    period-1          1           1           1           1
    period-2          0           1           1           1
    period-4          0           0           1           1
    chaos             0           0           0           0

i.e. beta_1(Y) flips from 0 to 1 at the window where the orbit first
closes. The pattern across rows IS the cascade signature.
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

DATA_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "baby_ueda"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GAMMAS = {
    "period_1": 9.000,
    "period_2": 5.250,
    "period_4": 5.375,
    "chaos":    7.500,
}

# Sample density: 200 per drive period (5x denser than the original
# Duffing experiment) so the cubical complex has multiple samples per
# box and the topological ring is robust.
SAMPLES_PER_T = 200
N_WINDOW_PERIODS = 8        # total trajectory covers 8 drive periods
WINDOWS = [1, 2, 4, 8]      # window sizes to test, in drive periods

# Ueda parameters
DELTA = 0.05
ALPHA = 0.0
BETA = 1.0
OMEGA = 1.0
TRANSIENT_TIME = 3000.0
RECORD_TIME = (N_WINDOW_PERIODS + 0.05) * 2 * np.pi  # 8T + small buffer

# Cycling-signature parameters. boxsize chosen large enough that each
# box contains multiple samples of the now-dense trajectory.
BOXSIZE = 0.5
SB_RADIUS = 1
R_MAX = 2.0


def simulate_full(gamma: float) -> np.ndarray:
    p = DuffingParams(delta=DELTA, alpha=ALPHA, beta=BETA, gamma=gamma, omega=OMEGA)
    return simulate(
        p,
        transient_time=TRANSIENT_TIME,
        record_time=RECORD_TIME,
        samples_per_drive_period=SAMPLES_PER_T,
        rtol=1e-11, atol=1e-13,
    )


def write_slice(positions: np.ndarray, base: str, n_samples: int) -> str:
    pos_slice = positions[:n_samples].copy()
    tangents = finite_difference_tangents(pos_slice)
    np.savetxt(DATA_DIR / f"{base}_positions.csv", pos_slice, delimiter=" ")
    np.savetxt(DATA_DIR / f"{base}_tangents.csv", tangents, delimiter=" ")
    np.save(DATA_DIR / f"{base}_positions.npy", pos_slice)
    np.save(DATA_DIR / f"{base}_tangents.npy", tangents)
    return base


def run_julia_beta1(base: str) -> int:
    out_prefix = f"subsegments_{base}"
    cmd = [
        "julia",
        f"--project=time series/cycling_signature",
        "chyll_v2/cycling_signature/julia/run_subsegments.jl",
        "--data-dir", str(DATA_DIR.relative_to(REPO_ROOT)),
        "--base", base,
        "--boxsize", str(BOXSIZE),
        "--sb-radius", str(SB_RADIUS),
        "--r-max", str(R_MAX),
        "--eval-radius", "0.5",
        "--segment-lengths", "20:10:100",
        "--n-runs", "20",
        "--max-rank", "1",
        "--out-prefix", out_prefix,
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        return -1
    for line in res.stdout.splitlines():
        if "comparison beta_1(Y)=" in line:
            return int(line.split("=")[-1].strip())
    return -1


def main() -> int:
    results: dict[str, dict[int, int]] = {}
    for regime, gamma in GAMMAS.items():
        print(f"\nSimulating gamma = {gamma}  ({regime}) ...", flush=True)
        full = simulate_full(gamma)
        print(f"  trajectory shape: {full.shape}", flush=True)
        results[regime] = {}
        for w in WINDOWS:
            n_samples = w * SAMPLES_PER_T + 1   # +1 to include closure point
            base = f"ueda_{regime}_g{gamma:.3f}_win{w}T".replace(".", "p")
            write_slice(full, base, n_samples)
            beta1 = run_julia_beta1(base)
            results[regime][w] = beta1
            print(f"  window={w}T ({n_samples} samples)  ->  beta_1(Y) = {beta1}", flush=True)

    print("\n" + "=" * 64)
    print("Fixed-window cycling-signature cascade detector  (Ueda equation)")
    print("=" * 64)
    print(f"{'regime':>10s} {'gamma':>7s}   " + "  ".join(f"win={w}T" for w in WINDOWS))
    print("-" * 64)
    for regime, gamma in GAMMAS.items():
        row = "  ".join(f"   {results[regime][w]:>2d}  " for w in WINDOWS)
        print(f"{regime:>10s} {gamma:>7.3f}   {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
