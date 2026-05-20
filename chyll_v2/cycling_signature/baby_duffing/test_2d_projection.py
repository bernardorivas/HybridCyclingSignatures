"""Test the 2D (x, xdot) projection of the Duffing cascade for cycling-signature.

The 4D embedding (x, xdot, cos t, sin t) gives beta_1(Y) = 1 for every
periodic orbit because the orbit is topologically S^1 in 4D regardless
of period. Dropping the drive-phase coordinates puts us in 2D (x, xdot)
where the orbit self-intersects (period-n has 2n stacked loops with
a single self-intersection point at the "neck"). The cubical comparison
space of a self-intersecting figure-8 should give beta_1 > 1.

This is a quick test: load existing positions, drop columns, redo
tangents in 2D, write lift, run subsegments.
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

GAMMA_VALUES = [1.0, 1.6, 2.6, 3.0]


def base_4d(gamma: float) -> str:
    return f"duffing_g{gamma:.2f}".replace(".", "p")


def base_2d(gamma: float) -> str:
    return f"duffing_2d_g{gamma:.2f}".replace(".", "p")


def main() -> int:
    for gamma in GAMMA_VALUES:
        b4 = base_4d(gamma)
        b2 = base_2d(gamma)
        positions_4d = np.load(DATA_DIR / f"{b4}_positions.npy")
        positions_2d = positions_4d[:, :2].copy()  # (x, xdot) only
        tangents_2d = finite_difference_tangents(positions_2d)
        np.savetxt(DATA_DIR / f"{b2}_positions.csv", positions_2d, delimiter=" ")
        np.savetxt(DATA_DIR / f"{b2}_tangents.csv", tangents_2d, delimiter=" ")
        np.save(DATA_DIR / f"{b2}_positions.npy", positions_2d)
        np.save(DATA_DIR / f"{b2}_tangents.npy", tangents_2d)
        print(f"gamma={gamma}: wrote 2D lift, shape={positions_2d.shape}")

        # Run subsegments with a boxsize-sweep mindset: small enough to see
        # figure-8 separation, large enough to keep loops connected.
        cmd = [
            "julia",
            f"--project=time series/cycling_signature",
            "chyll_v2/cycling_signature/julia/run_subsegments.jl",
            "--data-dir", str(DATA_DIR.relative_to(REPO_ROOT)),
            "--base", b2,
            "--boxsize", "0.1",
            "--sb-radius", "1",
            "--r-max", "0.5",
            "--eval-radius", "0.2",
            "--segment-lengths", "50:25:1500",
            "--n-runs", "100",
            "--out-prefix", f"subsegments_{b2}",
        ]
        print("running:", " ".join(cmd[:4]), b2)
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        meta = (DATA_DIR / f"subsegments_{b2}_metadata.txt").read_text()
        for line in meta.splitlines():
            if line.startswith("beta1_Y="):
                print(f"  ==> gamma={gamma}  2D projection  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
