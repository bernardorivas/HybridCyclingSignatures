"""Sanity check: does cycling-signature give beta_1(Y) = 1 on a perfect 2D circle?

A circle of radius R in R^2, with 1000 samples, tangent vectors pointing
in the orbital direction. Tube around it should have beta_1 = 1
regardless of boxsize (up to scale).

This is the simplest possible test of the framework, BEFORE we make
claims about 4D forced-oscillator orbits.

We sweep boxsize and report beta_1(Y) at each, so we can see whether
the framework consistently reports 1, or whether the construction is
fragile in some parameter regime.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "sanity_circle"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_circle(n: int = 1000, r: float = 5.0, include_closure: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return positions (N, 2) and unit tangents (N, 2) for a circle."""
    if include_closure:
        theta = np.linspace(0.0, 2 * np.pi, n + 1)  # closure included at end
    else:
        theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    tx = -np.sin(theta)
    ty = np.cos(theta)
    return np.stack([x, y], axis=1), np.stack([tx, ty], axis=1)


def run_julia(base: str, boxsize: float, sb_radius: int, r_max: float) -> int:
    cmd = [
        "julia",
        f"--project=time series/cycling_signature",
        "chyll_v2/cycling_signature/julia/run_subsegments.jl",
        "--data-dir", str(OUT_DIR.relative_to(REPO_ROOT)),
        "--base", base,
        "--boxsize", str(boxsize),
        "--sb-radius", str(sb_radius),
        "--r-max", str(r_max),
        "--eval-radius", str(r_max / 2),
        "--segment-lengths", "50:50:500",
        "--n-runs", "10",
        "--max-rank", "1",
        "--out-prefix", f"subsegments_{base}",
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        return -1
    for line in res.stdout.splitlines():
        if "comparison beta_1(Y)=" in line:
            return int(line.split("=")[-1].strip())
    return -1


def main() -> int:
    pos, tan = make_circle(n=1000, r=5.0, include_closure=True)
    base = "circle_2d_r5_n1001"
    np.savetxt(OUT_DIR / f"{base}_positions.csv", pos, delimiter=" ")
    np.savetxt(OUT_DIR / f"{base}_tangents.csv", tan, delimiter=" ")
    print(f"circle: shape={pos.shape}, radius=5.0")

    print(f"\n{'boxsize':>9s} {'sb_r':>5s} {'r_max':>7s}   beta_1(Y)")
    print("-" * 40)
    boxsizes = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]
    for bs in boxsizes:
        for sb in (1, 2):
            for r_max in (1.0, 5.0):
                beta = run_julia(base, bs, sb, r_max)
                print(f"{bs:>9.2f} {sb:>5d} {r_max:>7.1f}   {beta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
