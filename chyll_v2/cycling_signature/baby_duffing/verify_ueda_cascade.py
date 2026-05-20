"""Verify Ueda-equation cascade parameters before running any cycling-signature.

Ueda equation:
    x'' + delta * x' + x^3 = gamma * cos(omega * t)
(equivalently DuffingParams(alpha=0, beta=1, delta=0.05, omega=1.0) in our
parameterisation.)

Diagnostic per gamma:

    For each n in {1, 2, 4, 8}, compute the closure ratio
        r_n := |pos[0] - pos[n * T_samples]| / orbit_diameter
    where T_samples = samples per drive period and orbit_diameter is the
    range of |pos[k]| across the simulation.

    Period-1 orbit: r_1, r_2, r_4, r_8 all small AND r_n grows roughly
                    linearly with n (numerical drift only).
    Period-2 orbit: r_1 is ~ 1 (orbit not closed after T) but r_2 is small.
    Period-4 orbit: r_1, r_2 are ~ 1 but r_4 is small.
    Chaos:          r_n large and irregular for all n.

We sweep gamma at high resolution to find clean period-1, period-2, period-4
examples. No Julia is invoked. This is a setup-verification step only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from chyll_v2.cycling_signature.baby_duffing.simulate import (  # noqa: E402
    DuffingParams,
    simulate,
)

SAMPLES_PER_T = 40
TRANSIENT_TIME = 3000.0   # long, so the orbit has settled
RECORD_TIME = 100.0       # ~16 drive periods, plenty for n in {1,2,4,8}

GAMMA_VALUES = np.round(
    np.arange(5.25, 5.55, 0.025),
    3,
)


def closure_ratios(gamma: float) -> dict:
    p = DuffingParams(delta=0.05, alpha=0.0, beta=1.0, gamma=gamma, omega=1.0)
    pos = simulate(
        p,
        transient_time=TRANSIENT_TIME,
        record_time=RECORD_TIME,
        samples_per_drive_period=SAMPLES_PER_T,
        rtol=1e-11, atol=1e-13,  # very tight to suppress integration drift
    )
    diameter = max(
        np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)),
        1e-6,
    )
    ratios = {}
    for n in (1, 2, 4, 8):
        idx = n * SAMPLES_PER_T
        if idx >= len(pos):
            ratios[n] = float("nan")
        else:
            ratios[n] = float(np.linalg.norm(pos[0] - pos[idx]) / diameter)
    return {"gamma": gamma, "diameter": float(diameter), **{f"r_{n}": ratios[n] for n in (1, 2, 4, 8)}}


def classify(r: dict, drift_threshold: float = 0.01, closed_threshold: float = 0.01) -> str:
    """Heuristic regime classification from closure ratios.

    'period-n' means r_n is closed (< closed_threshold) and r_k is open
    (>= 5 * r_n) for k < n. Drift-only would have all r_n small and
    roughly proportional to n.
    """
    r1, r2, r4, r8 = r["r_1"], r["r_2"], r["r_4"], r["r_8"]
    # Drift detection: all small and r_n approx n * r_1.
    if max(r1, r2, r4, r8) < drift_threshold:
        # Could be period-1 with drift. Confirm linearity:
        if (r2 < 3 * r1 + 0.005 and r4 < 5 * r1 + 0.005):
            return "period-1"
        return "period-1 (drift)"
    # Period-2: r_2 small, r_1 large.
    if r2 < closed_threshold and r1 > 5 * r2:
        return "period-2"
    # Period-4: r_4 small, r_1, r_2 large.
    if r4 < closed_threshold and r1 > 5 * r4 and r2 > 5 * r4:
        return "period-4"
    # Period-8.
    if r8 < closed_threshold and r1 > 5 * r8 and r2 > 5 * r8 and r4 > 5 * r8:
        return "period-8"
    return "chaotic / other"


def main() -> int:
    print(f"{'gamma':>8s} {'diam':>7s} {'r_1':>9s} {'r_2':>9s} {'r_4':>9s} {'r_8':>9s}   regime")
    print("-" * 78)
    for gamma in GAMMA_VALUES:
        r = closure_ratios(float(gamma))
        regime = classify(r)
        print(
            f"{r['gamma']:>8.3f} {r['diameter']:>7.3f} "
            f"{r['r_1']:>9.5f} {r['r_2']:>9.5f} "
            f"{r['r_4']:>9.5f} {r['r_8']:>9.5f}   {regime}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
