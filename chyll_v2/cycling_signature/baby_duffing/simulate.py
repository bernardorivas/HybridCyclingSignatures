"""Forced Duffing oscillator simulator and trajectory export.

The forced Duffing oscillator is the standard textbook continuous flow
with a period-doubling cascade and *stacked-loop* orbit geometry:

    x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t)

The cascade is along ``gamma`` at fixed ``(delta, alpha, beta, omega)``.
With Holmes 1979 parameters ``delta = 0.3, alpha = -1, beta = 1,
omega = 1.0`` (double-well potential, weak damping):

    gamma = 0.20  period-1 limit cycle
    gamma = 0.28  period-2
    gamma = 0.30  period-4
    gamma = 0.34  period-8 (approximate)
    gamma = 0.40  chaotic band
    gamma = 0.50  chaos (Ueda-style strange attractor)

We promote the explicit time-dependence to an autonomous 4D state by
appending ``(cos(omega t), sin(omega t))`` (the drive phase as a point
on the unit circle, avoiding the discontinuity at ``omega t = 2 pi``):

    state = (x, x_dot, cos(omega t), sin(omega t)) in R^4

The orbit lives on R^2 x S^1 (a solid torus topologically). Period-n
orbits are closed curves winding n times around S^1; cycling-signature
on this orbit's stacked-loop structure is the test we want.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class DuffingParams:
    delta: float = 0.3
    alpha: float = -1.0
    beta: float = 1.0
    gamma: float = 0.40
    omega: float = 1.0


def duffing_rhs(t: float, state: np.ndarray, p: DuffingParams) -> np.ndarray:
    """Augmented 4D RHS with the drive phase on S^1.

    state = (x, x_dot, c, s) where (c, s) = (cos(omega t), sin(omega t)).
    The phase coordinates satisfy dc/dt = -omega s, ds/dt = omega c.
    """
    x, xdot, c, s = state
    return np.array([
        xdot,
        -p.delta * xdot - p.alpha * x - p.beta * x**3 + p.gamma * c,
        -p.omega * s,
        p.omega * c,
    ])


def simulate(
    p: DuffingParams,
    x0: np.ndarray | None = None,
    transient_time: float = 200.0,
    record_time: float = 400.0,
    tau: float | None = None,
    samples_per_drive_period: int = 40,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> np.ndarray:
    """Return an (N, 4) trajectory after discarding the transient.

    Default sampling: 40 samples per drive period so the orbit has
    plenty of resolution within one cycle. ``record_time`` is in time
    units, not periods.
    """
    if tau is None:
        tau = (2 * np.pi / p.omega) / samples_per_drive_period
    if x0 is None:
        x0 = np.array([0.5, 0.0, 1.0, 0.0], dtype=np.float64)

    sol = solve_ivp(
        lambda t, y: duffing_rhs(t, y, p),
        (0.0, transient_time),
        x0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if sol.status != 0:
        raise RuntimeError(f"Duffing transient integration failed: status {sol.status}")
    state_after = sol.y[:, -1]
    # Re-normalise the (c, s) coordinates to stay exactly on the unit
    # circle after the transient (numerical drift over long times).
    c, s = state_after[2], state_after[3]
    rho = np.hypot(c, s)
    state_after[2] /= rho
    state_after[3] /= rho

    n_steps = int(round(record_time / tau)) + 1
    t_eval = np.linspace(0.0, record_time, n_steps)
    sol = solve_ivp(
        lambda t, y: duffing_rhs(t, y, p),
        (0.0, record_time),
        state_after,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
    )
    if sol.status != 0:
        raise RuntimeError(f"Duffing record integration failed: status {sol.status}")
    return sol.y.T


def finite_difference_tangents(positions: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dz = np.diff(positions, axis=0)
    dz = np.vstack([dz, dz[-1:]])
    nrm = np.linalg.norm(dz, axis=1)
    bad = nrm < eps
    if bad.any():
        good = np.where(~bad)[0]
        if good.size == 0:
            raise ValueError("all tangent rows are numerically zero")
        for i in np.where(bad)[0]:
            prev = good[good < i]
            dz[i] = dz[prev[-1] if prev.size else good[0]]
        nrm = np.linalg.norm(dz, axis=1)
    return dz / nrm[:, None]


def write_lift(
    out_dir: Path,
    base: str,
    positions: np.ndarray,
    tangents: np.ndarray,
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / f"{base}_positions.csv", positions, delimiter=" ")
    np.savetxt(out_dir / f"{base}_tangents.csv", tangents, delimiter=" ")
    np.save(out_dir / f"{base}_positions.npy", positions)
    np.save(out_dir / f"{base}_tangents.npy", tangents)
    lines = [
        "Forced Duffing lift",
        "=" * 32,
        f"samples: {positions.shape[0]}",
        f"dim: {positions.shape[1]}",
    ]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    (out_dir / f"report_{base}.txt").write_text("\n".join(lines) + "\n")
