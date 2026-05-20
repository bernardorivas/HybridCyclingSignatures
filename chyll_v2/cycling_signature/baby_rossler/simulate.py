"""Rossler attractor simulator and trajectory export.

The Rossler system is the canonical 3D continuous ODE for a period-doubling
cascade to chaos:

    dx/dt = -y - z
    dy/dt =  x + a y
    dz/dt =  b + z (x - c)

With ``a = b = 0.2`` fixed, varying ``c`` traverses the cascade:

    c = 2.5   period-1 limit cycle
    c = 3.5   period-2
    c = 4.0   period-4
    c = 4.15  period-8
    c = 4.5   chaotic band
    c = 5.7   classic chaotic attractor (Rossler's reported value)

There is no hybrid structure here: the dynamics is purely continuous. We
sample the trajectory at uniform ``tau`` after discarding a transient,
then write the standard space-separated ``*_positions.csv`` and
``*_tangents.csv`` files consumed by
``chyll_v2/cycling_signature/julia/run_subsegments.jl``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class RosslerParams:
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7


def rossler_rhs(t: float, state: np.ndarray, p: RosslerParams) -> np.ndarray:
    x, y, z = state
    return np.array([
        -y - z,
        x + p.a * y,
        p.b + z * (x - p.c),
    ])


def simulate(
    p: RosslerParams,
    x0: np.ndarray = np.array([0.1, 0.0, 0.0]),
    transient_time: float = 200.0,
    record_time: float = 300.0,
    tau: float = 0.05,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> np.ndarray:
    """Return an (N, 3) trajectory sampled every ``tau`` after the transient.

    ``transient_time`` is integrated and discarded so the trajectory
    settles onto the attractor before recording. ``record_time`` is then
    sampled at ``N = round(record_time / tau) + 1`` uniform timestamps.
    """
    # Run the transient.
    sol = solve_ivp(
        lambda t, y: rossler_rhs(t, y, p),
        (0.0, transient_time),
        x0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    if sol.status != 0:
        raise RuntimeError(f"Rossler transient integration failed: status {sol.status}")
    state_after_transient = sol.y[:, -1]

    # Record at uniform sample times.
    n_steps = int(round(record_time / tau)) + 1
    t_eval = np.linspace(0.0, record_time, n_steps)
    sol = solve_ivp(
        lambda t, y: rossler_rhs(t, y, p),
        (0.0, record_time),
        state_after_transient,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
    )
    if sol.status != 0:
        raise RuntimeError(f"Rossler record integration failed: status {sol.status}")
    return sol.y.T  # (N, 3)


def finite_difference_tangents(positions: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Forward-difference unit tangents with a tiny-row repair."""
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
        "Rossler attractor lift",
        "=" * 32,
        f"samples: {positions.shape[0]}",
        f"dim: {positions.shape[1]}",
    ]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    (out_dir / f"report_{base}.txt").write_text("\n".join(lines) + "\n")
