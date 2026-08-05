"""
Roessler attractor: period-doubling route to chaos.

Generates timeseries with control over burn-in, integration method, and tolerance.
"""

from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class RoesslerRegime:
    """Period-doubling regime specification for the Roessler attractor."""
    label: str
    a: float
    b: float
    c: float
    expected_period: int | None
    x0: tuple


@dataclass
class Timeseries:
    """Continuous-time trajectory with unit tangent vectors and metadata."""
    t: np.ndarray      # (N,) time samples
    x: np.ndarray      # (N, 3) state trajectory
    v: np.ndarray      # (N, 3) unit tangent vectors
    meta: dict         # system, regime parameters, integration settings


# Roessler regimes: period-doubling cascade to chaos
ROESSLER_REGIMES = {
    "period1": RoesslerRegime(
        label="period1",
        a=0.1,
        b=0.1,
        c=4.0,
        expected_period=1,
        x0=(1.0, 1.0, 0.0),
    ),
    "period2": RoesslerRegime(
        label="period2",
        a=0.1,
        b=0.1,
        c=6.0,
        expected_period=2,
        x0=(1.0, 1.0, 0.0),
    ),
    "period4": RoesslerRegime(
        label="period4",
        a=0.1,
        b=0.1,
        c=8.5,
        expected_period=4,
        x0=(1.0, 1.0, 0.0),
    ),
    "period8": RoesslerRegime(
        label="period8",
        a=0.1,
        b=0.1,
        c=8.7,
        expected_period=8,
        x0=(1.0, 1.0, 0.0),
    ),
    "chaos": RoesslerRegime(
        label="chaos",
        a=0.1,
        b=0.1,
        c=9.0,
        expected_period=None,
        x0=(1.0, 1.0, 0.0),
    ),
}


def vector_field(t, state, a, b, c):
    """Roessler vector field.

    Parameters
    ----------
    t : float
        Time (unused, for compatibility with solve_ivp).
    state : array-like (3,)
        [x, y, z]
    a, b, c : float
        Roessler parameters.

    Returns
    -------
    array (3,)
        [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    return np.array([
        -y - z,
        x + a * y,
        b + z * (x - c),
    ])


def generate_timeseries(regime, t_span=500.0, dt=0.02, burn_in=2000.0,
                        method="DOP853", rtol=1e-10, atol=1e-12):
    """Generate a Roessler timeseries with burn-in and uniform sampling.

    Parameters
    ----------
    regime : RoesslerRegime
        Regime specification (a, b, c, x0).
    t_span : float
        Integration interval length (seconds).
    dt : float
        Uniform sampling interval.
    burn_in : float
        Burn-in integration time before main pass.
    method : str
        solve_ivp integration method.
    rtol, atol : float
        Integration tolerances for main pass.

    Returns
    -------
    Timeseries
        Sampled trajectory with unit tangent vectors and metadata.
    """
    a, b, c = regime.a, regime.b, regime.c
    x0 = np.array(regime.x0, dtype=float)

    # Burn-in: cheap integration to reach attractor
    sol_burn = solve_ivp(
        lambda t, s: vector_field(t, s, a, b, c),
        [0, burn_in],
        x0,
        method=method,
        rtol=1e-9,
        atol=1e-11,
        dense_output=False,
    )
    x_final_burn = sol_burn.y[:, -1]

    # Sample at uniform grid t_k = k*dt. Integrate exactly to the last grid
    # point so the dense interpolant is never extrapolated (t_span need not
    # be a multiple of dt).
    n_samples = int(np.round(t_span / dt)) + 1
    t_grid = np.arange(n_samples) * dt
    t_end = t_grid[-1]

    # Main pass: integrate with dense output
    sol = solve_ivp(
        lambda t, s: vector_field(t, s, a, b, c),
        [0, t_end],
        x_final_burn,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    x_grid = sol.sol(t_grid).T  # (N, 3)

    # Compute tangent vectors (vector field normalized to unit length)
    v_raw = np.array([
        vector_field(t, x, a, b, c)
        for t, x in zip(t_grid, x_grid)
    ])
    norms = np.linalg.norm(v_raw, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-300)  # Guard against zero-norm
    v_grid = v_raw / norms  # (N, 3)

    meta = {
        "system": "roessler",
        "label": regime.label,
        "a": float(regime.a),
        "b": float(regime.b),
        "c": float(regime.c),
        "expected_period": regime.expected_period,
        "x0": regime.x0,
        "t_span": float(t_span),
        "dt": float(dt),
        "burn_in": float(burn_in),
        "method": method,
        "rtol": rtol,
        "atol": atol,
        "n_samples": n_samples,
    }

    return Timeseries(t=t_grid, x=x_grid, v=v_grid, meta=meta)
