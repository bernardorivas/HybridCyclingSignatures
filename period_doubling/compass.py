"""Compass gait biped: hybrid system with guard and impact map.

Simulates a passive compass-gait walker (Goswami 2000) on a slope.
State: (theta_ns, theta_s, dtheta_ns, dtheta_s) -- non-stance leg, stance leg, angular velocities.
Guard: theta_ns + theta_s + 2*phi = 0 (heelstrike), active when theta_ns > theta_s.
Impact: angular momentum conservation + leg-role swap (Goswami Eq. A.34-A.37).

Reference: /Users/bdoprad/Work/Projects/hybrid-cycling-signatures/code/time series/compass gait/simulate.py
"""

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp


# Physical parameters (Goswami mu=2, beta=1, l=1 normalization)
M = 5.0          # leg mass
M_H = 10.0       # hip mass
L = 1.0          # leg length
A = 0.5          # foot-to-mass distance
B = 0.5          # hip-to-mass distance
G = 9.81         # gravity


def _get_matrices(q, dq, phi):
    """Mass, Coriolis, Gravity matrices for compass gait Lagrangian.

    Parameters
    ----------
    q : array-like (2,)
        (theta_ns, theta_s)
    dq : array-like (2,)
        (dtheta_ns, dtheta_s)
    phi : float
        Ground slope (radians)

    Returns
    -------
    M_mat : ndarray (2, 2)
        Mass matrix
    N_mat : ndarray (2, 2)
        Coriolis/centrifugal matrix
    G_vec : ndarray (2,)
        Gravity vector
    """
    th_ns, th_s = q
    dth_ns, dth_s = dq

    cos_diff = np.cos(th_s - th_ns)
    sin_diff = np.sin(th_s - th_ns)

    M_mat = np.array([
        [M * B**2,                    -M * L * B * cos_diff],
        [-M * L * B * cos_diff,       (M_H + M) * L**2 + M * A**2],
    ], dtype=float)

    N_mat = np.array([
        [0,                            M * L * B * sin_diff * dth_s],
        [-M * L * B * sin_diff * dth_ns, 0],
    ], dtype=float)

    G_vec = np.array([
        M * B * G * np.sin(th_ns),
        -(M_H * L + M * A + M * L) * G * np.sin(th_s),
    ], dtype=float)

    return M_mat, N_mat, G_vec


def _vector_field(t, state, phi):
    """Continuous vector field of compass gait in swing phase.

    Parameters
    ----------
    t : float
        Time (unused, for solve_ivp compatibility)
    state : array-like (4,)
        [theta_ns, theta_s, dtheta_ns, dtheta_s]
    phi : float
        Ground slope (radians)

    Returns
    -------
    dstate : ndarray (4,)
        Time derivative of state
    """
    q, dq = state[:2], state[2:]
    M_mat, N_mat, G_vec = _get_matrices(q, dq, phi)
    ddq = np.linalg.solve(M_mat, -(N_mat @ dq + G_vec))
    return np.array([dq[0], dq[1], ddq[0], ddq[1]], dtype=float)


def _reset_map(state, phi):
    """Impact map: leg role swap + angular momentum conservation.

    Applies Goswami impact equations (Eq. A.34-A.37) at heelstrike.

    Parameters
    ----------
    state : array-like (4,)
        [theta_ns, theta_s, dtheta_ns, dtheta_s] just before impact
    phi : float
        Ground slope (radians); used for guard check in caller, not here

    Returns
    -------
    state_plus : ndarray (4,)
        State just after impact
    """
    th_ns_m, th_s_m, dth_ns_m, dth_s_m = state

    # Angle swap: old stance becomes new non-stance, old non-stance becomes new stance
    th_ns_p = th_s_m
    th_s_p = th_ns_m

    # Velocity jump via angular momentum conservation
    alpha = (th_s_m - th_ns_m) / 2.0
    cos2a = np.cos(2 * alpha)

    Q_minus = np.array([
        [-M * A * B,    (M_H * L**2 + 2 * M * A * L) * cos2a - M * A * B],
        [0,             -M * A * B],
    ], dtype=float)

    Q_plus = np.array([
        [M * B * (B - L * cos2a),    M * L * (L - B * cos2a) + M_H * L**2 + M * A**2],
        [M * B**2,                   -M * B * L * cos2a],
    ], dtype=float)

    dq_minus = np.array([dth_ns_m, dth_s_m], dtype=float)
    dq_plus = np.linalg.solve(Q_plus, Q_minus @ dq_minus)

    return np.array([th_ns_p, th_s_p, dq_plus[0], dq_plus[1]], dtype=float)


@dataclass(frozen=True)
class CompassRegime:
    """Specification for a compass gait regime.

    Attributes
    ----------
    label : str
        Name of regime (e.g., "period1", "period2")
    phi_deg : float
        Ground slope in degrees
    phi : float
        Ground slope in radians
    expected_period : int or None
        Expected period-doubling level (1, 2, 4, 8, or None for chaos)
    ic : tuple of 4 floats
        Post-impact initial condition: (theta_ns, theta_s, dtheta_ns, dtheta_s)
    """
    label: str
    phi_deg: float
    phi: float
    expected_period: int | None
    ic: tuple


# Pre-computed return-map fixed points or chaos-set center, computed offline
phi_4_00 = np.radians(4.00)

COMPASS_REGIMES = {
    "period1": CompassRegime(
        label="period1",
        phi_deg=4.00,
        phi=phi_4_00,
        expected_period=1,
        ic=(-(phi_4_00 + 0.27), -(phi_4_00 - 0.27), -0.38, -1.09)
    ),
    "period2": CompassRegime(
        label="period2",
        phi_deg=4.75,
        phi=np.radians(4.75),
        expected_period=2,
        ic=(-0.381789947562, 0.215983668622, -0.153332476821, -1.153674078371)
    ),
    "period4": CompassRegime(
        label="period4",
        phi_deg=5.00,
        phi=np.radians(5.00),
        expected_period=4,
        ic=(-0.392348120036, 0.217815194837, -0.108006761939, -1.161444888863)
    ),
    "period8": CompassRegime(
        label="period8",
        phi_deg=5.02,
        phi=np.radians(5.02),
        expected_period=8,
        ic=(-0.394646896303, 0.219415839402, -0.097321131256, -1.162198996812)
    ),
    # Chaos IC is an on-attractor post-impact state harvested from a settled
    # chaotic rollout (the raw return-box midpoint falls within a few strides).
    "chaos": CompassRegime(
        label="chaos",
        phi_deg=5.20,
        phi=np.radians(5.20),
        expected_period=None,
        ic=(-0.411749943251, 0.230235701043, -0.040221385965, -1.171977694806)
    ),
}


@dataclass
class HybridTimeseries:
    """Timeseries with hybrid impact annotations.

    Attributes
    ----------
    t : ndarray (N,)
        Uniform time samples, 0 <= t <= t_span
    x : ndarray (N, 4)
        State samples [theta_ns, theta_s, dtheta_ns, dtheta_s]
    v : ndarray (N, 4)
        Unit-norm tangent vectors (normalized state derivatives)
    impact_times : ndarray (K,)
        Global time of each impact (in [0, t_span])
    jump_minus : ndarray (K, 4)
        State just before each impact
    jump_plus : ndarray (K, 4)
        State just after each impact (reset-map output)
    meta : dict
        Metadata: system, label, phi_deg, phi, expected_period, ic_used, attempt,
                  t_span, dt, burn_in_strides, n_strides_main, max_arc_time,
                  rtol, atol, max_step, n_samples
    """
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    impact_times: np.ndarray
    jump_minus: np.ndarray
    jump_plus: np.ndarray
    meta: dict


def generate_timeseries(regime, t_span=400.0, dt=0.02, burn_in_strides=80,
                       max_arc_time=5.0, rtol=1e-10, atol=1e-12,
                       max_step=0.01, max_retries=10, ic_sigma=0.002,
                       seed=20260805):
    """Generate a hybrid timeseries from compass gait dynamics.

    Algorithm:
      1. Retry loop (up to max_retries attempts):
         - Attempt 0: use regime.ic exactly
         - Retry j >= 1: perturb with Gaussian noise, seed = seed + j
      2. Burn-in: integrate arc-by-arc, detecting impacts, until burn_in_strides
         strides complete. Fall = RuntimeError.
      3. Main span: reset clock to t=0 at end of burn-in. Continue integrating
         arc-by-arc with dense output until accumulated time >= t_span. Fall = retry.
      4. Sampling: uniform grid t_k = k*dt, k=0..round(t_span/dt).
         Use each arc's dense interpolant for that arc's sample times.
         NEVER interpolate across impacts (positions may jump).
      5. Tangents: v[k] = normalized vector_field(x[k]).
      6. Metadata: system, label, phi_deg, phi, expected_period, ic_used, attempt,
                   t_span, dt, burn_in_strides, n_strides_main, max_arc_time,
                   rtol, atol, max_step, n_samples.

    Parameters
    ----------
    regime : CompassRegime
        Regime specification with label, phi, expected_period, ic
    t_span : float, default 400.0
        Duration of main integration (seconds)
    dt : float, default 0.02
        Sampling interval (seconds)
    burn_in_strides : int, default 80
        Number of impacts to discard before collecting main-span data
    max_arc_time : float, default 5.0
        Maximum integration time per arc; if no impact within this,
        walker has fallen. In practice ~0.7-0.8 s per stride.
    rtol, atol : float
        Relative and absolute tolerances for solve_ivp (default 1e-10, 1e-12)
    max_step : float, default 0.01
        Maximum step size for solve_ivp
    max_retries : int, default 10
        Number of IC perturbation attempts before raising RuntimeError
    ic_sigma : float, default 0.002
        Standard deviation of Gaussian perturbation to IC components
    seed : int, default 20260805
        Seed for IC perturbation RNG

    Returns
    -------
    HybridTimeseries
        Collected timeseries with impact annotations and metadata

    Raises
    ------
    RuntimeError
        If all attempts fail (walker falls during burn-in or main span)
    """
    phi = regime.phi

    # Create guard event closure over phi
    def guard_event(t, state):
        if state[0] - state[1] > 0.01:
            return (state[0] + state[1]) + 2 * phi
        return 1.0

    guard_event.terminal = True
    guard_event.direction = -1

    # Create vector field closure over phi
    def vector_field(t, state):
        return _vector_field(t, state, phi)

    # Retry loop
    for attempt in range(max_retries):
        # Choose initial condition
        if attempt == 0:
            ic_current = np.array(regime.ic, dtype=float)
        else:
            rng = np.random.default_rng(seed + attempt)
            ic_current = np.array(regime.ic, dtype=float) + rng.normal(0, ic_sigma, 4)

        ic_used = ic_current.copy()  # Record what we actually used

        try:
            x_current = ic_current.copy()

            # Burn-in phase: complete burn_in_strides impacts
            for _ in range(burn_in_strides):
                sol = solve_ivp(vector_field, [0, max_arc_time], x_current,
                               events=guard_event, dense_output=False,
                               max_step=max_step, rtol=rtol, atol=atol)

                if sol.status != 1:
                    # Guard never reached within max_arc_time
                    raise RuntimeError("Walker fell during burn-in")

                x_minus = sol.y[:, -1].copy()
                x_current = _reset_map(x_minus, phi)

            # Main integration phase with dense output and impact tracking
            impact_times = []
            jump_minus = []
            jump_plus = []
            arc_list = []  # (t_start_global, t_end_global, dense_interpolant)

            accumulated_t = 0.0
            while accumulated_t < t_span:
                sol = solve_ivp(vector_field, [0, max_arc_time], x_current,
                               events=guard_event, dense_output=True,
                               max_step=max_step, rtol=rtol, atol=atol)

                if sol.status != 1:
                    # Guard never reached; walker fell
                    raise RuntimeError("Walker fell during main span")

                arc_t_start = accumulated_t
                arc_t_end = accumulated_t + sol.t[-1]

                x_minus = sol.y[:, -1].copy()
                x_plus = _reset_map(x_minus, phi)

                impact_times.append(arc_t_end)
                jump_minus.append(x_minus.copy())
                jump_plus.append(x_plus.copy())
                arc_list.append((arc_t_start, arc_t_end, sol.sol))

                accumulated_t = arc_t_end
                x_current = x_plus

            # The final arc overshoots t_span; its impact lies beyond the
            # sampled window, so drop it from the bookkeeping (impact_times
            # stay within [0, t_span]). The arc itself is kept for sampling.
            n_in_span = sum(1 for it in impact_times if it <= t_span)
            impact_times = impact_times[:n_in_span]
            jump_minus = jump_minus[:n_in_span]
            jump_plus = jump_plus[:n_in_span]

            # Sampling phase: uniform time grid
            n_samples = int(np.round(t_span / dt)) + 1
            t_samples = np.linspace(0, t_span, n_samples)

            x_samples = np.zeros((n_samples, 4))
            v_samples = np.zeros((n_samples, 4))

            # Arcs and sample times are both increasing: advance an arc
            # pointer instead of rescanning the arc list per sample. Each
            # sample lives in [t0, t1) of its arc (final arc closed right).
            arc_idx = 0
            n_arcs = len(arc_list)
            for k, t_k in enumerate(t_samples):
                while arc_idx < n_arcs - 1 and t_k >= arc_list[arc_idx][1]:
                    arc_idx += 1

                arc_t0, arc_t1, interpolant = arc_list[arc_idx]
                if not (arc_t0 <= t_k <= arc_t1):
                    raise RuntimeError(
                        f"Sample time t={t_k:.6f} not covered by arcs "
                        f"[{arc_list[0][0]:.6f}, {arc_list[-1][1]:.6f}]"
                    )
                tau_local = t_k - arc_t0  # time within this arc, in [0, arc_duration]

                # Evaluate trajectory at local time
                x_k = interpolant(tau_local)
                x_samples[k] = x_k

                # Evaluate tangent: normalized vector field
                v_k = vector_field(tau_local, x_k)
                v_norm = np.linalg.norm(v_k)
                if v_norm > 1e-14:
                    v_k = v_k / v_norm
                else:
                    v_k = np.zeros(4)
                v_samples[k] = v_k

            # Build metadata
            meta = {
                "system": "compass_gait",
                "label": regime.label,
                "phi_deg": regime.phi_deg,
                "phi": float(regime.phi),
                "expected_period": regime.expected_period,
                "ic_used": tuple(ic_used),
                "attempt": attempt,
                "t_span": t_span,
                "dt": dt,
                "burn_in_strides": burn_in_strides,
                "n_strides_main": len(impact_times),
                "max_arc_time": max_arc_time,
                "rtol": rtol,
                "atol": atol,
                "max_step": max_step,
                "n_samples": len(t_samples),
            }

            return HybridTimeseries(
                t=t_samples,
                x=x_samples,
                v=v_samples,
                impact_times=np.array(impact_times, dtype=float),
                jump_minus=np.array(jump_minus, dtype=float),
                jump_plus=np.array(jump_plus, dtype=float),
                meta=meta
            )

        except RuntimeError:
            # This attempt failed; retry with perturbed IC
            continue

    # All attempts exhausted
    raise RuntimeError(
        f"All {max_retries} attempts failed: walker fell before completing "
        f"{burn_in_strides} strides in burn-in or accumulating {t_span}s in main span"
    )
