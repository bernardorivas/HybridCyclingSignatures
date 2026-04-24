"""
Compass gait simulation with ground-truth impact events.

Self-contained: reimplements the Lagrangian dynamics, guard, and impact map
from src/system.py so this folder has no dependency on src/.
"""

import numpy as np
from scipy.integrate import solve_ivp


# Default physical parameters (matches src/config.py)
PHI = 0.07       # ground slope (rad)
M = 5.0          # leg mass
M_H = 10.0       # hip mass
L = 1.0          # leg length
A = 0.5          # foot-to-mass distance
B = 0.5          # hip-to-mass distance
G = 9.81         # gravity


def _get_matrices(q, dq):
    """Mass, Coriolis, Gravity matrices for the compass gait Lagrangian."""
    th_ns, th_s = q
    dth_ns, dth_s = dq

    cos_diff = np.cos(th_s - th_ns)
    sin_diff = np.sin(th_s - th_ns)

    M_mat = np.array([
        [M * B**2,                    -M * L * B * cos_diff],
        [-M * L * B * cos_diff,       (M_H + M) * L**2 + M * A**2],
    ])

    N_mat = np.array([
        [0,                            M * L * B * sin_diff * dth_s],
        [-M * L * B * sin_diff * dth_ns, 0],
    ])

    G_vec = np.array([
        M * B * G * np.sin(th_ns),
        -(M_H * L + M * A + M * L) * G * np.sin(th_s),
    ])

    return M_mat, N_mat, G_vec


def _vector_field(t, state):
    q, dq = state[:2], state[2:]
    M_mat, N_mat, G_vec = _get_matrices(q, dq)
    ddq = np.linalg.solve(M_mat, -(N_mat @ dq + G_vec))
    return [dq[0], dq[1], ddq[0], ddq[1]]


def state_velocity(state):
    """Return the continuous-time compass vector field evaluated at ``state``."""
    return np.asarray(_vector_field(0.0, np.asarray(state, dtype=float)), dtype=float)


def _guard_event(t, state):
    """Guard: theta_ns + theta_s + 2*phi = 0, with theta_ns > theta_s."""
    if state[0] - state[1] > 0.01:
        return (state[0] + state[1]) + 2 * PHI
    return 1.0

_guard_event.terminal = True
_guard_event.direction = -1


def _reset_map(state):
    """Impact map: leg role swap + angular momentum conservation."""
    th_ns_m, th_s_m, dth_ns_m, dth_s_m = state

    # Angle swap
    th_ns_p = th_s_m
    th_s_p = th_ns_m

    # Velocity jump (Goswami Eq. A.34-A.37)
    alpha = (th_s_m - th_ns_m) / 2.0
    cos2a = np.cos(2 * alpha)

    Q_minus = np.array([
        [-M * A * B,    (M_H * L**2 + 2 * M * A * L) * cos2a - M * A * B],
        [0,             -M * A * B],
    ])

    Q_plus = np.array([
        [M * B * (B - L * cos2a),    M * L * (L - B * cos2a) + M_H * L**2 + M * A**2],
        [M * B**2,                   -M * B * L * cos2a],
    ])

    dq_minus = np.array([dth_ns_m, dth_s_m])
    dq_plus = np.linalg.solve(Q_plus, Q_minus @ dq_minus)

    return np.array([th_ns_p, th_s_p, dq_plus[0], dq_plus[1]])


def simulate_compass_gait(state0, n_impacts=10, t_max=20.0):
    """Simulate a compass gait trajectory.

    Parameters
    ----------
    state0 : array-like (4,)
        Initial [theta_ns, theta_s, dtheta_ns, dtheta_s].
    n_impacts : int
        Number of foot-strike events to simulate.
    t_max : float
        Max integration time per arc (safety cutoff).

    Returns
    -------
    segments : list of ndarray, each (K, 4)
        Continuous trajectory arcs between impacts.
    jump_pairs : list of (ndarray, ndarray)
        (x_minus, x_plus) at each impact.
    """
    state = np.array(state0, dtype=float)
    segments = []
    jump_pairs = []

    for _ in range(n_impacts):
        sol = solve_ivp(_vector_field, [0, t_max], state,
                        events=_guard_event, dense_output=True,
                        max_step=0.005, rtol=1e-10, atol=1e-12)
        seg = sol.y.T  # (K, 4)
        segments.append(seg)

        if sol.status != 1:
            break  # guard never reached

        x_minus = sol.y[:, -1].copy()
        x_plus = _reset_map(x_minus)
        jump_pairs.append((x_minus, x_plus))
        state = x_plus

    return segments, jump_pairs


def concatenate_timeseries(segments):
    """Concatenate segments into a single raw time series."""
    return np.concatenate(segments, axis=0)


def extract_jump_pairs(segments, jump_pairs):
    """Return concatenated base trajectory and jump pairs."""
    base_trajectory = np.concatenate(segments, axis=0)
    return base_trajectory, jump_pairs


# Known initial conditions near the limit cycle (from Bernardo's scripts)
LIMIT_CYCLE_IC = np.array([-(PHI + 0.27), -(PHI - 0.27), -0.38, -1.09])


def generate_diverse_dataset(n_trajectories=100, n_impacts_per=3):
    """Generate trajectories from perturbed limit-cycle ICs.

    The compass gait has a narrow basin of attraction, so we perturb
    around the known limit cycle rather than sampling uniformly.

    Parameters
    ----------
    n_trajectories : int
    n_impacts_per : int

    Returns
    -------
    all_segments : list of ndarray, each (K, 4)
    all_jump_pairs : list of (ndarray, ndarray)
    """
    all_segments = []
    all_jump_pairs = []
    n_success = 0

    for _ in range(n_trajectories * 3):  # oversample to compensate for failures
        if n_success >= n_trajectories:
            break

        # Perturb around limit cycle
        perturbation = np.random.normal(0, 0.03, size=4)
        ic = LIMIT_CYCLE_IC + perturbation

        segments, jump_pairs = simulate_compass_gait(
            ic, n_impacts=n_impacts_per, t_max=10.0
        )

        # Only keep trajectories that actually produced impacts
        if len(jump_pairs) >= 1:
            all_segments.extend(segments)
            all_jump_pairs.extend(jump_pairs)
            n_success += 1

    return all_segments, all_jump_pairs
