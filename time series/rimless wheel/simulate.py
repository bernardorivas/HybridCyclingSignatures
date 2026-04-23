"""
Rimless wheel simulation with ground-truth impact events.

Provides simulate_rimless_wheel() for a single trajectory and
generate_diverse_dataset() for broad state-space coverage.
"""

import numpy as np
from scipy.integrate import solve_ivp


# Default physical parameters
GAMMA = 0.2      # slope angle
ALPHA = 0.4      # half-angle between spokes
THETA_GUARD = ALPHA + GAMMA    # 0.6
THETA_RESET = 2 * GAMMA - THETA_GUARD  # -0.2
COS2A = np.cos(2 * ALPHA)


def simulate_rimless_wheel(theta0, omega0, n_impacts=10,
                           gamma=GAMMA, alpha=ALPHA):
    """Simulate a single rimless wheel trajectory.

    Returns
    -------
    segments : list of ndarray, each (K, 2)
        Continuous trajectory segments between impacts.
    jump_pairs : list of (ndarray, ndarray)
        Each entry is (x_minus, x_plus).
    """
    theta_guard = alpha + gamma
    theta_reset = 2 * gamma - theta_guard
    cos2a = np.cos(2 * alpha)

    def vector_field(t, state):
        theta, omega = state
        return [omega, np.sin(theta)]

    def guard_event(t, state):
        if state[1] > 0:
            return state[0] - theta_guard
        return 1.0
    guard_event.terminal = True
    guard_event.direction = 1

    def reset(state):
        return np.array([theta_reset, state[1] * cos2a])

    state = np.array([theta0, omega0])
    segments = []
    jump_pairs = []

    for _ in range(n_impacts):
        sol = solve_ivp(vector_field, [0, 50], state,
                        events=guard_event, dense_output=True, max_step=0.01)
        seg = sol.y.T  # (K, 2)
        segments.append(seg)

        if sol.status != 1:
            break  # guard was never reached

        x_minus = sol.y[:, -1].copy()
        x_plus = reset(x_minus)
        jump_pairs.append((x_minus, x_plus))
        state = x_plus

    return segments, jump_pairs


def extract_jump_pairs(segments, jump_pairs):
    """Return concatenated base trajectory and jump pairs."""
    base_trajectory = np.concatenate(segments, axis=0)
    return base_trajectory, jump_pairs


def concatenate_timeseries(segments):
    """Concatenate segments into a single raw time series.

    The resulting array includes the discontinuities: the last point
    of segment i (x^-) is followed by the first point of segment i+1 (x^+).
    """
    return np.concatenate(segments, axis=0)


def generate_diverse_dataset(n_trajectories=200, n_impacts_per=4,
                             theta_range=(-0.2, 0.6),
                             omega_range=(0.3, 2.0),
                             gamma=GAMMA, alpha=ALPHA):
    """Generate many short trajectories from diverse initial conditions.

    Samples ICs uniformly across the base state space so training data
    covers the full domain, not just the limit cycle neighborhood.

    Parameters
    ----------
    n_trajectories : int
        Number of independent trajectories to simulate.
    n_impacts_per : int
        Max impacts per trajectory (short to preserve IC diversity).
    theta_range : tuple
        (min, max) for initial theta sampling.
    omega_range : tuple
        (min, max) for initial omega sampling (positive only —
        the wheel must roll forward to reach the guard).
    gamma, alpha : float
        Physical parameters.

    Returns
    -------
    all_segments : list of ndarray, each (K, 2)
    all_jump_pairs : list of (ndarray, ndarray)
    """
    all_segments = []
    all_jump_pairs = []

    for _ in range(n_trajectories):
        th0 = np.random.uniform(*theta_range)
        om0 = np.random.uniform(*omega_range)

        segments, jump_pairs = simulate_rimless_wheel(
            th0, om0, n_impacts=n_impacts_per, gamma=gamma, alpha=alpha
        )
        all_segments.extend(segments)
        all_jump_pairs.extend(jump_pairs)

    return all_segments, all_jump_pairs
