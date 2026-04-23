"""
1D bouncing ball simulation with ground-truth impact events.

State: [x, v] (height, velocity), gravity g.
Guard: x = 0, v < 0 (hits ground moving downward).
Reset: v -> -e * v (coefficient of restitution).

All jumps occur at x=0, so interpolation segments in R^2 are
vertical lines on the v-axis — total overlap. This is the
minimal example where the eta dimension is genuinely necessary.
"""

import numpy as np
from scipy.integrate import solve_ivp


# Default parameters
GRAVITY = 9.81
RESTITUTION = 0.8  # coefficient of restitution


def _vector_field(t, state):
    x, v = state
    return [v, -GRAVITY]


def _guard_event(t, state):
    """Guard: x = 0 with v < 0 (falling)."""
    if state[1] < -0.01:
        return state[0]
    return 1.0

_guard_event.terminal = True
_guard_event.direction = -1


def _reset_map(state, e=RESTITUTION):
    return np.array([0.0, -e * state[1]])


def simulate_bouncing_ball(x0, v0, n_impacts=10, e=RESTITUTION):
    """Simulate a bouncing ball trajectory.

    Parameters
    ----------
    x0, v0 : float
        Initial height and velocity.
    n_impacts : int
    e : float
        Coefficient of restitution.

    Returns
    -------
    segments : list of ndarray, each (K, 2)
    jump_pairs : list of (ndarray, ndarray)
    """
    state = np.array([x0, v0])
    segments = []
    jump_pairs = []

    for _ in range(n_impacts):
        sol = solve_ivp(_vector_field, [0, 50], state,
                        events=_guard_event, dense_output=True, max_step=0.01)
        seg = sol.y.T
        segments.append(seg)

        if sol.status != 1:
            break

        x_minus = sol.y[:, -1].copy()
        x_plus = _reset_map(x_minus, e)
        jump_pairs.append((x_minus, x_plus))
        state = x_plus

    return segments, jump_pairs


def concatenate_timeseries(segments):
    return np.concatenate(segments, axis=0)


def extract_jump_pairs(segments, jump_pairs):
    base_trajectory = np.concatenate(segments, axis=0)
    return base_trajectory, jump_pairs


def generate_diverse_dataset(n_trajectories=100, n_impacts_per=5,
                             x_range=(0.5, 5.0), v_range=(0.0, 8.0),
                             e=RESTITUTION):
    """Generate trajectories from diverse ICs.

    Samples initial height and upward velocity uniformly.
    """
    all_segments = []
    all_jump_pairs = []

    for _ in range(n_trajectories):
        x0 = np.random.uniform(*x_range)
        v0 = np.random.uniform(*v_range)

        segments, jump_pairs = simulate_bouncing_ball(
            x0, v0, n_impacts=n_impacts_per, e=e
        )
        if len(jump_pairs) >= 1:
            all_segments.extend(segments)
            all_jump_pairs.extend(jump_pairs)

    return all_segments, all_jump_pairs
