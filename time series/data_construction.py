"""
Generic data construction for data-driven suspension representation.

Pipeline stage:
  time series -> jump detection -> augmented suspension dataset

System-specific simulation lives in subfolders (e.g. rimless wheel/).
This module provides the system-agnostic pieces.
"""

import numpy as np


def detect_jumps(timeseries, threshold):
    """Detect jump pairs from a raw time series via finite differences.

    Parameters
    ----------
    timeseries : ndarray (T, state_dim)
        Concatenated time series (may contain discontinuities).
    threshold : float
        Minimum L2 norm of consecutive difference to count as a jump.

    Returns
    -------
    base_trajectory : ndarray (T - n_jumps, state_dim)
        Time series with jump-destination points removed
        (each jump contributes x^- to base, x^+ starts the next arc).
    jump_pairs : list of (ndarray, ndarray)
        Detected (x_minus, x_plus) pairs.
    """
    diffs = np.linalg.norm(np.diff(timeseries, axis=0), axis=1)  # (T-1,)
    jump_idx = np.where(diffs > threshold)[0]  # indices k where jump occurs

    jump_pairs = []
    for k in jump_idx:
        jump_pairs.append((timeseries[k].copy(), timeseries[k + 1].copy()))

    # Base trajectory: all points that are not the post-jump duplicate
    # Keep all points except the ones right after a jump (they'll be
    # reached via the bridge landing instead)
    mask = np.ones(len(timeseries), dtype=bool)
    for k in jump_idx:
        mask[k + 1] = False
    base_trajectory = timeseries[mask]

    return base_trajectory, jump_pairs


def build_augmented_dataset(base_trajectory, jump_pairs, n_bridge_samples=20):
    """Construct the augmented suspension dataset D_base U D_bridge.

    Parameters
    ----------
    base_trajectory : ndarray (N, state_dim)
        Original trajectory samples (smooth arcs concatenated).
    jump_pairs : list of (x_minus, x_plus)
        Detected (or ground-truth) jump pairs.
    n_bridge_samples : int
        Number of interior samples per bridge (s uniformly in (0,1),
        plus the two endpoints s=0 and s=1).

    Returns
    -------
    dataset : dict with keys
        'base_points'  : ndarray (N, state_dim+1)  -- (x, s=0) for each base sample
        'bridge_points': ndarray (M, state_dim+1)  -- (x_minus, s) for each bridge sample
        'bridge_ids'   : ndarray (M,) int           -- which jump pair each bridge sample belongs to
        'jump_pairs'   : list of (x_minus, x_plus)
    """
    state_dim = base_trajectory.shape[1]

    # Base data: append s=0
    base_points = np.hstack([
        base_trajectory,
        np.zeros((len(base_trajectory), 1))
    ])

    # Bridge data
    bridge_points_list = []
    bridge_ids_list = []

    # s values: interior of [0,1] plus endpoints
    s_values = np.linspace(0, 1, n_bridge_samples + 2)  # includes 0 and 1

    for j, (x_minus, x_plus) in enumerate(jump_pairs):
        for s in s_values:
            # Hold base state at x^- throughout the cylinder; do NOT interpolate.
            # The encoder will learn the geometric bridge later.
            point = np.zeros(state_dim + 1)
            point[:state_dim] = x_minus
            point[state_dim] = s
            bridge_points_list.append(point)
            bridge_ids_list.append(j)

    bridge_points = np.array(bridge_points_list) if bridge_points_list else np.empty((0, state_dim + 1))
    bridge_ids = np.array(bridge_ids_list, dtype=int) if bridge_ids_list else np.empty(0, dtype=int)

    return {
        'base_points': base_points,
        'bridge_points': bridge_points,
        'bridge_ids': bridge_ids,
        'jump_pairs': jump_pairs,
    }
