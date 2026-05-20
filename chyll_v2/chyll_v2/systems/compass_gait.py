"""Compass-gait biped hybrid system on the mapping cylinder ``X' = X \\cup (G x [0,1])``.

Base state ``x = (theta_ns, theta_s, dtheta_ns, dtheta_s)``, ``ns`` is the
non-stance (swing) leg and ``s`` is the stance leg. Continuous dynamics is
the Lagrangian biped on a slope of angle ``phi`` under gravity:
``M(q) ddot{q} + N(q, dq) dq + G(q) = 0``. The guard surface is
``G = {theta_ns + theta_s = -2 phi, theta_ns > theta_s}`` (swing leg
ahead of stance). Foot-strike applies an angular-momentum-conserving
impact map with leg-role swap (Goswami Eq. 2.4, 2.10, A.34-A.37). The
mapping-cylinder relaxed space inserts a unit-time traversal at every
impact, cylinder coordinate ``s in [0, 1]``.

Canonical constants (Goswami, matching ``src/config.py``):
``phi = 0.07``, ``m = 5``, ``m_H = 10``, ``l = 1``, ``a = b = 0.5``,
``g = 9.81``.

**IC sampling note:** The compass gait has a narrow basin of attraction
around the stable limit cycle. Uniform sampling on
``[-0.5, 0.5] x [-2, 2]`` mostly produces divergent trajectories that
ruin the training data (verified empirically). Default IC mode is
``"limit_cycle"``: draw ``LIMIT_CYCLE_IC + N(0, sigma)`` with the
canonical fixed point and ``sigma = 0.03`` (Bernardo's value). The
generated trajectories are screened for divergence and ICs are
redrawn up to ``ic_max_retries`` times to keep the dataset clean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseHybridSystem, Trajectory
from .compass_gait_slope_configs import (
    CompassGaitSlopeConfig,
    get_compass_slope_config,
)


# Post-reset fixed point on the canonical limit cycle at phi = 0.07,
# matching ``time series/compass gait/simulate.py`` (Bernardo's value):
# ``[-(phi + 0.27), -(phi - 0.27), -0.38, -1.09]``.
def _limit_cycle_ic(phi: float) -> np.ndarray:
    return np.array(
        [-(phi + 0.27), -(phi - 0.27), -0.38, -1.09], dtype=np.float64
    )


class CompassGait(BaseHybridSystem):
    name = "compass_gait"
    base_dim = 4

    def __init__(
        self,
        phi: float = 0.07,
        slope_config: str | CompassGaitSlopeConfig | None = None,
        m: float = 5.0,
        m_H: float = 10.0,
        l: float = 1.0,
        a: float = 0.5,
        b: float = 0.5,
        g: float = 9.81,
        # IC sampling -----------------------------------------------------
        ic_mode: str = "limit_cycle",
        ic_sigma: float = 0.03,
        ic_max_theta: float = 1.5,
        ic_max_retries: int = 10,
        # Legacy uniform-box bounds (only used when ic_mode == "uniform_box").
        # Kept for reproducibility of the original 2026-05-14 chyll_v2
        # compass run that exhibited the divergence problem.
        theta_lo: float = -0.5,
        theta_hi: float = 0.5,
        omega_lo: float = -2.0,
        omega_hi: float = 2.0,
    ):
        if slope_config is not None:
            if isinstance(slope_config, str):
                slope_config = get_compass_slope_config(slope_config)
            phi = slope_config.phi
            ic_sigma = slope_config.ic_sigma
            ic_mode = "slope_config"

        if ic_mode not in ("limit_cycle", "uniform_box", "slope_config"):
            raise ValueError(
                "ic_mode must be 'limit_cycle', 'uniform_box', or "
                f"'slope_config', got {ic_mode!r}"
            )
        self.phi = phi
        self.slope_config = slope_config
        self.m = m
        self.m_H = m_H
        self.l = l
        self.a = a
        self.b = b
        self.g = g
        self.ic_mode = ic_mode
        self.ic_sigma = ic_sigma
        self.ic_max_theta = ic_max_theta
        self.ic_max_retries = ic_max_retries
        self.theta_lo = theta_lo
        self.theta_hi = theta_hi
        self.omega_lo = omega_lo
        self.omega_hi = omega_hi
        self._sampling_cloud = None
        if self.slope_config is not None and self.slope_config.sampling_cloud_path:
            cloud_path = Path(self.slope_config.sampling_cloud_path)
            self._sampling_cloud = np.load(cloud_path)
            if self._sampling_cloud.ndim != 2 or self._sampling_cloud.shape[1] != 4:
                raise ValueError(
                    f"sampling cloud must have shape (N, 4), got "
                    f"{self._sampling_cloud.shape}"
                )

    # --- dynamics ------------------------------------------------------

    def _get_matrices(
        self, q: np.ndarray, dq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        th_ns, th_s = q
        dth_ns, dth_s = dq
        m, m_H, a, b, l, g = self.m, self.m_H, self.a, self.b, self.l, self.g

        M = np.zeros((2, 2))
        M[0, 0] = m * b**2
        M[0, 1] = -m * l * b * np.cos(th_s - th_ns)
        M[1, 0] = -m * l * b * np.cos(th_s - th_ns)
        M[1, 1] = (m_H + m) * l**2 + m * a**2

        N = np.zeros((2, 2))
        N[0, 1] = m * l * b * np.sin(th_s - th_ns) * dth_s
        N[1, 0] = -m * l * b * np.sin(th_s - th_ns) * dth_ns

        G_vec = np.zeros(2)
        G_vec[0] = m * b * g * np.sin(th_ns)
        G_vec[1] = -(m_H * l + m * a + m * l) * g * np.sin(th_s)

        return M, N, G_vec

    def base_vector_field(self, x: np.ndarray) -> np.ndarray:
        q = x[:2]
        dq = x[2:]
        M, N, G_vec = self._get_matrices(q, dq)
        ddq = np.linalg.solve(M, -(N @ dq + G_vec))
        return np.array([dq[0], dq[1], ddq[0], ddq[1]])

    def is_guard_hit(self, t: float, x: np.ndarray) -> float:
        # Guard surface: theta_ns + theta_s + 2 phi = 0, with the swing leg
        # ahead of the stance leg by at least 0.01 rad. The sentinel +1 in
        # the early-step regime prevents a spurious trigger right after
        # impact, when the legs are nearly co-located.
        if x[0] - x[1] > 0.01:
            return (x[0] + x[1]) + 2 * self.phi
        return 1.0

    @property
    def guard_direction(self) -> int:
        return -1

    def reset_map(self, g: np.ndarray) -> np.ndarray:
        """Foot-strike impact map (Goswami)."""
        th_ns_minus, th_s_minus, dth_ns_minus, dth_s_minus = g
        m, m_H, a, b, l = self.m, self.m_H, self.a, self.b, self.l

        alpha = (th_s_minus - th_ns_minus) / 2.0
        cos2a = np.cos(2 * alpha)

        th_ns_plus = th_s_minus
        th_s_plus = th_ns_minus

        Q_minus = np.zeros((2, 2))
        Q_minus[0, 0] = -m * a * b
        Q_minus[0, 1] = (m_H * l**2 + 2 * m * a * l) * cos2a - m * a * b
        Q_minus[1, 0] = 0.0
        Q_minus[1, 1] = -m * a * b

        Q_plus = np.zeros((2, 2))
        Q_plus[0, 0] = m * b * (b - l * cos2a)
        Q_plus[0, 1] = m * l * (l - b * cos2a) + m_H * l**2 + m * a**2
        Q_plus[1, 0] = m * b**2
        Q_plus[1, 1] = -m * b * l * cos2a

        dq_minus = np.array([dth_ns_minus, dth_s_minus])
        dq_plus = np.linalg.solve(Q_plus, Q_minus @ dq_minus)

        return np.array([th_ns_plus, th_s_plus, dq_plus[0], dq_plus[1]])

    # --- IC sampling ----------------------------------------------------

    def sample_initial_condition(self, rng: np.random.Generator) -> np.ndarray:
        if self.ic_mode == "limit_cycle":
            pert = rng.normal(0.0, self.ic_sigma, size=4)
            return np.concatenate([_limit_cycle_ic(self.phi) + pert, [0.0]])
        if self.ic_mode == "slope_config":
            if self.slope_config is None:
                raise ValueError("slope_config IC mode requires a slope_config")
            if self.slope_config.fixed_points:
                idx = rng.integers(0, len(self.slope_config.fixed_points))
                x = np.array(self.slope_config.fixed_points[idx], dtype=np.float64)
                if self.ic_sigma > 0.0:
                    x = x + rng.normal(0.0, self.ic_sigma, size=4)
            elif self._sampling_cloud is not None:
                idx = rng.integers(0, self._sampling_cloud.shape[0])
                x = self._sampling_cloud[idx].astype(np.float64, copy=True)
                if self.ic_sigma > 0.0:
                    x = x + rng.normal(0.0, self.ic_sigma, size=4)
            elif self.slope_config.sampling_box is not None:
                lo = np.array(self.slope_config.sampling_box[0], dtype=np.float64)
                hi = np.array(self.slope_config.sampling_box[1], dtype=np.float64)
                x = rng.uniform(lo, hi)
            else:
                raise ValueError(
                    f"slope config {self.slope_config.label!r} has no IC support"
                )
            if self.slope_config.project_to_post_impact_guard:
                x[1] = -2.0 * self.phi - x[0]
            return np.concatenate([x, [0.0]])
        # Legacy uniform-box mode.
        q = rng.uniform(self.theta_lo, self.theta_hi, size=2)
        dq = rng.uniform(self.omega_lo, self.omega_hi, size=2)
        return np.array([q[0], q[1], dq[0], dq[1], 0.0])

    def _trajectory_is_bounded(self, traj: Trajectory) -> bool:
        return bool(np.max(np.abs(traj.states[:, :2])) <= self.ic_max_theta)

    def generate_dataset(
        self,
        n_trajectories: int,
        tau: float,
        n_steps: int,
        seed: int = 0,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ):
        """Generate trajectories, retrying ICs that yield divergent walks.

        A compass-gait IC is "divergent" if any ``|theta|`` along the
        simulated trajectory exceeds ``ic_max_theta`` (default 1.5 rad).
        In ``limit_cycle`` mode this catches the ~17% of canonical-sigma
        perturbations that escape the basin. In ``uniform_box`` mode the
        screening still applies and most ICs will be discarded, so this
        function will be slow there.
        """
        rng = np.random.default_rng(seed)
        out = []
        n_attempts = 0
        n_discarded = 0
        for _ in range(n_trajectories):
            for retry in range(self.ic_max_retries + 1):
                n_attempts += 1
                ic = self.sample_initial_condition(rng)
                traj = self.generate_trajectory(
                    ic, tau=tau, n_steps=n_steps, rtol=rtol, atol=atol,
                )
                if self._trajectory_is_bounded(traj):
                    out.append(traj)
                    break
                n_discarded += 1
            else:
                # Fell out of the for-else without break: keep the last
                # (divergent) trajectory rather than fail the whole run.
                # In practice this only triggers if max_retries is too low.
                out.append(traj)
        print(
            f"[compass_gait] generated {len(out)} trajectories in {n_attempts} "
            f"attempts (discarded {n_discarded} divergent)"
        )
        return out
