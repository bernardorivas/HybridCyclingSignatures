"""Mapping-cylinder hybrid system simulator.

The hybrid suspension construction of Section 2 of the manuscript augments
the state with a cylinder coordinate ``s in [0, 1]``. The tau-semiflow
``phi'`` on ``X' = X \\cup (G x [0,1])`` produces a sequence of augmented
states ``(x_k, s_k)`` at uniform real-time intervals ``tau``. The encoder
later sees these augmented states directly.

Subclasses implement the base-space physics, the guard event function,
the reset map, an initial-condition sampler, and a guard sampler used to
build the symbolic gluing pairs for ``L_g`` and ``L_v``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Trajectory:
    """One simulated trajectory on the augmented space ``X'``.

    Attributes
    ----------
    states : (T, base_dim + 1)
        Augmented states ``(x, s)`` sampled at intervals of ``tau``.
    times : (T,)
        Cumulative real-time stamps (uniform spacing).
    """

    states: np.ndarray
    times: np.ndarray

    def __len__(self) -> int:
        return self.states.shape[0]


class BaseHybridSystem:
    """Abstract base for hybrid systems on the mapping cylinder."""

    name: str
    base_dim: int

    @property
    def state_dim(self) -> int:
        """Dimension of the augmented state ``(x, s)`` fed to the encoder."""
        return self.base_dim + 1

    # --- physics hooks (override) ---------------------------------------

    def base_vector_field(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def is_guard_hit(self, t: float, x: np.ndarray) -> float:
        raise NotImplementedError

    def reset_map(self, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_initial_condition(self, rng: np.random.Generator) -> np.ndarray:
        """Sample an *augmented* initial condition ``(x_0, s_0)``."""
        raise NotImplementedError

    @property
    def guard_direction(self) -> int:
        raise NotImplementedError

    # --- semiflow on X' --------------------------------------------------

    def _simulate_base_to_guard(
        self, x0: np.ndarray, t_span: Tuple[float, float], rtol: float, atol: float
    ):
        ev = lambda t, y: self.is_guard_hit(t, y)
        ev.terminal = True
        ev.direction = self.guard_direction
        return solve_ivp(
            lambda t, y: self.base_vector_field(y),
            t_span,
            x0,
            method="RK45",
            events=ev,
            rtol=rtol,
            atol=atol,
            max_step=t_span[1] - t_span[0],
        )

    def tau_step(
        self,
        state_aug: np.ndarray,
        tau: float,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> np.ndarray:
        """One step of the tau-semiflow on X'.

        Mirrors the construction in ``src/system.py``: cylinder traversal
        when ``s > 0`` (or on the guard), base-space integration otherwise,
        with an event-driven transition between the two regimes.
        """
        d = self.base_dim
        x = state_aug[:d].copy()
        s = float(state_aug[d])

        on_guard = (
            abs(self.is_guard_hit(0.0, x)) < 1e-6 and s == 0.0
        )
        in_cylinder = s > 0.0 or on_guard

        if in_cylinder:
            time_to_glue = 1.0 - s
            if tau >= time_to_glue:
                rem = tau - time_to_glue
                x_post = self.reset_map(x)
                if rem > 0:
                    sol = self._simulate_base_to_guard(
                        x_post, (0.0, rem), rtol=rtol, atol=atol
                    )
                    if sol.status == 1:
                        # Hit the guard again within ``rem``: re-enter cylinder.
                        t_hit = float(sol.t[-1])
                        x_hit = sol.y[:, -1]
                        next_s = rem - t_hit
                        # If the second cylinder traversal also finishes inside
                        # ``rem``, recurse one more time. In practice this is
                        # extremely unlikely with reasonable ``tau``.
                        if next_s >= 1.0:
                            x_post2 = self.reset_map(x_hit)
                            out = np.zeros(d + 1)
                            out[:d] = x_post2
                            out[d] = 0.0
                            return out
                        out = np.zeros(d + 1)
                        out[:d] = x_hit
                        out[d] = next_s
                        return out
                    out = np.zeros(d + 1)
                    out[:d] = sol.y[:, -1]
                    out[d] = 0.0
                    return out
                out = np.zeros(d + 1)
                out[:d] = x_post
                out[d] = 0.0
                return out
            out = np.zeros(d + 1)
            out[:d] = x
            out[d] = s + tau
            return out

        # Base-space phase.
        sol = self._simulate_base_to_guard(x, (0.0, tau), rtol=rtol, atol=atol)
        if sol.status == 1:
            t_hit = float(sol.t[-1])
            rem = tau - t_hit
            x_hit = sol.y[:, -1]
            if rem >= 1.0:
                # Full cylinder traversal plus remaining base flow.
                x_post = self.reset_map(x_hit)
                rem2 = rem - 1.0
                if rem2 > 0:
                    sol2 = self._simulate_base_to_guard(
                        x_post, (0.0, rem2), rtol=rtol, atol=atol
                    )
                    out = np.zeros(d + 1)
                    out[:d] = sol2.y[:, -1]
                    out[d] = 0.0
                    return out
                out = np.zeros(d + 1)
                out[:d] = x_post
                out[d] = 0.0
                return out
            out = np.zeros(d + 1)
            out[:d] = x_hit
            out[d] = rem
            return out
        out = np.zeros(d + 1)
        out[:d] = sol.y[:, -1]
        out[d] = 0.0
        return out

    def generate_trajectory(
        self,
        x0_aug: np.ndarray,
        tau: float,
        n_steps: int,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> Trajectory:
        d = self.base_dim
        states = np.empty((n_steps, d + 1), dtype=np.float64)
        times = np.empty(n_steps, dtype=np.float64)
        states[0] = x0_aug
        times[0] = 0.0
        for k in range(1, n_steps):
            states[k] = self.tau_step(states[k - 1], tau, rtol=rtol, atol=atol)
            times[k] = k * tau
        return Trajectory(states=states, times=times)

    # --- dataset helpers ------------------------------------------------

    def generate_dataset(
        self,
        n_trajectories: int,
        tau: float,
        n_steps: int,
        seed: int = 0,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> List[Trajectory]:
        rng = np.random.default_rng(seed)
        return [
            self.generate_trajectory(
                self.sample_initial_condition(rng), tau=tau, n_steps=n_steps,
                rtol=rtol, atol=atol,
            )
            for _ in range(n_trajectories)
        ]

