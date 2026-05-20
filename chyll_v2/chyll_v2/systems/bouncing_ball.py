"""Bouncing ball hybrid system on the mapping cylinder ``X' = X \\cup (G x [0,1])``.

Base state ``x = (h, v)`` where ``h`` is height above ground and ``v`` is
vertical velocity. Continuous dynamics is free-fall under gravity:
``dot{x} = (v, -g)``. The guard ``G = {h = 0, v <= 0}`` triggers an
elastic-with-restitution collision: ``r(h, v) = (h, -alpha v)``. The
mapping-cylinder relaxed space inserts a unit-time traversal at every
impact, with cylinder coordinate ``s in [0, 1]``.
"""
from __future__ import annotations

import numpy as np

from .base import BaseHybridSystem


class BouncingBall(BaseHybridSystem):
    name = "bouncing_ball"
    base_dim = 2

    def __init__(
        self,
        g: float = 1.0,
        alpha: float = 0.8,
        h_max: float = 1.0,
        v_max: float = 2.0,
    ):
        self.g = g
        self.alpha = alpha
        self.h_max = h_max
        self.v_max = v_max

    def base_vector_field(self, x: np.ndarray) -> np.ndarray:
        _, v = x
        return np.array([v, -self.g])

    def is_guard_hit(self, t: float, x: np.ndarray) -> float:
        # Event is ``h = 0`` while descending (``v < 0``). Returning a
        # positive sentinel when ``v >= 0`` avoids spurious triggers as the
        # ball climbs through h = 0 after a bounce (which the cylinder
        # construction handles by setting s back to 0, so the post-reset
        # state never quite re-crosses zero on the rising side).
        if x[1] < 0:
            return x[0]
        return 1.0

    @property
    def guard_direction(self) -> int:
        # Event crosses zero from positive (above ground) to negative;
        # SciPy's convention is +1 means increasing, -1 means decreasing.
        return -1

    def reset_map(self, g: np.ndarray) -> np.ndarray:
        # ``g = (0, v_minus)`` with ``v_minus <= 0`` at impact.
        # Reset flips and damps the velocity.
        _, v = g
        return np.array([0.0, -self.alpha * v])

    def sample_initial_condition(self, rng: np.random.Generator) -> np.ndarray:
        # Start in the base space at s = 0, somewhere above ground with a
        # velocity that lets the trajectory reach the guard eventually.
        # Uniform on a box that covers above-ground + descending states.
        h = rng.uniform(0.1, self.h_max)
        v = rng.uniform(-self.v_max, self.v_max)
        return np.array([h, v, 0.0])
