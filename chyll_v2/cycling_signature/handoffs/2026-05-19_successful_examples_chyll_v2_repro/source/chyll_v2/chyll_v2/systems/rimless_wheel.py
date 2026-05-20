"""Rimless wheel hybrid system on the mapping cylinder ``X' = X \\cup (G x [0,1])``.

Base state ``x = (theta, omega)``; augmented state ``(x, s)`` with
``s in [0, 1]``. Constants match ``src/config.py`` (``alpha = 0.4``,
``gamma = 0.2``).
"""
from __future__ import annotations

import math

import numpy as np

from .base import BaseHybridSystem


class RimlessWheel(BaseHybridSystem):
    name = "rimless_wheel"
    base_dim = 2

    def __init__(
        self,
        alpha: float = 0.4,
        gamma: float = 0.2,
        theta_lo: float = -0.2,
        omega_lo: float = -1.0,
        omega_hi: float = 2.0,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.theta_guard = alpha + gamma                  # 0.6
        self.theta_reset = 2 * gamma - self.theta_guard   # -0.2
        self.omega_restitution = math.cos(2 * alpha)      # cos(0.8) ~ 0.697
        self.theta_lo = theta_lo
        self.theta_hi = self.theta_guard
        self.omega_lo = omega_lo
        self.omega_hi = omega_hi

    def base_vector_field(self, x: np.ndarray) -> np.ndarray:
        theta, omega = x
        return np.array([omega, np.sin(theta)])

    def is_guard_hit(self, t: float, x: np.ndarray) -> float:
        if x[1] > 0:
            return x[0] - self.theta_guard
        return 1.0

    @property
    def guard_direction(self) -> int:
        return +1

    def reset_map(self, g: np.ndarray) -> np.ndarray:
        _, omega = g
        return np.array([self.theta_reset, omega * self.omega_restitution])

    def sample_initial_condition(self, rng: np.random.Generator) -> np.ndarray:
        # Start in base space (s = 0) with forward angular velocity so the
        # trajectory eventually reaches the guard.
        theta = rng.uniform(self.theta_lo, self.theta_hi)
        omega = rng.uniform(max(0.2, self.omega_lo), self.omega_hi)
        return np.array([theta, omega, 0.0])
