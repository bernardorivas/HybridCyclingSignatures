"""Latent Neural ODE rollout via torchdiffeq.

Sangli et al. integrate the latent flow ``dz/dt = V_theta(z)`` with the
Neural ODE machinery of Chen et al. (2018), using the adjoint method
for memory-efficient backprop. We expose both ``odeint`` (autograd
through the solver, exact gradients) and ``odeint_adjoint`` (constant
memory) via a config flag.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from .config import CHyLLv2Config


class _VFieldWrapper(nn.Module):
    """torchdiffeq expects ``forward(t, z) -> dz/dt``; our V_theta is
    autonomous, so ``t`` is ignored."""

    def __init__(self, vfield: nn.Module):
        super().__init__()
        self.vfield = vfield

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.vfield(z)


def make_ode_func(vfield: nn.Module) -> nn.Module:
    return _VFieldWrapper(vfield)


def rollout(
    ode_func: nn.Module,
    z0: torch.Tensor,
    times: torch.Tensor,
    cfg: CHyLLv2Config,
) -> torch.Tensor:
    """Integrate the latent flow at the requested ``times``.

    Parameters
    ----------
    ode_func : nn.Module
        Output of ``make_ode_func``.
    z0 : (B, latent_dim)
        Initial latent state for each trajectory in the batch.
    times : (T,)
        Real-time stamps at which to return the integrated latent.
        Must be monotonically increasing and start at the same value
        for all trajectories.

    Returns
    -------
    z : (T, B, latent_dim)
        Latent trajectory; ``z[0] == z0`` up to solver tolerance.
    """
    method = cfg.ode_method
    fixed_step_methods = {"euler", "midpoint", "rk4", "explicit_adams"}

    solver = odeint_adjoint if cfg.ode_use_adjoint else odeint

    kwargs = {}
    if method in fixed_step_methods:
        kwargs["options"] = {"step_size": cfg.ode_step_size}
    else:
        kwargs["rtol"] = cfg.ode_rtol
        kwargs["atol"] = cfg.ode_atol

    z = solver(ode_func, z0, times, method=method, **kwargs)
    return z


def latent_velocity(vfield: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """V_theta(z) without going through the ODE solver."""
    return vfield(z)
