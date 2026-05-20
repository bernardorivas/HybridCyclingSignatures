"""Data pipeline for the cylinder-augmented CHyLL v2 baseline.

Training operates on a single stream of supervised data: contiguous
length-``horizon`` slices of augmented states ``(x, s)`` drawn from the
tau-semiflow ``phi'``. Within each slice, the dataloader also returns masks
over edges that flag pairs crossing the gluing identification
``(g, 1) \\sim (r(g), 0)`` and pairs entering either end of the cylinder seam.

The index sets are read directly off the ``s`` coordinate of the trajectory.
No symbolic knowledge of the guard set ``G`` or the reset map ``r`` is
required at training time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .systems.base import BaseHybridSystem, Trajectory


# ----------------------------------------------------------------------
# Gluing-index identification (data-driven, no G or r)
# ----------------------------------------------------------------------


def identify_gluing_indices(
    trajectory: Trajectory,
    s_high: float = 0.9,
    s_low: float = 0.1,
) -> np.ndarray:
    """Return indices ``k`` such that the pair ``(x_k, x_{k+1})`` crosses
    the gluing identification ``(g, 1) ~ (r(g), 0)``.

    Detection is purely from the trajectory's ``s`` coordinate: ``s_k``
    must be near the cylinder top (``>= s_high``) and ``s_{k+1}`` must
    have returned to the base (``<= s_low``).
    """
    s = trajectory.states[:, -1]
    mask = (s[:-1] >= s_high) & (s[1:] <= s_low)
    return np.where(mask)[0]


def identify_gluing_bool(
    trajectory: Trajectory,
    s_high: float = 0.9,
    s_low: float = 0.1,
) -> np.ndarray:
    """Same as :func:`identify_gluing_indices` but returns a boolean mask
    of length ``len(trajectory) - 1`` (one entry per edge)."""
    s = trajectory.states[:, -1]
    return (s[:-1] >= s_high) & (s[1:] <= s_low)


def identify_entry_bool(
    trajectory: Trajectory,
    s_base_tol: float = 1e-8,
    s_positive_tol: float = 1e-8,
) -> np.ndarray:
    """Return a boolean edge mask for base-flow to cylinder-entry edges."""
    s = trajectory.states[:, -1]
    return (s[:-1] <= s_base_tol) & (s[1:] > s_positive_tol)


def identify_seam_bool(
    trajectory: Trajectory,
    *,
    s_high: float = 0.9,
    s_low: float = 0.1,
    s_base_tol: float = 1e-8,
    s_positive_tol: float = 1e-8,
) -> np.ndarray:
    """Return the union of cylinder-entry and gluing-exit seam masks."""
    return identify_entry_bool(
        trajectory,
        s_base_tol=s_base_tol,
        s_positive_tol=s_positive_tol,
    ) | identify_gluing_bool(trajectory, s_high=s_high, s_low=s_low)


# ----------------------------------------------------------------------
# Trajectory-slice dataset
# ----------------------------------------------------------------------


@dataclass
class SliceBatch:
    """Stacked batch of length-``horizon`` trajectory slices."""

    states: torch.Tensor       # (B, T, base_dim + 1)
    times: torch.Tensor        # (B, T)
    glue_mask: torch.Tensor    # (B, T - 1)  bool
    seam_mask: torch.Tensor    # (B, T - 1)  bool


class TrajectorySliceDataset(Dataset):
    def __init__(
        self,
        trajectories: List[Trajectory],
        horizon: int,
        s_high: float = 0.9,
        s_low: float = 0.1,
    ):
        if horizon < 2:
            raise ValueError("horizon must be >= 2")
        self.trajectories = trajectories
        self.horizon = horizon
        self._glue_bool: List[np.ndarray] = [
            identify_gluing_bool(t, s_high=s_high, s_low=s_low)
            for t in trajectories
        ]
        self._seam_bool: List[np.ndarray] = [
            identify_seam_bool(t, s_high=s_high, s_low=s_low)
            for t in trajectories
        ]
        self._max_starts = [len(t) - horizon + 1 for t in trajectories]
        self._cumulative = np.cumsum([0] + self._max_starts)

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, idx: int):
        traj_idx = int(np.searchsorted(self._cumulative, idx, side="right") - 1)
        start = idx - int(self._cumulative[traj_idx])
        end = start + self.horizon
        traj = self.trajectories[traj_idx]
        states = torch.from_numpy(traj.states[start:end]).float()
        times = torch.from_numpy(traj.times[start:end]).float()
        glue = self._glue_bool[traj_idx][start:end - 1]
        seam = self._seam_bool[traj_idx][start:end - 1]
        glue_mask = torch.from_numpy(glue.copy())
        seam_mask = torch.from_numpy(seam.copy())
        return states, times, glue_mask, seam_mask


def collate_slices(batch) -> SliceBatch:
    states = torch.stack([b[0] for b in batch], dim=0)
    times = torch.stack([b[1] for b in batch], dim=0)
    glue_mask = torch.stack([b[2] for b in batch], dim=0)
    seam_mask = torch.stack([b[3] for b in batch], dim=0)
    return SliceBatch(
        states=states,
        times=times,
        glue_mask=glue_mask,
        seam_mask=seam_mask,
    )


def make_slice_loader(
    trajectories: List[Trajectory],
    horizon: int,
    batch_size: int,
    num_workers: int = 0,
    s_high: float = 0.9,
    s_low: float = 0.1,
) -> DataLoader:
    ds = TrajectorySliceDataset(
        trajectories=trajectories,
        horizon=horizon,
        s_high=s_high,
        s_low=s_low,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_slices,
        drop_last=True,
    )


# ----------------------------------------------------------------------
# One-shot helper
# ----------------------------------------------------------------------


def generate_trajectories(
    system: BaseHybridSystem,
    n_trajectories: int,
    tau: float,
    trajectory_steps: int,
    seed: int = 0,
    sim_rtol: float = 1e-8,
    sim_atol: float = 1e-10,
) -> List[Trajectory]:
    return system.generate_dataset(
        n_trajectories=n_trajectories,
        tau=tau,
        n_steps=trajectory_steps,
        seed=seed,
        rtol=sim_rtol,
        atol=sim_atol,
    )
