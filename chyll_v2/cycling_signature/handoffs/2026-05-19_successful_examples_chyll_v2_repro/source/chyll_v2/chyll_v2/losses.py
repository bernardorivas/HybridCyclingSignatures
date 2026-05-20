"""Five-term composite loss (Sangli et al. ICML 2026, eqs. 4-8), adapted
to mapping-cylinder data and made fully data-driven.

- ``L_x``: reconstruction MSE between ``D(z_predicted)`` and the augmented
  state ``(x, s)`` along a rollout.
- ``L_z``: latent-consistency MSE between the integrated ``z`` and the
  encoded ``z`` along the rollout.
- ``L_g``: gluing MSE between consecutive encoded points across a
  detected gluing transition in the trajectory.
- ``L_v``: seam-velocity MSE between the pre- and post-seam latent
  finite-difference velocities at both cylinder entry and gluing exit
  seams.
- ``L_c``: per-coordinate variance floor.

No symbolic knowledge of the guard set ``G`` or reset map ``r`` is used:
``L_g`` consumes the data-driven ``glue_mask`` produced by
:func:`chyll_v2.data.identify_gluing_bool`; ``L_v`` consumes the union seam
mask produced by :func:`chyll_v2.data.identify_seam_bool`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import CHyLLv2Config


@dataclass
class LossOutputs:
    total: torch.Tensor
    L_x: torch.Tensor
    L_z: torch.Tensor
    L_g: torch.Tensor
    L_v: torch.Tensor
    L_c: torch.Tensor


# ----------------------------------------------------------------------
# Individual term helpers
# ----------------------------------------------------------------------


def _masked_mse_over_vectors(
    diff: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean of ``||diff_k||^2`` over the entries flagged by ``mask``.

    ``diff`` has shape ``(B, K, d)``; ``mask`` has shape ``(B, K)`` bool.
    Returns a 0-dim tensor. If the mask is empty, returns 0.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if not mask.any():
        return diff.new_zeros(())
    selected = diff[mask]  # (N, d)
    return (selected ** 2).sum(dim=-1).mean()


def reconstruction_loss(
    states: torch.Tensor, states_decoded: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(states_decoded, states)


def latent_consistency_loss(
    z_predicted: torch.Tensor, z_encoded: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(z_predicted, z_encoded)


def gluing_loss(
    z_encoded: torch.Tensor, glue_mask: torch.Tensor
) -> torch.Tensor:
    """L_g: MSE on edges ``(z_k, z_{k+1})`` flagged by ``glue_mask``.

    ``z_encoded`` has shape ``(B, T, d)``;
    ``glue_mask`` has shape ``(B, T - 1)`` bool.
    """
    diff = z_encoded[:, 1:, :] - z_encoded[:, :-1, :]   # (B, T-1, d)
    return _masked_mse_over_vectors(diff, glue_mask)


def seam_velocity_loss(
    z_encoded: torch.Tensor, seam_mask: torch.Tensor, dt: float
) -> torch.Tensor:
    """L_v: MSE between pre- and post-seam latent finite-difference
    velocities.

    For each seam edge at index ``k`` (i.e. ``seam_mask[:, k] = True``),
    compare ``(z_k - z_{k-1}) / dt`` with ``(z_{k+2} - z_{k+1}) / dt``.
    The seam edge itself ``z_{k+1} - z_k`` is skipped. At gluing-exit
    edges this leap is handled by :func:`gluing_loss`; at cylinder-entry
    edges it is the data-driven analogue of the second mapping-cylinder
    seam condition.

    Indices ``k`` are valid only when ``1 <= k <= T - 3`` so that
    ``k - 1`` and ``k + 2`` are inside the slice.
    """
    B, T, _ = z_encoded.shape
    if T < 4 or not seam_mask.any():
        return z_encoded.new_zeros(())
    # Interior mask over edge indices ``k`` with k in [1, T-3].
    interior = torch.zeros_like(seam_mask)
    interior[:, 1:T - 2] = seam_mask[:, 1:T - 2]
    if not interior.any():
        return z_encoded.new_zeros(())
    # zdot^- at index k: (z_k - z_{k-1}) / dt, indexed by k in [1, T-3].
    zdot_minus = (z_encoded[:, 1:T - 2, :] - z_encoded[:, 0:T - 3, :]) / dt
    # zdot^+ at index k+1: (z_{k+2} - z_{k+1}) / dt, indexed by k in [1, T-3].
    zdot_plus = (z_encoded[:, 3:T, :] - z_encoded[:, 2:T - 1, :]) / dt
    # Sub-mask restricted to k in [1, T-3].
    sub_mask = interior[:, 1:T - 2]
    return _masked_mse_over_vectors(zdot_minus - zdot_plus, sub_mask)


def anti_collapse_loss(
    z_encoded: torch.Tensor, threshold: float
) -> torch.Tensor:
    z_flat = z_encoded.reshape(-1, z_encoded.shape[-1])
    per_coord_var = z_flat.var(dim=0, unbiased=False)
    return torch.relu(threshold - per_coord_var).sum()


# ----------------------------------------------------------------------
# Composite
# ----------------------------------------------------------------------


def compute_losses(
    cfg: CHyLLv2Config,
    states: torch.Tensor,           # (B, T, n+1)
    z_predicted: torch.Tensor,      # (B, T, m)
    z_encoded: torch.Tensor,        # (B, T, m)
    states_decoded: torch.Tensor,   # (B, T, n+1)
    glue_mask: torch.Tensor,        # (B, T-1) bool
    seam_mask: torch.Tensor | None = None,  # (B, T-1) bool
) -> LossOutputs:
    L_x = reconstruction_loss(states, states_decoded)
    L_z = latent_consistency_loss(z_predicted, z_encoded)
    L_g = gluing_loss(z_encoded, glue_mask)

    if cfg.w_v > 0:
        if seam_mask is None:
            seam_mask = glue_mask
        L_v = seam_velocity_loss(z_encoded, seam_mask, dt=cfg.tau)
    else:
        L_v = z_encoded.new_zeros(())

    L_c = anti_collapse_loss(z_encoded, cfg.collapse_threshold)

    total = (
        cfg.w_x * L_x
        + cfg.w_z * L_z
        + cfg.w_g * L_g
        + cfg.w_v * L_v
        + cfg.w_c * L_c
    )
    return LossOutputs(total=total, L_x=L_x, L_z=L_z, L_g=L_g, L_v=L_v, L_c=L_c)
