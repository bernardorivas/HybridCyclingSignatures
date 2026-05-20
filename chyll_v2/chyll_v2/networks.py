"""Networks for the CHyLL v2 baseline: Encoder, latent vector field, Decoder.

Each component is a simple MLP. The encoder optionally uses a sin
activation at the final hidden layer (their §7.4 ablation), with SIREN-style
initialization (Sitzmann et al., 2020).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .config import CHyLLv2Config


# ----------------------------------------------------------------------
# Sin activation (SIREN-style)
# ----------------------------------------------------------------------


class Sine(nn.Module):
    """Sine activation with a fixed frequency multiplier.

    Per Sitzmann et al. (2020), the preceding linear layer must be
    initialized in a frequency-aware way -- handled by ``init_sin_linear``.
    """

    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


def init_sin_linear(linear: nn.Linear, omega: float, first_layer: bool = False) -> None:
    """SIREN initialization for the Linear feeding a Sine activation."""
    with torch.no_grad():
        fan_in = linear.in_features
        if first_layer:
            bound = 1.0 / fan_in
        else:
            bound = (6.0 / fan_in) ** 0.5 / omega
        linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.zero_()


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden: int,
    n_layers: int,
    use_sin_last: bool = False,
    sin_omega: float = 30.0,
) -> nn.Sequential:
    """A simple feed-forward MLP with GELU activations.

    If ``use_sin_last`` is set, the final hidden activation is replaced by
    ``Sine(omega)`` and its preceding linear is re-initialized SIREN-style.
    """
    layers: List[nn.Module] = []
    prev = in_dim
    for layer_idx in range(n_layers):
        lin = nn.Linear(prev, hidden)
        layers.append(lin)
        is_last_hidden = layer_idx == n_layers - 1
        if is_last_hidden and use_sin_last:
            init_sin_linear(lin, omega=sin_omega, first_layer=(layer_idx == 0))
            layers.append(Sine(sin_omega))
        else:
            layers.append(nn.GELU())
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------
# Encoder / Decoder / Vector field
# ----------------------------------------------------------------------


class Encoder(nn.Module):
    """``f_theta : M -> R^m``.

    Implemented as a residual MLP around a zero-padded identity from
    state space into the latent space. At initialization this maps
    ``x -> pad(x)``, similar to the suspension-side encoder.
    """

    def __init__(self, cfg: CHyLLv2Config):
        super().__init__()
        self.state_dim = cfg.state_dim
        self.latent_dim = cfg.latent_dim
        if cfg.latent_dim < cfg.state_dim:
            raise ValueError(
                "latent_dim must be >= state_dim for the pad-identity skip"
            )
        self.mlp = build_mlp(
            in_dim=cfg.state_dim,
            out_dim=cfg.latent_dim,
            hidden=cfg.encoder_hidden,
            n_layers=cfg.encoder_layers,
            use_sin_last=cfg.use_sin_last_layer,
            sin_omega=cfg.sin_omega,
        )
        # Initialize the final linear small so f ~ pad(x) at start.
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., state_dim) -> z: (..., latent_dim)
        pad = torch.zeros(
            *x.shape[:-1], self.latent_dim, device=x.device, dtype=x.dtype
        )
        pad[..., : self.state_dim] = x
        return pad + self.mlp(x)


class VectorField(nn.Module):
    """``V_theta : R^m -> R^m``.

    Pure MLP with no skip; the latent flow is ``dz/dt = V_theta(z)``.
    """

    def __init__(self, cfg: CHyLLv2Config):
        super().__init__()
        self.mlp = build_mlp(
            in_dim=cfg.latent_dim,
            out_dim=cfg.latent_dim,
            hidden=cfg.vfield_hidden,
            n_layers=cfg.vfield_layers,
            use_sin_last=False,
            sin_omega=cfg.sin_omega,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class Decoder(nn.Module):
    """``f_theta^{-1} : Img(f_theta) -> M``.

    Residual around the projection ``pi_n : R^m -> R^n`` that drops the
    extra coordinates, so at initialization it returns the first
    ``state_dim`` components of ``z``.
    """

    def __init__(self, cfg: CHyLLv2Config):
        super().__init__()
        self.state_dim = cfg.state_dim
        self.latent_dim = cfg.latent_dim
        self.mlp = build_mlp(
            in_dim=cfg.latent_dim,
            out_dim=cfg.state_dim,
            hidden=cfg.decoder_hidden,
            n_layers=cfg.decoder_layers,
            use_sin_last=False,
            sin_omega=cfg.sin_omega,
        )
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        proj = z[..., : self.state_dim]
        return proj + self.mlp(z)


# ----------------------------------------------------------------------
# Bundle
# ----------------------------------------------------------------------


class CHyLLv2Networks(nn.Module):
    """Container for the three nets so they share device / state_dict."""

    def __init__(self, cfg: CHyLLv2Config):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.vfield = VectorField(cfg)
        self.decoder = Decoder(cfg)
