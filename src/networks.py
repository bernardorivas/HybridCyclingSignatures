"""
Learning architecture from Section 4 of the manuscript.

Three components:
  Encoder        E : R^{n+1} -> R^d      generic MLP, initialised approx identity
  Semiflow       F : R+ x R^d -> R^d     time-conditioned, F(y, dt) = y + MLP(y, dt)
  Decoder        D : R^d -> R^{n+1}      generic MLP, no s=0 forcing
"""
import torch
import torch.nn as nn
from config import config


def _build_mlp(in_dim, out_dim, hidden_dim, num_layers):
    """MLP with GELU activations."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """
    Generic encoder E : R^{n+1} -> R^d (manuscript Section 4, architecture block).

    Realized as a residual MLP around an identity-padded base:
        E(x) = pad(x, d) + MLP(x)
    with the final MLP layer initialised near zero so that E(x) approximates
    the identity (padded with small random noise in coords n+1..d) at the
    start of training. Nothing architecturally pins E at s = 0; injectivity
    on the base space is a learned property.
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_in = cfg.state_dim + 1            # n + 1
        self.d_out = cfg.embed_dim               # d
        self.net = _build_mlp(self.d_in, self.d_out, cfg.hidden_dim, cfg.num_layers)
        # Final layer near-zero init: approximate identity on input dims,
        # small random perturbation on padded dims.
        with torch.no_grad():
            last = self.net[-1]
            last.weight.mul_(0.01)
            last.bias.zero_()

    def forward(self, x):
        # x: (batch, n+1). Out-of-place padding keeps the function vmap-safe
        # (needed by the conformal loss's per-sample Jacobian).
        pad = self.d_out - self.d_in
        if pad > 0:
            zeros = torch.zeros(x.shape[0], pad, device=x.device, dtype=x.dtype)
            base = torch.cat([x, zeros], dim=-1)
        else:
            base = x
        return base + self.net(x)


class Decoder(nn.Module):
    """
    Generic decoder D : R^d -> R^{n+1} (manuscript Section 4).

    No s = 0 forcing; the decoder freely reconstructs the cylinder parameter.
    Residual form around a truncation to the first n+1 latent dims for
    training stability.
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_in = cfg.embed_dim                # d
        self.d_out = cfg.state_dim + 1           # n + 1
        self.net = _build_mlp(self.d_in, self.d_out, cfg.hidden_dim, cfg.num_layers)

    def forward(self, y):
        # y: (batch, d)
        base = y[:, :self.d_out]
        return base + self.net(y)


class FlowPredictor(nn.Module):
    """
    Time-conditioned discrete semiflow F : R+ x R^d -> R^d (manuscript Ψ).

    Implemented as F(y, dt) = y + MLP(concat(y, dt)). At inference / training
    time the data is always sampled at a fixed step dt = integration_tau,
    but the time-conditioning allows evaluation at other step sizes.
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = _build_mlp(cfg.embed_dim + 1, cfg.embed_dim,
                              cfg.hidden_dim, cfg.num_layers)

    def forward(self, y, dt):
        # y: (batch, d)
        # dt: float or tensor of shape (batch,) or (batch, 1)
        if isinstance(dt, (int, float)):
            dt_t = torch.full((y.shape[0], 1), float(dt),
                              device=y.device, dtype=y.dtype)
        else:
            dt_t = torch.as_tensor(dt, device=y.device, dtype=y.dtype)
            if dt_t.dim() == 0:
                dt_t = dt_t.expand(y.shape[0], 1)
            elif dt_t.dim() == 1:
                dt_t = dt_t.unsqueeze(-1)
        inp = torch.cat([y, dt_t], dim=-1)
        return y + self.net(inp)


class SuspensionNetworks(nn.Module):
    def __init__(self, cfg=config):
        super().__init__()
        self.E = Encoder(cfg)
        self.F = FlowPredictor(cfg)
        self.D = Decoder(cfg)
