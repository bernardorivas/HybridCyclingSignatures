import torch
import torch.nn as nn
from config import config


def _build_mlp(in_dim, out_dim, hidden_dim, num_layers):
    """Build an MLP with the given depth, using GELU activations between layers."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class ExtrusionEncoder(nn.Module):
    """
    Encoder E : R^{d+1} -> R^{embed_dim}

    Maps (state, s) into the embedding space. Identity at s=0 (padded with
    zeros when embed_dim > d+1); learned deformation scaled by s on the
    mapping cylinder (s > 0).
    """
    def __init__(self, cfg):
        super().__init__()
        self.d = cfg.state_dim
        self.e = cfg.embed_dim      # output dimension
        self.net = _build_mlp(self.d + 1, self.e, cfg.hidden_dim, cfg.num_layers)

    def forward(self, x):
        # x: [batch, d+1]  (state coords + cylinder parameter s)
        s = x[:, self.d : self.d + 1]
        base_mask = (s.squeeze(-1) == 0)

        y = torch.zeros(x.shape[0], self.e, device=x.device, dtype=x.dtype)

        # Base space (s=0): identity embedding (zero-padded if embed_dim > d+1)
        if base_mask.any():
            y[base_mask, :self.d + 1] = x[base_mask]

        # Cylinder (s>0): learned deformation scaled by s
        if (~base_mask).any():
            x_cyl = x[~base_mask]
            s_cyl = x_cyl[:, self.d : self.d + 1]
            base_point = torch.zeros(x_cyl.shape[0], self.e,
                                     device=x.device, dtype=x.dtype)
            base_point[:, :self.d + 1] = x_cyl
            base_point[:, self.d] = 0  # project s to 0
            y[~base_mask] = base_point + s_cyl * self.net(x_cyl)
        return y


class StabilizationDecoder(nn.Module):
    """
    Decoder D : R^{embed_dim} -> R^{d+1}

    Inverts E via small residual correction; forces s=0 in output.
    """
    def __init__(self, cfg):
        super().__init__()
        self.d = cfg.state_dim
        self.e = cfg.embed_dim
        self.net = _build_mlp(self.e, self.d + 1, cfg.hidden_dim, cfg.num_layers)

    def forward(self, y):
        # y: [batch, embed_dim]
        # Truncate to d+1 as the base prediction, then add learned correction
        x_base = y[:, :self.d + 1]
        correction = self.net(y)
        x_recon = x_base + (0.1 * correction)
        x_recon[:, self.d : self.d + 1] = 0  # Force s=0
        return x_recon


class FlowPredictor(nn.Module):
    """
    Discrete-time semiflow F : R^{embed_dim} -> R^{embed_dim}
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = _build_mlp(cfg.embed_dim, cfg.embed_dim,
                              cfg.hidden_dim, cfg.num_layers)

    def forward(self, y):
        return y + self.net(y)


class SuspensionNetworks(nn.Module):
    def __init__(self, cfg=config):
        super().__init__()
        self.E = ExtrusionEncoder(cfg)
        self.F = FlowPredictor(cfg)
        self.D = StabilizationDecoder(cfg)
