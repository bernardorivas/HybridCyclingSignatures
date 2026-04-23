"""
Learned bridge map Gamma_theta for the suspension construction.

System-agnostic: works for any state dimension n.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgeNet(nn.Module):
    """Learned bridge generator Gamma_theta(x_minus, x_plus, s) -> R^{n+1}.

    Architecture hard-codes:
      - endpoint attachment: Gamma(.,.,0) = (x_minus, 0), Gamma(.,.,1) = (x_plus, 0)
      - no premature return: eta = s(1-s) * (softplus(w) + eps) > 0 for s in (0,1)

    The learnable part is two small MLPs:
      u_net: R^{2n+1} -> R^n   (base-coordinate correction)
      w_net: R^{2n+1} -> R^1   (extra-coordinate pre-activation)
    """

    def __init__(self, state_dim, hidden=64, eps=1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.eps = eps
        input_dim = 2 * state_dim + 1  # [x_minus, x_plus, s]

        self.u_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
        )

        self.w_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_minus, x_plus, s):
        """
        Parameters
        ----------
        x_minus : (B, n)
        x_plus  : (B, n)
        s       : (B, 1)

        Returns
        -------
        gamma : (B, n+1)
            Bridge points (Psi_1, ..., Psi_n, eta).
        """
        inp = torch.cat([x_minus, x_plus, s], dim=-1)  # (B, 2n+1)

        u = self.u_net(inp)                              # (B, n)
        w = self.w_net(inp)                              # (B, 1)

        # Psi: (1-s)*x_minus + s*x_plus + s*(1-s)*u
        psi = (1 - s) * x_minus + s * x_plus + s * (1 - s) * u  # (B, n)

        # eta: s*(1-s) * (softplus(w) + eps)  -- positive for s in (0,1)
        v = nn.functional.softplus(w) + self.eps         # (B, 1), strictly positive
        eta = s * (1 - s) * v                            # (B, 1)

        return torch.cat([psi, eta], dim=-1)             # (B, n+1)

    def forward_raw(self, x_minus, x_plus, s):
        """Like forward, but also returns the raw u and w outputs.

        Returns
        -------
        gamma : (B, n+1)
        u : (B, n)
        w : (B, 1)
        """
        inp = torch.cat([x_minus, x_plus, s], dim=-1)
        u = self.u_net(inp)
        w = self.w_net(inp)

        psi = (1 - s) * x_minus + s * x_plus + s * (1 - s) * u
        v = nn.functional.softplus(w) + self.eps
        eta = s * (1 - s) * v

        return torch.cat([psi, eta], dim=-1), u, w


def regularization_loss(bridge_net, jump_pairs, n_s=10, lambda_u=1.0, lambda_w=1.0):
    """Penalize ||u_theta||^2 and ||w_theta||^2 with separate weights.

    Parameters
    ----------
    bridge_net : BridgeNet
    jump_pairs : list of (ndarray, ndarray)
    n_s : int
        Interior samples per bridge (s in (0,1), excluding endpoints).
    lambda_u : float
        Weight on base-coordinate correction penalty.
    lambda_w : float
        Weight on extra-coordinate pre-activation penalty.
        Set to 0 to leave eta free for separation.

    Returns
    -------
    loss : scalar tensor
    """
    s_vals = torch.linspace(0, 1, n_s + 2)[1:-1]

    u_sq_sum = torch.tensor(0.0)
    w_sq_sum = torch.tensor(0.0)
    count = 0

    for xm, xp in jump_pairs:
        xm_t = torch.tensor(xm, dtype=torch.float32).unsqueeze(0).expand(n_s, -1)
        xp_t = torch.tensor(xp, dtype=torch.float32).unsqueeze(0).expand(n_s, -1)
        s_t = s_vals.unsqueeze(-1)

        _, u, w = bridge_net.forward_raw(xm_t, xp_t, s_t)
        u_sq_sum = u_sq_sum + (u ** 2).sum()
        w_sq_sum = w_sq_sum + (w ** 2).sum()
        count += n_s

    return (lambda_u * u_sq_sum + lambda_w * w_sq_sum) / count


def separation_loss(bridge_net, jump_pairs, n_s=10, margin=0.1):
    """Hinge-based separation loss between all pairs of distinct bridges.

    Samples n_s interior points per bridge (s in (0,1), excluding endpoints),
    then penalizes any pair of points from different bridges that are closer
    than `margin`.

    Parameters
    ----------
    bridge_net : BridgeNet
    jump_pairs : list of (ndarray, ndarray)
    n_s : int
        Interior samples per bridge.
    margin : float
        Minimum allowed distance between points on different bridges.

    Returns
    -------
    loss : scalar tensor
    """
    n_bridges = len(jump_pairs)
    if n_bridges < 2:
        return torch.tensor(0.0)

    s_vals = torch.linspace(0, 1, n_s + 2)[1:-1]  # (n_s,)

    all_gamma = []
    for xm, xp in jump_pairs:
        xm_t = torch.tensor(xm, dtype=torch.float32).unsqueeze(0).expand(n_s, -1)
        xp_t = torch.tensor(xp, dtype=torch.float32).unsqueeze(0).expand(n_s, -1)
        s_t = s_vals.unsqueeze(-1)
        gamma = bridge_net(xm_t, xp_t, s_t)
        all_gamma.append(gamma)

    all_gamma = torch.stack(all_gamma)  # (n_bridges, n_s, n+1)
    flat = all_gamma.reshape(-1, all_gamma.shape[-1])
    dists = torch.cdist(flat, flat)

    bridge_idx = torch.arange(n_bridges).unsqueeze(1).expand(-1, n_s).reshape(-1)
    diff_bridge = bridge_idx.unsqueeze(1) != bridge_idx.unsqueeze(0)
    upper = torch.triu(diff_bridge, diagonal=1)

    violations = F.relu(margin - dists)
    loss = (violations[upper] ** 2).mean()

    return loss
