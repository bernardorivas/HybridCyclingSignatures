"""
Six-term loss from Section 4 of the manuscript (eq. loss_total):

    L_total = w_dyn  * L_dyn
            + w_glue * L_glue
            + w_seam * L_seam
            + w_conf * L_conf
            + w_coll * L_coll
            + w_recon* L_recon

L_utb is kept as a legacy one-sided tangent anchor (default weight 0);
L_seam subsumes it with a two-sided cosine-tangent constraint.

Phase I (manuscript Algorithm 1) trains E and F on all terms EXCEPT L_recon.
Phase II freezes E and F and trains D alone on masked L_recon with larger LR.
"""
import numpy as np
import torch
import torch.nn.functional as F

from config import SystemType


# ---------------------------------------------------------------------------
# Guard-point sampling (shared by glue and utb losses)
# ---------------------------------------------------------------------------

def _sample_guard_states(cfg, num_samples):
    """Return (num_samples, state_dim) array of guard states for the active system."""
    if cfg.system_type == SystemType.RIMLESS_WHEEL:
        w_samples = np.random.uniform(0.0, cfg.omega_bounds[1],
                                      size=(num_samples, 1))
        return np.column_stack([
            np.full((num_samples, 1), cfg.theta_guard),
            w_samples
        ])
    # COMPASS_GAIT
    th_s = np.random.uniform(cfg.cg_theta_bounds[0], -cfg.phi - 0.01,
                             size=(num_samples, 1))
    th_ns = -2 * cfg.phi - th_s   # ensures th_ns > -phi > th_s
    dq = np.random.uniform(*cfg.cg_omega_bounds, size=(num_samples, 2))
    return np.column_stack([th_ns, th_s, dq])


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def compute_dynamics_loss(model, x_i, x_next, cfg):
    """L_dyn = || Ψ(Δt, E(x_i)) - E(x_{i+1}) ||² with Δt = integration_tau."""
    y_i = model.E(x_i)
    y_next_true = model.E(x_next)
    dt = cfg.integration_tau
    y_next_pred = model.F(y_i, dt)
    return F.mse_loss(y_next_pred, y_next_true)


def compute_gluing_loss(model, sys, cfg, num_samples=256):
    """L_glue = || E(g, 1) - E(r(g), 0) ||² over sampled g on the guard."""
    device = next(model.parameters()).device
    g_states = _sample_guard_states(cfg, num_samples)
    x_guard = np.column_stack([g_states, np.ones(num_samples)])
    r_states = np.array([sys.reset_map(g) for g in g_states])
    x_reset = np.column_stack([r_states, np.zeros(num_samples)])

    x_guard_t = torch.as_tensor(x_guard, dtype=torch.float32, device=device)
    x_reset_t = torch.as_tensor(x_reset, dtype=torch.float32, device=device)
    return F.mse_loss(model.E(x_guard_t), model.E(x_reset_t))


def compute_reconstruction_loss_masked(model, x, cfg):
    """L_recon = mean || D(E(x)) - x ||² on samples outside an ε-neighborhood
    of the cylinder boundaries s ∈ {0, 1}.

    Keep: s == 0 (base-space points) OR ε < s < 1 - ε (interior of cylinder).
    Mask: s ∈ (0, ε] ∪ [1 - ε, 1].
    """
    s = x[:, -1]
    eps = cfg.recon_boundary_eps
    keep = (s == 0) | ((s > eps) & (s < 1.0 - eps))
    if keep.sum() == 0:
        # Phase II always backpropagates the returned scalar. Keep an empty
        # masked batch differentiable with respect to D instead of returning
        # a detached zero that makes ``backward()`` fail.
        return next(model.D.parameters()).sum() * 0.0
    x_k = x[keep]
    return F.mse_loss(model.D(model.E(x_k)), x_k)


def compute_conformal_loss(model, x, cfg, max_samples=64):
    """L_conf = mean || λ I - J^T J ||_F² with λ = tr(J^T J) / (n+1).

    Batched Jacobian via torch.func.jacrev + vmap. Subsampled to keep cost
    bounded per step.
    """
    n_plus_1 = cfg.state_dim + 1
    if x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_samples]
        x = x[idx]

    def E_single(xi):
        return model.E(xi.unsqueeze(0)).squeeze(0)

    # J has shape (batch, d, n+1)
    J = torch.vmap(torch.func.jacrev(E_single))(x)
    JTJ = J.transpose(-2, -1) @ J                              # (batch, n+1, n+1)
    lam = JTJ.diagonal(dim1=-2, dim2=-1).sum(-1) / n_plus_1    # (batch,)
    I = torch.eye(n_plus_1, device=x.device, dtype=x.dtype)
    target = lam.view(-1, 1, 1) * I.unsqueeze(0)               # (batch, n+1, n+1)
    return ((JTJ - target) ** 2).sum(dim=(-2, -1)).mean()


def compute_collapse_loss(model, x, cfg):
    """L_coll = Σ_i ReLU(Λ - Var(E(x)_i)) with Λ = collapse_threshold.

    Uses per-batch variance as an estimator of the dataset variance.
    """
    z = model.E(x)
    var = z.var(dim=0, unbiased=False)          # (d,)
    return F.relu(cfg.collapse_threshold - var).sum()


def compute_seam_loss(model, sys, cfg):
    """L_seam: cosine mismatch at BOTH cylinder-arc seams.

    Pre-jump (arc-end -> bridge-start, at base point g, s=0):
      arc-end tangent in latent  = grad_x E(g, 0) . v_X(g)
      bridge-start tangent       = ∂_s E(g, 0)         (anchored ~v_X(g) by L_utb)
    Post-jump (bridge-end -> next arc-start, at base point r(g), s=1):
      bridge-end tangent         = ∂_s E(g, 1)         (anchored ~v_X(r(g)) by L_utb)
      next-arc-start tangent     = grad_x E(r(g), 0) . v_X(r(g))

    L_utb anchors the bridge tangents but does not constrain grad_x E, so
    BOTH seams generically have a directional cusp. This loss directly
    closes both via `(1 - cos)` over sampled guard states.

    Without this, the most visible cusp can sit at the pre-jump (impact
    moment) and the trajectory's lift-tangent flips by ~140 deg even when
    L_utb is fully satisfied.
    """
    device = next(model.parameters()).device
    h = cfg.utb_finite_diff_h
    num_samples = cfg.utb_num_samples

    g_states = _sample_guard_states(cfg, num_samples)
    r_states = np.array([sys.reset_map(g) for g in g_states])
    v_g  = np.array([sys.base_vector_field(0,  g) for  g in g_states])
    v_rg = np.array([sys.base_vector_field(0, rg) for rg in r_states])

    g_t = torch.as_tensor(g_states, dtype=torch.float32, device=device)
    r_t = torch.as_tensor(r_states, dtype=torch.float32, device=device)
    vg_t  = torch.as_tensor(v_g,  dtype=torch.float32, device=device)
    vrg_t = torch.as_tensor(v_rg, dtype=torch.float32, device=device)
    zeros = torch.zeros(num_samples, 1, device=device)

    # Bridge-end tangent at s=1 (post-jump seam): finite-difference along s.
    x_g_1   = torch.cat([g_t, torch.ones(num_samples, 1, device=device)], dim=-1)
    x_g_1mh = torch.cat([g_t, torch.full((num_samples, 1), 1.0 - h, device=device)], dim=-1)
    v_bridge_end = (model.E(x_g_1) - model.E(x_g_1mh)) / h

    # Bridge-start tangent at s=0 (pre-jump seam): finite-difference along s.
    x_g_0 = torch.cat([g_t, zeros], dim=-1)
    x_g_h = torch.cat([g_t, torch.full((num_samples, 1), h, device=device)], dim=-1)
    v_bridge_start = (model.E(x_g_h) - model.E(x_g_0)) / h

    # Next-arc-start tangent: JVP of E at (r(g), 0) in direction (v_X(r(g)), 0).
    x_at_post = torch.cat([r_t, zeros], dim=-1)
    dir_post  = torch.cat([vrg_t, zeros], dim=-1)
    _, v_arc_post = torch.func.jvp(model.E, (x_at_post,), (dir_post,))

    # Arc-end tangent: JVP of E at (g, 0) in direction (v_X(g), 0).
    x_at_pre = torch.cat([g_t, zeros], dim=-1)
    dir_pre  = torch.cat([vg_t, zeros], dim=-1)
    _, v_arc_pre = torch.func.jvp(model.E, (x_at_pre,), (dir_pre,))

    eps = 1e-8
    def _cos_loss(a, b):
        ua = a / (a.norm(dim=-1, keepdim=True) + eps)
        ub = b / (b.norm(dim=-1, keepdim=True) + eps)
        return (1.0 - (ua * ub).sum(dim=-1)).mean()

    return _cos_loss(v_bridge_start, v_arc_pre) + _cos_loss(v_bridge_end, v_arc_post)


def compute_utb_loss(model, sys, cfg):
    """L_utb anchors the cylinder-direction velocity v_cyl(g, s) = ∂_s E(g, s)
    to the physical velocity at s=0 and s=1.

    v_X has dimension n; we pad to embed_dim by appending zeros (the latent
    s-direction and any extra embedding dims are not constrained).
    """
    device = next(model.parameters()).device
    n = cfg.state_dim
    d = cfg.embed_dim
    h = cfg.utb_finite_diff_h
    num_samples = cfg.utb_num_samples

    g_states = _sample_guard_states(cfg, num_samples)

    # Physical velocities at the two cylinder boundaries
    v_g = np.array([sys.base_vector_field(0, g) for g in g_states])
    r_states = np.array([sys.reset_map(g) for g in g_states])
    v_rg = np.array([sys.base_vector_field(0, rg) for rg in r_states])

    # Embed v_X into R^d by padding with zeros (state dims filled, rest zero)
    v_g_aug = np.zeros((num_samples, d))
    v_g_aug[:, :n] = v_g
    v_rg_aug = np.zeros((num_samples, d))
    v_rg_aug[:, :n] = v_rg

    # Finite-difference v_cyl at s=0 and s=1
    x_g_0   = np.column_stack([g_states, np.zeros(num_samples)])
    x_g_h   = np.column_stack([g_states, np.full(num_samples, h)])
    x_g_1mh = np.column_stack([g_states, np.full(num_samples, 1.0 - h)])
    x_g_1   = np.column_stack([g_states, np.ones(num_samples)])

    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    v_cyl_0 = (model.E(to_t(x_g_h))   - model.E(to_t(x_g_0)))   / h
    v_cyl_1 = (model.E(to_t(x_g_1))   - model.E(to_t(x_g_1mh))) / h

    loss_start = F.mse_loss(v_cyl_0, to_t(v_g_aug))
    loss_end   = F.mse_loss(v_cyl_1, to_t(v_rg_aug))
    return loss_start + loss_end


# ---------------------------------------------------------------------------
# Phase-aware composite
# ---------------------------------------------------------------------------

def calculate_composite_losses(model, sys, x_i, x_next, cfg, phase=1):
    """
    phase=1 : Phase I  — E, F only; drops L_recon
    phase=2 : Phase II — D only; only L_recon (masked)

    Returns (total_loss, metrics_dict) with per-term floats.
    """
    if phase == 2:
        loss_recon = compute_reconstruction_loss_masked(model, x_i, cfg)
        total = cfg.weight_recon * loss_recon
        return total, {'recon': loss_recon.item(), 'total': total.item()}

    # --- Phase I: five-or-six terms, no recon ---
    loss_dyn  = compute_dynamics_loss(model, x_i, x_next, cfg)
    loss_glue = compute_gluing_loss(model, sys, cfg)
    loss_conf = compute_conformal_loss(model, x_i, cfg)
    loss_coll = compute_collapse_loss(model, x_i, cfg)
    loss_utb  = compute_utb_loss(model, sys, cfg)

    total = (
        cfg.weight_dyn  * loss_dyn  +
        cfg.weight_glue * loss_glue +
        cfg.weight_conf * loss_conf +
        cfg.weight_coll * loss_coll +
        cfg.weight_utb  * loss_utb
    )

    metrics = {
        'dyn':   loss_dyn.item(),
        'glue':  loss_glue.item(),
        'conf':  loss_conf.item(),
        'coll':  loss_coll.item(),
        'utb':   loss_utb.item(),
    }

    # L_seam: cosine mismatch of bridge-end vs next arc-start tangents.
    # It is active in the canonical configuration and remains weight-controlled.
    if getattr(cfg, 'weight_seam', 0.0) > 0.0:
        loss_seam = compute_seam_loss(model, sys, cfg)
        total = total + cfg.weight_seam * loss_seam
        metrics['seam'] = loss_seam.item()

    metrics['total'] = total.item()
    return total, metrics
