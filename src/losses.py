import torch
import torch.nn.functional as F
import numpy as np


def compute_gluing_loss(model, sys, cfg, num_samples=256):
    """
    Gluing loss: enforces the quotient identification (g,1) ~ ι(r(g)).
    Generalized to any system.
    """
    device = next(model.parameters()).device
    d = cfg.state_dim

    # Sample pre-impact states g on the guard set
    # In general, this depends on the system.
    from config import SystemType
    if cfg.system_type == SystemType.RIMLESS_WHEEL:
        w_samples = np.random.uniform(*cfg.omega_bounds, size=(num_samples, 1))
        g_states = np.column_stack([
            np.full((num_samples, 1), cfg.theta_guard),
            w_samples
        ])
    else:
        # Compass gait guard: theta_ns + theta_s = -2*phi, theta_ns > theta_s
        th_s = np.random.uniform(cfg.cg_theta_bounds[0], -cfg.phi - 0.01, size=(num_samples, 1))
        th_ns = -2*cfg.phi - th_s  # ensures th_ns > -phi > th_s
        dq = np.random.uniform(*cfg.cg_omega_bounds, size=(num_samples, 2))
        g_states = np.column_stack([th_ns, th_s, dq])

    # Pre-impact at s=1
    x_guard = np.column_stack([g_states, np.ones(num_samples)])
    
    # Post-impact at s=0
    r_states = np.array([sys.reset_map(g) for g in g_states])
    x_reset = np.column_stack([r_states, np.zeros(num_samples)])

    x_guard_t = torch.tensor(x_guard, dtype=torch.float32).to(device)
    x_reset_t = torch.tensor(x_reset, dtype=torch.float32).to(device)

    return F.mse_loss(model.E(x_guard_t), model.E(x_reset_t))


def calculate_composite_losses(model, sys, x_i, x_next, cfg):
    """
    Composite loss for learning the hybrid suspension embedding.
    """
    y_i = model.E(x_i)
    y_next_true = model.E(x_next)

    # 1. Commutativity: F(E(x_i)) ≈ E(x_{i+1})
    y_next_pred = model.F(y_i)
    loss_comm = F.mse_loss(y_next_pred, y_next_true)

    # 2. Gluing: E(g,1) ≈ E(ι(r(g)))
    loss_glue = compute_gluing_loss(model, sys, cfg)

    # 3. Reconstruction: D(E(x)) ≈ x
    x_recon = model.D(y_i)
    loss_recon = F.mse_loss(x_recon, x_i)

    # 4. Weighted total
    total_loss = (
        cfg.weight_comm * loss_comm +
        cfg.weight_glue * loss_glue +
        cfg.weight_recon * loss_recon
    )

    metrics = {
        'comm': loss_comm.item(),
        'glue': loss_glue.item(),
        'recon': loss_recon.item(),
        'total': total_loss.item()
    }

    return total_loss, metrics
