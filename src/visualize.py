from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

# Directories
ROOT = Path(__file__).parent.parent
FIGURES_DIR = ROOT / "figures"

# Colorblind-safe Okabe-Ito palette
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00',
             '#56B4E9', '#CC79A7', '#F0E442', '#000000']

# Publication-quality matplotlib style
PUB_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': '0.8',
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

# =============================================================================
# HELPER
# =============================================================================

def generate_massive_background_cloud(cfg, root_resolution=100):
    """Generates a dense point cloud for visualizing the manifold."""
    if cfg.state_dim == 2:
        v_bg = np.linspace(cfg.theta_bounds[0], cfg.theta_bounds[1], root_resolution)
        u_bg = np.linspace(cfg.omega_bounds[0], cfg.omega_bounds[1], root_resolution)
        V, U = np.meshgrid(v_bg, u_bg)
        X_base = np.column_stack([V.ravel(), U.ravel(), np.zeros_like(V.ravel())])
        
        # Cylinder
        s_steps = np.linspace(0.01, 1.0, 50)
        C_v = np.full(root_resolution * len(s_steps), cfg.theta_guard)
        C_u = np.tile(u_bg, len(s_steps))
        C_s = np.repeat(s_steps, root_resolution)
        X_cyl = np.column_stack([C_v, C_u, C_s])
        return np.vstack([X_base, X_cyl])
    else:
        # For 4D+, just return a small random sample around meaningful ranges
        X_base = np.random.uniform(-0.5, 0.5, size=(root_resolution**2, cfg.state_dim))
        X_base = np.column_stack([X_base, np.zeros(root_resolution**2)])
        return X_base

# =============================================================================
# RIMLESS WHEEL VISUALIZATION
# =============================================================================

def compute_deep_crossing_validation_rimless(model, sys, cfg):
    """Evaluates Rimless Wheel semiflow across the mapped cylinder."""
    model.eval()
    x0 = [0.0, 1.5, 0.0]
    tau = cfg.integration_tau
    n_steps = int(4.0 / tau)

    true_orbit_x = sys.generate_tau_timeseries(x0, tau, n_steps)
    true_orbit_t = torch.tensor(true_orbit_x, dtype=torch.float32)
    device = next(model.parameters()).device
    true_orbit_t = true_orbit_t.to(device)

    with torch.no_grad():
        true_orbit_y = model.E(true_orbit_t)
        pred_orbit_y = [true_orbit_y[0]]
        y_curr = true_orbit_y[0].unsqueeze(0)
        for _ in range(len(true_orbit_t) - 1):
            y_curr = model.F(y_curr)
            pred_orbit_y.append(y_curr.squeeze(0))
        pred_orbit_y = torch.stack(pred_orbit_y)
        recon_pred_orbit_x = model.D(pred_orbit_y)

    y_mse = F.mse_loss(true_orbit_y, pred_orbit_y).item()
    x_mse = F.mse_loss(true_orbit_t, recon_pred_orbit_x).item()

    print("\n" + "="*50)
    print("RIMLESS WHEEL VALIDATION")
    print(f"Path MSE: Y={y_mse:.6f}, X={x_mse:.6f}")
    print("="*50 + "\n")


def plot_hybrid_suspension_rimless(model, hds, cfg, loss_history):
    matplotlib.rcParams.update(PUB_STYLE)
    model.eval()
    device = next(model.parameters()).device
    out_dir = FIGURES_DIR / "rimless_wheel"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Rimless Wheel visualizations...")
    X_bg = generate_massive_background_cloud(cfg)
    X_bg_t = torch.tensor(X_bg, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        Y_bg = model.E(X_bg_t).cpu().numpy()

    # Initial conditions for orbits
    initial_conditions = [[cfg.theta_reset, 0.5, 0.0], [0.0, 1.2, 0.0], [cfg.theta_reset, 2.0, 0.0]]
    colors = [OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2]]
    viz_tau = cfg.viz_tau
    n_steps_viz = int(cfg.viz_orbit_duration / viz_tau)
    pred_tau = cfg.integration_tau
    n_steps_pred = int(cfg.viz_orbit_duration / pred_tau)

    orbits_X_true = []
    orbits_X_pred = []
    orbits_Y_true = []
    orbits_Y_pred = []

    for x0 in initial_conditions:
        true_X = hds.generate_tau_timeseries(x0, viz_tau, n_steps_viz)
        orbits_X_true.append(true_X)
        with torch.no_grad():
            y_curr = model.E(torch.tensor(x0, dtype=torch.float32).to(device).unsqueeze(0))
            pred_Y = [y_curr.squeeze().cpu().numpy()]
            for _ in range(n_steps_pred - 1):
                y_curr = model.F(y_curr)
                pred_Y.append(y_curr.squeeze().cpu().numpy())
            pred_Y_t = torch.tensor(np.array(pred_Y), dtype=torch.float32).to(device)
            orbits_Y_pred.append(pred_Y_t.cpu().numpy())
            orbits_X_pred.append(model.D(pred_Y_t).cpu().numpy())
            true_X_pred_res = hds.generate_tau_timeseries(x0, pred_tau, n_steps_pred)
            orbits_Y_true.append(model.E(torch.tensor(true_X_pred_res, dtype=torch.float32).to(device)).cpu().numpy())

    # FIGURE 1: 2D Phase Space
    fig1, ax1 = plt.subplots(figsize=(4.0, 3.5))
    ax1.axvline(x=cfg.theta_reset, color='black', linestyle=':', linewidth=1.0, alpha=0.8)
    y_max_ax1 = cfg.omega_bounds[1] + 0.1
    ax1.vlines(x=cfg.theta_guard, ymin=0, ymax=y_max_ax1, color='#FF0000', linestyle='--', linewidth=1.2)
    
    idx_to_plot = 1
    def plot_split_trajectory(ax, data, fmt, **kwargs):
        if len(data) == 0: return
        theta = data[:, 0]
        jumps = np.where(np.abs(np.diff(theta)) > 0.1)[0]
        start_idx = 0
        for jump in jumps:
            ax.plot(data[start_idx:jump+1, 0], data[start_idx:jump+1, 1], fmt, **kwargs)
            start_idx = jump + 1
        ax.plot(data[start_idx:, 0], data[start_idx:, 1], fmt, **kwargs)

    plot_split_trajectory(ax1, orbits_X_true[idx_to_plot][orbits_X_true[idx_to_plot][:,2]<1e-3], '-', color=OKABE_ITO[0], linewidth=2.0)
    plot_split_trajectory(ax1, orbits_X_pred[idx_to_plot][orbits_X_pred[idx_to_plot][:,2]<1e-3], '--', color='#333333', linewidth=1.5, alpha=0.9)
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$\dot{\theta}$')
    ax1.set_xlim(cfg.theta_bounds[0] - 0.05, cfg.theta_bounds[1] + 0.05)
    ax1.set_ylim(cfg.omega_bounds[0] - 0.1, cfg.omega_bounds[1] + 0.1)
    ax1.grid(True, color='#F0F0F0', linewidth=0.5)
    fig1.savefig(out_dir / "fig_rimless_phase_2d.png", dpi=300)
    plt.close(fig1)

    # FIGURE 2: 3D Manifold
    fig2 = plt.figure(figsize=(5.5, 5.0))
    ax2 = fig2.add_subplot(111, projection='3d')
    base_mask = X_bg[:, 2] == 0.0
    cyl_mask = X_bg[:, 2] > 0.0
    ax2.plot_trisurf(Y_bg[base_mask, 0], Y_bg[base_mask, 1], Y_bg[base_mask, 2], color='#CCDDEE', alpha=0.3, shade=True)
    s_col = ax2.scatter(Y_bg[cyl_mask, 0], Y_bg[cyl_mask, 1], Y_bg[cyl_mask, 2], c=X_bg[cyl_mask, 2], cmap='plasma', s=2.5, alpha=0.8)
    ax2.plot(orbits_Y_true[idx_to_plot][:, 0], orbits_Y_true[idx_to_plot][:, 1], orbits_Y_true[idx_to_plot][:, 2], '-', color=OKABE_ITO[0], linewidth=2.5)
    ax2.plot(orbits_Y_pred[idx_to_plot][:, 0], orbits_Y_pred[idx_to_plot][:, 1], orbits_Y_pred[idx_to_plot][:, 2], '--', color='#333333', linewidth=1.5, alpha=0.9)
    ax2.view_init(elev=30, azim=-60)
    fig2.savefig(out_dir / "fig_rimless_suspension_mapped_3d.png", dpi=300)
    plt.close(fig2)

    # FIGURE 3: Losses
    fig3, ax3 = plt.subplots(figsize=(4.0, 3.0))
    ax3.plot(loss_history, color=OKABE_ITO[0])
    ax3.set_yscale('log')
    ax3.set_title('Optimization Convergence')
    fig3.savefig(out_dir / "fig_rimless_optimization_losses.png", dpi=300)
    plt.close(fig3)

# =============================================================================
# COMPASS GAIT VISUALIZATION
# =============================================================================

def plot_hybrid_suspension_compass(model, hds, cfg, loss_history):
    matplotlib.rcParams.update(PUB_STYLE)
    model.eval()
    device = next(model.parameters()).device
    out_dir = FIGURES_DIR / "compass_gait"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Compass-Gait visualizations...")
    
    # 1. Trajectories
    # Start near a typical limit cycle
    x0 = [0.2, -0.2, -0.5, 0.0, 0.0] 
    viz_tau = cfg.viz_tau
    n_steps_viz = int(cfg.viz_orbit_duration / viz_tau)
    pred_tau = cfg.integration_tau
    n_steps_pred = int(cfg.viz_orbit_duration / pred_tau)

    true_X = hds.generate_tau_timeseries(x0, viz_tau, n_steps_viz)
    
    with torch.no_grad():
        y_curr = model.E(torch.tensor(x0, dtype=torch.float32).to(device).unsqueeze(0))
        pred_Y = [y_curr.squeeze().cpu().numpy()]
        for _ in range(n_steps_pred - 1):
            y_curr = model.F(y_curr)
            pred_Y.append(y_curr.squeeze().cpu().numpy())
        pred_Y_t = torch.tensor(np.array(pred_Y), dtype=torch.float32).to(device)
        orbits_X_pred = model.D(pred_Y_t).cpu().numpy()

    # FIGURE 1: Phase Portrait (theta_s vs theta_ns)
    fig1, ax1 = plt.subplots(figsize=(4.0, 3.5))
    ax1.plot(true_X[:, 1], true_X[:, 0], '-', color=OKABE_ITO[0], linewidth=1.5, label='True')
    ax1.plot(orbits_X_pred[:, 1], orbits_X_pred[:, 0], '--', color='black', linewidth=1.0, alpha=0.8, label='Pred')
    ax1.set_xlabel(r'$\theta_s$ (rad)')
    ax1.set_ylabel(r'$\theta_{ns}$ (rad)')
    ax1.set_title(r'Compass-Gait Gait Cycle ($\phi$=%.2f$^\circ$)' % np.degrees(cfg.phi))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.savefig(out_dir / "fig_compass_gait_phase.png", dpi=300)
    plt.close(fig1)

    # FIGURE 2: Loss History
    fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
    ax2.plot(loss_history, color=OKABE_ITO[0])
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Compass-Gait Training Loss')
    fig2.savefig(out_dir / "fig_compass_optimization_losses.png", dpi=300)
    plt.close(fig2)

    print(f"Compass-Gait figures saved to {out_dir}")
