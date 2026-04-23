"""
Sanity-check visualization for the augmented suspension dataset.

Plots the base trajectory at s=0 and the bridges connecting each jump pair
in (theta, omega, s) space.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_augmented_suspension(dataset, title="Augmented suspension trajectory",
                              azim=-60, elev=25, figsize=(10, 7)):
    """Plot base trajectory and bridges in 3D (theta, omega, s).

    Parameters
    ----------
    dataset : dict
        Output of build_augmented_dataset().
    title : str
    azim, elev : float
        Viewing angles.
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    base = dataset['base_points']
    bridge = dataset['bridge_points']
    bridge_ids = dataset['bridge_ids']
    jump_pairs = dataset['jump_pairs']

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Base trajectory: (theta, omega, s=0)
    ax.plot(base[:, 0], base[:, 1], base[:, 2],
            color='#0072B2', lw=0.8, alpha=0.7, label='base trajectory')

    # Bridges — one color per jump pair
    n_jumps = len(jump_pairs)
    cmap = plt.cm.Set1
    for j in range(n_jumps):
        mask = bridge_ids == j
        pts = bridge[mask]
        # Sort by s for a clean line
        order = np.argsort(pts[:, -1])
        pts = pts[order]
        color = cmap(j / max(n_jumps, 1))
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=color, lw=1.5, marker='.', markersize=3,
                label=f'bridge {j}' if j < 6 else None)

        # Mark jump endpoints
        x_minus, x_plus = jump_pairs[j]
        ax.scatter(*x_minus, 0, color=color, s=40, marker='v', zorder=5)
        ax.scatter(*x_plus, 0, color=color, s=40, marker='^', zorder=5)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\omega$')
    ax.set_zlabel(r'$s$')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    return fig, ax


def plot_bridges_sanity(bridge_net, dataset, n_s=50,
                        title="Learned bridges (random weights)",
                        azim=-60, elev=25, figsize=(10, 7), max_bridges=50):
    """Plot base trajectory and Gamma_theta bridge curves in (theta, omega, eta).

    Parameters
    ----------
    bridge_net : BridgeNet
        May have random (untrained) weights.
    dataset : dict
        Output of build_augmented_dataset().
    n_s : int
        Number of s samples per bridge curve.
    max_bridges : int
        Cap on bridges to plot (for readability).

    Returns
    -------
    fig, ax
    """
    base = dataset['base_points']
    jump_pairs = dataset['jump_pairs']

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Base trajectory at eta=0
    ax.plot(base[:, 0], base[:, 1], np.zeros(len(base)),
            color='#0072B2', lw=0.8, alpha=0.7, label='base trajectory')

    # Evaluate bridge curves
    s_vals = torch.linspace(0, 1, n_s).unsqueeze(-1)  # (n_s, 1)
    cmap = plt.cm.Set1
    n_plot = min(len(jump_pairs), max_bridges)

    bridge_net.eval()
    with torch.no_grad():
        for j in range(n_plot):
            xm, xp = jump_pairs[j]
            xm_t = torch.tensor(xm, dtype=torch.float32).unsqueeze(0).expand(n_s, -1)
            xp_t = torch.tensor(xp, dtype=torch.float32).unsqueeze(0).expand(n_s, -1)

            gamma = bridge_net(xm_t, xp_t, s_vals).numpy()  # (n_s, n+1)
            color = cmap(j / max(n_plot, 1))
            ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2],
                    color=color, lw=1.5, alpha=0.8,
                    label=f'bridge {j}' if j < 6 else None)

            # Mark endpoints
            ax.scatter(*xm, 0, color=color, s=40, marker='v', zorder=5)
            ax.scatter(*xp, 0, color=color, s=40, marker='^', zorder=5)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\omega$')
    ax.set_zlabel(r'$\eta$')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    return fig, ax


def plot_rollout_comparison(segments, jump_pairs, bridge_net,
                            title="Rollout comparison",
                            figsize=(8, 6)):
    """Compare ground-truth and suspension-readout trajectories in (theta, omega).

    Ground truth: continuous arcs + dashed jump lines.
    Suspension readout: continuous arcs + bridge endpoint landings
    (skip bridge interior, fast-forward from x^- to Gamma(x^-,x^+,1)[:n]).

    Parameters
    ----------
    segments : list of ndarray, each (K, 2)
        Smooth trajectory arcs from simulate_rimless_wheel().
    jump_pairs : list of (ndarray, ndarray)
    bridge_net : BridgeNet (trained)

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ground truth: arcs + jump dashes
    for i, seg in enumerate(segments):
        ax.plot(seg[:, 0], seg[:, 1], color='#0072B2', lw=1.5,
                label='ground truth' if i == 0 else None)
    for xm, xp in jump_pairs:
        ax.plot([xm[0], xp[0]], [xm[1], xp[1]],
                color='#0072B2', ls='--', lw=0.8, alpha=0.5)

    # Suspension readout: arcs + bridge landings
    bridge_net.eval()
    with torch.no_grad():
        for i, seg in enumerate(segments):
            ax.plot(seg[:, 0], seg[:, 1], color='#D55E00', lw=1.5, ls='-',
                    alpha=0.7, label='suspension readout' if i == 0 else None)

            if i < len(jump_pairs):
                xm, xp = jump_pairs[i]
                xm_t = torch.tensor(xm, dtype=torch.float32).unsqueeze(0)
                xp_t = torch.tensor(xp, dtype=torch.float32).unsqueeze(0)
                s1 = torch.ones(1, 1)
                landing = bridge_net(xm_t, xp_t, s1).numpy().flatten()
                landing_base = landing[:len(xm)]  # drop eta
                ax.plot([xm[0], landing_base[0]], [xm[1], landing_base[1]],
                        color='#D55E00', ls='-', lw=1.2, alpha=0.7)

    # Mark jump points
    for xm, xp in jump_pairs:
        ax.plot(xm[0], xm[1], 'v', color='k', markersize=6, zorder=5)
        ax.plot(xp[0], xp[1], '^', color='k', markersize=6, zorder=5)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\omega$')
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax


def suspension_diagnostics(segments, jump_pairs, bridge_net, n_s=50):
    """Quantitative diagnostics for the learned suspension.

    1. Continuity: gap at each base->bridge and bridge->base junction.
    2. Bridge separation: minimum pairwise distance between bridge interiors.

    Parameters
    ----------
    segments : list of ndarray, each (K, state_dim)
    jump_pairs : list of (ndarray, ndarray)
    bridge_net : BridgeNet (trained)
    n_s : int
        Interior samples per bridge for separation check.

    Returns
    -------
    report : dict with keys
        'junction_gaps'   : list of float, L2 gap at each junction
        'min_pairwise'    : float, minimum distance between any two bridge interiors
        'pairwise_matrix' : ndarray (n_jumps, n_jumps), min distance between each pair
        'bridge_curves'   : list of ndarray (n_s, state_dim+1), sampled bridge curves
    """
    state_dim = segments[0].shape[1]
    n_jumps = len(jump_pairs)

    # --- 1. Continuity: stitch base arcs and bridges, measure gaps ---
    # The full augmented trajectory goes:
    #   seg[0] at s=0  ->  bridge[0] from (x-_0, 0) to (x+_0, 0)  ->  seg[1] at s=0  -> ...
    junction_gaps = []

    bridge_net.eval()
    s_interior = torch.linspace(0, 1, n_s + 2)  # includes endpoints
    bridge_curves = []

    with torch.no_grad():
        for j, (xm, xp) in enumerate(jump_pairs):
            xm_t = torch.tensor(xm, dtype=torch.float32).unsqueeze(0).expand(n_s + 2, -1)
            xp_t = torch.tensor(xp, dtype=torch.float32).unsqueeze(0).expand(n_s + 2, -1)
            s_t = s_interior.unsqueeze(-1)
            curve = bridge_net(xm_t, xp_t, s_t).numpy()  # (n_s+2, state_dim+1)
            bridge_curves.append(curve)

            # Gap: end of base arc -> start of bridge (s=0)
            if j < len(segments):
                seg_end = segments[j][-1]  # (state_dim,)
                bridge_start = curve[0, :state_dim]  # base coords at s=0
                gap_in = np.linalg.norm(seg_end - bridge_start)
                junction_gaps.append(gap_in)

            # Gap: end of bridge (s=1) -> start of next base arc
            if j + 1 < len(segments):
                bridge_end = curve[-1, :state_dim]  # base coords at s=1
                next_seg_start = segments[j + 1][0]
                gap_out = np.linalg.norm(bridge_end - next_seg_start)
                junction_gaps.append(gap_out)

    # --- 2. Bridge separation: pairwise min distances ---
    # Use only interior points (exclude s=0 and s=1 endpoints)
    interior_curves = [c[1:-1] for c in bridge_curves]  # list of (n_s, d+1)

    pairwise_matrix = np.full((n_jumps, n_jumps), np.inf)
    min_pairwise = np.inf

    for i in range(n_jumps):
        for j in range(i + 1, n_jumps):
            # cdist between interior points of bridge i and bridge j
            dists = np.linalg.norm(
                interior_curves[i][:, None, :] - interior_curves[j][None, :, :],
                axis=-1
            )  # (n_s, n_s)
            d_min = dists.min()
            pairwise_matrix[i, j] = d_min
            pairwise_matrix[j, i] = d_min
            if d_min < min_pairwise:
                min_pairwise = d_min

    return {
        'junction_gaps': junction_gaps,
        'min_pairwise': float(min_pairwise),
        'pairwise_matrix': pairwise_matrix,
        'bridge_curves': bridge_curves,
    }


def plot_diagnostics(diag, jump_pairs, state_labels=None,
                     figsize=(16, 5)):
    """Visualize suspension diagnostics.

    Three panels:
      1. Junction gap magnitudes (bar chart)
      2. Bridge separation heatmap
      3. 2D projections of bridge interiors (state_dim cols vs eta)

    Parameters
    ----------
    diag : dict from suspension_diagnostics()
    jump_pairs : list of (ndarray, ndarray)
    state_labels : list of str, optional
    figsize : tuple

    Returns
    -------
    fig, axes
    """
    gaps = diag['junction_gaps']
    pw = diag['pairwise_matrix']
    curves = diag['bridge_curves']
    n_jumps = len(jump_pairs)
    state_dim = curves[0].shape[1] - 1

    if state_labels is None:
        state_labels = [f'x{i}' for i in range(state_dim)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Junction gaps
    ax = axes[0]
    if len(gaps) > 0:
        ax.bar(range(len(gaps)), gaps, color='#0072B2', alpha=0.8)
        ax.set_xlabel('Junction index')
        ax.set_ylabel('L2 gap')
        ax.set_title(f'Junction continuity (max={max(gaps):.2e})')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    else:
        ax.text(0.5, 0.5, 'No junctions', ha='center', va='center',
                transform=ax.transAxes)

    # Panel 2: Bridge separation heatmap
    ax = axes[1]
    # Replace inf with nan for display
    pw_display = pw.copy()
    pw_display[pw_display == np.inf] = np.nan
    n_show = min(n_jumps, 30)  # cap for readability
    im = ax.imshow(pw_display[:n_show, :n_show], cmap='viridis',
                   aspect='equal')
    fig.colorbar(im, ax=ax, shrink=0.8, label='min distance')
    ax.set_xlabel('Bridge j')
    ax.set_ylabel('Bridge i')
    ax.set_title(f'Pairwise min dist (global min={diag["min_pairwise"]:.4f})')

    # Panel 3: Bridge projections — each state coordinate vs eta
    ax = axes[2]
    cmap = plt.cm.tab10
    # Pick up to 2 most informative state dims to plot
    # (first and third, typically theta and dtheta)
    dims_to_plot = [0, min(2, state_dim - 1)]
    for j, curve in enumerate(curves[:20]):  # cap bridges
        interior = curve[1:-1]  # exclude endpoints
        color = cmap(j % 10)
        for d in dims_to_plot:
            ax.plot(interior[:, d], interior[:, -1],
                    color=color, lw=0.8, alpha=0.6)
    ax.set_xlabel(f'{state_labels[dims_to_plot[0]]} / {state_labels[dims_to_plot[1]]}')
    ax.set_ylabel(r'$\eta$')
    ax.set_title('Bridge interiors projected to (state, eta)')

    plt.suptitle('Suspension quality diagnostics', fontsize=12)
    plt.tight_layout()
    return fig, axes
