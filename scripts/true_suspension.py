"""
Exact visualization of the hybrid suspension construction.
No learning — the quotient embedding is constructed analytically,
dynamics are computed via ODE integration.

Usage:
    python scripts/true_suspension.py
    python scripts/true_suspension.py --height 0.5 --azim -60 --elev 30
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
ROOT = Path(__file__).parent.parent
FIGURES_DIR = ROOT / "figures"

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from config import config
from system import RimlessWheelHybridSystem


matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Colorblind-safe, print-friendly (Okabe-Ito derived)
# Linestyle encodes dynamics: solid = continuous flow, dashed = cylinder transit
ORBIT_COLORS = ["#0072B2", "#D55E00", "#009E73"]  # blue, vermillion, green
ORBIT_LABELS = [r"Orbit 1", r"Orbit 2", r"Orbit 3"]


def analytical_embedding(x, cfg, h=0.3):
    x = np.atleast_2d(x)
    y = np.zeros_like(x)
    s = x[:, 2]
    base = s == 0.0
    cyl = ~base
    y[base] = x[base]
    if cyl.any():
        omega = x[cyl, 1]
        sc = x[cyl, 2]
        y[cyl, 0] = (1 - sc) * cfg.theta_guard + sc * cfg.theta_reset
        y[cyl, 1] = (1 - sc) * omega + sc * omega * cfg.omega_restitution
        y[cyl, 2] = h * np.sin(np.pi * sc)
    return y


def generate_surfaces(cfg, resolution=200):
    th = np.linspace(cfg.theta_bounds[0], cfg.theta_bounds[1], resolution)
    om = np.linspace(cfg.omega_bounds[0], cfg.omega_bounds[1], resolution)
    TH, OM = np.meshgrid(th, om)
    s_vals = np.linspace(0.01, 1.0, 80)
    om_cyl = np.linspace(0.0, cfg.omega_bounds[1], resolution)
    S, OC = np.meshgrid(s_vals, om_cyl)
    return TH, OM, S, OC


def main(h=0.3, azim=-50, elev=20):
    cfg = config
    sys = RimlessWheelHybridSystem(cfg)

    viz_tau = cfg.viz_tau
    n_steps = int(cfg.viz_orbit_duration / viz_tau)
    res = 200

    print("Generating surfaces...")
    TH, OM, S, OC = generate_surfaces(cfg, resolution=res)

    Y_base_0, Y_base_1, Y_base_2 = TH, OM, np.zeros_like(TH)

    Y_cyl_0 = (1 - S) * cfg.theta_guard + S * cfg.theta_reset
    Y_cyl_1 = (1 - S) * OC + S * OC * cfg.omega_restitution
    Y_cyl_2 = h * np.sin(np.pi * S)

    E_base = 0.5 * OM**2 + np.cos(TH)

    initial_conditions = [
        [cfg.theta_reset, 0.5, 0.0],
        [0.0, 1.2, 0.0],
        [cfg.theta_reset, 2.0, 0.0],
    ]

    orbits_X, orbits_Y = [], []
    for x0 in initial_conditions:
        orbit = sys.generate_tau_timeseries(x0, viz_tau, n_steps)
        orbits_X.append(orbit)
        orbits_Y.append(analytical_embedding(orbit, cfg, h))

    # ==========================================================
    # FIGURE 1: 2D Phase Portrait
    # ==========================================================
    print("Building 2D phase portrait...")
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.0))

    # Energy level sets -- very faint background texture
    levels = np.linspace(E_base.min(), E_base.max(), 15)
    ax1.contour(TH, OM, E_base, levels=levels, colors="0.88", linewidths=0.3)

    # Guard (red, prominent)
    om_guard = np.linspace(0.0, cfg.omega_bounds[1], 100)
    ax1.plot(np.full_like(om_guard, cfg.theta_guard), om_guard,
             color="#C0392B", linewidth=1.8,
             label=r"$G = \{\theta_g\} \times [\,0,\infty)$")
    # Reset image
    ax1.plot(np.full_like(om_guard, cfg.theta_reset),
             om_guard * cfg.omega_restitution,
             color="0.55", linewidth=1.0, linestyle=":",
             label=r"$r(G)$")

    # Reset arrows (dashed -- discontinuous jump)
    for om_a in [0.5, 1.0, 1.5]:
        om_p = om_a * cfg.omega_restitution
        ax1.annotate("", xy=(cfg.theta_reset, om_p),
                     xytext=(cfg.theta_guard, om_a),
                     arrowprops=dict(arrowstyle="-|>", color="0.40",
                                     lw=0.9, shrinkA=1, shrinkB=1,
                                     linestyle="--",
                                     connectionstyle="arc3,rad=-0.15"))

    # Orbits -- colored, on top of everything
    for i, orbit in enumerate(orbits_X):
        col = ORBIT_COLORS[i]
        base_pts = orbit[orbit[:, 2] < 1e-6]
        if len(base_pts):
            ax1.plot(base_pts[:, 0], base_pts[:, 1],
                     color=col, ls="-", lw=1.8, zorder=4,
                     label=ORBIT_LABELS[i])
            ax1.plot(base_pts[0, 0], base_pts[0, 1], "o",
                     color=col, markersize=5,
                     markeredgecolor="black", markeredgewidth=0.6,
                     zorder=5)

    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\omega$")
    ax1.legend(loc="upper left", framealpha=0.95, edgecolor="0.7")
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "fig_phase_portrait.pdf", bbox_inches="tight")
    fig1.savefig(FIGURES_DIR / "fig_phase_portrait.png", dpi=300, bbox_inches="tight")
    print("  -> figures/fig_phase_portrait.pdf / .png")

    # ==========================================================
    # FIGURE 2: 3D Embedded Suspension
    # ==========================================================
    print("Building 3D suspension visualization...")
    fig2 = plt.figure(figsize=(4.5, 4.0))
    ax2 = fig2.add_subplot(111, projection="3d")

    # Clean 3D panes for publication style
    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#CCCCCC')
        pane.set_linewidth(0.4)
    ax2.grid(True, color='#DDDDDD', linewidth=0.3, linestyle=':')

    # Base surface: light gray, low alpha
    ax2.plot_surface(Y_base_0, Y_base_1, Y_base_2,
                     color="0.90", alpha=0.20, shade=False,
                     rstride=5, cstride=5, zorder=1)

    # Cylinder surface: single sequential colormap by s
    norm_s = Normalize(vmin=0.0, vmax=1.0)
    cyl_colors = cm.Greys(norm_s(S) * 0.5 + 0.15)
    cyl_colors[..., 3] = 0.40
    ax2.plot_surface(Y_cyl_0, Y_cyl_1, Y_cyl_2,
                     facecolors=cyl_colors, shade=False,
                     rstride=2, cstride=2, zorder=2)

    # Guard line on base plane
    om_line = np.linspace(0.0, cfg.omega_bounds[1], 100)
    ax2.plot(np.full_like(om_line, cfg.theta_guard), om_line,
             np.zeros_like(om_line), color="#C0392B", linewidth=1.2,
             zorder=3)

    # Wireframe edges of cylinder at s=0 and s=1 for clarity
    om_w = np.linspace(0.0, cfg.omega_bounds[1], 100)
    # s=0 edge (guard line, already drawn)
    # s=1 edge (reset line on base)
    ax2.plot(np.full_like(om_w, cfg.theta_reset),
             om_w * cfg.omega_restitution,
             np.zeros_like(om_w),
             color="0.4", linewidth=0.8, linestyle=":", zorder=3)

    # A few iso-omega wires on the cylinder for depth
    for om_wire in [0.3, 0.8, 1.3, 1.8]:
        s_w = np.linspace(0.01, 1.0, 80)
        y0 = (1 - s_w) * cfg.theta_guard + s_w * cfg.theta_reset
        y1 = (1 - s_w) * om_wire + s_w * om_wire * cfg.omega_restitution
        y2 = h * np.sin(np.pi * s_w)
        ax2.plot(y0, y1, y2, color="0.55", linewidth=0.4, zorder=2)

    # Orbits: solid = continuous flow, dashed = cylinder transit
    for i in range(len(orbits_Y)):
        col = ORBIT_COLORS[i]
        oY = orbits_Y[i]
        oX = orbits_X[i]
        base_mask = oX[:, 2] < 1e-6

        segments = np.split(np.arange(len(oY)),
                            np.where(np.diff(base_mask.astype(int)))[0] + 1)
        for seg in segments:
            if len(seg) < 2:
                continue
            if base_mask[seg[0]]:
                ax2.plot(oY[seg, 0], oY[seg, 1], oY[seg, 2],
                         "-", color=col, lw=1.4, zorder=5)
            else:
                ax2.plot(oY[seg, 0], oY[seg, 1], oY[seg, 2],
                         "--", color=col, lw=1.4, zorder=5)

        ax2.plot(oY[0, 0], oY[0, 1], oY[0, 2], "o",
                 color=col, markersize=4,
                 markeredgecolor="black", markeredgewidth=0.5, zorder=6)

        ax2.plot([], [], [], color=col, ls="-", lw=1.4,
                 label=ORBIT_LABELS[i])

    ax2.set_xlabel(r"$y_1$", labelpad=6)
    ax2.set_ylabel(r"$y_2$", labelpad=6)
    ax2.set_zlabel(r"$y_3$", labelpad=6)
    ax2.legend(loc="upper left", framealpha=0.9, edgecolor="0.7", fontsize=9)
    ax2.view_init(elev=elev, azim=azim)
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "fig_suspension_3d.pdf", bbox_inches="tight")
    fig2.savefig(FIGURES_DIR / "fig_suspension_3d.png", dpi=300, bbox_inches="tight")
    print("  -> figures/fig_suspension_3d.pdf / .png")

    print("Interactive windows open.")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=0.3,
                        help="Cylinder lift amplitude h in h*sin(pi*s)")
    parser.add_argument("--azim", type=float, default=-50)
    parser.add_argument("--elev", type=float, default=20)
    args = parser.parse_args()
    main(h=args.height, azim=args.azim, elev=args.elev)
