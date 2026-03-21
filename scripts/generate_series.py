import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "rimless_wheel"
FIGURES_DIR = ROOT / "figures" / "rimless_wheel"

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import HybridSuspensionConfig
from networks import SuspensionNetworks
from system import RimlessWheelHybridSystem

# Colorblind-safe Okabe-Ito palette
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00',
             '#56B4E9', '#CC79A7', '#F0E442', '#000000']

# Publication-quality matplotlib style
PUB_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
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
}

N_STEPS = 1_000_000
TOTAL_TIME = 1000.0
TAU = TOTAL_TIME / N_STEPS  # 0.001s
X0 = [0.0, 1.5, 0.0]

# --- Load model ---
cfg = HybridSuspensionConfig()
sys_obj = RimlessWheelHybridSystem(cfg)
model = SuspensionNetworks(cfg)
model.load_state_dict(torch.load(ROOT / "runs" / "rimless_wheel" / "model.pt", map_location="cpu"))
model.eval()

# --- True trajectory ---
print("Generating true trajectory...")
true_X = sys_obj.generate_tau_timeseries(X0, TAU, N_STEPS)
# true_X: [N_STEPS+1, 3]

with torch.no_grad():
    true_X_tensor = torch.tensor(true_X, dtype=torch.float32)
    true_Y = model.E(true_X_tensor).numpy()
# true_Y: [N_STEPS+1, 3]

# --- Learned trajectory (iterate F) ---
print("Generating learned trajectory...")
x0_tensor = torch.tensor([X0], dtype=torch.float32)
with torch.no_grad():
    y = model.E(x0_tensor)  # [1, 3]
    pred_Y = [y.squeeze().numpy()]
    for _ in range(N_STEPS):
        y = model.F(y)
        pred_Y.append(y.squeeze().numpy())
pred_Y = np.array(pred_Y)  # [N_STEPS+1, 3]

# --- Save to disk ---
for name, arr, cols in [
    ("series_true_X", true_X, ["theta", "omega", "s"]),
    ("series_true_Y", true_Y, ["y1", "y2", "y3"]),
    ("series_pred_Y", pred_Y, ["y1", "y2", "y3"]),
]:
    np.save(DATA_DIR / f"{name}.npy", arr)
    np.savetxt(DATA_DIR / f"{name}.csv", arr, delimiter=",", header=",".join(cols), comments="")
print(f"Saved .npy and .csv files: shapes {true_X.shape}, {true_Y.shape}, {pred_Y.shape}")

matplotlib.rcParams.update(PUB_STYLE)
# For plotting, subsample dense trajectories to avoid excessive render time
plot_stride = max(1, (N_STEPS + 1) // 100_000)

t = np.arange(N_STEPS + 1)
t_plot = t[::plot_stride]
true_Y_plot = true_Y[::plot_stride]
pred_Y_plot = pred_Y[::plot_stride]

# --- Figure 1: 3D trajectory ---
fig = plt.figure(figsize=(4.5, 4.0))
ax = fig.add_subplot(111, projection="3d")

# Clean 3D panes for publication style
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#CCCCCC')
    pane.set_linewidth(0.4)
ax.grid(True, color='#DDDDDD', linewidth=0.3, linestyle=':')

sc1 = ax.scatter(
    true_Y_plot[:, 0], true_Y_plot[:, 1], true_Y_plot[:, 2],
    c=t_plot, cmap="viridis", s=0.5, alpha=0.5, label=r'True trajectory $E(x_n)$'
)
sc2 = ax.scatter(
    pred_Y_plot[:, 0], pred_Y_plot[:, 1], pred_Y_plot[:, 2],
    c=t_plot, cmap="plasma", s=0.5, alpha=0.5, label=r'Predicted $\hat{F}^n(y_0)$'
)

cb1 = fig.colorbar(sc1, ax=ax, shrink=0.5, aspect=20, pad=0.05)
cb1.ax.tick_params(labelsize=6)
cb1.set_label("Step (true)", fontsize=7)
cb2 = fig.colorbar(sc2, ax=ax, shrink=0.5, aspect=20, pad=0.15)
cb2.ax.tick_params(labelsize=6)
cb2.set_label("Step (learned)", fontsize=7)

ax.set_xlabel(r'$y_1$', labelpad=4)
ax.set_ylabel(r'$y_2$', labelpad=4)
ax.set_zlabel(r'$y_3$', labelpad=4)
ax.set_title(r'$10^6$-step trajectory in embedding space')
ax.legend(loc="upper left", markerscale=3, fontsize=7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_long_series_3d.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_long_series_3d.png")

# --- Figure 2: pointwise drift error ---
errors = np.linalg.norm(true_Y - pred_Y, axis=1)  # [N_STEPS+1]

# Detect guard crossings from theta drops > 0.2
theta = true_X[:, 0]
theta_diff = np.diff(theta)
guard_crossings = np.where(theta_diff < -0.2)[0]  # index n where drop happens between n and n+1

fig, ax = plt.subplots(figsize=(7.0, 2.5))
ax.semilogy(t, errors, linewidth=0.6, color=OKABE_ITO[0])

# Limit guard crossing markers to at most 100
for gc in guard_crossings[:100]:
    ax.axvline(x=gc, color='#AAAAAA', linestyle='--', linewidth=0.3, alpha=0.4)

ax.set_xlabel(r'Step $n$')
ax.set_ylabel(r'$\|y_n^{\rm true} - \hat{y}_n\|_2$')
ax.set_title(r'Pointwise drift error (embedding space, log scale)')
ax.grid(True, which='both', color='#EEEEEE', linewidth=0.4, linestyle=':')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_long_series_error.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig_long_series_error.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_long_series_error.pdf / .png")

# --- Error summary markdown ---
pct = [0, 1, 5, 25, 50, 75, 95, 99, 100]
percentiles = np.percentile(errors, pct)
n_crossings = len(guard_crossings)

with open(DATA_DIR / "series_error_report.md", "w") as f:
    f.write("# Long Time Series: Drift Error Report\n\n")
    f.write(f"- **Steps**: {N_STEPS:,}\n")
    f.write(f"- **Total time**: {TOTAL_TIME:.1f} s\n")
    f.write(f"- **tau**: {TAU:.4f} s\n")
    f.write(f"- **Initial condition**: theta={X0[0]}, omega={X0[1]}, s={X0[2]}\n")
    f.write(f"- **Guard crossings detected**: {n_crossings}\n\n")
    f.write("## Error statistics (L2 norm in embedding space)\n\n")
    f.write("| Percentile | Error |\n")
    f.write("|---|---|\n")
    for p, v in zip(pct, percentiles):
        f.write(f"| {p}% | {v:.6e} |\n")
    f.write(f"\n**Mean**: {errors.mean():.6e}  \n")
    f.write(f"**Std**: {errors.std():.6e}  \n\n")
    f.write("## Guard crossings (step indices)\n\n")
    if n_crossings == 0:
        f.write("None detected.\n")
    else:
        f.write(", ".join(str(g) for g in guard_crossings[:50]))
        if n_crossings > 50:
            f.write(f", ... ({n_crossings - 50} more)")
        f.write("\n")

print("Saved series_error_report.md")
