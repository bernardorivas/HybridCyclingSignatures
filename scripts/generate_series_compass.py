"""
Generate a long time series for the compass-gait biped, analogous to
generate_series.py for the rimless wheel. Produces:
  - data/compass_gait/series_true_X.npy   (true trajectory in state+s space)
  - data/compass_gait/series_true_Y.npy   (true trajectory through E)
  - data/compass_gait/series_pred_Y.npy   (predicted trajectory iterating F)
  - figures: drift error plot
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "compass_gait"
FIGURES_DIR = ROOT / "figures" / "compass_gait"

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from config import config, SystemType
from networks import SuspensionNetworks
from system import CompassGaitHybridSystem
from visualize import OKABE_ITO, PUB_STYLE

matplotlib.rcParams.update(PUB_STYLE)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_STEPS = 400
TAU = 0.05  # Must match config.integration_tau (what F was trained on)
# 400 * 0.05 = 20s of simulated walking

config.system_type = SystemType.COMPASS_GAIT
phi = config.phi

# Start near the limit cycle (post-reset state, s=0)
# These values are approximate; the dynamics will converge quickly.
X0 = [-(phi + 0.27), -(phi - 0.27), -0.38, -1.09, 0.0]

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
if config.device == "mps" and torch.backends.mps.is_available():
    device = torch.device("mps")
elif config.device == "cuda" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

hds = CompassGaitHybridSystem(config)
model = SuspensionNetworks(config).to(device)
model.load_state_dict(
    torch.load(ROOT / "runs" / "compass_gait" / "model.pt", map_location=device))
model.eval()
print(f"Model loaded (embed_dim={config.embed_dim}, device={device})")

# ---------------------------------------------------------------------------
# True trajectory
# ---------------------------------------------------------------------------
print(f"Generating true trajectory ({N_STEPS:,} steps, tau={TAU:.4f}s)...")
true_X = hds.generate_tau_timeseries(X0, TAU, N_STEPS)
print(f"  shape: {true_X.shape}")

with torch.no_grad():
    # Encode in chunks to avoid OOM
    chunk = 50_000
    true_Y_parts = []
    for i in range(0, len(true_X), chunk):
        x_chunk = torch.tensor(true_X[i:i+chunk], dtype=torch.float32).to(device)
        true_Y_parts.append(model.E(x_chunk).cpu().numpy())
    true_Y = np.vstack(true_Y_parts)
print(f"  encoded shape: {true_Y.shape}")

# ---------------------------------------------------------------------------
# Learned trajectory (iterate F from same IC)
# ---------------------------------------------------------------------------
print("Generating learned trajectory (iterating F)...")
with torch.no_grad():
    x0_tensor = torch.tensor([X0], dtype=torch.float32).to(device)
    y = model.E(x0_tensor)
    pred_Y = [y.squeeze().cpu().numpy()]
    for step in range(N_STEPS):
        y = model.F(y)
        pred_Y.append(y.squeeze().cpu().numpy())
        if (step + 1) % 200_000 == 0:
            print(f"  {step+1:,} / {N_STEPS:,}")
pred_Y = np.array(pred_Y)
print(f"  shape: {pred_Y.shape}")

# ---------------------------------------------------------------------------
# Save to disk
# ---------------------------------------------------------------------------
for name, arr in [
    ("series_true_X", true_X),
    ("series_true_Y", true_Y),
    ("series_pred_Y", pred_Y),
]:
    np.save(DATA_DIR / f"{name}.npy", arr)
print(f"Saved .npy files to {DATA_DIR}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
errors = np.linalg.norm(true_Y - pred_Y, axis=1)

# Detect guard crossings: large jump in theta_ns (reset swaps angles)
theta_ns = true_X[:, 0]
theta_ns_diff = np.abs(np.diff(theta_ns))
guard_crossings = np.where(theta_ns_diff > 0.1)[0]

t = np.arange(N_STEPS + 1)

# --- Drift error plot ---
fig, ax = plt.subplots(figsize=(7.0, 2.5))
ax.semilogy(t, errors, linewidth=0.5, color=OKABE_ITO[0])

for gc in guard_crossings[:200]:
    ax.axvline(x=gc, color='#AAAAAA', linestyle='--', linewidth=0.3, alpha=0.3)

ax.set_xlabel(r'Step $n$')
ax.set_ylabel(r'$\|y_n^{\rm true} - \hat{y}_n\|_2$')
ax.set_title(r'Pointwise drift error (embedding space)')
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_long_series_error.png", dpi=200)
plt.close()
print(f"Saved drift error figure")

# --- Error report ---
pct = [0, 1, 5, 25, 50, 75, 95, 99, 100]
percentiles = np.percentile(errors, pct)
n_crossings = len(guard_crossings)

with open(DATA_DIR / "series_error_report.md", "w") as f:
    f.write("# Compass-Gait Long Time Series: Drift Error Report\n\n")
    f.write(f"- **Steps**: {N_STEPS:,}\n")
    f.write(f"- **Total time**: {N_STEPS * TAU:.1f} s\n")
    f.write(f"- **tau**: {TAU:.6f} s\n")
    f.write(f"- **IC**: {X0}\n")
    f.write(f"- **Guard crossings detected**: {n_crossings}\n\n")
    f.write("## Error statistics (L2 in embedding space)\n\n")
    f.write("| Percentile | Error |\n|---|---|\n")
    for p, v in zip(pct, percentiles):
        f.write(f"| {p}% | {v:.6e} |\n")
    f.write(f"\n**Mean**: {errors.mean():.6e}  \n")
    f.write(f"**Std**: {errors.std():.6e}  \n")

print("Saved error report")
print(f"\nDone. Mean drift error: {errors.mean():.4e}")
