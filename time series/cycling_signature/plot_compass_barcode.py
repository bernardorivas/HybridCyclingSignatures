"""
Plot the compass-gait cycling-signature sweep across variants.

Reads canonical sweep CSVs from data/compass_gait/ and writes a single
paper-facing comparison figure to figures/compass_gait/.

Usage:
    python "time series/cycling_signature/plot_compass_barcode.py"

Inputs (all in data/compass_gait/):
    barcode_H1_analytic.csv          baseline (parabolic eta, affine Psi)
    barcode_H1_analytic_tgt025.csv   best hand-tuned analytic
    barcode_H1_relaxed.csv           learned relaxed-space encoder, 5 impacts
    barcode_H1_relaxed_n20.csv       learned relaxed-space encoder, 20 impacts

Output:
    figures/compass_gait/fig_compass_cycling_signature.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "compass_gait")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "compass_gait")
os.makedirs(FIG_DIR, exist_ok=True)


def parse_sweep_header(path):
    """Extract the commented parameter-sweep rows as dict keyed by (boxsize, sb_radius)."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line.startswith("# "):
                continue
            payload = line[2:].strip()
            if payload.startswith("boxsize"):
                continue
            parts = payload.split(",")
            if len(parts) != 7:
                continue
            try:
                boxsize = float(parts[0])
                sb_radius = int(float(parts[1]))
                beta1 = int(parts[3])
                rank = int(parts[4])
                max_birth = parts[6]
                out[(boxsize, sb_radius)] = {
                    "beta1": beta1, "rank": rank, "max_birth": max_birth,
                }
            except ValueError:
                continue
    return out


variants = [
    ("Learned (20 impacts)",    "barcode_H1_relaxed_n20.csv",     "#009E73", "D"),
    ("Learned (5 impacts)",     "barcode_H1_relaxed.csv",         "#0072B2", "^"),
    ("Analytic tuned",          "barcode_H1_analytic_tgt025.csv", "#D55E00", "s"),
    ("Analytic baseline",       "barcode_H1_analytic.csv",        "#888888", "o"),
]

data = {}
for label, fname, color, marker in variants:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"[WARN] missing: {path}")
        continue
    data[label] = parse_sweep_header(path)
    print(f"loaded {label}: {len(data[label])} cells")

from matplotlib.ticker import FixedLocator, FixedFormatter

sb_radii = [1, 2, 4]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)

for i, sb in enumerate(sb_radii):
    ax = axes[i]
    for (label, _, color, marker) in variants:
        if label not in data:
            continue
        d = data[label]
        xs = sorted({bs for (bs, sb_) in d.keys() if sb_ == sb})
        ys = [d[(bs, sb)]["rank"] for bs in xs]
        ax.plot(xs, ys, label=label, color=color, marker=marker,
                markersize=7, linewidth=1.6)
    ax.axhline(1, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("cubical boxsize")
    ax.set_title(r"$\mathrm{sb\_radius} = " + str(sb) + r"$")
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.3]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.05", "0.10", "0.20", "0.30"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0", "1", "2", "5", "10", "20"])
    ax.grid(True, which="both", alpha=0.25)

axes[0].set_ylabel(r"cycling-signature rank ($H_1$)")
axes[0].legend(loc="upper right", fontsize=8, framealpha=0.95)

fig.suptitle(
    r"Compass gait: $H_1$ cycling-signature rank across cover parameters",
    fontsize=12,
)
fig.tight_layout()

out = os.path.join(FIG_DIR, "fig_compass_cycling_signature.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
