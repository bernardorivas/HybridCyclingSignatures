"""
Experimental heatmap figure for the embed9 variant (latent dim d = 9).

Same 2-panel layout and discrete colormap as the baseline plotter, but reads
the embed9 barcode CSVs and writes to experimental output paths. Baseline
figure files are not touched.

Inputs (under data/compass_gait/):
    barcode_H1_relaxed_embed9.csv       (5-impact)
    barcode_H1_relaxed_embed9_n20.csv   (20-impact)

Outputs (under figures/compass_gait/):
    fig_compass_cycling_rank_heatmap_embed9.pdf
    fig_compass_cycling_rank_heatmap_embed9.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.titlepad": 5.0,
    "axes.labelpad": 3.0,
})

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "compass_gait")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "compass_gait")
os.makedirs(FIG_DIR, exist_ok=True)


def parse_sweep(path):
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
                rank = int(parts[4])
                out[(boxsize, sb_radius)] = rank
            except ValueError:
                continue
    return out


panels = [
    ("Learned, 5 impacts  (d = 9)",  "barcode_H1_relaxed_embed9.csv"),
    ("Learned, 20 impacts  (d = 9)", "barcode_H1_relaxed_embed9_n20.csv"),
]

boxsizes = [0.30, 0.20, 0.10, 0.05]
sb_radii = [1, 2, 4]

bounds = [-0.5, 0.5, 1.5, 4.5, 9.5, 1000.5]
cmap = ListedColormap(["#E5E5E5", "#2CA25F", "#FFE08A", "#F58C3B", "#B83030"])
norm = BoundaryNorm(bounds, cmap.N)

XLABEL = "cubical cover scale"
YLABEL = "sphere-bundle cover radius"


def _text_color(rank):
    return "white" if rank >= 5 else "black"


def _plot_panel(ax, label, data_path, show_ylabel):
    d = parse_sweep(data_path)
    Z = np.full((len(sb_radii), len(boxsizes)), -1, dtype=int)
    for i, sb in enumerate(sb_radii):
        for j, bs in enumerate(boxsizes):
            Z[i, j] = d.get((bs, sb), -1)

    ax.imshow(Z, cmap=cmap, norm=norm, aspect="equal", origin="lower")
    for i in range(len(sb_radii)):
        for j in range(len(boxsizes)):
            ax.text(j, i, str(int(Z[i, j])), ha="center", va="center",
                    color=_text_color(int(Z[i, j])), fontsize=9.5)

    ax.set_xticks(range(len(boxsizes)))
    ax.set_xticklabels([f"{b:.2f}" for b in boxsizes])
    ax.set_yticks(range(len(sb_radii)))
    ax.set_yticklabels([str(sb) for sb in sb_radii])
    ax.set_xlabel(XLABEL)
    if show_ylabel:
        ax.set_ylabel(YLABEL)
    ax.set_title(label)
    ax.tick_params(bottom=False, left=False, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
for k, (ax, (label, fname)) in enumerate(zip(axes, panels)):
    _plot_panel(ax, label, os.path.join(DATA_DIR, fname), show_ylabel=(k == 0))
fig.subplots_adjust(wspace=0.22)

pdf = os.path.join(FIG_DIR, "fig_compass_cycling_rank_heatmap_embed9.pdf")
png = os.path.join(FIG_DIR, "fig_compass_cycling_rank_heatmap_embed9.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"wrote {pdf}")
print(f"wrote {png}")
