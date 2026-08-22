"""
NON-RUNNABLE with the current repository artifacts.

Manuscript figure: compass cycling-signature rank heatmaps. The renderer
unconditionally requires all three inputs listed below, including the analytic
baseline that the compass pipeline does not generate and the repository does
not store. See pipelines/compass_gait_relaxed/README.md.

Small-multiple heatmaps of the cycling-signature rank over the discrete
cover-parameter grid (cubical cover scale, sphere-bundle cover radius).

Main figure (2 panels):
    left    Learned, 5 impacts
    right   Learned, 20 impacts

Appendix figure (3 panels):
    main panels + Analytic reference

"Learned" here refers to the relaxed-space embedding E_theta trained on the
compass relaxed semiflow; see pipelines/compass_gait_relaxed/SPEC.md.

Inputs (under data/compass_gait/):
    barcode_H1_relaxed.csv
    barcode_H1_relaxed_n20.csv
    barcode_H1_analytic_tgt025.csv

Outputs (under figures/compass_gait/):
    fig_compass_cycling_rank_heatmap.pdf           (main, 2 panels)
    fig_compass_cycling_rank_heatmap.png
    fig_compass_cycling_rank_heatmap_appendix.pdf  (3 panels, with analytic)
    fig_compass_cycling_rank_heatmap_appendix.png
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
    """Return dict (boxsize, sb_radius) -> rank from the commented CSV header."""
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


learned_panels = [
    ("Learned, 5 impacts",  "barcode_H1_relaxed.csv"),
    ("Learned, 20 impacts", "barcode_H1_relaxed_n20.csv"),
]
analytic_panel = ("Analytic reference", "barcode_H1_analytic_tgt025.csv")

boxsizes = [0.30, 0.20, 0.10, 0.05]   # coarse -> fine, left to right
sb_radii = [1, 2, 4]                  # ascending; plotted with origin='lower'

# Discrete colormap.
# rank 0        : gray    (cover contractible, no cycling signature)
# rank 1        : green   (expected dominant loop)
# rank 2..4     : light yellow (mild over-counting)
# rank 5..9     : orange  (notable over-counting)
# rank >= 10    : red     (heavy over-counting)
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


def build_figure(panels, out_base, figsize, wspace):
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]
    for k, (ax, (label, fname)) in enumerate(zip(axes, panels)):
        _plot_panel(ax, label, os.path.join(DATA_DIR, fname), show_ylabel=(k == 0))
    fig.subplots_adjust(wspace=wspace)
    pdf = os.path.join(FIG_DIR, out_base + ".pdf")
    png = os.path.join(FIG_DIR, out_base + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    plt.close(fig)


# Main figure: 2 learned-embedding panels.
build_figure(
    learned_panels,
    "fig_compass_cycling_rank_heatmap",
    figsize=(6.8, 2.9),
    wspace=0.22,
)

# Appendix / supplementary figure: learned panels + analytic reference.
build_figure(
    learned_panels + [analytic_panel],
    "fig_compass_cycling_rank_heatmap_appendix",
    figsize=(10.0, 2.9),
    wspace=0.24,
)
