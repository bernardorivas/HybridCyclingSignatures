"""Plot beta_1(Y) of the UTB comparison space as a function of (boxsize, sb_radius).

Reads the CSV output of ``julia/sweep_beta1_comparison_space.jl`` and produces
a stacked-row plot, one row per lift, with one line per sb_radius. Highlights
the plateau where beta_1(Y) = 1 (the genuine periodic-orbit signature) and
flags the transient mid-scale region where beta_1(Y) > 1 (Vietoris-Rips
artifacts before all sub-loops fuse).

Usage:
    python chyll_v2/cycling_signature/plot/plot_beta1_comparison_space_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "chyll_v2" / "cycling_signature" / "data"
FIG_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "figures"


def load_sweep(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["sb_radius", "boxsize"]).reset_index(drop=True)
    return df


PANELS = [
    {
        "csv": DATA / "rimless_wheel" / "subsegments_chyll_v2_rimless_phaseB_beta1_sweep.csv",
        "title": "Rimless wheel (Phase-B, 1500 samples)",
        "xlim": (0.04, 0.55),
    },
    {
        "csv": DATA / "bouncing_ball" / "subsegments_chyll_v2_bb_phaseB_beta1_sweep.csv",
        "title": "Bouncing ball Phase-B (5 impacts, 1069 samples)",
        "xlim": (0.08, 1.05),
    },
    {
        "csv": DATA / "bouncing_ball" / "subsegments_chyll_v2_bb_phaseB_long_beta1_sweep.csv",
        "title": "Bouncing ball Phase-B (15 impacts, 2004 samples)",
        "xlim": (0.04, 0.55),
    },
]


def main() -> int:
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(7, 2.4 * len(PANELS)), sharey=False)
    for ax, panel in zip(axes, PANELS):
        df = load_sweep(panel["csv"])
        for sb, group in df.groupby("sb_radius"):
            ax.plot(
                group["boxsize"], group["beta1_Y"],
                marker="o", lw=1.2, label=f"sb_radius = {sb}",
            )
        ax.axhline(1, color="C7", ls=":", lw=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlim(panel["xlim"])
        ax.set_yticks([0, 1, 2, 3, 5, 8, 13, 21])
        ax.set_ylabel(r"$\beta_1(Y)$")
        ax.set_title(panel["title"], fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("boxsize (cubical-complex cell size)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "fig_beta1_comparison_space_sweep.pdf"
    out_png = FIG_DIR / "fig_beta1_comparison_space_sweep.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
