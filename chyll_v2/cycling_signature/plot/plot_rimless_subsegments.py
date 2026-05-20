"""Publication-style pilot figure for CHyLL v2 rimless subsegments.

Reads the CSV summaries produced by
``chyll_v2/cycling_signature/julia/run_subsegments.jl`` and writes a compact
two-column comparison:

  orbit tangent lift  vs.  flow tangent lift

Each column shows a rank-one frequency heatmap over segment span and
thickening radius, followed by a rank-count summary at the evaluation radius.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = (
    REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "rimless_wheel"
)
DEFAULT_FIG_DIR = (
    REPO_ROOT / "chyll_v2" / "cycling_signature" / "figures" / "rimless_wheel"
)


STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "savefig.bbox": "tight",
}


def read_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key] = value
    return out


def read_heatmap(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.reader(path.open()))
    lengths = np.array([float(x) for x in rows[0][1:]], dtype=float)
    radii = np.array([float(row[0]) for row in rows[1:]], dtype=float)
    values = np.array(
        [[float(x) for x in row[1:]] for row in rows[1:]],
        dtype=float,
    )
    return radii, lengths, values


def read_rank_at_radius(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths, rank0, rank1 = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths.append(float(row["segment_length"]))
            rank0.append(float(row["rank0"]))
            rank1.append(float(row["rank1"]))
    return np.array(lengths), np.array(rank0), np.array(rank1)


def centers_to_edges(x: np.ndarray) -> np.ndarray:
    if len(x) == 1:
        dx = max(abs(x[0]), 1.0)
        return np.array([x[0] - dx / 2, x[0] + dx / 2])
    mids = 0.5 * (x[:-1] + x[1:])
    first = x[0] - (mids[0] - x[0])
    last = x[-1] + (x[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def panel_paths(data_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "heat": data_dir / f"{prefix}_rank_heatmap_rank1.csv",
        "rank": data_dir / f"{prefix}_rank_at_radius.csv",
        "meta": data_dir / f"{prefix}_metadata.txt",
    }


def plot_heatmap(ax, paths: dict[str, Path], title: str, norm: Normalize):
    radii, lengths, values = read_heatmap(paths["heat"])
    meta = read_metadata(paths["meta"])
    mesh = ax.pcolormesh(
        centers_to_edges(lengths),
        centers_to_edges(radii),
        values,
        cmap="viridis",
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_ylabel("thickening radius")
    ax.set_xlim(lengths.min(), lengths.max())
    ax.set_ylim(radii.min(), radii.max())
    ax.text(
        0.03,
        0.94,
        rf"$\beta_1(Y)={meta.get('beta1_Y', '?')}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )
    return mesh


def plot_rank_counts(ax, paths: dict[str, Path], show_ylabel: bool):
    lengths, rank0, rank1 = read_rank_at_radius(paths["rank"])
    meta = read_metadata(paths["meta"])
    n_runs = max(float(meta.get("n_runs", "1")), 1.0)
    bar_width = 0.72 * np.min(np.diff(lengths)) if len(lengths) > 1 else 10.0

    ax.bar(
        lengths,
        rank0,
        width=bar_width,
        color="#D9D9D9",
        edgecolor="#666666",
        linewidth=0.35,
        label="rank 0",
    )
    ax.bar(
        lengths,
        rank1,
        bottom=rank0,
        width=bar_width,
        color="#2CA25F",
        edgecolor="#1B7837",
        linewidth=0.35,
        label="rank 1",
    )
    ax.set_xlim(lengths.min() - bar_width, lengths.max() + bar_width)
    ax.set_ylim(0, n_runs)
    ax.set_xlabel("segment time span (samples)")
    if show_ylabel:
        ax.set_ylabel("# sampled segments")
    ax.text(
        0.97,
        0.92,
        rf"$r={float(meta.get('eval_radius', 'nan')):.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def build_figure(data_dir: Path, out_dir: Path, diff_prefix: str, vfield_prefix: str, out_base: str):
    plt.rcParams.update(STYLE)
    diff = panel_paths(data_dir, diff_prefix)
    vfield = panel_paths(data_dir, vfield_prefix)
    for paths in (diff, vfield):
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)

    all_values = []
    for paths in (diff, vfield):
        _, _, values = read_heatmap(paths["heat"])
        all_values.append(values)
    vmax = max(float(v.max()) for v in all_values)
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0))

    fig = plt.figure(figsize=(7.2, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.12],
        height_ratios=[1.0, 0.58],
    )

    ax_h0 = fig.add_subplot(gs[0, 0])
    ax_h1 = fig.add_subplot(gs[0, 1], sharey=ax_h0)
    ax_b0 = fig.add_subplot(gs[1, 0])
    ax_b1 = fig.add_subplot(gs[1, 1], sharey=ax_b0)
    cax = fig.add_subplot(gs[0, 2])
    ax_leg = fig.add_subplot(gs[1, 2])

    mesh = plot_heatmap(ax_h0, diff, "orbit tangent lift", norm)
    plot_heatmap(ax_h1, vfield, "flow tangent lift", norm)
    ax_h1.set_ylabel("")
    plt.setp(ax_h1.get_yticklabels(), visible=False)

    plot_rank_counts(ax_b0, diff, show_ylabel=True)
    plot_rank_counts(ax_b1, vfield, show_ylabel=False)
    plt.setp(ax_b1.get_yticklabels(), visible=False)
    ax_b1.set_ylabel("")
    handles, labels = ax_b1.get_legend_handles_labels()
    ax_leg.axis("off")
    ax_leg.legend(handles, labels, loc="center", frameon=False, handlelength=1.8)

    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("# rank-1 segments")
    cb.outline.set_linewidth(0.6)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{out_base}.pdf"
    png = out_dir / f"{out_base}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--diff-prefix", default="subsegments_chyll_v2_diff_pilot")
    parser.add_argument("--vfield-prefix", default="subsegments_chyll_v2_vfield_pilot")
    parser.add_argument("--out-base", default="fig_rimless_chyll_v2_subsegments_pilot")
    args = parser.parse_args()

    build_figure(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        diff_prefix=args.diff_prefix,
        vfield_prefix=args.vfield_prefix,
        out_base=args.out_base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
