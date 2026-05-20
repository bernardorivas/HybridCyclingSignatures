"""Manuscript-grade cycling-signature figures for the rimless CHyLL v2 run.

This builds two figures from the random-subsegment summaries:

1. fixed-radius rank distributions, following the style of David et al. Fig. 10;
2. radius-sensitive rank heatmaps, following the style of David et al. Fig. 15.
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
from matplotlib.patches import Patch


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
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "legend.title_fontsize": 7.0,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "savefig.bbox": "tight",
}

RANK_COLORS = {
    0: "#111111",
    1: "#7B3294",
}


def read_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                out[key] = value
    return out


def read_heatmap(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.reader(path.open()))
    lengths = np.array([float(x) for x in rows[0][1:]], dtype=float)
    radii = np.array([float(row[0]) for row in rows[1:]], dtype=float)
    values = np.array([[float(x) for x in row[1:]] for row in rows[1:]], dtype=float)
    return radii, lengths, values


def read_rank_at_radius(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    lengths: list[float] = []
    ranks: dict[int, list[float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        rank_keys = [key for key in reader.fieldnames or [] if key.startswith("rank")]
        for key in rank_keys:
            ranks[int(key.removeprefix("rank"))] = []
        for row in reader:
            lengths.append(float(row["segment_length"]))
            for key in rank_keys:
                ranks[int(key.removeprefix("rank"))].append(float(row[key]))
    return np.array(lengths), {key: np.array(value) for key, value in ranks.items()}


def centers_to_edges(x: np.ndarray) -> np.ndarray:
    if len(x) == 1:
        dx = max(abs(x[0]), 1.0)
        return np.array([x[0] - dx / 2, x[0] + dx / 2])
    mids = 0.5 * (x[:-1] + x[1:])
    first = x[0] - (mids[0] - x[0])
    last = x[-1] + (x[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def paths(data_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "rank0": data_dir / f"{prefix}_rank_heatmap_rank0.csv",
        "rank1": data_dir / f"{prefix}_rank_heatmap_rank1.csv",
        "rank_at": data_dir / f"{prefix}_rank_at_radius.csv",
        "meta": data_dir / f"{prefix}_metadata.txt",
    }


def check_paths(*panels: dict[str, Path]) -> None:
    for panel in panels:
        for path in panel.values():
            if not path.exists():
                raise FileNotFoundError(path)


def row_label(ax, text: str) -> None:
    ax.text(
        -0.18,
        0.5,
        text,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.0,
        fontweight="bold",
    )


def plot_rank_distribution(
    data_dir: Path,
    out_dir: Path,
    diff_prefix: str,
    flow_prefix: str,
    out_base: str,
    radius_scale: float = 1e3,
    radius_scale_label: str = r"$10^{-3}$",
) -> None:
    plt.rcParams.update(STYLE)
    diff = paths(data_dir, diff_prefix)
    flow = paths(data_dir, flow_prefix)
    check_paths(diff, flow)

    panels = [
        ("orbit tangent lift", diff),
        ("flow tangent lift", flow),
    ]

    fig = plt.figure(figsize=(6.5, 2.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.18], height_ratios=[1, 1])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    ax_leg = fig.add_subplot(gs[:, 1])

    for i, (name, panel) in enumerate(panels):
        ax = axes[i]
        meta = read_metadata(panel["meta"])
        lengths, rank_counts = read_rank_at_radius(panel["rank_at"])
        n_runs = max(float(meta.get("n_runs", "1")), 1.0)
        eval_radius = float(meta.get("eval_radius", "nan"))
        bar_width = 0.82 * np.min(np.diff(lengths)) if len(lengths) > 1 else 8.0
        bottom = np.zeros_like(lengths)
        for rank in sorted(rank_counts):
            values = rank_counts[rank]
            ax.bar(
                lengths,
                values,
                bottom=bottom,
                width=bar_width,
                color=RANK_COLORS.get(rank, "#999999"),
                edgecolor="white",
                linewidth=0.18,
            )
            bottom = bottom + values
        ax.set_xlim(lengths.min() - bar_width, lengths.max() + bar_width)
        ax.set_ylim(0, n_runs)
        ax.set_ylabel("# segments")
        row_label(ax, name)
        ax.text(
            0.98,
            0.86,
            (
                rf"$r={eval_radius * radius_scale:.1f}\times$ {radius_scale_label}"
                if radius_scale_label
                else rf"$r={eval_radius * radius_scale:.3g}$"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.3,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("segment time span (samples)")

    ax_leg.axis("off")
    handles = [
        Patch(facecolor=RANK_COLORS[0], edgecolor="none", label="rank 0"),
        Patch(facecolor=RANK_COLORS[1], edgecolor="none", label="rank 1"),
    ]
    ax_leg.legend(handles=handles, loc="center", frameon=True, title="rank")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        path = out_dir / f"{out_base}.{ext}"
        fig.savefig(path, **kwargs)
        print(f"wrote {path}")
    plt.close(fig)


def plot_rank_heatmaps(
    data_dir: Path,
    out_dir: Path,
    diff_prefix: str,
    flow_prefix: str,
    out_base: str,
    radius_scale: float = 1e3,
    radius_scale_label: str = r"$10^{-3}$",
) -> None:
    plt.rcParams.update(STYLE)
    diff = paths(data_dir, diff_prefix)
    flow = paths(data_dir, flow_prefix)
    check_paths(diff, flow)

    panels = [
        ("orbit tangent lift", diff),
        ("flow tangent lift", flow),
    ]

    all_values = []
    for _, panel in panels:
        for rank_key in ("rank0", "rank1"):
            _, _, values = read_heatmap(panel[rank_key])
            all_values.append(values)
    vmax = max(float(v.max()) for v in all_values)
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0))

    fig = plt.figure(figsize=(6.5, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 0.075],
        height_ratios=[1, 1],
    )
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
    cax = fig.add_subplot(gs[:, 2])

    mesh = None
    for i, (name, panel) in enumerate(panels):
        meta = read_metadata(panel["meta"])
        for j, rank_key in enumerate(("rank0", "rank1")):
            ax = axes[i][j]
            radii, lengths, values = read_heatmap(panel[rank_key])
            radii_scaled = radii * radius_scale
            mesh = ax.pcolormesh(
                centers_to_edges(lengths),
                centers_to_edges(radii_scaled),
                values,
                cmap="viridis",
                norm=norm,
                shading="flat",
                rasterized=True,
            )
            ax.set_xlim(lengths.min(), lengths.max())
            ax.set_ylim(radii_scaled.min(), radii_scaled.max())
            if i == 0:
                ax.set_title(f"rank {j}")
            if j == 0:
                ax.set_ylabel(
                    f"filtration radius ({radius_scale_label})"
                    if radius_scale_label
                    else "filtration radius"
                )
                row_label(ax, name)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if i == 1:
                ax.set_xlabel("segment time span (samples)")
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            if j == 1:
                ax.text(
                    0.97,
                    0.88,
                    rf"$\beta_1(Y)={meta.get('beta1_Y', '?')}$",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.2,
                    color="white" if meta.get("beta1_Y", "0") != "0" else "black",
                    bbox={
                        "boxstyle": "round,pad=0.16",
                        "facecolor": "black" if meta.get("beta1_Y", "0") != "0" else "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                    },
                )

    if mesh is None:
        raise RuntimeError("No heatmap was drawn")
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("# segments")
    cb.outline.set_linewidth(0.55)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        path = out_dir / f"{out_base}.{ext}"
        fig.savefig(path, **kwargs)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--diff-prefix", default="subsegments_chyll_v2_diff_zoom100")
    parser.add_argument("--flow-prefix", default="subsegments_chyll_v2_vfield_zoom100")
    parser.add_argument("--rank-out-base", default="fig_rimless_signature_rank_distribution")
    parser.add_argument("--heatmap-out-base", default="fig_rimless_signature_rank_heatmaps")
    parser.add_argument(
        "--radius-scale", type=float, default=1e3,
        help=(
            "Multiplicative factor applied to the y-axis radius values. "
            "Default 1000 prints radii in units of 10^-3 (rimless scale). "
            "Pass 1 to print the raw value (bouncing-ball scale)."
        ),
    )
    parser.add_argument(
        "--radius-scale-label", type=str, default=r"$10^{-3}$",
        help=(
            "Unit label appended to the radius axis title and annotation. "
            "Pass an empty string for no unit when --radius-scale 1."
        ),
    )
    args = parser.parse_args()

    plot_rank_distribution(
        args.data_dir,
        args.out_dir,
        args.diff_prefix,
        args.flow_prefix,
        args.rank_out_base,
        radius_scale=args.radius_scale,
        radius_scale_label=args.radius_scale_label,
    )
    plot_rank_heatmaps(
        args.data_dir,
        args.out_dir,
        args.diff_prefix,
        args.flow_prefix,
        args.heatmap_out_base,
        radius_scale=args.radius_scale,
        radius_scale_label=args.radius_scale_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
