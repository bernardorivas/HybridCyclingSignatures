#!/usr/bin/env python3
"""
Matplotlib reproductions of Figures 10, 11, 13, 14, 15 from the cycling-signatures paper.

Reads cycling-signature CSVs from Julia output and generates publication-quality plots:
- Fig 10: Rank distribution (stacked bar chart)
- Fig 11: Rank-1 cycling spaces (stacked line plot)
- Fig 13: Rank-2 cycling spaces (stacked line plot)
- Fig 14: Rank-1 to Rank-2 inclusion graphs (bipartite)
- Fig 15: Rank heatmaps grid

Data layout: {data_root}/roessler/signatures/ with files like
  subsegments_roessler_{regime}_rank_at_radius.csv
  subsegments_roessler_{regime}_rank1_spaces_at_radius.csv
  subsegments_roessler_{regime}_rank2_spaces_at_radius.csv
  subsegments_roessler_{regime}_s21_inclusion.csv
  subsegments_roessler_{regime}_rank_heatmap_rank{k}.csv
  roessler_{regime}.npz (for dt extraction)
  subsegments_roessler_{regime}_metadata.txt (for stride)

Segment lengths are in post-stride samples; convert to time spans via
  tau = length * dt * stride

Usage:
  python plot_signatures.py
  python plot_signatures.py --data-root /path/to/data --fig-dir /path/to/figures
  python plot_signatures.py --eval-radius-suffix r0p8
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

REGIMES = ["period1", "period2", "period4", "period8", "chaos"]
RANK_COLORS = {
    0: "#d3d3d3",  # light gray
    1: "#1f77b4",  # tab:blue
    2: "#ff7f0e",  # tab:orange
    3: "#d62728",  # tab:red
}
# Extended colors for rank 4+: use more reds/magentas
RANK_COLORS.update({i: plt.cm.Reds(0.4 + 0.5 * (i - 4) / 3) for i in range(4, 8)})

# matplotlib.tab10 cycled for space colors (V1..V6, W1..W6)
SPACE_COLORS_TAB10 = plt.cm.tab10(np.arange(10))
# Create extended palette by cycling
SPACE_COLORS = {i: SPACE_COLORS_TAB10[i % 10] for i in range(20)}


# ============================================================================
# Utility Functions
# ============================================================================

def load_json_from_npz(npz_path, key="meta_json"):
    """Extract and parse JSON string from numpy npz file."""
    data = np.load(npz_path, allow_pickle=True)
    meta_json = data[key]
    # Handle numpy 0-d array containing string
    if isinstance(meta_json, np.ndarray):
        meta_json = meta_json.item()
    if isinstance(meta_json, bytes):
        meta_json = meta_json.decode("utf-8")
    return json.loads(meta_json)


def read_stride_from_metadata(metadata_path):
    """Read stride from metadata.txt file. Default 1 if absent."""
    if not os.path.exists(metadata_path):
        return 1
    try:
        with open(metadata_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("stride="):
                    return int(line.split("=")[1])
    except Exception:
        pass
    return 1


def load_rank_at_radius_csv(csv_path):
    """Load rank_at_radius.csv as dict with lists.

    Returns:
        dict with keys: 'segment_length' (list), 'rank0' (list), 'rank1' (list), ...
    """
    if not os.path.exists(csv_path):
        return None

    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for key in reader.fieldnames:
            data[key] = []
        for row in reader:
            for key, val in row.items():
                try:
                    data[key].append(int(val))
                except ValueError:
                    data[key].append(float(val))

    return data


def load_rank_spaces_csv(csv_path):
    """Load rank{k}_spaces_at_radius.csv.

    Returns:
        spaces: list of space indices (1-indexed)
        counts: dict {space_idx: [counts per segment_length]}
    """
    if not os.path.exists(csv_path):
        return None, None

    spaces = []
    counts = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        segment_lengths = [col for col in reader.fieldnames if col not in ("space_index", "space_matrix")]
        for row in reader:
            space_id = int(row["space_index"])
            spaces.append(space_id)
            counts[space_id] = [int(row[col]) for col in segment_lengths]
    return spaces, counts


def load_inclusion_csv(csv_path):
    """Load s21_inclusion.csv.

    Returns:
        dict: {rank1_idx: [list of rank2_indices_where_included]}
    """
    if not os.path.exists(csv_path):
        return None

    inclusion = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rank2_cols = [col for col in reader.fieldnames if col not in ("rank1_index", "rank1_space")]
        for row in reader:
            r1_idx = int(row["rank1_index"])
            included_in = [int(col) for col in rank2_cols if int(row[col]) == 1]
            inclusion[r1_idx] = included_in
    return inclusion


def load_rank_heatmap_csv(csv_path):
    """Load rank_heatmap_rank{k}.csv.

    Returns:
        radii: list of filtration radii
        segment_lengths: list of segment lengths
        counts: (n_radii, n_lengths) matrix
    """
    if not os.path.exists(csv_path):
        return None, None, None

    radii = []
    counts_list = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        segment_lengths_cols = [col for col in reader.fieldnames if col != "radius"]
        segment_lengths = [int(col) for col in segment_lengths_cols]

        for row in reader:
            radii.append(float(row["radius"]))
            counts_list.append([int(row[col]) for col in segment_lengths_cols])

    counts = np.array(counts_list)
    return radii, segment_lengths, counts


# ============================================================================
# Figure Generation Functions
# ============================================================================

def fig_10_rank_stacked(regimes_data, time_spans_dict, out_path, dpi=200):
    """
    Figure 10: Rank distribution (stacked bar chart).

    One row of 5 panels (period1..chaos), each a stacked bar plot with:
    x = time span, bar segments = fraction/count of segments with each rank
    """
    n_regimes = len(regimes_data)
    fig, axes = plt.subplots(1, n_regimes, figsize=(15, 3.5), sharey=True)
    if n_regimes == 1:
        axes = [axes]

    for ax, regime, data_dict in zip(axes, REGIMES, regimes_data):
        if data_dict is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(regime)
            continue

        rank_data = data_dict["rank_at_radius"]
        if rank_data is None:
            ax.text(0.5, 0.5, "No rank data", ha="center", va="center")
            ax.set_title(regime)
            continue

        segment_lengths = np.array(rank_data["segment_length"])
        time_spans = time_spans_dict.get(regime, segment_lengths)

        # Extract rank columns (rank0, rank1, rank2, ...)
        rank_cols = [col for col in rank_data.keys() if col.startswith("rank")]
        rank_indices = sorted([int(col[4:]) for col in rank_cols])

        # Build stacked bar data
        bottom = np.zeros(len(segment_lengths))
        for rank in rank_indices:
            col_name = f"rank{rank}"
            if col_name in rank_data:
                values = np.array(rank_data[col_name])
                color = RANK_COLORS.get(rank, RANK_COLORS[3])
                width = np.mean(np.diff(time_spans)) * 0.8 if len(time_spans) > 1 else 5
                ax.bar(time_spans, values, bottom=bottom, label=f"rank {rank}",
                       color=color, width=width)
                bottom += values

        ax.set_xlabel("segment time span")
        ax.set_ylabel("#segments" if ax == axes[0] else "")
        ax.set_title(regime)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.15, 0.5))
    fig.suptitle("Distribution of cycling ranks", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def fig_11_sig1_stacked(regimes_data, time_spans_dict, out_path, dpi=200):
    """
    Figure 11: Rank-1 cycling spaces (stacked line plot).

    One row of 5 panels, per panel stacked bars of counts of each distinct
    rank-1 cycling space vs time span (one color per space index, consistent).
    Labels: V1, V2, ...
    """
    n_regimes = len(regimes_data)
    fig, axes = plt.subplots(1, n_regimes, figsize=(15, 3.5), sharey=True)
    if n_regimes == 1:
        axes = [axes]

    all_space_labels = {}  # Track which label (V1, V2, ...) each space_idx gets
    label_counter = [1]  # Counter for generating V1, V2, ...

    for ax, regime, data_dict in zip(axes, REGIMES, regimes_data):
        if data_dict is None or data_dict["rank1_spaces"] == (None, None):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(regime)
            continue

        spaces, counts_dict = data_dict["rank1_spaces"]
        if spaces is None:
            ax.text(0.5, 0.5, "No rank-1 spaces", ha="center", va="center")
            ax.set_title(regime)
            continue

        rank_data = data_dict["rank_at_radius"]
        if rank_data is not None:
            segment_lengths = np.array(rank_data["segment_length"])
            time_spans = time_spans_dict.get(regime, segment_lengths)
        else:
            # Infer from counts_dict
            segment_lengths = np.arange(len(next(iter(counts_dict.values()))))
            time_spans = segment_lengths

        # Assign labels V1, V2, ... to space indices
        for space_idx in spaces:
            if space_idx not in all_space_labels:
                all_space_labels[space_idx] = f"V{label_counter[0]}"
                label_counter[0] += 1

        # Plot stacked bars for each space
        bottom = np.zeros(len(segment_lengths))
        for space_idx in spaces:
            label = all_space_labels[space_idx]
            values = np.array(counts_dict[space_idx])
            color = SPACE_COLORS[space_idx % 10]
            width = np.mean(np.diff(time_spans)) * 0.8 if len(time_spans) > 1 else 5
            ax.bar(time_spans, values, bottom=bottom, label=label, color=color, width=width)
            bottom += values

        ax.set_xlabel("segment time span")
        ax.set_ylabel("#segments" if ax == axes[0] else "")
        ax.set_title(regime)

    # Legend for all spaces found
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.15, 0.5),
                   title="1d cycling space")
    fig.suptitle("Distribution of rank-1 segments", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def fig_13_sig2_stacked(regimes_data, time_spans_dict, out_path, dpi=200):
    """
    Figure 13: Rank-2 cycling spaces (stacked line plot).

    Same layout as Fig 11, but for rank-2 spaces with labels W1, W2, ...
    """
    n_regimes = len(regimes_data)
    fig, axes = plt.subplots(1, n_regimes, figsize=(15, 3.5), sharey=True)
    if n_regimes == 1:
        axes = [axes]

    all_space_labels = {}
    label_counter = [1]

    for ax, regime, data_dict in zip(axes, REGIMES, regimes_data):
        if data_dict is None or data_dict["rank2_spaces"] == (None, None):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(regime)
            continue

        spaces, counts_dict = data_dict["rank2_spaces"]
        if spaces is None:
            ax.text(0.5, 0.5, "No rank-2 spaces", ha="center", va="center")
            ax.set_title(regime)
            continue

        rank_data = data_dict["rank_at_radius"]
        if rank_data is not None:
            segment_lengths = np.array(rank_data["segment_length"])
            time_spans = time_spans_dict.get(regime, segment_lengths)
        else:
            segment_lengths = np.arange(len(next(iter(counts_dict.values()))))
            time_spans = segment_lengths

        for space_idx in spaces:
            if space_idx not in all_space_labels:
                all_space_labels[space_idx] = f"W{label_counter[0]}"
                label_counter[0] += 1

        bottom = np.zeros(len(segment_lengths))
        for space_idx in spaces:
            label = all_space_labels[space_idx]
            values = np.array(counts_dict[space_idx])
            color = SPACE_COLORS[space_idx % 10]
            width = np.mean(np.diff(time_spans)) * 0.8 if len(time_spans) > 1 else 5
            ax.bar(time_spans, values, bottom=bottom, label=label, color=color, width=width)
            bottom += values

        ax.set_xlabel("segment time span")
        ax.set_ylabel("#segments" if ax == axes[0] else "")
        ax.set_title(regime)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.15, 0.5),
                   title="2d cycling space")
    fig.suptitle("Distribution of rank-2 segments", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def fig_14_inclusion_graphs(regimes_data, out_path, dpi=200):
    """
    Figure 14: Inclusion graphs (bipartite).

    One row of 5 panels; per panel a bipartite graph:
    - Bottom row: frequent rank-1 spaces (labeled V1, V2, ...)
    - Top row: frequent rank-2 spaces (labeled W1, W2, ...)
    - Edges where inclusion CSV has 1
    - Nodes as circles with labels, no axes
    - If no rank-2 spaces exist, print 'no rank-2 segments' centered
    """
    n_regimes = len(regimes_data)
    fig, axes = plt.subplots(1, n_regimes, figsize=(15, 4), sharey=False)
    if n_regimes == 1:
        axes = [axes]

    v_labels = {}  # {space_idx: "V1", ...}
    w_labels = {}  # {space_idx: "W1", ...}
    v_counter = [1]
    w_counter = [1]

    for ax, regime, data_dict in zip(axes, REGIMES, regimes_data):
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 2.5)
        ax.axis("off")
        ax.set_aspect("equal")

        if (data_dict is None or
            data_dict["rank1_spaces"][0] is None or
            data_dict["rank2_spaces"][0] is None or
            data_dict["inclusion"] is None):
            ax.text(3, 1, "no rank-2 segments", ha="center", va="center", fontsize=10)
            ax.set_title(regime)
            continue

        inclusion = data_dict["inclusion"]
        rank1_spaces = data_dict["rank1_spaces"][0]
        rank2_spaces = data_dict["rank2_spaces"][0]

        if not inclusion or not rank2_spaces:
            ax.text(3, 1, "no rank-2 segments", ha="center", va="center", fontsize=10)
            ax.set_title(regime)
            continue

        # Assign labels
        for space_idx in rank1_spaces:
            if space_idx not in v_labels:
                v_labels[space_idx] = f"V{v_counter[0]}"
                v_counter[0] += 1

        for space_idx in rank2_spaces:
            if space_idx not in w_labels:
                w_labels[space_idx] = f"W{w_counter[0]}"
                w_counter[0] += 1

        # Position nodes: V's at y=0.5, W's at y=1.5
        n_v = len(rank1_spaces)
        n_w = len(rank2_spaces)
        v_x = {space_idx: i for i, space_idx in enumerate(rank1_spaces)}
        w_x = {space_idx: i for i, space_idx in enumerate(rank2_spaces)}

        # Draw edges
        for v_idx, w_indices in inclusion.items():
            if v_idx not in v_x:
                continue
            x1 = v_x[v_idx]
            for w_idx in w_indices:
                if w_idx in w_x:
                    x2 = w_x[w_idx]
                    ax.plot([x1, x2], [0.5, 1.5], "k-", alpha=0.3, zorder=1)

        # Draw V nodes (bottom)
        for space_idx, label in v_labels.items():
            if space_idx in v_x:
                x = v_x[space_idx]
                color = SPACE_COLORS[space_idx % 10]
                circle = mpatches.Circle((x, 0.5), 0.2, color=color, ec="black", linewidth=1.5, zorder=3)
                ax.add_patch(circle)
                ax.text(x, 0.5, label, ha="center", va="center", fontsize=9, weight="bold", zorder=4)

        # Draw W nodes (top)
        for space_idx, label in w_labels.items():
            if space_idx in w_x:
                x = w_x[space_idx]
                color = SPACE_COLORS[space_idx % 10]
                circle = mpatches.Circle((x, 1.5), 0.2, color=color, ec="black", linewidth=1.5, zorder=3)
                ax.add_patch(circle)
                ax.text(x, 1.5, label, ha="center", va="center", fontsize=9, weight="bold", zorder=4)

        ax.set_title(regime)

    fig.suptitle("Inclusion graphs (rank-1 to rank-2)", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def fig_15_rank_heatmaps(regimes_data, time_spans_dict, out_path, dpi=200):
    """
    Figure 15: Rank heatmaps grid.

    Grid: rows = 5 regimes, cols = ranks 0..2 (or max available)
    Per cell: heatmap of counts, x = time span, y = radius, imshow origin lower
    Shared colormap (viridis), per-row y in SAME radius scale
    One shared colorbar

    time_spans_dict maps regime -> time spans matching the regime's
    segment-length list; used to convert heatmap x from samples to time.
    """
    n_regimes = len(REGIMES)
    max_rank = 3

    fig, axes = plt.subplots(n_regimes, max_rank, figsize=(12, 14))

    # Determine global vmin/vmax for shared colorbar
    all_values = []
    for data_dict in regimes_data:
        if data_dict is not None:
            for rank in range(max_rank):
                heatmap_data = data_dict.get(f"rank{rank}_heatmap")
                if heatmap_data is not None:
                    radii, segment_lengths, counts = heatmap_data
                    all_values.extend(counts.flatten())

    if all_values:
        vmin, vmax = np.percentile(all_values, [0, 95])  # 95th percentile for better contrast
    else:
        vmin, vmax = 0, 1

    for i, (regime, data_dict) in enumerate(zip(REGIMES, regimes_data)):
        for rank in range(max_rank):
            ax = axes[i, rank]

            if data_dict is None:
                ax.text(0.5, 0.5, "Missing data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{regime} rank-{rank}")
                continue

            heatmap_data = data_dict.get(f"rank{rank}_heatmap")
            if heatmap_data is None:
                ax.text(0.5, 0.5, f"No rank-{rank}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{regime} rank-{rank}")
                continue

            radii, segment_lengths, counts = heatmap_data

            # Convert x from post-stride sample counts to time spans using
            # the same dt*stride factor as the other figures
            spans = time_spans_dict.get(regime)
            if spans is not None and len(spans) > 0:
                factor = spans[-1] / segment_lengths[-1]
            else:
                factor = 1.0

            # Create heatmap (origin=lower means radii increase upward)
            im = ax.imshow(counts, aspect="auto", cmap="viridis", origin="lower",
                          extent=[min(segment_lengths) * factor,
                                 max(segment_lengths) * factor,
                                 min(radii), max(radii)],
                          vmin=vmin, vmax=vmax, interpolation="nearest")

            ax.set_xlabel("segment time span" if i == n_regimes - 1 else "")
            ax.set_ylabel("filtration radius" if rank == 0 else "")
            ax.set_title(f"{regime} rank-{rank}", fontsize=10)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("count", rotation=270, labelpad=15)

    fig.suptitle("Cycling rank distributions by radius", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def load_all_data(data_root, prefix, eval_radius_suffix=""):
    """Load all CSV and heatmap data for all regimes.

    Returns:
        list of dicts, one per regime, with keys:
        - rank_at_radius: dict or None
        - rank1_spaces: (spaces, counts_dict)
        - rank2_spaces: (spaces, counts_dict)
        - inclusion: dict or None
        - rank0_heatmap, rank1_heatmap, rank2_heatmap: (radii, lengths, counts)
    """
    regimes_data = []
    time_spans_dict = {}

    for regime in REGIMES:
        data_dict = {}

        # Build file paths
        base_prefix = f"subsegments_{prefix}_{regime}"
        if eval_radius_suffix:
            # Accept both "r0p8" (as the driver writes it) and bare "0p8".
            s = eval_radius_suffix
            suffix = "_" + s if s.startswith("r") else "_r" + s
        else:
            suffix = ""

        rank_at_radius_path = os.path.join(data_root, f"{base_prefix}_rank_at_radius{suffix}.csv")
        rank1_spaces_path = os.path.join(data_root, f"{base_prefix}_rank1_spaces_at_radius{suffix}.csv")
        rank2_spaces_path = os.path.join(data_root, f"{base_prefix}_rank2_spaces_at_radius{suffix}.csv")
        inclusion_path = os.path.join(data_root, f"{base_prefix}_s21_inclusion{suffix}.csv")
        # npz lifts live one level above the signatures/ subdirectory
        npz_path = os.path.join(os.path.dirname(data_root), f"{prefix}_{regime}.npz")
        metadata_path = os.path.join(data_root, f"{base_prefix}_metadata.txt")

        # Load rank_at_radius
        rank_dict = load_rank_at_radius_csv(rank_at_radius_path)
        if rank_dict is None:
            print(f"Warning: Missing {rank_at_radius_path}, skipping {regime}")
            regimes_data.append(None)
            continue

        data_dict["rank_at_radius"] = rank_dict

        # Load rank1 and rank2 spaces
        data_dict["rank1_spaces"] = load_rank_spaces_csv(rank1_spaces_path)
        data_dict["rank2_spaces"] = load_rank_spaces_csv(rank2_spaces_path)
        data_dict["inclusion"] = load_inclusion_csv(inclusion_path)

        # Load heatmaps for ranks 0, 1, 2 (heatmaps span the full radius
        # grid and are written unsuffixed regardless of eval radius)
        for rank in range(3):
            hmap_path = os.path.join(data_root, f"{base_prefix}_rank_heatmap_rank{rank}.csv")
            hmap_data = load_rank_heatmap_csv(hmap_path)
            data_dict[f"rank{rank}_heatmap"] = hmap_data

        # Get dt and stride to compute time spans
        dt = 0.02  # Default
        stride = 1  # Default
        try:
            meta = load_json_from_npz(npz_path)
            dt = meta.get("dt", dt)
        except Exception as e:
            print(f"Warning: Could not read dt from {npz_path}: {e}")

        stride = read_stride_from_metadata(metadata_path)

        # Compute time spans from segment_length
        segment_lengths = np.array(rank_dict["segment_length"])
        time_spans = segment_lengths * dt * stride
        time_spans_dict[regime] = time_spans

        regimes_data.append(data_dict)

    return regimes_data, time_spans_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate cycling-signature matplotlib reproductions (Figs 10, 11, 13, 14, 15)"
    )
    parser.add_argument(
        "--data-root",
        default="/Users/bdoprad/Work/Projects/hybrid-cycling-signatures/code/period_doubling/data",
        help="Root data directory containing roessler/ subdirectory",
    )
    parser.add_argument(
        "--fig-dir",
        default="/Users/bdoprad/Work/Projects/hybrid-cycling-signatures/code/period_doubling/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--eval-radius-suffix",
        default="",
        help="Suffix for multi-radius variants (e.g., 'r0p8' for _r0p8 files)",
    )
    args = parser.parse_args()

    data_root = os.path.join(args.data_root, "roessler", "signatures")
    fig_dir = args.fig_dir

    os.makedirs(fig_dir, exist_ok=True)

    # Check if data directory exists
    if not os.path.isdir(data_root):
        print(f"Warning: Data directory not found: {data_root}")
        print("This is expected if cycling signatures have not been computed yet.")
        return

    print(f"Loading signatures from {data_root}")

    # Use "roessler" as the prefix
    regimes_data, time_spans_dict = load_all_data(data_root, "roessler", args.eval_radius_suffix)

    # Check if we got any data
    if all(d is None for d in regimes_data):
        print("Error: No valid regime data loaded. Exiting.")
        return

    # Generate figures
    if args.eval_radius_suffix:
        s = args.eval_radius_suffix
        suffix = "_" + s if s.startswith("r") else "_r" + s
    else:
        suffix = ""

    fig10_path = os.path.join(fig_dir, f"roessler_fig10_rank_stacked{suffix}.png")
    fig_10_rank_stacked(regimes_data, time_spans_dict, fig10_path)

    fig11_path = os.path.join(fig_dir, f"roessler_fig11_sig1_stacked{suffix}.png")
    fig_11_sig1_stacked(regimes_data, time_spans_dict, fig11_path)

    fig13_path = os.path.join(fig_dir, f"roessler_fig13_sig2_stacked{suffix}.png")
    fig_13_sig2_stacked(regimes_data, time_spans_dict, fig13_path)

    fig14_path = os.path.join(fig_dir, f"roessler_fig14_inclusion{suffix}.png")
    fig_14_inclusion_graphs(regimes_data, fig14_path)

    fig15_path = os.path.join(fig_dir, f"roessler_fig15_rank_heatmaps{suffix}.png")
    fig_15_rank_heatmaps(regimes_data, time_spans_dict, fig15_path)

    print(f"All figures generated in {fig_dir}")


if __name__ == "__main__":
    main()
