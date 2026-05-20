"""Within-piece tangent-coherence diagnostics for rimless CHyLL v2 lifts.

The cycling-signature export concatenates base-flow arcs and bridge interiors.
This script reconstructs those pieces and measures how quickly the exported
unit tangents rotate inside each piece.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from chyll_v2.cycling_signature.export.prepare_rimless_lift import (  # noqa: E402
    LIMIT_CYCLE_IC,
    load_config,
    simulate_limit_cycle,
)
from chyll_v2.chyll_v2.systems.rimless_wheel import RimlessWheel  # noqa: E402


DEFAULT_DATA_DIR = (
    REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "rimless_wheel"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "rimless_wheel"
)
DEFAULT_BASES = (
    "continuous_lift_chyll_v2_phaseB",
    "continuous_lift_chyll_v2_phaseB_vfield",
)


def reconstruct_piece_slices(
    *,
    config_path: Path,
    n_impacts: int,
    n_s: int,
    max_time: float,
    max_step: float,
) -> list[tuple[str, int, int, int]]:
    cfg = load_config(config_path)
    system = RimlessWheel()
    segments, jump_pairs = simulate_limit_cycle(
        system=system,
        x0=LIMIT_CYCLE_IC,
        n_impacts=n_impacts,
        max_time=max_time,
        max_step=max_step,
        rtol=cfg.sim_rtol,
        atol=cfg.sim_atol,
    )

    pieces: list[tuple[str, int, int, int]] = []
    start = 0
    for j, seg in enumerate(segments):
        end = start + len(seg)
        pieces.append(("arc", j, start, end))
        start = end
        if j < len(jump_pairs):
            end = start + n_s
            pieces.append(("bridge", j, start, end))
            start = end
    return pieces


def angles_inside_piece(
    tangents: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    if end - start < 2:
        return np.empty(0, dtype=float)
    a = tangents[start:end - 1]
    b = tangents[start + 1:end]
    dots = np.sum(a * b, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def summarize(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "count": 0,
            "median": np.nan,
            "mean": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "frac_gt_5": np.nan,
            "frac_gt_10": np.nan,
            "frac_gt_20": np.nan,
            "frac_gt_45": np.nan,
        }
    return {
        "count": float(len(values)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "frac_gt_5": float(np.mean(values > 5.0)),
        "frac_gt_10": float(np.mean(values > 10.0)),
        "frac_gt_20": float(np.mean(values > 20.0)),
        "frac_gt_45": float(np.mean(values > 45.0)),
    }


def histogram_counts(values: np.ndarray, bins: np.ndarray) -> list[int]:
    counts, _ = np.histogram(values, bins=bins)
    return [int(x) for x in counts]


def fmt(x: float) -> str:
    if np.isnan(x):
        return "nan"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def analyze_base(
    data_dir: Path,
    base: str,
    pieces: list[tuple[str, int, int, int]],
) -> tuple[list[dict[str, object]], list[str]]:
    path = data_dir / f"{base}_tangents.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    tangents = np.loadtxt(path)
    expected_n = max(end for _, _, _, end in pieces)
    if len(tangents) != expected_n:
        raise ValueError(f"{path}: expected {expected_n} tangents, found {len(tangents)}")

    rows: list[dict[str, object]] = []
    lines = [f"{base}", "-" * len(base)]
    by_type: dict[str, list[np.ndarray]] = defaultdict(list)
    all_angles = []

    for piece_type, piece_index, start, end in pieces:
        angles = angles_inside_piece(tangents, start, end)
        by_type[piece_type].append(angles)
        all_angles.append(angles)
        stats = summarize(angles)
        rows.append(
            {
                "base": base,
                "piece_type": piece_type,
                "piece_index": piece_index,
                "start": start,
                "end": end,
                **stats,
            }
        )

    for label, arrays in [
        ("all", all_angles),
        ("arc", by_type["arc"]),
        ("bridge", by_type["bridge"]),
    ]:
        values = np.concatenate(arrays) if arrays else np.empty(0, dtype=float)
        stats = summarize(values)
        rows.append(
            {
                "base": base,
                "piece_type": label,
                "piece_index": "aggregate",
                "start": "",
                "end": "",
                **stats,
            }
        )
        lines.append(
            f"  {label:6s} n={int(stats['count']):4d} "
            f"median={fmt(stats['median'])} deg "
            f"p95={fmt(stats['p95'])} deg "
            f"p99={fmt(stats['p99'])} deg "
            f"max={fmt(stats['max'])} deg "
            f">10={fmt(100 * stats['frac_gt_10'])}% "
            f">20={fmt(100 * stats['frac_gt_20'])}%"
        )

    bins = np.array([0, 1, 2, 5, 10, 20, 45, 90, 180], dtype=float)
    all_values = np.concatenate(all_angles) if all_angles else np.empty(0, dtype=float)
    counts = histogram_counts(all_values, bins)
    labels = [f"[{bins[i]:g},{bins[i + 1]:g})" for i in range(len(bins) - 1)]
    lines.append("  histogram: " + ", ".join(f"{lab}={cnt}" for lab, cnt in zip(labels, counts)))

    boundary_by_type: dict[str, list[float]] = defaultdict(list)
    for left, right in zip(pieces[:-1], pieces[1:]):
        left_type, left_index, _left_start, left_end = left
        right_type, right_index, right_start, _right_end = right
        dot = float(np.dot(tangents[left_end - 1], tangents[right_start]))
        angle = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
        boundary_label = f"{left_type}->{right_type}"
        boundary_by_type[boundary_label].append(angle)
        rows.append(
            {
                "base": base,
                "piece_type": f"boundary:{boundary_label}",
                "piece_index": f"{left_index}->{right_index}",
                "start": left_end - 1,
                "end": right_start,
                **summarize(np.array([angle], dtype=float)),
            }
        )

    for boundary_label, values_list in sorted(boundary_by_type.items()):
        values = np.array(values_list, dtype=float)
        stats = summarize(values)
        lines.append(
            f"  boundary {boundary_label:12s} n={int(stats['count']):2d} "
            f"median={fmt(stats['median'])} deg "
            f"p95={fmt(stats['p95'])} deg "
            f"max={fmt(stats['max'])} deg"
        )
    lines.append("")
    return rows, lines


def write_stats_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "base",
        "piece_type",
        "piece_index",
        "start",
        "end",
        "count",
        "median",
        "mean",
        "p90",
        "p95",
        "p99",
        "max",
        "frac_gt_5",
        "frac_gt_10",
        "frac_gt_20",
        "frac_gt_45",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "chyll_v2" / "runs" / "rimless_wheel_phaseB_finetune" / "config.json",
    )
    parser.add_argument("--n-impacts", type=int, default=5)
    parser.add_argument("--n-s", type=int, default=50)
    parser.add_argument("--max-time", type=float, default=50.0)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--bases", nargs="+", default=list(DEFAULT_BASES))
    parser.add_argument("--out-prefix", default="tangent_coherence")
    args = parser.parse_args()

    pieces = reconstruct_piece_slices(
        config_path=args.config,
        n_impacts=args.n_impacts,
        n_s=args.n_s,
        max_time=args.max_time,
        max_step=args.max_step,
    )
    rows: list[dict[str, object]] = []
    report_lines = [
        "Within-piece tangent-coherence diagnostic",
        "=" * 43,
        f"pieces: {len(pieces)}",
        f"samples: {max(end for _, _, _, end in pieces)}",
        "angle = arccos(<v_i, v_{i+1}>) in degrees, same piece only",
        "",
    ]

    for base in args.bases:
        base_rows, base_lines = analyze_base(args.data_dir, base, pieces)
        rows.extend(base_rows)
        report_lines.extend(base_lines)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    txt = args.out_dir / f"{args.out_prefix}.txt"
    csv_path = args.out_dir / f"{args.out_prefix}.csv"
    txt.write_text("\n".join(report_lines) + "\n")
    write_stats_csv(csv_path, rows)
    print(txt)
    print(csv_path)
    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
