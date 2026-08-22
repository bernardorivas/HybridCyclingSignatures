#!/usr/bin/env python3
"""Apply a preregistered period-8 branch-separation resolution criterion."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSTIMPACT = (
    CODE_ROOT
    / "chyll_v2"
    / "compass_analysis"
    / "compass_gait_cascade"
    / "phi_3_latent_postimpact_z.npy"
)
DEFAULT_HEATMAP = (
    CODE_ROOT
    / "period_doubling"
    / "data"
    / "compass_gait_latent"
    / "signatures"
    / "subsegments_compass_period8_rank_heatmap_rank1.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "period8_resolution"
    / "criterion.json"
)


def branch_statistics(
    points: np.ndarray,
    n_branches: int,
    burn_impacts: int,
) -> dict[str, object]:
    points = points[burn_impacts:]
    if len(points) < 4 * n_branches:
        raise ValueError("too few post-impact points for branch statistics")
    groups = [points[index::n_branches] for index in range(n_branches)]
    centroids = np.vstack([group.mean(axis=0) for group in groups])
    centroid_distances = []
    for i in range(n_branches):
        for j in range(i + 1, n_branches):
            centroid_distances.append(np.linalg.norm(centroids[i] - centroids[j]))
    within_p95 = [
        float(np.percentile(np.linalg.norm(group - center, axis=1), 95))
        for group, center in zip(groups, centroids)
    ]
    minimum = float(min(centroid_distances))
    noise_allowance = 2.0 * max(within_p95)
    return {
        "n_postimpact_points": int(len(points)),
        "n_branches": n_branches,
        "minimum_centroid_separation": minimum,
        "within_branch_p95": within_p95,
        "noise_allowance": noise_allowance,
        "effective_separation": max(0.0, minimum - noise_allowance),
    }


def closing_radius_floor(
    path: Path,
    success_fraction: float,
    n_runs: int,
) -> dict[str, float | int]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"empty rank heatmap: {path}")
    values = np.asarray([[float(item) for item in row] for row in rows[1:]])
    radii = values[:, 0]
    counts = values[:, 1:]
    if np.any(counts < 0) or np.any(counts > n_runs):
        raise ValueError(
            f"rank-1 counts must lie between zero and --n-runs={n_runs}"
        )
    threshold = success_fraction * n_runs
    qualifying = radii[np.max(counts, axis=1) >= threshold]
    if len(qualifying) == 0:
        raise ValueError("no radius reaches the requested closing success fraction")
    return {
        "radius_floor": float(qualifying.min()),
        "n_runs": n_runs,
        "success_fraction": success_fraction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latent-postimpact",
        type=Path,
        default=DEFAULT_POSTIMPACT,
        help=(
            "learned encoding of reference-simulator post-impact states; "
            "do not substitute sampled post-bridge arc starts"
        ),
    )
    parser.add_argument("--rank1-heatmap", type=Path, default=DEFAULT_HEATMAP)
    parser.add_argument("--n-branches", type=int, default=8)
    parser.add_argument("--burn-impacts", type=int, default=16)
    parser.add_argument("--success-fraction", type=float, default=0.9)
    parser.add_argument(
        "--n-runs",
        type=int,
        default=150,
        help="subsegment trials per heatmap cell; must match its metadata",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.success_fraction <= 1.0:
        raise ValueError("--success-fraction must lie in (0, 1]")
    if args.n_runs < 1:
        raise ValueError("--n-runs must be positive")
    postimpact = np.load(args.latent_postimpact, allow_pickle=False)
    if postimpact.ndim != 2 or postimpact.shape[1] < 2:
        raise ValueError(
            "--latent-postimpact must be a two-dimensional encoded-state array"
        )
    if not np.all(np.isfinite(postimpact)):
        raise ValueError("--latent-postimpact contains non-finite values")
    branch = branch_statistics(postimpact, args.n_branches, args.burn_impacts)
    radius = closing_radius_floor(
        args.rank1_heatmap, args.success_fraction, args.n_runs
    )

    # Conservative criterion: even after a two-sided within-branch noise
    # allowance, branch separation must exceed the smallest reliably closing
    # signature radius.  Equality is not counted as resolved.
    resolved = branch["effective_separation"] > radius["radius_floor"]
    result = {
        "criterion": (
            "effective branch separation (minimum centroid separation minus "
            "twice the largest within-branch p95 radius) must be strictly "
            "greater than the empirical closing-radius floor"
        ),
        "latent_postimpact": str(args.latent_postimpact.resolve()),
        "rank1_heatmap": str(args.rank1_heatmap.resolve()),
        **branch,
        **radius,
        "resolved": bool(resolved),
        "report_label": "resolved" if resolved else "partially resolved",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
