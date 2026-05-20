"""Analyze the compass-gait period-doubling cascade in CHyLL latent space.

Primary output: post-impact latent cluster counts for the four Goswami slopes.
The script simulates the exact compass-gait return map, encodes each
post-impact state with the trained CHyLL v2 encoder, clusters the resulting
latent section points, and plots the section point cloud.

It also reports a lightweight section-crossing count: the number of
post-impact crossings before the encoded return sequence closes. This is the
Python-only "winding-number pairing" diagnostic; plain beta_1 is intentionally
not used here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chyll_v2.chyll_v2.config import CHyLLv2Config, make_default  # noqa: E402
from chyll_v2.chyll_v2.networks import CHyLLv2Networks  # noqa: E402
from chyll_v2.chyll_v2.systems.compass_gait import CompassGait  # noqa: E402
from chyll_v2.chyll_v2.systems.compass_gait_slope_configs import (  # noqa: E402
    GOSWAMI_COMPASS_SLOPE_CONFIGS,
    CompassGaitSlopeConfig,
)

try:  # noqa: E402
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("This analysis script requires torch.") from exc


RUNS = {
    "phi_1": REPO_ROOT / "chyll_v2" / "runs" / "compass_gait_phi_1_4.75deg",
    "phi_2": REPO_ROOT / "chyll_v2" / "runs" / "compass_gait_phi_2_5deg",
    "phi_3": REPO_ROOT / "chyll_v2" / "runs" / "compass_gait_phi_3_5.02deg",
    "phi_4_cloud": (
        REPO_ROOT / "chyll_v2" / "runs" / "compass_gait_phi_4_cloud_5.2deg"
    ),
}


def load_config(path: Path) -> CHyLLv2Config:
    if not path.exists():
        return make_default("compass_gait")
    payload = json.loads(path.read_text())
    allowed = {f.name for f in fields(CHyLLv2Config)}
    payload = {k: v for k, v in payload.items() if k in allowed}
    cfg = CHyLLv2Config(**payload)
    cfg.device = "cpu"
    return cfg


def load_networks(run_dir: Path) -> tuple[CHyLLv2Config, CHyLLv2Networks]:
    cfg = load_config(run_dir / "config.json")
    nets = CHyLLv2Networks(cfg)
    try:
        state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(run_dir / "model.pt", map_location="cpu")
    nets.load_state_dict(state)
    nets.eval()
    return cfg, nets


def first_ic(slope: CompassGaitSlopeConfig) -> np.ndarray:
    if slope.fixed_points:
        return np.array(slope.fixed_points[0], dtype=np.float64)
    if slope.sampling_cloud_path:
        cloud = np.load(REPO_ROOT / slope.sampling_cloud_path)
        return np.array(cloud[0], dtype=np.float64)
    if slope.sampling_box is not None:
        lo = np.array(slope.sampling_box[0], dtype=np.float64)
        hi = np.array(slope.sampling_box[1], dtype=np.float64)
        return 0.5 * (lo + hi)
    raise ValueError(f"slope config {slope.label!r} has no IC source")


def return_map_step(system: CompassGait, x: np.ndarray) -> np.ndarray:
    sol = system._simulate_base_to_guard(x, (0.0, 10.0), rtol=1e-10, atol=1e-12)
    if sol.status != 1:
        raise RuntimeError(f"no guard hit from x={x}; status={sol.status}")
    if len(sol.y_events[0]):
        x_minus = sol.y_events[0][0]
    else:
        x_minus = sol.y[:, -1]
    return system.reset_map(x_minus)


def simulate_post_impacts(
    slope: CompassGaitSlopeConfig,
    *,
    n_burn: int,
    n_keep: int,
) -> np.ndarray:
    system = CompassGait(phi=slope.phi)
    x = first_ic(slope)
    out: list[np.ndarray] = []
    for k in range(n_burn + n_keep):
        x = return_map_step(system, x)
        if k >= n_burn:
            out.append(x.copy())
    return np.asarray(out, dtype=np.float64)


def encode_post_impacts(nets: CHyLLv2Networks, posts: np.ndarray) -> np.ndarray:
    aug = np.hstack([posts, np.zeros((len(posts), 1), dtype=np.float64)])
    with torch.no_grad():
        z = nets.encoder(torch.as_tensor(aug, dtype=torch.float32)).cpu().numpy()
    return z


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def dbscan_labels(dist: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = dist.shape[0]
    labels = np.full(n, -99, dtype=int)
    cluster_id = 0
    neighbors = [np.flatnonzero(dist[i] <= eps) for i in range(n)]

    for i in range(n):
        if labels[i] != -99:
            continue
        if len(neighbors[i]) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue: deque[int] = deque(int(j) for j in neighbors[i] if j != i)
        while queue:
            j = queue.popleft()
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -99:
                continue
            labels[j] = cluster_id
            if len(neighbors[j]) >= min_samples:
                queue.extend(int(k) for k in neighbors[j] if labels[k] in (-99, -1))
        cluster_id += 1
    labels[labels == -99] = -1
    return labels


def cluster_summary(z: np.ndarray, eps_fraction: float, min_samples: int):
    dist = pairwise_distances(z)
    diameter = float(dist.max())
    eps = eps_fraction * diameter
    labels = dbscan_labels(dist, eps, min_samples=min_samples)
    clusters = sorted(c for c in set(labels.tolist()) if c >= 0)
    counts = {c: int(np.sum(labels == c)) for c in clusters}
    noise_count = int(np.sum(labels < 0))
    return labels, diameter, eps, counts, noise_count


def pca2(z: np.ndarray) -> np.ndarray:
    centered = z - z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def section_crossing_count(labels: np.ndarray) -> str:
    good = labels[labels >= 0]
    if len(good) < 2:
        return "many"
    unique = sorted(set(good.tolist()))
    if len(unique) > 16:
        return "many"

    seq = good.tolist()
    for period in range(1, min(16, len(seq) // 2) + 1):
        ok = all(seq[i] == seq[i % period] for i in range(len(seq)))
        if ok:
            return str(period)
    return "many"


def plot_clusters(
    out_path: Path,
    z: np.ndarray,
    labels: np.ndarray,
    title: str,
    expected: int | None,
    cluster_count: int,
) -> None:
    xy = pca2(z)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    clusters = sorted(c for c in set(labels.tolist()) if c >= 0)
    cmap = plt.get_cmap("tab10")
    for idx, c in enumerate(clusters):
        mask = labels == c
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=16,
            color=cmap(idx % 10),
            alpha=0.85,
            label=f"C{c} ({int(mask.sum())})",
        )
    if np.any(labels < 0):
        mask = labels < 0
        ax.scatter(xy[mask, 0], xy[mask, 1], s=12, color="0.65", label="noise")
    expected_label = "chaos" if expected is None else str(expected)
    ax.set_title(f"{title}: {cluster_count} clusters, expected {expected_label}")
    ax.set_xlabel("latent PCA 1")
    ax.set_ylabel("latent PCA 2")
    if len(clusters) <= 10:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_panel(rows: Iterable[dict], out_path: Path) -> None:
    rows = list(rows)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    for ax, row in zip(axes.ravel(), rows):
        xy = row["xy"]
        labels = row["labels"]
        clusters = sorted(c for c in set(labels.tolist()) if c >= 0)
        cmap = plt.get_cmap("tab10")
        for idx, c in enumerate(clusters):
            mask = labels == c
            ax.scatter(xy[mask, 0], xy[mask, 1], s=9, color=cmap(idx % 10), alpha=0.8)
        if np.any(labels < 0):
            mask = labels < 0
            ax.scatter(xy[mask, 0], xy[mask, 1], s=8, color="0.65", alpha=0.7)
        ax.set_title(
            f"{row['label']} ({row['phi_deg']:g} deg): {row['cluster_display']}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.18)
    fig.suptitle("Post-impact latent section clusters", y=0.98)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "chyll_v2" / "figures" / "compass_gait_cascade",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "chyll_v2" / "compass_analysis" / "compass_gait_cascade",
    )
    parser.add_argument("--n-burn", type=int, default=32)
    parser.add_argument("--n-keep", type=int, default=256)
    parser.add_argument(
        "--eps-fraction",
        type=float,
        default=0.01,
        help=(
            "DBSCAN section tolerance as a fraction of latent diameter. "
            "The default 1 percent resolves the period-8 slope while still "
            "classifying the chaotic section as many local components."
        ),
    )
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict] = []
    panel_rows: list[dict] = []
    for label, run_dir in RUNS.items():
        slope = GOSWAMI_COMPASS_SLOPE_CONFIGS[label]
        cfg, nets = load_networks(run_dir)
        posts = simulate_post_impacts(slope, n_burn=args.n_burn, n_keep=args.n_keep)
        z = encode_post_impacts(nets, posts)
        labels, diameter, eps, counts, noise = cluster_summary(
            z,
            eps_fraction=args.eps_fraction,
            min_samples=args.min_samples,
        )
        cluster_count = len(counts)
        display = "many" if slope.expected_period is None and cluster_count > 16 else str(cluster_count)
        crossing = section_crossing_count(labels)

        base = f"{label}_latent_postimpact"
        np.save(args.data_dir / f"{base}_states.npy", posts)
        np.save(args.data_dir / f"{base}_z.npy", z)
        np.save(args.data_dir / f"{base}_labels.npy", labels)

        plot_clusters(
            args.out_dir / f"{base}_clusters.png",
            z,
            labels,
            title=f"{label} phi={slope.phi_deg:g} deg",
            expected=slope.expected_period,
            cluster_count=cluster_count,
        )
        table_rows.append(
            {
                "label": label,
                "phi_deg": slope.phi_deg,
                "phi_rad": slope.phi,
                "expected_period": slope.expected_period or "chaos",
                "latent_diameter": diameter,
                "dbscan_eps": eps,
                "cluster_count": cluster_count,
                "cluster_display": display,
                "noise_count": noise,
                "section_crossings_per_loop": crossing,
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            }
        )
        panel_rows.append(
            {
                "label": label,
                "phi_deg": slope.phi_deg,
                "labels": labels,
                "xy": pca2(z),
                "cluster_display": display,
            }
        )
        print(
            f"{label}: phi={slope.phi_deg:g} deg clusters={display} "
            f"noise={noise} section_crossings={crossing} eps={eps:.4g}"
        )

    plot_panel(panel_rows, args.out_dir / "postimpact_latent_clusters_panel.png")

    csv_path = args.data_dir / "compass_cascade_latent_cluster_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    md_path = args.data_dir / "compass_cascade_latent_cluster_table.md"
    lines = [
        "| label | phi (deg) | expected | latent clusters | section crossings | noise |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['label']} | {r['phi_deg']:.2f} | {r['expected_period']} | "
            f"{r['cluster_display']} | {r['section_crossings_per_loop']} | "
            f"{r['noise_count']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {args.out_dir / 'postimpact_latent_clusters_panel.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
