"""Encode the generated Goswami trajectories with trained CHyLL-v2 models.

The physical CSVs contain continuous compass-gait arcs and explicit pre/post
impact samples.  This exporter inserts the mapping-cylinder bridge at every
impact, sends the resulting augmented states ``(x, s)`` through the learned
encoder, and evaluates the learned continuous latent vector field.

The two space-delimited files ``*_positions.csv`` and ``*_tangents.csv`` are
direct inputs to the existing Julia cycling-signature scripts.  Additional
NumPy arrays, metadata, manifests, and diagnostics retain the information
needed for Conley-style downstream work.

Run from the repository root with the scientific environment:

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl-compass \
      ~/.venvs/sci/bin/python \
      scripts/goswami_period_doubling/export_learned_latents.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chyll_v2.chyll_v2.networks import CHyLLv2Networks  # noqa: E402
from chyll_v2.cycling_signature.export.prepare_rimless_lift import (  # noqa: E402
    load_config,
    load_state_dict,
    model_call,
    normalize_tangent_rows,
)

try:  # noqa: E402
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required. Use ~/.venvs/sci/bin/python for this repository."
    ) from exc


CASE_INFO = OrderedDict(
    [
        (
            "period1",
            {
                "expected_period": 1,
                "phi_deg": 4.00,
                "run_dir": "chyll_v2/runs/compass_gait_phi007",
                "model_note": (
                    "Nearby nominal period-1 model trained at phi=0.07 rad "
                    "(4.010704565 deg); evaluated on the 4.00 deg trajectory."
                ),
            },
        ),
        (
            "period2",
            {
                "expected_period": 2,
                "phi_deg": 4.75,
                "run_dir": "chyll_v2/runs/compass_gait_phi_1_4.75deg",
                "model_note": "Slope-matched period-2 model.",
            },
        ),
        (
            "period4",
            {
                "expected_period": 4,
                "phi_deg": 5.00,
                "run_dir": "chyll_v2/runs/compass_gait_phi_2_5deg",
                "model_note": "Slope-matched period-4 model.",
            },
        ),
        (
            "period8",
            {
                "expected_period": 8,
                "phi_deg": 5.02,
                "run_dir": "chyll_v2/runs/compass_gait_phi_3_5.02deg",
                "model_note": "Slope-matched period-8 model.",
            },
        ),
        (
            "chaos",
            {
                "expected_period": None,
                "phi_deg": 5.20,
                "run_dir": "chyll_v2/runs/compass_gait_phi_4_cloud_5.2deg",
                "model_note": "Slope-matched chaotic-cloud model.",
            },
        ),
    ]
)


def read_timeseries(path: Path) -> OrderedDict[int, dict[str, np.ndarray]]:
    """Read physical trajectory CSV and preserve its stepwise arc structure."""
    columns: dict[int, dict[str, list]] = OrderedDict()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            item = columns.setdefault(step, {"t": [], "x": [], "event": []})
            item["t"].append(float(row["t"]))
            item["x"].append(
                [
                    float(row["theta_ns"]),
                    float(row["theta_s"]),
                    float(row["dtheta_ns"]),
                    float(row["dtheta_s"]),
                ]
            )
            item["event"].append(row["event"])

    arcs: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
    for step, item in columns.items():
        t = np.asarray(item["t"], dtype=np.float64)
        x = np.asarray(item["x"], dtype=np.float64)
        if len(t) < 2 or item["event"][0] != "post_impact":
            raise ValueError(f"malformed arc step {step} in {path}")
        if item["event"][-1] != "pre_impact":
            raise ValueError(f"arc step {step} has no pre-impact endpoint")
        if np.any(np.diff(t) < 0):
            raise ValueError(f"nonmonotone time in arc step {step}")
        arcs[step] = {"t": t, "x": x}
    return arcs


def read_returns(path: Path) -> dict[int, dict[str, np.ndarray | float]]:
    """Read post-impact states; keys use the one-based impact/step number."""
    out: dict[int, dict[str, np.ndarray | float]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            out[step] = {
                "time": float(row["impact_time"]),
                "period": float(row["step_period"]),
                "x": np.array(
                    [
                        float(row["theta_ns"]),
                        float(row["theta_s"]),
                        float(row["dtheta_ns"]),
                        float(row["dtheta_s"]),
                    ],
                    dtype=np.float64,
                ),
            }
    return out


def build_relaxed_input(
    arcs: OrderedDict[int, dict[str, np.ndarray]],
    returns: dict[int, dict[str, np.ndarray | float]],
    n_bridge: int,
) -> tuple[np.ndarray, list[dict], list[tuple[np.ndarray, np.ndarray]]]:
    """Insert unit-time mapping-cylinder bridges into the physical curve."""
    pieces: list[np.ndarray] = []
    metadata: list[dict] = []
    jump_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    steps = list(arcs)

    for position, step in enumerate(steps):
        arc = arcs[step]
        impact_number = step + 1
        if impact_number not in returns:
            raise ValueError(f"missing return row {impact_number}")

        t = arc["t"]
        x = arc["x"]
        x_arc = np.hstack([x, np.zeros((len(x), 1), dtype=np.float64)])
        pieces.append(x_arc)
        for j, physical_t in enumerate(t):
            event = "flow"
            if j == 0:
                event = "post_impact"
            elif j == len(t) - 1:
                event = "pre_impact"
            metadata.append(
                {
                    "relaxed_time": float(physical_t + step),
                    "physical_time": float(physical_t),
                    "piece": "arc",
                    "step": step,
                    "impact": impact_number,
                    "s": 0.0,
                    "event": event,
                }
            )

        x_minus = x[-1].copy()
        x_plus = np.asarray(returns[impact_number]["x"], dtype=np.float64)
        jump_pairs.append((x_minus, x_plus))
        physical_t = float(returns[impact_number]["time"])
        s_values = np.linspace(0.0, 1.0, n_bridge + 1, dtype=np.float64)[1:]
        x_bridge = np.hstack(
            [np.tile(x_minus, (len(s_values), 1)), s_values[:, None]]
        )
        pieces.append(x_bridge)
        for s in s_values:
            metadata.append(
                {
                    "relaxed_time": physical_t + step + float(s),
                    "physical_time": physical_t,
                    "piece": "bridge",
                    "step": step,
                    "impact": impact_number,
                    "s": float(s),
                    "event": "bridge_end" if np.isclose(s, 1.0) else "bridge",
                }
            )

        # The next arc supplies the reset state for intermediate impacts.  Add
        # the last reset explicitly so the exported ordered curve does not end
        # on the guard-side representative of the quotient seam.
        if position == len(steps) - 1:
            pieces.append(np.r_[x_plus, 0.0][None, :])
            metadata.append(
                {
                    "relaxed_time": physical_t + step + 1.0,
                    "physical_time": physical_t,
                    "piece": "arc",
                    "step": step + 1,
                    "impact": impact_number,
                    "s": 0.0,
                    "event": "terminal_post_impact",
                }
            )

    x_aug = np.vstack(pieces)
    if len(x_aug) != len(metadata):
        raise AssertionError("augmented-state/metadata length mismatch")
    return x_aug, metadata, jump_pairs


def load_network(run_dir: Path):
    cfg = load_config(run_dir / "config.json")
    cfg.device = "cpu"
    nets = CHyLLv2Networks(cfg)
    nets.load_state_dict(load_state_dict(run_dir / "model.pt"))
    nets.eval()
    return cfg, nets


def tag_aware_differences(
    z: np.ndarray, metadata: list[dict], eps: float = 1e-12
) -> tuple[np.ndarray, int]:
    """Finite-difference tangents that never create cross-piece chords."""
    dz = np.zeros_like(z)
    start = 0
    while start < len(z):
        key = (metadata[start]["piece"], metadata[start]["step"])
        end = start + 1
        while end < len(z) and (
            metadata[end]["piece"], metadata[end]["step"]
        ) == key:
            end += 1
        if end - start >= 2:
            local = np.diff(z[start:end], axis=0)
            dz[start : end - 1] = local
            dz[end - 1] = local[-1]
        start = end
    return normalize_tangent_rows(dz, eps, "tag-aware latent differences")


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def summarize_alignment(
    tangent_a: np.ndarray,
    tangent_b: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Summarize signed cosines between two arrays of unit tangents."""
    cosines = np.sum(tangent_a[mask] * tangent_b[mask], axis=1)
    return {
        **summarize(cosines),
        "fraction_positive": float(np.mean(cosines > 0.0)),
    }


def pca(z: np.ndarray, n_components: int = 3) -> np.ndarray:
    centered = z - z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:n_components].T


def plot_latent(
    path: Path, label: str, z: np.ndarray, metadata: list[dict], phi_deg: float
) -> None:
    xyz = pca(z, 3)
    arc = np.array([m["piece"] == "arc" for m in metadata], dtype=bool)
    bridge = ~arc
    color = np.array([m["relaxed_time"] for m in metadata])

    fig = plt.figure(figsize=(10.5, 4.6), constrained_layout=True)
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.scatter(xyz[arc, 0], xyz[arc, 1], c=color[arc], cmap="viridis",
                s=4, alpha=0.8, rasterized=True)
    ax2.scatter(xyz[bridge, 0], xyz[bridge, 1], color="#D55E00",
                s=5, alpha=0.65, label="learned bridge", rasterized=True)
    ax2.set(xlabel="latent PCA 1", ylabel="latent PCA 2")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.22)

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#0072B2",
             lw=0.7, alpha=0.75)
    ax3.scatter(xyz[bridge, 0], xyz[bridge, 1], xyz[bridge, 2],
                color="#D55E00", s=3, alpha=0.5)
    ax3.set(xlabel="PCA 1", ylabel="PCA 2", zlabel="PCA 3")
    fig.suptitle(f"{label}: learned continuous latent trajectory "
                 f"($\\phi={phi_deg:.2f}^\\circ$)", fontsize=14)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def write_metadata(path: Path, metadata: list[dict]) -> None:
    fields = ["index", "relaxed_time", "physical_time", "piece", "step",
              "impact", "s", "event"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, item in enumerate(metadata):
            writer.writerow({"index": index, **item})


def write_successor_edges(path: Path, metadata: list[dict]) -> np.ndarray:
    """Write the ordered directed graph used by set-oriented workflows."""
    edges = np.column_stack(
        [np.arange(len(metadata) - 1), np.arange(1, len(metadata))]
    ).astype(np.int64)
    fields = [
        "source", "target", "delta_relaxed_time", "delta_physical_time",
        "transition",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source, target in edges:
            left = metadata[source]
            right = metadata[target]
            if left["piece"] == right["piece"]:
                transition = left["piece"]
            elif left["piece"] == "arc":
                transition = "enter_bridge"
            else:
                transition = "quotient_seam"
            writer.writerow(
                {
                    "source": source,
                    "target": target,
                    "delta_relaxed_time": (
                        right["relaxed_time"] - left["relaxed_time"]
                    ),
                    "delta_physical_time": (
                        right["physical_time"] - left["physical_time"]
                    ),
                    "transition": transition,
                }
            )
    return edges


def export_case(
    label: str,
    info: dict,
    input_dir: Path,
    out_dir: Path,
    n_bridge: int,
) -> dict:
    timeseries_path = input_dir / f"{label}_timeseries.csv"
    returns_path = input_dir / f"{label}_returns.csv"
    run_dir = REPO_ROOT / info["run_dir"]
    for path in (timeseries_path, returns_path, run_dir / "model.pt",
                 run_dir / "config.json"):
        if not path.exists():
            raise FileNotFoundError(path)

    arcs = read_timeseries(timeseries_path)
    returns = read_returns(returns_path)
    x_aug, metadata, jump_pairs = build_relaxed_input(arcs, returns, n_bridge)
    cfg, nets = load_network(run_dir)

    z = model_call(nets.encoder, x_aug)
    decoded = model_call(nets.decoder, z)
    velocity = model_call(nets.vfield, z)
    tangents_vfield, tiny_vfield = normalize_tangent_rows(
        velocity.copy(), 1e-12, "learned latent vector field"
    )
    tangents_diff, tiny_diff = tag_aware_differences(z, metadata)

    g1 = np.vstack([np.r_[g, 1.0] for g, _ in jump_pairs])
    r0 = np.vstack([np.r_[r, 0.0] for _, r in jump_pairs])
    gluing = np.linalg.norm(
        model_call(nets.encoder, g1) - model_call(nets.encoder, r0), axis=1
    )
    reconstruction = np.linalg.norm(decoded - x_aug, axis=1)
    velocity_norm = np.linalg.norm(velocity, axis=1)

    arc_mask = np.array([m["piece"] == "arc" for m in metadata], dtype=bool)
    bridge_mask = ~arc_mask
    # Only compare a forward finite difference to V(z) when the next sample
    # belongs to the same flow arc or the same mapping-cylinder bridge.
    within_piece = np.zeros(len(metadata), dtype=bool)
    for i in range(len(metadata) - 1):
        within_piece[i] = (
            metadata[i]["piece"], metadata[i]["step"]
        ) == (
            metadata[i + 1]["piece"], metadata[i + 1]["step"]
        )
    post_mask = np.array(
        [m["event"] in ("post_impact", "terminal_post_impact") for m in metadata],
        dtype=bool,
    )
    post_z = z[post_mask]
    expected_period = info["expected_period"]
    period_error = None
    if expected_period is not None and len(post_z) > expected_period:
        period_error = summarize(
            np.linalg.norm(post_z[expected_period:] - post_z[:-expected_period], axis=1)
        )

    base = f"continuous_lift_goswami_{label}_vfield"
    case_dir = out_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)

    np.save(case_dir / f"{base}.npy", z)
    np.save(case_dir / f"{base}_tangents.npy", tangents_vfield)
    np.savetxt(case_dir / f"{base}_positions.csv", z, delimiter=" ")
    np.savetxt(case_dir / f"{base}_tangents.csv", tangents_vfield, delimiter=" ")
    np.save(case_dir / f"{base}_augmented_inputs.npy", x_aug)
    np.save(case_dir / f"{base}_decoded.npy", decoded)
    np.save(case_dir / f"{base}_latent_velocity.npy", velocity)
    np.save(case_dir / f"{base}_diff_tangents.npy", tangents_diff)
    np.save(case_dir / f"{base}_postimpact_latent.npy", post_z)
    write_metadata(case_dir / f"{base}_metadata.csv", metadata)
    successor_edges = write_successor_edges(
        case_dir / f"{base}_successor_edges.csv", metadata
    )
    np.save(case_dir / f"{base}_successor_edges.npy", successor_edges)
    plot_latent(case_dir / f"{base}_pca.png", label, z, metadata, info["phi_deg"])

    report = {
        "case": label,
        "phi_deg": info["phi_deg"],
        "expected_period": expected_period,
        "input_timeseries": str(timeseries_path.relative_to(REPO_ROOT)),
        "input_returns": str(returns_path.relative_to(REPO_ROOT)),
        "model": str((run_dir / "model.pt").relative_to(REPO_ROOT)),
        "config": str((run_dir / "config.json").relative_to(REPO_ROOT)),
        "model_note": info["model_note"],
        "latent_dim": int(z.shape[1]),
        "n_samples": int(len(z)),
        "n_successor_edges": int(len(successor_edges)),
        "n_arcs": int(len(arcs)),
        "n_impacts": int(len(jump_pairs)),
        "n_bridge_samples_per_impact": n_bridge,
        "physical_time_end": float(max(m["physical_time"] for m in metadata)),
        "relaxed_time_end": float(max(m["relaxed_time"] for m in metadata)),
        "gluing_error": summarize(gluing),
        "reconstruction_error_all": summarize(reconstruction),
        "reconstruction_error_arc": summarize(reconstruction[arc_mask]),
        "reconstruction_error_bridge": summarize(reconstruction[bridge_mask]),
        "latent_velocity_norm": summarize(velocity_norm),
        "trajectory_vfield_alignment": {
            "definition": (
                "signed cosine between the learned vector-field tangent and "
                "the forward tangent of the encoded ordered curve"
            ),
            "arc": summarize_alignment(
                tangents_vfield, tangents_diff, within_piece & arc_mask
            ),
            "bridge": summarize_alignment(
                tangents_vfield, tangents_diff, within_piece & bridge_mask
            ),
        },
        "period_closure_error": period_error,
        "tiny_vfield_tangent_repairs": tiny_vfield,
        "tiny_diff_tangent_repairs": tiny_diff,
        "cycling_signature_base": base,
    }
    (case_dir / f"report_{base}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"[{label}] z={z.shape} arcs={len(arcs)} impacts={len(jump_pairs)} "
        f"glue_median={np.median(gluing):.3e}"
    )
    return report


def write_validation_summary(path: Path, reports: list[dict]) -> None:
    fields = [
        "case", "phi_deg", "expected_period", "n_samples", "n_impacts",
        "gluing_error_median", "gluing_error_p95",
        "arc_vfield_cosine_median", "arc_vfield_cosine_p05",
        "bridge_vfield_cosine_median", "bridge_vfield_cosine_p05",
        "period_closure_error_p95",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            alignment = report["trajectory_vfield_alignment"]
            closure = report["period_closure_error"]
            writer.writerow(
                {
                    "case": report["case"],
                    "phi_deg": report["phi_deg"],
                    "expected_period": report["expected_period"],
                    "n_samples": report["n_samples"],
                    "n_impacts": report["n_impacts"],
                    "gluing_error_median": report["gluing_error"]["median"],
                    "gluing_error_p95": report["gluing_error"]["p95"],
                    "arc_vfield_cosine_median": alignment["arc"]["median"],
                    "arc_vfield_cosine_p05": alignment["arc"]["p05"],
                    "bridge_vfield_cosine_median": alignment["bridge"]["median"],
                    "bridge_vfield_cosine_p05": alignment["bridge"]["p05"],
                    "period_closure_error_p95": "" if closure is None else closure["p95"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=REPO_ROOT / "compass_goswami_output",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=(REPO_ROOT / "chyll_v2" / "cycling_signature" / "data"
                 / "compass_gait_goswami_csv"),
    )
    parser.add_argument("--n-bridge", type=int, default=50)
    parser.add_argument(
        "--cases", default=",".join(CASE_INFO),
        help="comma-separated subset of period1,period2,period4,period8,chaos",
    )
    args = parser.parse_args()

    selected = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(CASE_INFO))
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")
    if args.n_bridge < 2:
        raise ValueError("--n-bridge must be at least 2")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for label in selected:
        reports.append(
            export_case(
                label, CASE_INFO[label], args.input_dir, args.out_dir, args.n_bridge
            )
        )

    manifest = {
        "format_version": 1,
        "description": (
            "Learned continuous CHyLL-v2 latent trajectories exported from "
            "the generated Goswami physical CSV trajectories."
        ),
        "input_dir": str(args.input_dir),
        "n_bridge_samples_per_impact": args.n_bridge,
        "cases": reports,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_validation_summary(args.out_dir / "validation_summary.csv", reports)
    print(f"wrote manifest: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
