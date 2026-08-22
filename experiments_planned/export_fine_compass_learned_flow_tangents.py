#!/usr/bin/env python3
"""Export learned-vector-field tangents on the frozen fine Compass lifts.

The stored fine Compass positions are encoded exact-simulator suspension
paths.  This control leaves those positions, their order, timestamps,
arc/bridge labels, and duration-window manifests unchanged, but replaces the
encoder-JVP path tangents with unit vectors ``V_theta(z) / ||V_theta(z)||``.

The default mode is read-only: it validates all five source lifts, loads the
exact checkpoint named by each lift, and evaluates a bounded diagnostic probe.
``--materialize`` is required to evaluate every row and write tangent CSVs.
Outputs are confined to ``experiments_planned/outputs/`` and existing paths
are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(CODE_ROOT))

from chyll_v2.chyll_v2.networks import CHyLLv2Networks  # noqa: E402
from chyll_v2.cycling_signature.export.prepare_rimless_lift import (  # noqa: E402
    load_config,
    load_state_dict,
)


REGIMES = ("period1", "period2", "period4", "period8", "chaos")
MODEL_RUNS = {
    "period1": "compass_gait_phi007",
    "period2": "compass_gait_phi_1_4.75deg",
    "period4": "compass_gait_phi_2_5deg",
    "period8": "compass_gait_phi_3_5.02deg",
    "chaos": "compass_gait_phi_4_cloud_5.2deg",
}
CANONICAL_MANIFEST_SHA256 = {
    "period1": "c7500cc6010a2499c463981d1796734365370286705a52fc87642bb53dc33f3d",
    "period2": "0541f3b8dd6efdb7a33b9155569fb2dbe659d4e1f8a2456aac4f63a99118670e",
    "period4": "54327c7a059f8943b6fd9085092656e00a0f234b25f27ffa73a45f9d17da00c0",
    "period8": "93d2d15c2ac662991d3bf54062bdebb7a25467bffb70fb5396d0cff2fb81be61",
    "chaos": "604efccfd445c38de488797bc4ba3646a64a5532a9b07be608ee1cd3377df179",
}
MANIFEST_HEADER = (
    "target_duration",
    "split",
    "run_index",
    "start_index",
    "end_index",
    "realized_duration",
    "duration_error",
)


@dataclass(frozen=True)
class LiftSource:
    regime: str
    archive: Path
    positions_csv: Path
    source_tangents_csv: Path
    checkpoint: Path
    config: Path
    positions: np.ndarray
    source_tangents: np.ndarray
    timestamps: np.ndarray
    piece_kind: np.ndarray
    source_meta: dict[str, object]


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _require_output_path(path: Path) -> Path:
    allowed = (CODE_ROOT / "experiments_planned" / "outputs").resolve()
    resolved = _resolved(path)
    if resolved == allowed or not resolved.is_relative_to(allowed):
        raise ValueError(f"output directory must be a child of {allowed}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _code_repo_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=CODE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip().lower()
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise ValueError(f"git returned an invalid code-repository HEAD: {head!r}")
    return head


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: math.nan for key in ("min", "median", "mean", "p95", "max")}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cosine = np.sum(left * right, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _parse_regimes(spec: str) -> tuple[str, ...]:
    if spec == "all":
        return REGIMES
    regimes = tuple(item.strip() for item in spec.split(",") if item.strip())
    if not regimes or len(set(regimes)) != len(regimes):
        raise ValueError("--regimes must name one or more distinct regimes")
    unknown = sorted(set(regimes) - set(REGIMES))
    if unknown:
        raise ValueError(f"unknown regimes: {', '.join(unknown)}")
    return regimes


def _validate_manifest(
    path: Path,
    regime: str,
    lift_length: int,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing real window manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_HEADER:
            raise ValueError(f"unexpected window-manifest header: {path}")
        raw_rows = list(reader)
    if len(raw_rows) != 600:
        raise ValueError(f"expected 600 frozen windows: {path}")

    rows: list[dict[str, float | int | str]] = []
    for raw in raw_rows:
        row: dict[str, float | int | str] = {
            "target_duration": float(raw["target_duration"]),
            "split": raw["split"],
            "run_index": int(raw["run_index"]),
            "start_index": int(raw["start_index"]),
            "end_index": int(raw["end_index"]),
            "realized_duration": float(raw["realized_duration"]),
            "duration_error": float(raw["duration_error"]),
        }
        numeric = [value for key, value in row.items() if key != "split"]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"nonfinite window-manifest value: {path}")
        rows.append(row)

    expected_durations = [0.25 * index for index in range(1, 31)]
    expected_order = [
        (duration, run_index)
        for duration in expected_durations
        for run_index in range(1, 21)
    ]
    actual_order = [
        (float(row["target_duration"]), int(row["run_index"])) for row in rows
    ]
    if actual_order != expected_order:
        raise ValueError(f"window rows are not the frozen 30 x 20 order: {path}")
    for row in rows:
        run_index = int(row["run_index"])
        expected_split = "tune" if run_index <= 10 else "validate"
        if row["split"] != expected_split:
            raise ValueError(f"unexpected tune/validate assignment: {path}")
        realized = float(row["realized_duration"])
        target = float(row["target_duration"])
        error = float(row["duration_error"])
        if not math.isclose(realized - target, error, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"inconsistent duration error: {path}")
        if abs(error) > 0.0050000001:
            raise ValueError(f"duration realization exceeds one dt: {path}")

    end_indices = [int(row["end_index"]) for row in rows]
    start_indices = [int(row["start_index"]) for row in rows]
    if min(start_indices) < 1 or any(a >= b for a, b in zip(start_indices, end_indices)):
        raise ValueError(f"invalid one-based window indices: {path}")
    if max(end_indices) > lift_length:
        raise ValueError(f"window manifest exceeds its lift: {path}")
    for run_index in range(1, 21):
        run_rows = [row for row in rows if int(row["run_index"]) == run_index]
        if len({int(row["start_index"]) for row in run_rows}) != 1:
            raise ValueError(f"paired start changed across durations: {path}")
        ends = [int(row["end_index"]) for row in run_rows]
        if any(left >= right for left, right in zip(ends, ends[1:])):
            raise ValueError(f"duration windows are not strictly nested: {path}")
    sha256 = _file_sha256(path)
    if sha256 != CANONICAL_MANIFEST_SHA256[regime]:
        raise ValueError(
            f"window manifest is not the frozen {regime} manifest: {path}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256,
        "rows": len(rows),
        "target_durations": expected_durations,
        "starts_per_duration": 20,
        "tune_starts": 10,
        "validate_starts": 10,
        "max_end_index": max(end_indices),
    }


def _load_source(regime: str, data_dir: Path, runs_dir: Path) -> LiftSource:
    base = f"compass_{regime}"
    archive = data_dir / f"{base}.npz"
    positions_csv = data_dir / f"{base}_positions.csv"
    source_tangents_csv = data_dir / f"{base}_tangents.csv"
    for label, path in (
        ("latent archive", archive),
        ("positions CSV", positions_csv),
        ("source tangents CSV", source_tangents_csv),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing real {label}: {path}")

    with np.load(archive, allow_pickle=False) as payload:
        expected_keys = {"t", "x", "v", "piece_kind", "meta_json"}
        if set(payload.files) != expected_keys:
            raise ValueError(f"unexpected archive schema in {archive}: {payload.files}")
        timestamps = np.asarray(payload["t"], dtype=np.float64)
        positions = np.asarray(payload["x"], dtype=np.float64)
        source_tangents = np.asarray(payload["v"], dtype=np.float64)
        piece_kind = np.asarray(payload["piece_kind"], dtype=np.uint8)
        source_meta = json.loads(str(payload["meta_json"]))

    if positions.ndim != 2 or positions.shape[1] != 11:
        raise ValueError(f"expected N x 11 latent positions: {archive}")
    if source_tangents.shape != positions.shape:
        raise ValueError(f"source position/tangent shape mismatch: {archive}")
    if timestamps.shape != (len(positions),) or piece_kind.shape != (len(positions),):
        raise ValueError(f"timestamp or piece-label length mismatch: {archive}")
    if not np.isfinite(positions).all() or not np.isfinite(source_tangents).all():
        raise ValueError(f"nonfinite stored lift: {archive}")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"timestamps must be finite and strictly increasing: {archive}")
    if not set(np.unique(piece_kind)).issubset({0, 1}) or set(
        np.unique(piece_kind)
    ) != {0, 1}:
        raise ValueError(f"piece_kind must contain both arc=0 and bridge=1: {archive}")

    positions_from_csv = np.loadtxt(positions_csv, dtype=np.float64)
    tangents_from_csv = np.loadtxt(source_tangents_csv, dtype=np.float64)
    if not np.array_equal(positions, positions_from_csv):
        raise ValueError(f"positions CSV does not exactly match archive order: {archive}")
    if not np.array_equal(source_tangents, tangents_from_csv):
        raise ValueError(f"source tangent CSV does not exactly match archive: {archive}")
    tangent_norms = np.linalg.norm(source_tangents, axis=1)
    if not np.allclose(tangent_norms, 1.0, rtol=0.0, atol=2e-7):
        raise ValueError(f"stored source tangents are not unit length: {archive}")

    model_run = MODEL_RUNS[regime]
    if source_meta.get("model_run") != model_run:
        raise ValueError(
            f"{archive} names {source_meta.get('model_run')!r}, expected {model_run!r}"
        )
    if source_meta.get("tangent_source") != "encoder_jvp":
        raise ValueError(f"{archive} is not the expected encoder-JVP source lift")
    if source_meta.get("lift") != "chyll_v2_latent_suspension":
        raise ValueError(f"unexpected lift construction in {archive}")
    if int(source_meta.get("latent_dim", -1)) != positions.shape[1]:
        raise ValueError(f"latent dimension metadata mismatch: {archive}")
    if int(source_meta.get("n_arc_samples", -1)) != int((piece_kind == 0).sum()):
        raise ValueError(f"arc count metadata mismatch: {archive}")
    if int(source_meta.get("n_bridge_samples", -1)) != int((piece_kind == 1).sum()):
        raise ValueError(f"bridge count metadata mismatch: {archive}")

    run_dir = runs_dir / model_run
    checkpoint = run_dir / "model.pt"
    config = run_dir / "config.json"
    for label, path in (("checkpoint", checkpoint), ("configuration", config)):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing real {label}: {path}")
    recorded_model = _resolved(Path(str(source_meta.get("model_path", ""))))
    if recorded_model != checkpoint.resolve():
        raise ValueError(
            f"archive checkpoint provenance differs from the expected run: {archive}"
        )

    return LiftSource(
        regime=regime,
        archive=archive.resolve(),
        positions_csv=positions_csv.resolve(),
        source_tangents_csv=source_tangents_csv.resolve(),
        checkpoint=checkpoint.resolve(),
        config=config.resolve(),
        positions=positions,
        source_tangents=source_tangents,
        timestamps=timestamps,
        piece_kind=piece_kind,
        source_meta=source_meta,
    )


def _load_network(source: LiftSource) -> tuple[CHyLLv2Networks, float]:
    cfg = load_config(source.config)
    if cfg.system_name != "compass_gait":
        raise ValueError(f"non-Compass configuration: {source.config}")
    if cfg.state_dim != 5 or cfg.latent_dim != source.positions.shape[1]:
        raise ValueError(f"network/lift dimension mismatch: {source.config}")
    config_w_v = float(cfg.w_v)
    if config_w_v != 0.0:
        raise ValueError(
            f"learned-flow tangent control requires saved w_v=0: {source.config}"
        )
    cfg.device = "cpu"
    networks = CHyLLv2Networks(cfg)
    networks.load_state_dict(load_state_dict(source.checkpoint), strict=True)
    networks.eval()
    return networks, config_w_v


def _evaluate_velocity(
    networks: CHyLLv2Networks,
    positions: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            stop = min(start + batch_size, len(positions))
            z = torch.as_tensor(positions[start:stop], dtype=torch.float32)
            chunk = networks.vfield(z).cpu().numpy().astype(np.float64)
            chunks.append(chunk)
    velocity = np.vstack(chunks)
    if velocity.shape != positions.shape or not np.isfinite(velocity).all():
        raise ValueError("learned vector field returned invalid velocities")
    return velocity


def _unit_velocity(
    networks: CHyLLv2Networks,
    positions: np.ndarray,
    batch_size: int,
    min_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = _evaluate_velocity(networks, positions, batch_size)
    speeds = np.linalg.norm(velocity, axis=1)
    if np.any(speeds <= min_speed):
        index = int(np.flatnonzero(speeds <= min_speed)[0])
        raise ValueError(
            f"learned speed {speeds[index]:.3e} at row {index + 1} is not above "
            f"--min-speed={min_speed:.3e}; refusing a silent tangent repair"
        )
    tangents = velocity / speeds[:, None]
    if not np.allclose(
        np.linalg.norm(tangents, axis=1), 1.0, rtol=0.0, atol=5e-15
    ):
        raise ValueError("failed to normalize learned vector-field tangents")
    return tangents, speeds


def _probe_indices(piece_kind: np.ndarray, probe_count: int) -> np.ndarray:
    uniform = np.linspace(
        0, len(piece_kind) - 1, min(probe_count, len(piece_kind)), dtype=np.int64
    )
    boundaries = np.flatnonzero(piece_kind[:-1] != piece_kind[1:])
    return np.unique(np.concatenate((uniform, boundaries, boundaries + 1)))


def _diagnostics(
    source: LiftSource,
    tangents: np.ndarray,
    speeds: np.ndarray,
    row_indices: np.ndarray,
) -> dict[str, object]:
    source_probe = source.source_tangents[row_indices]
    signed_cosine = np.sum(tangents * source_probe, axis=1)
    kinds = source.piece_kind[row_indices]
    diagnostics: dict[str, object] = {
        "evaluated_rows": int(len(row_indices)),
        "speed": _summary(speeds),
        "flow_vs_encoder_jvp_signed_cosine": _summary(signed_cosine),
        "flow_vs_encoder_jvp_angle_degrees": _summary(
            _angle_degrees(tangents, source_probe)
        ),
        "arc_flow_vs_encoder_jvp_angle_degrees": _summary(
            _angle_degrees(tangents[kinds == 0], source_probe[kinds == 0])
        ),
        "bridge_flow_vs_encoder_jvp_angle_degrees": _summary(
            _angle_degrees(tangents[kinds == 1], source_probe[kinds == 1])
        ),
    }

    lookup = {int(row): index for index, row in enumerate(row_indices)}
    boundary_left = np.flatnonzero(source.piece_kind[:-1] != source.piece_kind[1:])
    boundary_endpoints = np.unique(
        np.concatenate((boundary_left, boundary_left + 1))
    )
    sampled_boundary = np.isin(row_indices, boundary_endpoints)
    diagnostics["boundary_endpoint_flow_vs_encoder_jvp_angle_degrees"] = _summary(
        _angle_degrees(
            tangents[sampled_boundary], source_probe[sampled_boundary]
        )
    )
    diagnostics["nonboundary_flow_vs_encoder_jvp_angle_degrees"] = _summary(
        _angle_degrees(
            tangents[~sampled_boundary], source_probe[~sampled_boundary]
        )
    )
    left_local = np.array([lookup[int(row)] for row in boundary_left], dtype=int)
    right_local = np.array([lookup[int(row + 1)] for row in boundary_left], dtype=int)
    flow_turns = _angle_degrees(tangents[left_local], tangents[right_local])
    source_turns = _angle_degrees(
        source.source_tangents[boundary_left],
        source.source_tangents[boundary_left + 1],
    )
    arc_to_bridge = source.piece_kind[boundary_left] == 0
    diagnostics["piece_boundaries"] = int(len(boundary_left))
    diagnostics["flow_boundary_turn_degrees"] = _summary(flow_turns)
    diagnostics["source_boundary_turn_degrees"] = _summary(source_turns)
    diagnostics["flow_arc_to_bridge_turn_degrees"] = _summary(
        flow_turns[arc_to_bridge]
    )
    diagnostics["flow_bridge_to_arc_turn_degrees"] = _summary(
        flow_turns[~arc_to_bridge]
    )
    return diagnostics


def _source_record(
    source: LiftSource,
    diagnostics: dict[str, object],
    manifests: Iterable[dict[str, object]],
    config_w_v: float,
) -> dict[str, object]:
    checkpoint_sha256 = _file_sha256(source.checkpoint)
    config_sha256 = _file_sha256(source.config)
    positions_csv_sha256 = _file_sha256(source.positions_csv)
    positions_array_sha256 = _array_sha256(source.positions)
    record: dict[str, object] = {
        "regime": source.regime,
        "rows": len(source.positions),
        "latent_dim": source.positions.shape[1],
        "source_archive": str(source.archive),
        "source_archive_sha256": _file_sha256(source.archive),
        "source_positions_csv": str(source.positions_csv),
        "source_positions_csv_sha256": positions_csv_sha256,
        "source_encoder_jvp_tangents_csv": str(source.source_tangents_csv),
        "source_encoder_jvp_tangents_csv_sha256": _file_sha256(
            source.source_tangents_csv
        ),
        "positions_array_sha256": positions_array_sha256,
        "timestamps_array_sha256": _array_sha256(source.timestamps),
        "piece_kind_array_sha256": _array_sha256(source.piece_kind),
        "checkpoint": str(source.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "config": str(source.config),
        "config_sha256": config_sha256,
        "config_w_v": config_w_v,
        "model_run": MODEL_RUNS[source.regime],
        "source_meta": source.source_meta,
        "window_manifests": list(manifests),
        "diagnostics": diagnostics,
        "tangent_generation_inputs": {
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "config_w_v": config_w_v,
            "source_positions_csv_sha256": positions_csv_sha256,
            "source_positions_array_sha256": positions_array_sha256,
        },
    }
    return record


def _format_statistic(summary: dict[str, float]) -> str:
    return (
        f"median={summary['median']:.4g}, mean={summary['mean']:.4g}, "
        f"p95={summary['p95']:.4g}, max={summary['max']:.4g}"
    )


def _report(document: dict[str, object]) -> str:
    lines = [
        "Fine Compass learned-flow-tangent control",
        "===========================================",
        f"status: {document['status']}",
        f"exporter script sha256: {document['exporter_script_sha256']}",
        f"code repository HEAD: {document['code_repo_head']}",
        "",
        "This control preserves the frozen latent positions and evaluates",
        "V_theta(z)/||V_theta(z)|| at those exact rows. It does not integrate",
        "the learned flow, rebuild the suspension path, or run signatures.",
        "",
    ]
    for record in document["regimes"]:
        diagnostics = record["diagnostics"]
        lines.extend(
            [
                f"{record['regime']}: {record['rows']} x {record['latent_dim']}",
                f"  saved config w_v: {record['config_w_v']}",
                "  speed: " + _format_statistic(diagnostics["speed"]),
                "  angle from encoder-JVP tangent (degrees): "
                + _format_statistic(
                    diagnostics["flow_vs_encoder_jvp_angle_degrees"]
                ),
                "  nonboundary angle from encoder-JVP tangent (degrees): "
                + _format_statistic(
                    diagnostics[
                        "nonboundary_flow_vs_encoder_jvp_angle_degrees"
                    ]
                ),
                "  learned-flow turn at arc/bridge boundaries (degrees): "
                + _format_statistic(diagnostics["flow_boundary_turn_degrees"]),
            ]
        )
    lines.extend(
        [
            "",
            "Scientific caveats",
            "------------------",
            "- These are vector-field directions on an encoded exact-simulator",
            "  path, not tangents of a trajectory generated by that vector field.",
            "- Agreement between V_theta(z) and the stored path derivative is an",
            "  empirical diagnostic, not guaranteed by this substitution.",
            "  The encoder-JVP direction is a path-tangent proxy corroborated by",
            "  the stored tag-aware finite-difference control, not an identity.",
            "- The saved models used w_v=0, so seam-velocity compatibility was",
            "  not directly penalized during training.",
            "- Unit normalization discards learned speed. It is suitable only for",
            "  the direction component of the present UTB comparison.",
            "- Each regime uses a separately trained model; cross-regime numerical",
            "  scales still require the paper's stated comparability decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    default_data = CODE_ROOT / "period_doubling" / "data_fine" / "compass_gait_latent"
    default_runs = CODE_ROOT / "chyll_v2" / "runs"
    default_output = (
        CODE_ROOT
        / "experiments_planned"
        / "outputs"
        / "fine_compass_learned_flow_tangents"
    )
    default_manifests = (
        CODE_ROOT
        / "experiments_planned"
        / "outputs"
        / "compass_c0p2_to_c0p8_diagnostic"
        / "manifests"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--runs-dir", type=Path, default=default_runs)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--regimes", default="all")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        action="append",
        help=(
            "frozen duration-window manifest directory to validate and hash; "
            f"default: {default_manifests}"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--probe-count",
        type=int,
        default=512,
        help="uniform check-only rows per regime, plus every piece boundary",
    )
    parser.add_argument("--min-speed", type=float, default=1e-12)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="write all five tangent CSVs and provenance; otherwise check only",
    )
    parser.set_defaults(default_manifest_dir=default_manifests)
    return parser


def main() -> int:
    args = _parser().parse_args()
    exporter_script_sha256 = _file_sha256(SCRIPT_PATH)
    code_repo_head = _code_repo_head()
    if args.batch_size < 1 or args.probe_count < 2:
        raise ValueError("--batch-size must be positive and --probe-count at least 2")
    if not math.isfinite(args.min_speed) or args.min_speed <= 0:
        raise ValueError("--min-speed must be positive and finite")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    regimes = _parse_regimes(args.regimes)
    data_dir = _resolved(args.data_dir)
    runs_dir = _resolved(args.runs_dir)
    output_dir = _require_output_path(args.output_dir)
    manifest_dirs = args.manifest_dir or [args.default_manifest_dir]
    manifest_dirs = [_resolved(path) for path in manifest_dirs]
    if len(set(manifest_dirs)) != len(manifest_dirs):
        raise ValueError("duplicate --manifest-dir")
    if args.materialize and regimes != REGIMES:
        raise ValueError("materialization requires --regimes all")
    if args.materialize and (output_dir.exists() or output_dir.is_symlink()):
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    sources = [_load_source(regime, data_dir, runs_dir) for regime in regimes]
    records: list[dict[str, object]] = []
    full_tangents: dict[str, np.ndarray] = {}
    for source in sources:
        manifest_records = [
            _validate_manifest(
                directory / f"{source.regime}.csv",
                source.regime,
                len(source.positions),
            )
            for directory in manifest_dirs
        ]
        networks, config_w_v = _load_network(source)
        if args.materialize:
            row_indices = np.arange(len(source.positions), dtype=np.int64)
            tangents, speeds = _unit_velocity(
                networks, source.positions, args.batch_size, args.min_speed
            )
            full_tangents[source.regime] = tangents
        else:
            row_indices = _probe_indices(source.piece_kind, args.probe_count)
            tangents, speeds = _unit_velocity(
                networks,
                source.positions[row_indices],
                args.batch_size,
                args.min_speed,
            )
        diagnostics = _diagnostics(source, tangents, speeds, row_indices)
        records.append(
            _source_record(
                source,
                diagnostics,
                manifest_records,
                config_w_v,
            )
        )
        turns = diagnostics["flow_boundary_turn_degrees"]
        print(
            f"{source.regime}: validated {len(source.positions)} rows; "
            f"evaluated {len(row_indices)}; flow boundary-turn "
            f"median={turns['median']:.2f} deg, max={turns['max']:.2f} deg"
        )

    status = "materialized" if args.materialize else "checked_not_materialized"
    if _file_sha256(SCRIPT_PATH) != exporter_script_sha256:
        raise RuntimeError("exporter script changed while the control was running")
    document: dict[str, object] = {
        "schema_version": 2,
        "kind": "fine_compass_learned_flow_tangent_control",
        "status": status,
        "created_utc": datetime.now(UTC).isoformat(),
        "method": "unit(V_theta(z)) evaluated in float32; normalized in float64",
        "batch_size": args.batch_size,
        "min_speed": args.min_speed,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "exporter_script": str(SCRIPT_PATH),
        "exporter_script_sha256": exporter_script_sha256,
        "code_repo_head": code_repo_head,
        "code_root": str(CODE_ROOT),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "regimes": records,
        "signature_status": "not_run",
    }
    if not args.materialize:
        print(json.dumps(document, indent=2))
        print("check-only complete; pass --materialize to write tangent CSVs")
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.partial-"
    ) as temporary:
        staging = Path(temporary)
        for source, record in zip(sources, records):
            output_path = staging / f"compass_{source.regime}_tangents.csv"
            np.savetxt(
                output_path,
                full_tangents[source.regime],
                delimiter=" ",
                fmt="%.18e",
            )
            reloaded = np.loadtxt(output_path, dtype=np.float64)
            if not np.array_equal(reloaded, full_tangents[source.regime]):
                raise ValueError(f"tangent CSV did not round-trip exactly: {output_path}")
            output_csv_sha256 = _file_sha256(output_path)
            output_array_sha256 = _array_sha256(full_tangents[source.regime])
            record.update(
                {
                    "learned_flow_tangents_csv": str(
                        output_dir / output_path.name
                    ),
                    "learned_flow_tangents_csv_sha256": output_csv_sha256,
                    "learned_flow_tangents_array_sha256": output_array_sha256,
                    "learned_flow_tangent_binding": {
                        **record["tangent_generation_inputs"],
                        "exporter_script_sha256": exporter_script_sha256,
                        "code_repo_head": code_repo_head,
                        "rows": record["rows"],
                        "latent_dim": record["latent_dim"],
                        "output_csv_sha256": output_csv_sha256,
                        "output_array_sha256": output_array_sha256,
                    },
                }
            )
        (staging / "provenance.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "report.txt").write_text(_report(document), encoding="utf-8")
        os.replace(staging, output_dir)
    print(f"wrote learned-flow tangent control to {output_dir}")
    print("cycling signatures were not run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
