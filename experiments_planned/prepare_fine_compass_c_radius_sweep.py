#!/usr/bin/env python3
"""Prepare paired physical-duration manifests for a fine Compass C/r sweep.

Nothing is simulated, trained, or signed by this planner.  Without
``--materialize`` it only prints the bounded command plan.  Materialization
writes manifests and a command list below ``experiments_planned/outputs/``;
the expensive Julia commands remain unexecuted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np


REGIMES = ("period1", "period2", "period4", "period8", "chaos")
FLOW_MODEL_RUNS = {
    "period1": "compass_gait_phi007",
    "period2": "compass_gait_phi_1_4.75deg",
    "period4": "compass_gait_phi_2_5deg",
    "period8": "compass_gait_phi_3_5.02deg",
    "chaos": "compass_gait_phi_4_cloud_5.2deg",
}
MANIFEST_FIELDS = (
    "target_duration",
    "split",
    "run_index",
    "start_index",
    "end_index",
    "realized_duration",
    "duration_error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def number_token(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _real_hashed_file(
    path: Path,
    expected_hash: object,
    label: str,
) -> tuple[Path, str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing real {label}: {path}")
    resolved = path.resolve()
    actual_hash = sha256(resolved)
    if str(expected_hash) != actual_hash:
        raise ValueError(f"{label} hash mismatch: {resolved}")
    return resolved, actual_hash


def validate_tangent_provenance(
    code_root: Path,
    tangents_dir: Path,
    expected_manifest_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    tangents_dir = tangents_dir.resolve()
    provenance_path = tangents_dir / "provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise FileNotFoundError(
            f"override tangents require real provenance.json: {provenance_path}"
        )
    document = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != 2
        or document.get("kind") != "fine_compass_learned_flow_tangent_control"
        or document.get("status") != "materialized"
        or document.get("signature_status") != "not_run"
    ):
        raise ValueError(f"unsupported tangent provenance: {provenance_path}")
    if Path(str(document.get("code_root", ""))).resolve() != code_root.resolve():
        raise ValueError("tangent provenance belongs to a different code checkout")
    if Path(str(document.get("output_dir", ""))).resolve() != tangents_dir:
        raise ValueError("tangent provenance names a different output directory")
    exporter_script = (
        code_root
        / "experiments_planned"
        / "export_fine_compass_learned_flow_tangents.py"
    ).resolve()
    recorded_exporter, exporter_hash = _real_hashed_file(
        Path(str(document.get("exporter_script", ""))),
        document.get("exporter_script_sha256"),
        "tangent exporter script",
    )
    if recorded_exporter != exporter_script:
        raise ValueError("tangent provenance names a different exporter script")
    code_repo_head = subprocess.run(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if document.get("code_repo_head") != code_repo_head:
        raise ValueError("tangent provenance names a different code HEAD")

    data_root = (
        code_root / "period_doubling" / "data_fine" / "compass_gait_latent"
    ).resolve()
    runs_root = (code_root / "chyll_v2" / "runs").resolve()
    if Path(str(document.get("data_dir", ""))).resolve() != data_root:
        raise ValueError("tangent provenance names a different source lift")
    records = document.get("regimes")
    if not isinstance(records, list) or len(records) != len(REGIMES):
        raise ValueError("tangent provenance must contain exactly five regimes")
    by_regime: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("tangent provenance regime must be an object")
        regime = str(record.get("regime", ""))
        if regime not in REGIMES or regime in by_regime:
            raise ValueError(f"invalid or duplicate provenance regime: {regime!r}")
        if int(record.get("latent_dim", 0)) != 11 or int(record.get("rows", 0)) <= 0:
            raise ValueError(f"invalid tangent dimensions for {regime}")
        if record.get("model_run") != FLOW_MODEL_RUNS[regime]:
            raise ValueError(f"unexpected model run for {regime}")
        if float(record.get("config_w_v", math.nan)) != 0.0:
            raise ValueError(f"expected saved config w_v=0 for {regime}")

        expected_paths = {
            "learned_flow_tangents_csv": (
                tangents_dir / f"compass_{regime}_tangents.csv"
            ),
            "source_positions_csv": data_root / f"compass_{regime}_positions.csv",
            "source_encoder_jvp_tangents_csv": (
                data_root / f"compass_{regime}_tangents.csv"
            ),
            "source_archive": data_root / f"compass_{regime}.npz",
            "checkpoint": runs_root / FLOW_MODEL_RUNS[regime] / "model.pt",
            "config": runs_root / FLOW_MODEL_RUNS[regime] / "config.json",
        }
        hashes: dict[str, str] = {}
        for key, expected_path in expected_paths.items():
            recorded_path = Path(str(record.get(key, ""))).resolve()
            if recorded_path != expected_path.resolve():
                raise ValueError(f"{regime} provenance names a different {key}")
            hash_key = f"{key}_sha256"
            _, hashes[key] = _real_hashed_file(
                expected_path,
                record.get(hash_key),
                f"{regime} {key}",
            )
        config_document = json.loads(
            expected_paths["config"].read_text(encoding="utf-8")
        )
        if float(config_document.get("w_v", math.nan)) != 0.0:
            raise ValueError(f"saved config does not contain w_v=0 for {regime}")
        positions_values = np.loadtxt(
            expected_paths["source_positions_csv"], dtype=np.float64
        )
        tangent_values = np.loadtxt(
            expected_paths["learned_flow_tangents_csv"], dtype=np.float64
        )
        expected_shape = (int(record["rows"]), int(record["latent_dim"]))
        if positions_values.shape != expected_shape or tangent_values.shape != expected_shape:
            raise ValueError(f"bound array shape differs for {regime}")
        positions_array_hash = array_sha256(positions_values)
        tangent_array_hash = array_sha256(tangent_values)
        if positions_array_hash != record.get("positions_array_sha256"):
            raise ValueError(f"source positions array hash mismatch for {regime}")
        if tangent_array_hash != record.get(
            "learned_flow_tangents_array_sha256"
        ):
            raise ValueError(f"learned-flow tangent array hash mismatch for {regime}")
        if not np.allclose(
            np.linalg.norm(tangent_values, axis=1),
            1.0,
            rtol=0.0,
            atol=5e-15,
        ):
            raise ValueError(f"learned-flow tangents are not unit length for {regime}")

        generation_inputs = record.get("tangent_generation_inputs")
        binding = record.get("learned_flow_tangent_binding")
        generation_keys = {
            "checkpoint_sha256",
            "config_sha256",
            "config_w_v",
            "source_positions_csv_sha256",
            "source_positions_array_sha256",
        }
        binding_keys = generation_keys | {
            "exporter_script_sha256",
            "code_repo_head",
            "output_csv_sha256",
            "output_array_sha256",
            "rows",
            "latent_dim",
        }
        if (
            not isinstance(generation_inputs, dict)
            or set(generation_inputs) != generation_keys
            or not isinstance(binding, dict)
            or set(binding) != binding_keys
        ):
            raise ValueError(f"incomplete learned-flow binding for {regime}")
        expected_generation = {
            "checkpoint_sha256": hashes["checkpoint"],
            "config_sha256": hashes["config"],
            "config_w_v": 0.0,
            "source_positions_csv_sha256": hashes["source_positions_csv"],
            "source_positions_array_sha256": record.get(
                "positions_array_sha256"
            ),
        }
        if generation_inputs != expected_generation:
            raise ValueError(f"generation inputs do not bind sources for {regime}")
        expected_binding = {
            **expected_generation,
            "exporter_script_sha256": exporter_hash,
            "code_repo_head": code_repo_head,
            "output_csv_sha256": hashes["learned_flow_tangents_csv"],
            "output_array_sha256": record.get(
                "learned_flow_tangents_array_sha256"
            ),
            "rows": int(record["rows"]),
            "latent_dim": int(record["latent_dim"]),
        }
        if binding != expected_binding:
            raise ValueError(f"learned-flow output binding differs for {regime}")

        manifests = record.get("window_manifests")
        if not isinstance(manifests, list) or not manifests:
            raise ValueError(f"{regime} provenance lacks window manifests")
        bound_manifest_hashes: set[str] = set()
        for manifest in manifests:
            if not isinstance(manifest, dict):
                raise ValueError(f"malformed {regime} window-manifest provenance")
            manifest_path, manifest_hash = _real_hashed_file(
                Path(str(manifest.get("path", ""))),
                manifest.get("sha256"),
                f"{regime} window manifest",
            )
            if not manifest_path.is_relative_to(
                (code_root / "experiments_planned" / "outputs").resolve()
            ):
                raise ValueError(f"{regime} manifest lies outside planned outputs")
            bound_manifest_hashes.add(manifest_hash)
        if expected_manifest_hashes is not None and bound_manifest_hashes != {
            expected_manifest_hashes[regime]
        }:
            raise ValueError(
                f"{regime} tangent provenance does not bind the planned manifest"
            )
        by_regime[regime] = {
            "tangents": str(expected_paths["learned_flow_tangents_csv"].resolve()),
            "tangents_sha256": hashes["learned_flow_tangents_csv"],
            "positions": str(expected_paths["source_positions_csv"].resolve()),
            "positions_sha256": hashes["source_positions_csv"],
            "source_archive": str(expected_paths["source_archive"].resolve()),
            "source_archive_sha256": hashes["source_archive"],
            "source_encoder_jvp_tangents": str(
                expected_paths["source_encoder_jvp_tangents_csv"].resolve()
            ),
            "source_encoder_jvp_tangents_sha256": hashes[
                "source_encoder_jvp_tangents_csv"
            ],
            "checkpoint": str(expected_paths["checkpoint"].resolve()),
            "checkpoint_sha256": hashes["checkpoint"],
            "config": str(expected_paths["config"].resolve()),
            "config_sha256": hashes["config"],
            "manifest_sha256": next(iter(bound_manifest_hashes)),
        }
    if set(by_regime) != set(REGIMES):
        raise ValueError("tangent provenance does not bind all five regimes")
    return {
        "path": str(provenance_path.resolve()),
        "sha256": sha256(provenance_path),
        "kind": document["kind"],
        "schema_version": document["schema_version"],
        "status": document["status"],
        "exporter_script": str(exporter_script),
        "exporter_script_sha256": exporter_hash,
        "code_repo_head": code_repo_head,
        "regimes": by_regime,
    }


def float_grid(spec: str) -> list[float]:
    parts = [float(value) for value in spec.split(":")]
    if len(parts) != 3 or parts[1] <= 0 or parts[2] < parts[0]:
        raise argparse.ArgumentTypeError("expected start:step:stop")
    step_count = (parts[2] - parts[0]) / parts[1]
    rounded_steps = round(step_count)
    if not math.isclose(step_count, rounded_steps, rel_tol=0.0, abs_tol=1e-10):
        raise argparse.ArgumentTypeError("grid stop must lie on the stated step")
    count = int(rounded_steps) + 1
    return [round(parts[0] + index * parts[1], 12) for index in range(count)]


def safe_output_root(code_root: Path, requested: Path) -> Path:
    allowed = (code_root / "experiments_planned" / "outputs").resolve()
    resolved = requested.resolve()
    if resolved != allowed and not resolved.is_relative_to(allowed):
        raise ValueError(f"output must stay below {allowed}")
    return resolved


def existing_plan_root(code_root: Path, requested: Path) -> Path:
    resolved = safe_output_root(code_root, requested)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"manifest source must be a real plan directory: {resolved}")
    return resolved


def parse_diagnostic_arm(spec: str) -> dict[str, object]:
    parts = spec.split(":", 3)
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "expected LABEL:BOXSIZE:SB_RADIUS[:TANGENTS_DIR]"
        )
    label = parts[0]
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None:
        raise argparse.ArgumentTypeError("arm label must be filename-safe")
    try:
        boxsize = float(parts[1])
        sb_radius = int(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "arm boxsize and sb-radius must be numeric"
        ) from error
    if not math.isfinite(boxsize) or boxsize <= 0 or sb_radius <= 0:
        raise argparse.ArgumentTypeError(
            "arm boxsize and sb-radius must be positive"
        )
    tangents_dir = None
    if len(parts) == 4 and parts[3] not in ("", "default"):
        tangents_dir = Path(parts[3]).expanduser().resolve()
    return {
        "label": label,
        "boxsize": boxsize,
        "sb_radius": sb_radius,
        "tangents_dir": tangents_dir,
    }


def load_reused_manifests(
    source_root: Path,
    durations: list[float],
    n_starts: int,
) -> tuple[dict[str, Path], dict[str, str]]:
    source_plan_path = source_root / "plan.json"
    if source_plan_path.is_symlink() or not source_plan_path.is_file():
        raise FileNotFoundError(f"missing real source plan: {source_plan_path}")
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    source_durations = [
        float(value) for value in source_plan.get("duration_grid_seconds", [])
    ]
    if source_durations != durations:
        raise ValueError("reused manifest duration grid differs from this plan")
    if int(source_plan.get("n_starts_per_duration", 0)) != n_starts:
        raise ValueError("reused manifest start count differs from this plan")

    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for regime in REGIMES:
        path = source_root / "manifests" / f"{regime}.csv"
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing real source manifest: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise ValueError(f"unexpected source manifest header: {path}")
            rows = list(reader)
        if len(rows) != len(durations) * n_starts:
            raise ValueError(f"unexpected source manifest row count: {path}")
        for duration in durations:
            duration_rows = [
                row
                for row in rows
                if math.isclose(
                    float(row["target_duration"]),
                    duration,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            if len(duration_rows) != n_starts:
                raise ValueError(f"incomplete duration {duration:g}: {path}")
            if {row["split"] for row in duration_rows} != {"tune", "validate"}:
                raise ValueError(f"missing tune/validate rows: {path}")
        paths[regime] = path.resolve()
        hashes[regime] = sha256(path)
    return paths, hashes


def parse_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"duplicate metadata key {key}: {path}")
        values[key] = value
    return values


def load_external_comparator(
    code_root: Path,
    requested_plan: Path,
    label: str,
    c_scale: float,
    boxsize: float,
    sb_radius: int,
    expected_manifest_hashes: dict[str, str],
) -> dict[str, object]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None:
        raise ValueError("external comparator label must be filename-safe")
    plan_root = existing_plan_root(code_root, requested_plan)
    jobs_path = plan_root / "jobs.json"
    if jobs_path.is_symlink() or not jobs_path.is_file():
        raise FileNotFoundError(f"missing real comparator jobs file: {jobs_path}")
    document = json.loads(jobs_path.read_text(encoding="utf-8"))
    raw_jobs = document.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("comparator jobs document has no job list")

    results: dict[str, dict[str, object]] = {}
    for regime in REGIMES:
        matches = [
            job
            for job in raw_jobs
            if isinstance(job, dict)
            and job.get("regime") == regime
            and math.isclose(float(job.get("C", -1)), c_scale, abs_tol=1e-12)
            and math.isclose(float(job.get("boxsize", -1)), boxsize, abs_tol=1e-12)
            and int(job.get("sb_radius", 0)) == sb_radius
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one comparator job for {regime}; found {len(matches)}"
            )
        job = matches[0]
        out_dir = Path(str(job["out_dir"])).resolve()
        if not out_dir.is_relative_to(plan_root / "signatures"):
            raise ValueError(f"comparator output lies outside its plan: {out_dir}")
        prefix = str(job["out_prefix"])
        metadata_path = out_dir / f"{prefix}_metadata.txt"
        births_path = out_dir / f"{prefix}_births.csv"
        if any(
            path.is_symlink() or not path.is_file()
            for path in (metadata_path, births_path)
        ):
            raise FileNotFoundError(f"incomplete comparator output: {out_dir}")
        metadata = parse_metadata(metadata_path)
        if metadata.get("split") != "all":
            raise ValueError(f"comparator is not a combined split: {metadata_path}")
        if not math.isclose(float(metadata["C"]), c_scale, abs_tol=1e-12):
            raise ValueError(f"comparator metric C mismatch: {metadata_path}")
        if not math.isclose(float(metadata["boxsize"]), boxsize, abs_tol=1e-12):
            raise ValueError(f"comparator boxsize mismatch: {metadata_path}")
        if int(metadata["sb_radius"]) != sb_radius:
            raise ValueError(f"comparator sb-radius mismatch: {metadata_path}")
        manifest_path = Path(metadata["manifest"]).resolve()
        manifest_hash = sha256(manifest_path)
        if manifest_hash != expected_manifest_hashes[regime]:
            raise ValueError(f"comparator manifest differs for {regime}")
        positions_path = Path(metadata["positions"]).resolve()
        tangents_path = Path(metadata["tangents"]).resolve()
        results[regime] = {
            "metadata": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "births": str(births_path),
            "births_sha256": sha256(births_path),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "positions": str(positions_path),
            "positions_sha256": sha256(positions_path),
            "tangents": str(tangents_path),
            "tangents_sha256": sha256(tangents_path),
        }
    return {
        "label": label,
        "plan_root": str(plan_root),
        "C": c_scale,
        "metric_mode": "cover_default",
        "cover_default_C": boxsize * sb_radius,
        "boxsize": boxsize,
        "sb_radius": sb_radius,
        "results": results,
    }


def physical_clock_and_arc_mask(raw_path: Path, latent_path: Path):
    with np.load(raw_path, allow_pickle=False) as raw, np.load(
        latent_path, allow_pickle=False
    ) as latent:
        raw_t = np.asarray(raw["t"], dtype=float)
        impacts = np.asarray(raw["impact_times"], dtype=float)
        piece_kind = np.asarray(latent["piece_kind"], dtype=np.uint8)

    clock = np.empty(len(piece_kind), dtype=float)
    arc_mask = piece_kind == 0
    if int(arc_mask.sum()) != len(raw_t):
        raise ValueError(f"arc/time mismatch: {latent_path}")
    clock[arc_mask] = raw_t
    bridge_starts = np.flatnonzero(
        (piece_kind == 1) & np.r_[True, piece_kind[:-1] == 0]
    )
    if len(bridge_starts) != len(impacts):
        raise ValueError(f"bridge/impact mismatch: {latent_path}")
    for bridge_start, impact_time in zip(bridge_starts, impacts):
        bridge_end = bridge_start
        while bridge_end < len(piece_kind) and piece_kind[bridge_end] == 1:
            bridge_end += 1
        clock[bridge_start:bridge_end] = impact_time
    if np.any(np.diff(clock) < -1e-12):
        raise ValueError(f"nonmonotone physical clock: {latent_path}")
    return clock, arc_mask


def make_rows(
    clock: np.ndarray,
    arc_mask: np.ndarray,
    durations: list[float],
    n_starts: int,
    seed: int,
):
    max_duration = max(durations)
    eligible = np.flatnonzero(arc_mask & (clock <= clock[-1] - max_duration))
    if len(eligible) < n_starts:
        raise ValueError("not enough paired arc-time starts")
    rng = np.random.default_rng(seed)
    starts = rng.choice(eligible, size=n_starts, replace=False)
    rows = []
    for target_duration in durations:
        for run_index, start in enumerate(starts, start=1):
            target_time = clock[start] + target_duration
            end = int(np.searchsorted(clock, target_time, side="left"))
            while end < len(clock) and not arc_mask[end]:
                end += 1
            if end == len(clock):
                raise ValueError("no arc sample reaches the duration target")
            if end <= start:
                raise ValueError("duration target produced a degenerate window")
            realized = float(clock[end] - clock[start])
            rows.append(
                {
                    "target_duration": target_duration,
                    "split": "tune" if run_index <= n_starts // 2 else "validate",
                    "run_index": run_index,
                    "start_index": int(start + 1),
                    "end_index": int(end + 1),
                    "realized_duration": realized,
                    "duration_error": realized - target_duration,
                }
            )
    return rows


def job_for(
    code_root: Path,
    output_root: Path,
    arm: str,
    regime: str,
    c_scale: float,
    boxsize: float,
    sb_radius: int,
    rho_max: float,
    metric_mode: str = "cover_default",
    tangents_dir: Path | None = None,
) -> dict[str, object]:
    suffix = number_token(c_scale)
    tied_construction = math.isclose(
        boxsize, c_scale, rel_tol=0.0, abs_tol=1e-12
    ) and sb_radius == 1 and metric_mode == "cover_default"
    legacy_layout = arm == "primary" and tangents_dir is None
    if tied_construction and legacy_layout:
        construction_suffix = f"C_{suffix}"
    else:
        box_suffix = number_token(boxsize)
        metric_suffix = "_metric_explicit" if metric_mode == "explicit" else ""
        construction_suffix = (
            f"C_{suffix}{metric_suffix}_bs_{box_suffix}_sb{sb_radius}"
        )
    if legacy_layout:
        out_dir = output_root / "signatures" / regime / construction_suffix
    else:
        out_dir = output_root / "signatures" / arm / regime / construction_suffix
    out_prefix = f"fine_{regime}_{construction_suffix}"
    data_dir = (
        code_root / "period_doubling" / "data_fine" / "compass_gait_latent"
    )
    tangents = (
        data_dir / f"compass_{regime}_tangents.csv"
        if tangents_dir is None
        else tangents_dir / f"compass_{regime}_tangents.csv"
    ).resolve()
    positions = (data_dir / f"compass_{regime}_positions.csv").resolve()
    argv = [
        "julia",
        "--project=period_doubling/julia",
        "experiments_planned/run_duration_c_radius.jl",
        "--data-dir",
        str(data_dir),
        "--base",
        f"compass_{regime}",
        "--manifest",
        str(output_root / "manifests" / f"{regime}.csv"),
        "--boxsize",
        f"{boxsize:g}",
        "--sb-radius",
        str(sb_radius),
    ]
    if metric_mode == "explicit":
        argv.extend(["--metric-c", f"{c_scale:g}"])
    if tangents_dir is not None:
        argv.extend(["--tangents", str(tangents)])
    argv.extend(
        [
            "--rho-max",
            f"{rho_max:g}",
            "--out-dir",
            str(out_dir),
            "--out-prefix",
            out_prefix,
        ]
    )
    return {
        "job_id": f"{arm}__{regime}__{construction_suffix}",
        "arm": arm,
        "regime": regime,
        "C": c_scale,
        "metric_mode": metric_mode,
        "cover_default_C": boxsize * sb_radius,
        "boxsize": boxsize,
        "sb_radius": sb_radius,
        "rho_max": rho_max,
        "data_dir": str(data_dir),
        "positions": str(positions),
        "tangents": str(tangents),
        "tangent_source": "default" if tangents_dir is None else "override",
        "manifest": str(output_root / "manifests" / f"{regime}.csv"),
        "out_dir": str(out_dir),
        "out_prefix": out_prefix,
        "argv": argv,
    }


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=code_root / "experiments_planned" / "outputs" / "fine_compass_c_radius",
    )
    parser.add_argument("--c-grid", default="0.10:0.025:0.30")
    parser.add_argument(
        "--fixed-boxsize",
        type=float,
        help=(
            "Keep this positional-cover boxsize across the C grid.  Without "
            "--fixed-sb-radius, each integer sb-radius=C/boxsize is derived."
        ),
    )
    parser.add_argument(
        "--fixed-sb-radius",
        type=int,
        help=(
            "Use one explicit sphere-cover radius with --fixed-boxsize; this "
            "form requires a one-value C grid."
        ),
    )
    parser.add_argument(
        "--explicit-metric",
        action="store_true",
        help=(
            "Treat each C-grid value as an explicit DynamicDistance coefficient "
            "independent of the comparison cover."
        ),
    )
    parser.add_argument(
        "--diagnostic-arm",
        action="append",
        type=parse_diagnostic_arm,
        metavar="LABEL:BOXSIZE:SB_RADIUS[:TANGENTS_DIR]",
        help=(
            "Prepare a named arm; repeat to share the same manifests across "
            "cover resolutions and tangent sources. Requires --explicit-metric."
        ),
    )
    parser.add_argument(
        "--reuse-manifests-from",
        type=Path,
        help=(
            "Copy manifests byte-for-byte from an existing plan below "
            "experiments_planned/outputs and record their hashes."
        ),
    )
    parser.add_argument(
        "--external-comparator-plan",
        type=Path,
        help="Record and hash one completed comparator from another plan.",
    )
    parser.add_argument(
        "--external-comparator-label",
        default="external_comparator",
    )
    parser.add_argument("--external-comparator-boxsize", type=float)
    parser.add_argument("--external-comparator-sb-radius", type=int)
    parser.add_argument(
        "--curve-bound-pair",
        action="append",
        metavar="LEFT:RIGHT",
        help=(
            "Require exact rowwise curve_bound equality after both named arms "
            "are complete; either name may be an external comparator label."
        ),
    )
    parser.add_argument("--duration-grid", default="0.25:0.25:7.50")
    parser.add_argument("--rho-max", type=float, default=1.75)
    parser.add_argument("--n-starts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Write manifests and commands; never runs the commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    code_root = Path(__file__).resolve().parents[1]
    output_root = safe_output_root(code_root, args.output_root)
    c_values = float_grid(args.c_grid)
    durations = float_grid(args.duration_grid)
    if any(value <= 0 for value in c_values):
        raise ValueError("every C value must be positive")
    if any(value <= 0 for value in durations):
        raise ValueError("every target duration must be positive")
    if args.n_starts < 2 or args.n_starts % 2:
        raise ValueError("--n-starts must be an even integer >= 2")
    if args.rho_max <= 0:
        raise ValueError("--rho-max must be positive")
    diagnostic_specs = args.diagnostic_arm or []
    if diagnostic_specs and not args.explicit_metric:
        raise ValueError("--diagnostic-arm requires --explicit-metric")
    diagnostic_labels = [str(spec["label"]) for spec in diagnostic_specs]
    if len(diagnostic_labels) != len(set(diagnostic_labels)):
        raise ValueError("diagnostic arm labels must be unique")

    fixed_construction = args.fixed_boxsize is not None
    if args.fixed_sb_radius is not None and not fixed_construction:
        raise ValueError("--fixed-sb-radius requires --fixed-boxsize")
    if fixed_construction and args.fixed_boxsize <= 0:
        raise ValueError("fixed boxsize must be positive")
    if args.fixed_sb_radius is not None:
        if len(c_values) != 1:
            raise ValueError("an explicit fixed sb-radius requires exactly one C")
        if args.fixed_sb_radius <= 0:
            raise ValueError("fixed sb-radius must be positive")
    if diagnostic_specs and (
        args.fixed_boxsize is not None or args.fixed_sb_radius is not None
    ):
        raise ValueError(
            "diagnostic arms state their own covers; omit fixed cover options"
        )
    if args.explicit_metric and not diagnostic_specs and (
        args.fixed_boxsize is None or args.fixed_sb_radius is None
    ):
        raise ValueError(
            "an explicit metric requires both fixed cover options or diagnostic arms"
        )

    for spec in diagnostic_specs:
        tangents_dir = spec["tangents_dir"]
        if tangents_dir is None:
            continue
        if not tangents_dir.is_dir() or tangents_dir.is_symlink():
            raise ValueError(
                f"arm tangents directory must be real and existing: {tangents_dir}"
            )
        for regime in REGIMES:
            tangent_path = tangents_dir / f"compass_{regime}_tangents.csv"
            if tangent_path.is_symlink() or not tangent_path.is_file():
                raise FileNotFoundError(f"missing real arm tangents: {tangent_path}")

    primary_constructions = []
    experiment_arms = []
    if diagnostic_specs:
        for c_scale in c_values:
            for spec in diagnostic_specs:
                experiment_arms.append(
                    {
                        "label": spec["label"],
                        "C": c_scale,
                        "metric_mode": "explicit",
                        "cover_default_C": (
                            float(spec["boxsize"]) * int(spec["sb_radius"])
                        ),
                        "boxsize": spec["boxsize"],
                        "sb_radius": spec["sb_radius"],
                        "tangent_source": (
                            "default"
                            if spec["tangents_dir"] is None
                            else "override"
                        ),
                        "tangents_dir": (
                            None
                            if spec["tangents_dir"] is None
                            else str(spec["tangents_dir"])
                        ),
                    }
                )
            primary_constructions.append(dict(experiment_arms[-len(diagnostic_specs)]))
        control_constructions = []
    else:
        for c_scale in c_values:
            boxsize = args.fixed_boxsize if fixed_construction else c_scale
            if args.fixed_sb_radius is not None:
                sb_radius = args.fixed_sb_radius
            elif fixed_construction:
                ratio = c_scale / boxsize
                sb_radius = round(ratio)
                if sb_radius <= 0 or not math.isclose(
                    ratio, sb_radius, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "every C/fixed-boxsize ratio must be a positive integer"
                    )
            else:
                sb_radius = 1
            metric_mode = "explicit" if args.explicit_metric else "cover_default"
            if metric_mode == "cover_default" and not math.isclose(
                boxsize * sb_radius,
                c_scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "cover-default construction does not satisfy C=boxsize*sb"
                )
            construction = {
                "C": c_scale,
                "metric_mode": metric_mode,
                "cover_default_C": boxsize * sb_radius,
                "boxsize": boxsize,
                "sb_radius": sb_radius,
                "label": (
                    f"primary_bs_{boxsize:.12g}_sb_{sb_radius}".replace(".", "p")
                ),
                "tangent_source": "default",
                "tangents_dir": None,
            }
            primary_constructions.append(construction)
            experiment_arms.append({**construction, "label": "primary"})

        control_constructions = [] if args.explicit_metric else [
            {
                "label": "control_bs_half_sb_2",
                "boxsize_over_C": 0.5,
                "sb_radius": 2,
            }
        ]
        if fixed_construction and not args.explicit_metric:
            control_constructions.append(
                {
                    "label": "control_bs_equal_C_sb_1",
                    "boxsize_over_C": 1.0,
                    "sb_radius": 1,
                }
            )

    reused_manifest_paths: dict[str, Path] | None = None
    reused_manifest_hashes: dict[str, str] | None = None
    manifests: dict[str, list[dict[str, object]]] = {}
    if args.reuse_manifests_from is not None:
        source_root = existing_plan_root(code_root, args.reuse_manifests_from)
        if source_root == output_root:
            raise ValueError("manifest source and output root must differ")
        reused_manifest_paths, reused_manifest_hashes = load_reused_manifests(
            source_root,
            durations,
            args.n_starts,
        )
    else:
        source_root = None
        for regime_index, regime in enumerate(REGIMES):
            data_root = code_root / "period_doubling" / "data_fine"
            clock, arc_mask = physical_clock_and_arc_mask(
                data_root / "compass_gait" / f"compass_{regime}.npz",
                data_root / "compass_gait_latent" / f"compass_{regime}.npz",
            )
            manifests[regime] = make_rows(
                clock,
                arc_mask,
                durations,
                args.n_starts,
                args.seed + regime_index,
            )

    override_arms = [
        arm for arm in experiment_arms if arm.get("tangent_source") == "override"
    ]
    if override_arms and reused_manifest_hashes is None:
        raise ValueError(
            "override tangent arms require byte-for-byte reused manifests so "
            "provenance can bind the exact windows"
        )
    provenance_by_directory: dict[Path, dict[str, object]] = {}
    for arm in override_arms:
        directory = Path(str(arm["tangents_dir"])).resolve()
        provenance = provenance_by_directory.get(directory)
        if provenance is None:
            provenance = validate_tangent_provenance(
                code_root,
                directory,
                reused_manifest_hashes,
            )
            provenance_by_directory[directory] = provenance
        arm["tangent_provenance"] = provenance
    for primary in primary_constructions:
        match = next(
            (
                arm
                for arm in experiment_arms
                if arm["label"] == primary["label"]
                and math.isclose(
                    float(arm["C"]), float(primary["C"]), abs_tol=1e-12
                )
            ),
            None,
        )
        if match is not None and "tangent_provenance" in match:
            primary["tangent_provenance"] = match["tangent_provenance"]

    external_comparators: list[dict[str, object]] = []
    if args.external_comparator_plan is not None:
        if reused_manifest_hashes is None:
            raise ValueError(
                "an external comparator requires --reuse-manifests-from"
            )
        if len(c_values) != 1:
            raise ValueError("an external comparator requires a one-value C grid")
        if (
            args.external_comparator_boxsize is None
            or args.external_comparator_sb_radius is None
        ):
            raise ValueError(
                "external comparator boxsize and sb-radius are required"
            )
        external_comparators.append(
            load_external_comparator(
                code_root,
                args.external_comparator_plan,
                args.external_comparator_label,
                c_values[0],
                args.external_comparator_boxsize,
                args.external_comparator_sb_radius,
                reused_manifest_hashes,
            )
        )
    elif (
        args.external_comparator_boxsize is not None
        or args.external_comparator_sb_radius is not None
    ):
        raise ValueError("external comparator cover options require its plan")

    known_pair_names = set(diagnostic_labels) | {
        str(item["label"]) for item in external_comparators
    }
    curve_bound_invariants = []
    for spec in args.curve_bound_pair or []:
        parts = spec.split(":")
        if len(parts) != 2 or any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", part) is None
            for part in parts
        ):
            raise ValueError("curve-bound pair must be LEFT:RIGHT")
        if any(part not in known_pair_names for part in parts):
            raise ValueError(f"curve-bound pair names an unknown arm: {spec}")
        curve_bound_invariants.append(
            {"left": parts[0], "right": parts[1], "comparison": "exact_rowwise"}
        )

    jobs = [
        job_for(
            code_root,
            output_root,
            str(arm["label"]),
            regime,
            float(arm["C"]),
            float(arm["boxsize"]),
            int(arm["sb_radius"]),
            args.rho_max,
            str(arm["metric_mode"]),
            None if arm["tangents_dir"] is None else Path(str(arm["tangents_dir"])),
        )
        for arm in experiment_arms
        for regime in REGIMES
    ]
    for job in jobs:
        positions_path = Path(str(job["positions"]))
        tangents_path = Path(str(job["tangents"]))
        if positions_path.is_symlink() or not positions_path.is_file():
            raise FileNotFoundError(f"missing real positions: {positions_path}")
        if tangents_path.is_symlink() or not tangents_path.is_file():
            raise FileNotFoundError(f"missing real tangents: {tangents_path}")
        job["positions_sha256"] = sha256(positions_path)
        job["tangents_sha256"] = sha256(tangents_path)
        if reused_manifest_hashes is not None:
            job["manifest_sha256"] = reused_manifest_hashes[str(job["regime"])]
        matching_arm = next(
            arm
            for arm in experiment_arms
            if arm["label"] == job["arm"]
            and math.isclose(float(arm["C"]), float(job["C"]), abs_tol=1e-12)
        )
        provenance = matching_arm.get("tangent_provenance")
        if provenance is not None:
            regime_provenance = provenance["regimes"][str(job["regime"])]
            if regime_provenance["tangents_sha256"] != job["tangents_sha256"]:
                raise ValueError("planned tangent hash differs from its provenance")
            if regime_provenance["positions_sha256"] != job["positions_sha256"]:
                raise ValueError("planned positions hash differs from tangent provenance")
            job["tangent_provenance"] = provenance["path"]
            job["tangent_provenance_sha256"] = provenance["sha256"]
            job["tangent_provenance_regime"] = regime_provenance
            argument_index = job["argv"].index("--tangents") + 2
            job["argv"][argument_index:argument_index] = [
                "--tangent-provenance",
                str(provenance["path"]),
            ]
    commands = [shlex.join(job["argv"]) for job in jobs]
    plan = {
        "status": "prepared_not_executed",
        "stage": (
            "diagnostic_arms_not_executed"
            if diagnostic_specs
            else "combined_tune_and_validation_rows"
        ),
        "c_grid": c_values,
        "metric_policy": "explicit" if args.explicit_metric else "cover_default",
        "fixed_primary_construction": fixed_construction,
        "primary_constructions": primary_constructions,
        "experiment_arms": experiment_arms if diagnostic_specs else None,
        "external_comparators": external_comparators,
        "curve_bound_invariants": curve_bound_invariants,
        "control_constructions": control_constructions,
        "sb_radius": primary_constructions[0]["sb_radius"],
        "rho_grid": [0.0, 0.025, args.rho_max],
        "duration_grid_seconds": durations,
        "n_starts_per_duration": args.n_starts,
        "tune_starts": args.n_starts // 2,
        "validation_starts": args.n_starts // 2,
        "seed": args.seed,
        "manifest_provenance": (
            {
                "mode": "byte_for_byte_copy",
                "source_root": str(source_root),
                "sha256": reused_manifest_hashes,
            }
            if source_root is not None
            else {
                "mode": "generated",
                "seed": args.seed,
            }
        ),
        "jobs_file": "jobs.json",
        "commands": commands,
    }
    jobs_document = {
        "schema_version": 1,
        "kind": "fine_compass_duration_c_radius_jobs",
        "status": "prepared_not_executed",
        "plan_root": str(output_root),
        "working_directory": str(code_root),
        "jobs": jobs,
    }

    print(json.dumps({key: value for key, value in plan.items() if key != "commands"}, indent=2))
    print(f"prepared {len(commands)} Julia commands; none executed")
    if not args.materialize:
        for command in commands:
            print(command)
        return 0
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing planned output: {output_root}"
        )

    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    if reused_manifest_paths is not None:
        for regime, source in reused_manifest_paths.items():
            target = manifest_root / f"{regime}.csv"
            shutil.copyfile(source, target)
            if sha256(target) != reused_manifest_hashes[regime]:
                raise RuntimeError(f"manifest copy hash mismatch: {target}")
    else:
        fieldnames = list(next(iter(manifests.values()))[0])
        for regime, rows in manifests.items():
            with (manifest_root / f"{regime}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    (output_root / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "jobs.json").write_text(
        json.dumps(jobs_document, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "commands.sh").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )
    print(f"wrote plan beneath {output_root}; commands remain unexecuted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
