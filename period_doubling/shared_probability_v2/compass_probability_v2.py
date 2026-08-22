#!/usr/bin/env python3
"""Strict Compass-gait counterpart of the David-family v2 analysis.

The default ``check`` action is read-only.  It validates a hash-bound input
bundle and invokes the unchanged Julia kernel in ``--check-only`` mode.  The
``materialize`` and ``execute`` actions are explicit and write only to a new
named directory below ``experiments_planned/outputs/shared_coauthor_protocol``.

The bundled frozen-path control is deliberately represented by a replaceable
manifest.  A future smoothed/Fourier Compass bundle can therefore be bound by
an immutable manifest without changing the probability statistic or kernel.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
HERE = SCRIPT_PATH.parent
CODE_ROOT = HERE.parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
DEFAULT_CASES_PATH = HERE / "compass_cases.json"
SAFE_OUTPUT_ROOT = (
    CODE_ROOT / "experiments_planned" / "outputs" / "shared_coauthor_protocol"
)
JULIA_KERNEL = CODE_ROOT / "period_doubling" / "julia" / "run_shared_probability.jl"
JULIA_BETA_PREFLIGHT = HERE / "preflight_compass_bundle_v2.jl"
JULIA_PROJECT = CODE_ROOT / "period_doubling" / "julia" / "Project.toml"
JULIA_MANIFEST = CODE_ROOT / "period_doubling" / "julia" / "Manifest.toml"
CYCLING_REPO = CODE_ROOT / "CyclingSignatures.jl"
KERNEL_INTERNAL_PROTOCOL_LABEL = "coauthor_roessler_probability_v1"

EXPECTED_CASES = (
    ("period1", "phi = 4.00 deg", 4.00, 1, 0.880, 0.7482409701092134),
    ("period2", "phi = 4.75 deg", 4.75, 2, 1.760, 1.5021396458781595),
    ("period4", "phi = 5.00 deg", 5.00, 4, 3.520, 3.0019139378989905),
    ("period8", "phi = 5.02 deg", 5.02, 8, 7.045, 6.004311980796729),
    ("chaos", "phi = 5.20 deg", 5.20, None, None, None),
)


@dataclass(frozen=True)
class Configuration:
    cases_path: Path
    protocol_path: Path
    bundle_manifest_path: Path
    cases: dict[str, Any]
    protocol: dict[str, Any]


@dataclass(frozen=True)
class ValidatedBundle:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    cases: tuple[dict[str, Any], ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as error:
        raise ValueError(f"missing JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def reject_symlink_components(path: Path, label: str) -> None:
    probe = path.absolute()
    while True:
        if probe.is_symlink():
            raise ValueError(f"{label} contains a symlink: {probe}")
        parent = probe.parent
        if parent == probe:
            return
        probe = parent


def resolve_existing_file(path: Path, label: str) -> Path:
    reject_symlink_components(path, label)
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing {label}: {resolved}")
    return resolved


def resolve_code_relative(value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or value in {"", ".", ".."} or ".." in relative.parts:
        raise ValueError(f"{label} must be a nonempty code-relative path")
    resolved = (CODE_ROOT / relative).resolve()
    if not resolved.is_relative_to(CODE_ROOT):
        raise ValueError(f"{label} escapes the code root: {value!r}")
    return resolve_existing_file(resolved, label)


def command_output(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def execution_environment(julia_bin: str) -> dict[str, Any]:
    resolved_julia = shutil.which(julia_bin)
    if resolved_julia is None:
        raise ValueError(f"Julia executable not found: {julia_bin}")
    nested_status = command_output(["git", "status", "--porcelain"], CYCLING_REPO)
    if nested_status:
        raise ValueError("CyclingSignatures.jl must be clean")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "julia_executable": str(Path(resolved_julia).resolve()),
        "julia_version": command_output([resolved_julia, "--version"], CODE_ROOT),
        "julia_project": str(JULIA_PROJECT.resolve()),
        "julia_project_sha256": sha256(JULIA_PROJECT),
        "julia_manifest": str(JULIA_MANIFEST.resolve()),
        "julia_manifest_sha256": sha256(JULIA_MANIFEST),
        "julia_kernel": str(JULIA_KERNEL.resolve()),
        "julia_kernel_sha256": sha256(JULIA_KERNEL),
        "julia_beta_preflight": str(JULIA_BETA_PREFLIGHT.resolve()),
        "julia_beta_preflight_sha256": sha256(JULIA_BETA_PREFLIGHT),
        "cycling_repo": str(CYCLING_REPO.resolve()),
        "cycling_repo_head": command_output(
            ["git", "rev-parse", "--verify", "HEAD"], CYCLING_REPO
        ),
        "cycling_repo_status": "clean",
    }


def segment_lengths(protocol: dict[str, Any]) -> list[int]:
    spec = protocol["segment_lengths"]
    start, step, stop = int(spec["start"]), int(spec["step"]), int(spec["stop"])
    if start < 2 or step <= 0 or stop < start or (stop - start) % step:
        raise ValueError("invalid segment-length grid")
    return list(range(start, stop + 1, step))


def validate_protocol(protocol: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version", "protocol_id", "statistic", "segment_lengths",
        "n_runs", "sampling", "seed", "duration_convention",
        "effective_sample_dt", "tangent_normalization", "position_boxsize",
        "sphere_box_resolution", "metric_c", "r_min", "r_max", "r_step",
        "r_subdivisions", "field_prime", "require_sample_radius_below_r_max",
        "plot_horizontal_radius_guide", "plot_vertical_nominal_suspension_guides",
    }
    if set(protocol) != expected_keys:
        raise ValueError(f"unexpected protocol keys: {sorted(set(protocol) ^ expected_keys)}")
    if int(protocol["schema_version"]) != 2:
        raise ValueError("Compass protocol requires schema_version=2")
    if protocol["protocol_id"] != "compass_fourier_embedded_probability_linf_david_grid_v2":
        raise ValueError("unexpected Compass protocol id")
    if protocol["statistic"] != "P(rank > 0)":
        raise ValueError("statistic must be P(rank > 0)")
    if segment_lengths(protocol) != list(range(400, 4801, 80)):
        raise ValueError(
            "refined Compass grid must be lengths 400:80:4800, preserving "
            "David's displayed durations 1:0.2:12"
        )
    expected: dict[str, Any] = {
        "n_runs": 20,
        "sampling": "independent_uniform_with_replacement_per_length",
        "seed": 20260820,
        "duration_convention": "segment_length_times_effective_dt",
        "effective_sample_dt": 0.0025,
        "tangent_normalization": "linf",
        "position_boxsize": 5.0,
        "sphere_box_resolution": 1,
        "metric_c": 5.0,
        "r_min": 0.0,
        "r_max": 5.0,
        "r_step": 0.025,
        "r_subdivisions": 201,
        "field_prime": 43,
        "require_sample_radius_below_r_max": True,
        "plot_horizontal_radius_guide": False,
        "plot_vertical_nominal_suspension_guides": True,
    }
    for key, wanted in expected.items():
        actual = protocol[key]
        matches = (
            math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(wanted, float)
            else actual == wanted
        )
        if not matches:
            raise ValueError(f"protocol {key}={actual!r}, expected {wanted!r}")


def validate_cases(document: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version", "analysis_id", "protocol", "bundle_manifest",
        "suggested_output_root", "figure_filename", "cases",
    }
    if set(document) != expected_keys:
        raise ValueError(f"unexpected cases keys: {sorted(set(document) ^ expected_keys)}")
    expected_scalars = {
        "schema_version": 2,
        "analysis_id": "compass_fourier_embedded_probability_linf_v2_david_grid",
        "protocol": "compass_protocol.json",
        "figure_filename": "compassgait_C5p0.pdf",
        "suggested_output_root": (
            "experiments_planned/outputs/shared_coauthor_protocol/"
            "compass_fourier_embedded_probability_linf_v2_david_grid"
        ),
    }
    for key, wanted in expected_scalars.items():
        if document[key] != wanted:
            raise ValueError(f"unexpected cases value for {key}")
    if not isinstance(document["bundle_manifest"], str):
        raise ValueError("cases must name a bundle manifest")
    actual_cases = document["cases"]
    if not isinstance(actual_cases, list) or len(actual_cases) != 5:
        raise ValueError("cases must contain period1/2/4/8 and chaos")
    keys = {
        "id", "title", "phi_deg", "q", "nominal_suspension_period",
        "physical_return_seconds",
    }
    for actual, expected in zip(actual_cases, EXPECTED_CASES):
        case_id, title, phi_deg, q, nominal_period, physical_period = expected
        if set(actual) != keys or actual["id"] != case_id or actual["title"] != title:
            raise ValueError(f"unexpected case record for {case_id}")
        if actual["q"] != q or not math.isclose(
            float(actual["phi_deg"]), phi_deg, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"unexpected phi/q for {case_id}")
        for key, wanted in (
            ("nominal_suspension_period", nominal_period),
            ("physical_return_seconds", physical_period),
        ):
            if wanted is None:
                if actual[key] is not None:
                    raise ValueError(f"chaos must not have {key}")
            elif not math.isclose(
                float(actual[key]), wanted, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"unexpected {key} for {case_id}")


def load_configuration(cases_path: Path, bundle_override: Path | None) -> Configuration:
    cases_path = resolve_existing_file(cases_path, "Compass cases document")
    cases = load_json(cases_path)
    validate_cases(cases)
    protocol_path = resolve_existing_file(
        cases_path.parent / str(cases["protocol"]), "Compass protocol"
    )
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    bundle_path = resolve_existing_file(
        bundle_override if bundle_override is not None else cases_path.parent / str(cases["bundle_manifest"]),
        "Compass input bundle",
    )
    return Configuration(cases_path, protocol_path, bundle_path, cases, protocol)


def validate_hash_record(record: Any, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} must bind path and sha256")
    path = resolve_code_relative(str(record["path"]), label)
    actual_hash = sha256(path)
    if actual_hash != record["sha256"]:
        raise ValueError(f"{label} hash changed: {path}")
    return path, actual_hash


def validate_tangent_provenance(
    record: dict[str, Any], manifest_record: dict[str, Any], positions: Path, tangents: Path
) -> None:
    case_id = str(record["id"])
    provenance_path = resolve_code_relative(
        str(manifest_record["path"]), "learned-flow tangent provenance"
    )
    if sha256(provenance_path) != manifest_record["sha256"]:
        raise ValueError("learned-flow tangent provenance hash changed")
    provenance = load_json(provenance_path)
    if provenance.get("kind") != manifest_record["kind"]:
        raise ValueError("unexpected learned-flow provenance kind")
    matches = [item for item in provenance.get("regimes", []) if item.get("regime") == case_id]
    if len(matches) != 1:
        raise ValueError(f"{case_id}: missing unique tangent provenance record")
    item = matches[0]
    if float(item.get("config_w_v", math.nan)) != float(manifest_record["config_w_v"]):
        raise ValueError(f"{case_id}: learned-flow bundle must use w_v=0")
    expected = (
        ("source_positions_csv", "source_positions_csv_sha256", positions),
        ("learned_flow_tangents_csv", "learned_flow_tangents_csv_sha256", tangents),
        ("source_archive", "source_archive_sha256", None),
        ("source_encoder_jvp_tangents_csv", "source_encoder_jvp_tangents_csv_sha256", None),
        ("checkpoint", "checkpoint_sha256", None),
        ("config", "config_sha256", None),
    )
    for path_key, hash_key, bound_path in expected:
        path = resolve_code_relative(str(Path(item[path_key]).resolve().relative_to(CODE_ROOT)), path_key)
        if bound_path is not None and path != bound_path:
            raise ValueError(f"{case_id}: provenance binds another {path_key}")
        if sha256(path) != item[hash_key]:
            raise ValueError(f"{case_id}: provenance hash failed for {path_key}")
    exporter = resolve_existing_file(Path(provenance["exporter_script"]), "tangent exporter")
    if sha256(exporter) != provenance["exporter_script_sha256"]:
        raise ValueError("learned-flow tangent exporter changed")


def global_curve_bound(positions: np.ndarray, tangents: np.ndarray, metric_c: float) -> float:
    norms = np.max(np.abs(tangents), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("analysis tangents must be finite and nonzero")
    normalized = tangents / norms[:, np.newaxis]
    dx = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dv = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    return float(max(float(np.max(dx)), metric_c * float(np.max(dv))))


def validate_return(
    archive_path: Path, case_id: str, phi_deg: float, q: int, expected: float, trim: int
) -> dict[str, Any]:
    with np.load(archive_path, allow_pickle=False) as source:
        required = {"impact_times", "jump_plus", "meta_json"}
        if not required.issubset(source.files):
            raise ValueError(f"{case_id}: physical archive lacks return arrays")
        impact_times = np.asarray(source["impact_times"], dtype=float)
        jump_plus = np.asarray(source["jump_plus"], dtype=float)
        meta = json.loads(str(source["meta_json"].item()))
    if meta.get("label") != case_id or int(meta.get("expected_period", -1)) != q:
        raise ValueError(f"{case_id}: physical archive metadata changed")
    if not math.isclose(float(meta["phi_deg"]), phi_deg, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{case_id}: physical archive phi changed")
    if len(impact_times) <= trim + 2 * q or len(jump_plus) != len(impact_times):
        raise ValueError(f"{case_id}: insufficient impact returns")
    durations = impact_times[trim + q :] - impact_times[trim:-q]
    measured = float(np.median(durations))
    if not math.isclose(measured, expected, rel_tol=0.0, abs_tol=5e-13):
        raise ValueError(f"{case_id}: measured return changed ({measured:.17g})")
    closure = np.linalg.norm(jump_plus[trim + q :] - jump_plus[trim:-q], axis=1)
    proper_divisors = [lag for lag in range(1, q) if q % lag == 0]
    divisor_medians = {
        str(lag): float(np.median(np.linalg.norm(
            jump_plus[trim + lag :] - jump_plus[trim:-lag], axis=1
        ))) for lag in proper_divisors
    }
    return {
        "method": "event_time_recurrence",
        "trim_impacts": trim,
        "full_return_lag_impacts": q,
        "median_return_seconds": measured,
        "median_recurrence_residual": float(np.median(closure)),
        "proper_divisor_median_residuals": divisor_medians,
    }


def display_extract_spec(archive_path: Path, case_id: str, q: int | None) -> dict[str, Any]:
    with np.load(archive_path, allow_pickle=False) as source:
        t = np.asarray(source["t"], dtype=float)
        positions = np.asarray(source["x"], dtype=float)
        piece_kind = np.asarray(source["piece_kind"], dtype=np.uint8)
    if q is None:
        span = int(round(50.0 / 0.005))
        stop = len(t) - 1
        start = stop - span
        kind = "late_50_unit_frozen_encoded_path"
    else:
        transitions = np.flatnonzero(np.diff(piece_kind) != 0) + 1
        run_starts = np.r_[0, transitions]
        arc_starts = run_starts[piece_kind[run_starts] == 0]
        if len(arc_starts) < q + 3:
            raise ValueError(f"{case_id}: insufficient complete encoded strides")
        start = int(arc_starts[-(q + 2)])
        stop = int(arc_starts[-2])
        kind = "one_full_q_impact_frozen_encoded_orbit"
    if not (0 <= start < stop < len(t)):
        raise ValueError(f"{case_id}: invalid display extract")
    return {
        "kind": kind,
        "source_start_index_zero_based": start,
        "source_end_index_zero_based_inclusive": stop,
        "n_rows": stop - start + 1,
        "nominal_suspension_duration": float(t[stop] - t[start]),
        "positions": positions[start : stop + 1],
        "piece_kind": piece_kind[start : stop + 1],
        "time": t[start : stop + 1] - t[start],
    }


def validate_nominal_suspension_period(
    archive_path: Path, case_id: str, q: int, raw_dt: float, expected: float
) -> dict[str, Any]:
    """Validate a plotted guide in the kernel's literal stored-row clock."""
    with np.load(archive_path, allow_pickle=False) as source:
        piece_kind = np.asarray(source["piece_kind"], dtype=np.uint8)
    transitions = np.flatnonzero(np.diff(piece_kind) != 0) + 1
    run_starts = np.r_[0, transitions]
    arc_starts = run_starts[piece_kind[run_starts] == 0]
    if len(arc_starts) <= q:
        raise ValueError(f"{case_id}: insufficient arc starts for suspension guide")
    row_spans = arc_starts[q:] - arc_starts[:-q]
    measured = float(np.median(row_spans) * raw_dt)
    if not math.isclose(measured, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{case_id}: nominal suspension period changed ({measured:.17g})"
        )
    return {
        "method": "median_q_impact_arc_start_row_span_times_raw_sample_dt",
        "clock": "literal_pre_stride_stored_row_suspension_clock",
        "q": q,
        "raw_sample_dt": raw_dt,
        "n_spans": int(len(row_spans)),
        "median_row_span": float(np.median(row_spans)),
        "nominal_suspension_period": measured,
    }


def _validate_frozen_path_bundle_deprecated(configuration: Configuration) -> ValidatedBundle:
    manifest = load_json(configuration.bundle_manifest_path)
    expected_keys = {
        "schema_version", "bundle_id", "status", "source_role", "scientific_scope",
        "raw_sample_dt", "analysis_stride", "effective_sample_dt", "dimension",
        "tangent_provenance", "return_provenance",
        "suspension_period_provenance", "cases",
    }
    if set(manifest) != expected_keys:
        raise ValueError(f"unexpected bundle keys: {sorted(set(manifest) ^ expected_keys)}")
    if int(manifest["schema_version"]) != 1 or manifest["status"] != "complete":
        raise ValueError("Compass bundle must be schema 1 and complete")
    for key in ("bundle_id", "source_role", "scientific_scope"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"Compass bundle lacks {key}")
    raw_dt = float(manifest["raw_sample_dt"])
    stride = int(manifest["analysis_stride"])
    effective_dt = float(manifest["effective_sample_dt"])
    dimension = int(manifest["dimension"])
    if not math.isclose(raw_dt * stride, effective_dt, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("bundle cadence/stride mismatch")
    if not math.isclose(
        effective_dt, float(configuration.protocol["effective_sample_dt"]),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("bundle effective dt differs from the frozen protocol")
    if dimension < 1 or stride < 1:
        raise ValueError("invalid bundle dimension/stride")
    tangent_provenance = manifest["tangent_provenance"]
    if not isinstance(tangent_provenance, dict) or set(tangent_provenance) != {
        "path", "sha256", "kind", "config_w_v"
    }:
        raise ValueError("unexpected tangent-provenance binding")
    if float(tangent_provenance["config_w_v"]) != 0.0:
        raise ValueError("Compass control requires w_v=0")
    return_provenance = manifest["return_provenance"]
    if not isinstance(return_provenance, dict) or set(return_provenance) != {
        "validator", "validator_sha256", "method", "trim_impacts"
    }:
        raise ValueError("unexpected return-provenance binding")
    validator = resolve_code_relative(str(return_provenance["validator"]), "return validator")
    if sha256(validator) != return_provenance["validator_sha256"]:
        raise ValueError("return validator changed")
    trim = int(return_provenance["trim_impacts"])
    if trim != 10:
        raise ValueError("return provenance must discard ten impacts")
    suspension_provenance = manifest["suspension_period_provenance"]
    if not isinstance(suspension_provenance, dict) or set(suspension_provenance) != {
        "method", "clock", "plotted_as_vertical_guide"
    }:
        raise ValueError("unexpected suspension-period provenance binding")
    if suspension_provenance["plotted_as_vertical_guide"] is not True:
        raise ValueError("nominal suspension periods must be the plotted guides")

    records = manifest["cases"]
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("bundle must contain five cases")
    by_id = {str(record.get("id")): record for record in records}
    if set(by_id) != {case[0] for case in EXPECTED_CASES}:
        raise ValueError("bundle case ids changed")
    validated: list[dict[str, Any]] = []
    case_documents = {case["id"]: case for case in configuration.cases["cases"]}
    for case_id, _, phi_deg, q, nominal_period, physical_period in EXPECTED_CASES:
        record = by_id[case_id]
        if set(record) != {
            "id", "phi_deg", "q", "nominal_suspension_period",
            "physical_return_seconds", "n_samples",
            "positions", "tangents", "latent_archive", "physical_archive",
        }:
            raise ValueError(f"{case_id}: unexpected bundle case schema")
        if record["q"] != q or not math.isclose(
            float(record["phi_deg"]), phi_deg, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{case_id}: bundle phi/q mismatch")
        for key in ("nominal_suspension_period", "physical_return_seconds"):
            if record[key] != case_documents[case_id][key]:
                raise ValueError(f"{case_id}: bundle/cases {key} mismatch")
        positions_path, positions_hash = validate_hash_record(record["positions"], f"{case_id} positions")
        tangents_path, tangents_hash = validate_hash_record(record["tangents"], f"{case_id} tangents")
        latent_path, latent_hash = validate_hash_record(record["latent_archive"], f"{case_id} latent archive")
        physical_path, physical_hash = validate_hash_record(record["physical_archive"], f"{case_id} physical archive")
        validate_tangent_provenance(record, tangent_provenance, positions_path, tangents_path)

        n_samples = int(record["n_samples"])
        positions = np.loadtxt(positions_path, dtype=float)
        tangents = np.loadtxt(tangents_path, dtype=float)
        if positions.shape != (n_samples, dimension) or tangents.shape != positions.shape:
            raise ValueError(f"{case_id}: analysis matrix shape changed")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(tangents)):
            raise ValueError(f"{case_id}: analysis matrices contain nonfinite values")
        with np.load(latent_path, allow_pickle=False) as latent:
            required = {"t", "x", "piece_kind", "meta_json"}
            if not required.issubset(latent.files):
                raise ValueError(f"{case_id}: latent archive lacks display arrays")
            latent_t = np.asarray(latent["t"], dtype=float)
            latent_x = np.asarray(latent["x"], dtype=float)
            meta = json.loads(str(latent["meta_json"].item()))
        if not np.array_equal(latent_x, positions):
            raise ValueError(f"{case_id}: positions CSV differs from latent archive")
        if len(latent_t) != n_samples or not np.allclose(
            np.diff(latent_t), raw_dt, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{case_id}: latent suspension cadence changed")
        if meta.get("label") != case_id or meta.get("expected_period") != q:
            raise ValueError(f"{case_id}: latent metadata changed")
        if not math.isclose(float(meta["phi_deg"]), phi_deg, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{case_id}: latent phi changed")
        analysis_positions = positions[::stride]
        analysis_tangents = tangents[::stride]
        if len(analysis_positions) < max(segment_lengths(configuration.protocol)):
            raise ValueError(f"{case_id}: insufficient analysis samples")
        curve_bound = global_curve_bound(
            analysis_positions, analysis_tangents, float(configuration.protocol["metric_c"])
        )
        if not curve_bound < float(configuration.protocol["r_max"]):
            raise ValueError(f"{case_id}: curve bound {curve_bound:g} is not below r_max")
        return_certificate = None
        suspension_period_certificate = None
        if q is not None and physical_period is not None and nominal_period is not None:
            return_certificate = validate_return(
                physical_path, case_id, phi_deg, q, physical_period, trim
            )
            suspension_period_certificate = validate_nominal_suspension_period(
                latent_path, case_id, q, raw_dt, nominal_period
            )
        display = display_extract_spec(latent_path, case_id, q)
        validated.append({
            "id": case_id,
            "title": case_documents[case_id]["title"],
            "phi_deg": phi_deg,
            "q": q,
            "nominal_suspension_period": nominal_period,
            "physical_return_seconds": physical_period,
            "raw_sample_dt": raw_dt,
            "stride": stride,
            "effective_sample_dt": effective_dt,
            "dimension": dimension,
            "n_samples": n_samples,
            "analysis_n_samples": len(analysis_positions),
            "positions": str(positions_path),
            "positions_sha256": positions_hash,
            "tangents": str(tangents_path),
            "tangents_sha256": tangents_hash,
            "latent_archive": str(latent_path),
            "latent_archive_sha256": latent_hash,
            "physical_archive": str(physical_path),
            "physical_archive_sha256": physical_hash,
            "global_curve_bound": curve_bound,
            "return_certificate": return_certificate,
            "suspension_period_certificate": suspension_period_certificate,
            "display_extract": display,
        })
    return ValidatedBundle(
        configuration.bundle_manifest_path,
        sha256(configuration.bundle_manifest_path),
        manifest,
        tuple(validated),
    )


def resolve_bundle_record(
    bundle_root: Path, record: Any, label: str, extra_keys: set[str] | None = None
) -> tuple[Path, str]:
    expected = {"path", "sha256"} | (extra_keys or set())
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError(f"{label} has an unexpected record schema")
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
        raise ValueError(f"{label} must be bundle-relative")
    path = resolve_existing_file(bundle_root / relative, label)
    if not path.is_relative_to(bundle_root):
        raise ValueError(f"{label} escapes the bundle")
    actual = sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"{label} hash changed")
    return path, actual


def validate_bundle_inventory(manifest: dict[str, Any], bundle_root: Path) -> None:
    records = manifest["files"]
    if not isinstance(records, list) or not records:
        raise ValueError("bundle inventory must be nonempty")
    expected_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
            raise ValueError("invalid bundle inventory record")
        relative = str(record["path"])
        if relative in expected_paths:
            raise ValueError(f"duplicate bundle inventory path: {relative}")
        path, _ = resolve_bundle_record(
            bundle_root, {"path": relative, "sha256": record["sha256"]},
            "bundle inventory file",
        )
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"bundle inventory byte count changed: {relative}")
        expected_paths.add(relative)
    actual_paths = {
        str(path.relative_to(bundle_root))
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    if actual_paths != expected_paths:
        raise ValueError(
            f"bundle inventory mismatch: missing={sorted(expected_paths-actual_paths)}, "
            f"extra={sorted(actual_paths-expected_paths)}"
        )


def validate_bundle(configuration: Configuration) -> ValidatedBundle:
    """Validate the final Fourier-closed periodic/mixed-chaos bundle."""
    manifest = load_json(configuration.bundle_manifest_path)
    expected_keys = {
        "schema_version", "bundle_id", "status", "created_utc", "generator",
        "scientific_scope", "analysis_sample_dt", "analysis_duration",
        "analysis_n_samples", "dimension", "metric_c_preflight",
        "maximum_global_curve_bound", "cases", "summary", "files",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            f"unexpected Fourier bundle keys: {sorted(set(manifest) ^ expected_keys)}"
        )
    if (
        int(manifest["schema_version"]) != 1
        or manifest["bundle_id"] != "compass_embedded_fourier_orbits_v1"
        or manifest["status"] != "complete"
    ):
        raise ValueError("unexpected Fourier bundle identity/status")
    bundle_root = configuration.bundle_manifest_path.parent.resolve()
    reject_symlink_components(bundle_root, "Fourier bundle")
    validate_bundle_inventory(manifest, bundle_root)
    generator = manifest["generator"]
    if not isinstance(generator, dict) or set(generator) != {
        "path", "sha256", "python_version", "numpy_version"
    }:
        raise ValueError("unexpected Fourier generator record")
    generator_path = resolve_code_relative(str(generator["path"]), "Fourier builder")
    if sha256(generator_path) != generator["sha256"]:
        raise ValueError("Fourier builder changed")
    scope = manifest["scientific_scope"]
    if not isinstance(scope, dict) or set(scope) != {
        "periodic_positions", "periodic_tangents", "chaos_positions",
        "chaos_tangents", "not_a_learned_rollout",
    }:
        raise ValueError("unexpected Fourier scientific-scope record")
    if scope["not_a_learned_rollout"] is not True:
        raise ValueError("bundle must state that it is not a learned rollout")
    analysis_dt = float(manifest["analysis_sample_dt"])
    analysis_duration = float(manifest["analysis_duration"])
    analysis_n_samples = int(manifest["analysis_n_samples"])
    dimension = int(manifest["dimension"])
    if not (
        math.isclose(analysis_dt, 0.0025, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(analysis_duration, 469.0, rel_tol=0.0, abs_tol=1e-12)
        and analysis_n_samples == 187601
        and dimension == 11
        and math.isclose(float(manifest["metric_c_preflight"]), 5.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("Fourier bundle cadence/duration/dimension changed")
    if not math.isclose(
        analysis_dt, float(configuration.protocol["effective_sample_dt"]),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("Fourier bundle dt differs from the protocol")
    summary_path, _ = resolve_bundle_record(bundle_root, manifest["summary"], "bundle summary")
    if not summary_path.name.endswith(".csv"):
        raise ValueError("bundle summary must be CSV")

    records = manifest["cases"]
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("Fourier bundle must contain five cases")
    by_id = {str(record.get("id")): record for record in records}
    if set(by_id) != {case[0] for case in EXPECTED_CASES}:
        raise ValueError("Fourier bundle case ids changed")
    case_documents = {case["id"]: case for case in configuration.cases["cases"]}
    validated: list[dict[str, Any]] = []
    observed_max_bound = 0.0
    for case_id, _, phi_deg, q, nominal_period, physical_period in EXPECTED_CASES:
        record = by_id[case_id]
        if set(record) != {
            "id", "kind", "phi_deg", "q", "nominal_suspension_period",
            "physical_return_seconds", "dimension", "analysis_sample_dt",
            "analysis_n_samples", "analysis_duration", "tangent_semantics",
            "positions", "tangents", "display", "fourier", "certificate",
            "global_curve_bound",
        }:
            raise ValueError(f"{case_id}: unexpected Fourier case schema")
        expected_kind = (
            "periodic_fourier_closed" if q is not None
            else "chaos_interpolated_frozen_path"
        )
        if record["id"] != case_id or record["kind"] != expected_kind or record["q"] != q:
            raise ValueError(f"{case_id}: kind/q changed")
        for key, wanted in (
            ("phi_deg", phi_deg),
            ("nominal_suspension_period", nominal_period),
            ("physical_return_seconds", physical_period),
            ("analysis_sample_dt", analysis_dt),
            ("analysis_duration", analysis_duration),
        ):
            actual = record[key]
            if wanted is None:
                if actual is not None:
                    raise ValueError(f"{case_id}: {key} must be null")
            elif not math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{case_id}: {key} changed")
        if int(record["dimension"]) != dimension or int(record["analysis_n_samples"]) != analysis_n_samples:
            raise ValueError(f"{case_id}: analysis shape metadata changed")
        if record["nominal_suspension_period"] != case_documents[case_id]["nominal_suspension_period"] or record["physical_return_seconds"] != case_documents[case_id]["physical_return_seconds"]:
            raise ValueError(f"{case_id}: cases/bundle period binding changed")
        positions_path, positions_hash = resolve_bundle_record(
            bundle_root, record["positions"], f"{case_id} positions"
        )
        tangents_path, tangents_hash = resolve_bundle_record(
            bundle_root, record["tangents"], f"{case_id} tangents"
        )
        display_path, display_hash = resolve_bundle_record(
            bundle_root, record["display"], f"{case_id} display",
            {"kind", "n_rows"},
        )
        certificate_path, certificate_hash = resolve_bundle_record(
            bundle_root, record["certificate"], f"{case_id} certificate"
        )
        certificate = load_json(certificate_path)
        if (
            certificate.get("case_id") != case_id
            or certificate.get("status") != "certified_derived_control"
            or certificate.get("q") != q
        ):
            raise ValueError(f"{case_id}: certificate identity changed")
        if certificate.get("analysis", {}).get("n_samples") != analysis_n_samples:
            raise ValueError(f"{case_id}: certificate analysis length changed")
        if q is not None:
            if record["tangent_semantics"] != "analytic_fourier_derivative":
                raise ValueError(f"{case_id}: periodic tangents must be analytic Fourier derivatives")
            fourier_path, fourier_hash = resolve_bundle_record(
                bundle_root, record["fourier"], f"{case_id} Fourier coefficients",
                {"harmonic_cutoff"},
            )
            if int(record["fourier"]["harmonic_cutoff"]) != 6 * q:
                raise ValueError(f"{case_id}: Fourier cutoff must be H=6q")
            fourier_certificate = certificate.get("fourier", {})
            selection = certificate.get("selection_rule", {})
            if (
                int(fourier_certificate.get("harmonics_per_impact", -1)) != 6
                or int(fourier_certificate.get("harmonic_cutoff", -1)) != 6 * q
                or int(selection.get("n_cycles", -1)) != 32
            ):
                raise ValueError(f"{case_id}: Fourier/selection certificate changed")
            expected_beta1 = 1
            fourier_record: dict[str, Any] | None = {
                **record["fourier"], "absolute_path": str(fourier_path),
                "sha256": fourier_hash,
            }
        else:
            if record["fourier"] is not None:
                raise ValueError("chaos must be explicitly nonperiodic")
            if record["tangent_semantics"] != "interpolated_learned_flow_direction_on_interpolated_frozen_path":
                raise ValueError("chaos tangent semantics changed")
            if "mixed semantics" not in certificate.get("analysis_tangent_semantics", ""):
                raise ValueError("chaos certificate must disclose mixed tangent semantics")
            expected_beta1 = None
            fourier_record = None

        positions = np.loadtxt(positions_path, dtype=float)
        tangents = np.loadtxt(tangents_path, dtype=float)
        if positions.shape != (analysis_n_samples, dimension) or tangents.shape != positions.shape:
            raise ValueError(f"{case_id}: analysis matrix shape changed")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(tangents)):
            raise ValueError(f"{case_id}: analysis matrix contains nonfinite values")
        curve_bound = global_curve_bound(
            positions, tangents, float(configuration.protocol["metric_c"])
        )
        if not math.isclose(
            curve_bound, float(record["global_curve_bound"]),
            rel_tol=0.0, abs_tol=2e-10,
        ):
            raise ValueError(f"{case_id}: bundle/driver curve bound mismatch")
        if not curve_bound < float(configuration.protocol["r_max"]):
            raise ValueError(f"{case_id}: curve bound is outside r_max")
        observed_max_bound = max(observed_max_bound, curve_bound)
        with display_path.open(encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
            row_count = sum(1 for _ in handle)
        if header[:2] != ["nominal_suspension_time", "z0"] or row_count != int(record["display"]["n_rows"]):
            raise ValueError(f"{case_id}: display schema/row count changed")
        validated.append({
            "id": case_id,
            "title": case_documents[case_id]["title"],
            "kind": expected_kind,
            "phi_deg": phi_deg,
            "q": q,
            "nominal_suspension_period": nominal_period,
            "physical_return_seconds": physical_period,
            "raw_sample_dt": analysis_dt,
            "stride": 1,
            "effective_sample_dt": analysis_dt,
            "dimension": dimension,
            "n_samples": analysis_n_samples,
            "analysis_n_samples": analysis_n_samples,
            "analysis_duration": analysis_duration,
            "positions": str(positions_path),
            "positions_sha256": positions_hash,
            "tangents": str(tangents_path),
            "tangents_sha256": tangents_hash,
            "tangent_semantics": record["tangent_semantics"],
            "global_curve_bound": curve_bound,
            "expected_beta1": expected_beta1,
            "certificate": {
                **record["certificate"], "absolute_path": str(certificate_path),
                "sha256": certificate_hash,
            },
            "fourier": fourier_record,
            "display_extract": {
                **record["display"], "path": str(display_path),
                "sha256": display_hash,
            },
        })
    if not math.isclose(
        observed_max_bound, float(manifest["maximum_global_curve_bound"]),
        rel_tol=0.0, abs_tol=2e-10,
    ):
        raise ValueError("bundle maximum curve bound changed")
    return ValidatedBundle(
        configuration.bundle_manifest_path,
        sha256(configuration.bundle_manifest_path),
        manifest,
        tuple(validated),
    )


def guard_output_root(path: Path, must_not_exist: bool) -> Path:
    reject_symlink_components(path, "Compass v2 output root")
    resolved = path.resolve(strict=False)
    safe = SAFE_OUTPUT_ROOT.resolve()
    if resolved == safe or not resolved.is_relative_to(safe):
        raise ValueError(f"output must be a named child of {safe}")
    if must_not_exist and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(f"refusing to overwrite output root: {resolved}")
    return resolved


def suggested_output_root(configuration: Configuration) -> Path:
    return guard_output_root(
        CODE_ROOT / str(configuration.cases["suggested_output_root"]), False
    )


def julia_command(
    case: dict[str, Any], protocol: dict[str, Any], output_dir: Path,
    julia_bin: str, check_only: bool = False,
) -> list[str]:
    lengths = segment_lengths(protocol)
    command = [
        julia_bin, "--startup-file=no", f"--project={JULIA_PROJECT.parent}",
        str(JULIA_KERNEL), "--positions", case["positions"], "--tangents",
        case["tangents"], "--stride", str(case["stride"]), "--sample-dt",
        str(case["raw_sample_dt"]), "--segment-lengths",
        f"{lengths[0]}:{lengths[1] - lengths[0]}:{lengths[-1]}", "--n-runs",
        str(protocol["n_runs"]), "--seed", str(protocol["seed"]),
        "--tangent-normalization", str(protocol["tangent_normalization"]),
        "--boxsize", str(protocol["position_boxsize"]), "--sb-radius",
        str(protocol["sphere_box_resolution"]), "--metric-c",
        str(protocol["metric_c"]), "--r-max", str(protocol["r_max"]),
        "--r-subdivisions", str(protocol["r_subdivisions"]), "--field-prime",
        str(protocol["field_prime"]), "--require-sample-radius-below-r-max",
        "true", "--parallel-inner", "false", "--progress",
        "false" if check_only else "true", "--out-dir", str(output_dir),
        "--out-prefix", case["id"],
    ]
    if check_only:
        command.append("--check-only")
    return command


def beta_preflight_command(case: dict[str, Any], protocol: dict[str, Any], julia_bin: str) -> list[str]:
    command = [
        julia_bin, "--startup-file=no", f"--project={JULIA_PROJECT.parent}",
        str(JULIA_BETA_PREFLIGHT), "--positions", case["positions"],
        "--tangents", case["tangents"], "--stride", str(case["stride"]),
        "--tangent-normalization", str(protocol["tangent_normalization"]),
        "--boxsize", str(protocol["position_boxsize"]), "--sb-radius",
        str(protocol["sphere_box_resolution"]), "--metric-c",
        str(protocol["metric_c"]), "--r-max", str(protocol["r_max"]),
        "--expected-beta1",
        "" if case["expected_beta1"] is None else str(case["expected_beta1"]),
    ]
    return command


def run_beta_preflights(
    bundle: ValidatedBundle, protocol: dict[str, Any], julia_bin: str
) -> None:
    for case in bundle.cases:
        print(f"Beta/curve preflight: {case['id']}", flush=True)
        subprocess.run(
            beta_preflight_command(case, protocol, julia_bin),
            cwd=WORKSPACE_ROOT,
            check=True,
        )


def display_csv_bytes(case: dict[str, Any]) -> bytes:
    extract = case["display_extract"]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["nominal_suspension_time", "piece_kind", *[
        f"z{index}" for index in range(case["dimension"])
    ]])
    for time, kind, row in zip(extract["time"], extract["piece_kind"], extract["positions"]):
        writer.writerow([f"{float(time):.17g}", int(kind), *[
            f"{float(value):.17g}" for value in row
        ]])
    return output.getvalue().encode("utf-8")


def serializable_case(case: dict[str, Any]) -> dict[str, Any]:
    result = dict(case)
    extract = dict(result["display_extract"])
    for key in ("time", "piece_kind", "positions"):
        extract.pop(key, None)
    result["display_extract"] = extract
    return result


def commands_text(jobs: list[dict[str, Any]]) -> str:
    return "#!/bin/sh\nset -eu\n" + "\n".join(
        shlex.join(job["command"]) for job in jobs
    ) + "\n"


def write_exclusive(path: Path, data: str | bytes) -> None:
    binary = isinstance(data, bytes)
    with path.open("xb" if binary else "x", **({} if binary else {"encoding": "utf-8"})) as handle:
        handle.write(data)


def materialize(
    configuration: Configuration, bundle: ValidatedBundle, output_root: Path,
    julia_bin: str,
) -> Path:
    root = guard_output_root(output_root, True)
    environment = execution_environment(julia_bin)
    run_beta_preflights(
        bundle, configuration.protocol, environment["julia_executable"]
    )
    root.mkdir(parents=True)
    try:
        jobs: list[dict[str, Any]] = []
        for case in bundle.cases:
            job = serializable_case(case)
            job["output_dir"] = str(root / "signatures" / case["id"])
            job["log"] = str(root / "logs" / f"{case['id']}.log")
            job["command"] = julia_command(
                job, configuration.protocol, Path(job["output_dir"]),
                environment["julia_executable"], False,
            )
            jobs.append(job)
        commands = commands_text(jobs)
        commands_path = root / "commands.sh"
        plan = {
            "schema_version": 2,
            "status": "materialized_not_executed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_id": configuration.cases["analysis_id"],
            "figure_filename": configuration.cases["figure_filename"],
            "output_root": str(root),
            "orchestrator": str(SCRIPT_PATH),
            "orchestrator_sha256": sha256(SCRIPT_PATH),
            "cases_path": str(configuration.cases_path),
            "cases_sha256": sha256(configuration.cases_path),
            "protocol_path": str(configuration.protocol_path),
            "protocol_sha256": sha256(configuration.protocol_path),
            "protocol": configuration.protocol,
            "bundle_manifest": str(bundle.manifest_path),
            "bundle_manifest_sha256": bundle.manifest_sha256,
            "bundle_id": bundle.manifest["bundle_id"],
            "scientific_scope": bundle.manifest["scientific_scope"],
            "environment": environment,
            "kernel_internal_protocol_label": KERNEL_INTERNAL_PROTOCOL_LABEL,
            "commands": str(commands_path),
            "commands_sha256": sha256_bytes(commands.encode("utf-8")),
            "jobs": jobs,
        }
        plan_path = root / "plan.json"
        write_exclusive(plan_path, json.dumps(plan, indent=2, sort_keys=True) + "\n")
        write_exclusive(root / "plan.sha256", f"{sha256(plan_path)}  plan.json\n")
        write_exclusive(commands_path, commands)
        commands_path.chmod(0o755)
    except BaseException:
        marker = root / "MATERIALIZATION_INCOMPLETE"
        if not marker.exists():
            write_exclusive(marker, "Compass v2 materialization did not complete\n")
        raise
    print(f"Materialized {len(jobs)} inert Compass v2 jobs at {root}")
    print("No cycling signatures were executed.")
    return plan_path


def check_all(
    configuration: Configuration, bundle: ValidatedBundle, julia_bin: str,
    run_kernel_checks: bool,
) -> None:
    environment = execution_environment(julia_bin)
    print(f"protocol={configuration.protocol['protocol_id']}")
    print(f"bundle={bundle.manifest['bundle_id']}")
    print(f"bundle_manifest_sha256={bundle.manifest_sha256}")
    print("source_role=Fourier-closed periodic derivatives; interpolated frozen-path V_theta chaos control")
    print("lengths=400:80:4800 durations=1:0.2:12 n=20 normalization=linf cover=5x1 C=5 radii=0:0.025:5 F_43")
    print("horizontal_radius_guide=none")
    print("periodic_vertical_guides=median q-impact periods in nominal stored-row suspension time")
    for case in bundle.cases:
        period = case["nominal_suspension_period"]
        guide = "none" if period is None else f"{period:.15g}"
        physical = case["physical_return_seconds"]
        physical_text = "none" if physical is None else f"{physical:.15g}"
        print(
            f"{case['id']}: analysis_samples={case['analysis_n_samples']} "
            f"curve_bound={case['global_curve_bound']:.15g} guide={guide} "
            f"physical_return_context={physical_text}"
        )
    print(f"kernel_sha256={environment['julia_kernel_sha256']}")
    print(f"manifest_sha256={environment['julia_manifest_sha256']}")
    print(f"cycling_repo_head={environment['cycling_repo_head']}")
    if not run_kernel_checks:
        print("Python-only check complete; Julia --check-only was explicitly skipped.")
        return
    check_root = SAFE_OUTPUT_ROOT / ".compass-v2-check-only"
    for case in bundle.cases:
        print(f"Julia check-only: {case['id']}")
        subprocess.run(
            julia_command(
                case, configuration.protocol, check_root / case["id"],
                environment["julia_executable"], True,
            ),
            cwd=WORKSPACE_ROOT,
            check=True,
        )
    run_beta_preflights(
        bundle, configuration.protocol, environment["julia_executable"]
    )
    print("CHECK ONLY complete: no comparison spaces built and no files written.")


def load_plan(plan_path: Path, julia_bin: str) -> tuple[Path, dict[str, Any]]:
    plan_path = resolve_existing_file(plan_path, "Compass v2 plan")
    root = guard_output_root(plan_path.parent, False)
    if plan_path != root / "plan.json":
        raise ValueError("plan must be the canonical plan.json")
    if (root / "MATERIALIZATION_INCOMPLETE").exists():
        raise ValueError("refusing incomplete materialization")
    sidecar = resolve_existing_file(root / "plan.sha256", "plan hash sidecar")
    if sidecar.read_text(encoding="utf-8") != f"{sha256(plan_path)}  plan.json\n":
        raise ValueError("plan hash sidecar mismatch")
    plan = load_json(plan_path)
    if int(plan.get("schema_version", -1)) != 2 or plan.get("status") != "materialized_not_executed":
        raise ValueError("unexpected Compass v2 plan schema/status")
    if plan.get("output_root") != str(root):
        raise ValueError("plan output root changed")
    if Path(plan.get("orchestrator", "")).resolve() != SCRIPT_PATH or plan.get("orchestrator_sha256") != sha256(SCRIPT_PATH):
        raise ValueError("Compass v2 orchestrator changed")
    cases_path = resolve_existing_file(Path(plan["cases_path"]), "planned cases")
    protocol_path = resolve_existing_file(Path(plan["protocol_path"]), "planned protocol")
    bundle_path = resolve_existing_file(Path(plan["bundle_manifest"]), "planned bundle")
    for path, key in (
        (cases_path, "cases_sha256"), (protocol_path, "protocol_sha256"),
        (bundle_path, "bundle_manifest_sha256"),
    ):
        if sha256(path) != plan[key]:
            raise ValueError(f"planned input changed: {path}")
    configuration = load_configuration(cases_path, bundle_path)
    if configuration.protocol_path != protocol_path or configuration.protocol != plan["protocol"]:
        raise ValueError("planned protocol binding changed")
    bundle = validate_bundle(configuration)
    current_environment = execution_environment(julia_bin)
    if current_environment != plan["environment"]:
        raise ValueError("Python/Julia/CyclingSignatures environment changed")
    if bundle.manifest["bundle_id"] != plan["bundle_id"] or bundle.manifest["scientific_scope"] != plan["scientific_scope"]:
        raise ValueError("bundle identity/scientific scope changed")
    commands_path = resolve_existing_file(root / "commands.sh", "planned commands")
    if plan["commands"] != str(commands_path) or sha256(commands_path) != plan["commands_sha256"]:
        raise ValueError("planned commands changed")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 5:
        raise ValueError("plan does not contain five jobs")
    current = {case["id"]: case for case in bundle.cases}
    for job in jobs:
        case = current.get(job.get("id"))
        if case is None:
            raise ValueError("unknown job in plan")
        for key in (
            "positions_sha256", "tangents_sha256", "global_curve_bound",
            "tangent_semantics", "expected_beta1",
        ):
            if job[key] != case[key]:
                raise ValueError(f"{job['id']}: planned bundle binding changed")
        display_path = resolve_existing_file(Path(job["display_extract"]["path"]), "display extract")
        if sha256(display_path) != job["display_extract"]["sha256"]:
            raise ValueError(f"{job['id']}: display extract changed")
        certificate_path = resolve_existing_file(
            Path(job["certificate"]["absolute_path"]), "case certificate"
        )
        if sha256(certificate_path) != job["certificate"]["sha256"]:
            raise ValueError(f"{job['id']}: case certificate changed")
    if commands_path.read_text(encoding="utf-8") != commands_text(jobs):
        raise ValueError("commands file cannot be reconstructed from jobs")
    return root, plan


def result_paths(job: dict[str, Any]) -> dict[str, Path]:
    root = Path(job["output_dir"])
    prefix = str(job["id"])
    return {
        "births": root / f"{prefix}_births.csv",
        "starts": root / f"{prefix}_segment_starts.csv",
        "rank0": root / f"{prefix}_rank0_heatmap.csv",
        "metadata": root / f"{prefix}_metadata.txt",
    }


def parse_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in result:
                raise ValueError(f"duplicate metadata key: {key}")
            result[key] = value
    return result


def validate_raw_result(job: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    paths = result_paths(job)
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError(f"{job['id']}: incomplete raw result")
    metadata = parse_metadata(paths["metadata"])
    expected = {
        "protocol": KERNEL_INTERNAL_PROTOCOL_LABEL,
        "positions_sha256": job["positions_sha256"],
        "tangents_sha256": job["tangents_sha256"],
        "driver_sha256": plan["environment"]["julia_kernel_sha256"],
        "dimension": str(job["dimension"]),
        "source_samples": str(job["n_samples"]),
        "analysis_samples": str(job["analysis_n_samples"]),
        "stride": str(job["stride"]),
        "raw_sample_dt": str(job["raw_sample_dt"]),
        "effective_sample_dt": str(job["effective_sample_dt"]),
        "segment_lengths": ",".join(map(str, segment_lengths(plan["protocol"]))),
        "n_runs": str(plan["protocol"]["n_runs"]),
        "seed": str(plan["protocol"]["seed"]),
        "tangent_normalization": "linf",
        "boxsize": str(float(plan["protocol"]["position_boxsize"])),
        "sb_radius": str(plan["protocol"]["sphere_box_resolution"]),
        "metric_C": str(float(plan["protocol"]["metric_c"])),
        "r_max": str(float(plan["protocol"]["r_max"])),
        "r_subdivisions": str(plan["protocol"]["r_subdivisions"]),
        "field_prime": str(plan["protocol"]["field_prime"]),
    }
    for key, wanted in expected.items():
        if metadata.get(key) != wanted:
            raise ValueError(f"{job['id']}: metadata {key}={metadata.get(key)!r}, expected {wanted!r}")
    beta1 = int(metadata.get("beta1_Y", "-1"))
    if job["expected_beta1"] is not None and beta1 != int(job["expected_beta1"]):
        raise ValueError(
            f"{job['id']}: beta1(Y)={beta1}, expected {job['expected_beta1']}"
        )
    if not math.isclose(float(metadata["global_curve_bound"]), float(job["global_curve_bound"]), rel_tol=1e-11, abs_tol=1e-11):
        raise ValueError(f"{job['id']}: Python/Julia curve bound mismatch")
    for name, metadata_key in (
        ("births", "births_sha256"), ("starts", "segment_starts_sha256"),
        ("rank0", "rank0_heatmap_sha256"),
    ):
        if metadata.get(metadata_key) != sha256(paths[name]):
            raise ValueError(f"{job['id']}: raw output hash mismatch")
    lengths = segment_lengths(plan["protocol"])
    n_runs = int(plan["protocol"]["n_runs"])
    expected_trials = {
        (length, run) for length in lengths for run in range(1, n_runs + 1)
    }
    first_birth: dict[tuple[int, int], float] = {}
    windows: dict[tuple[int, int], tuple[float, int, int]] = {}
    with paths["births"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_header = (
            "segment_length", "segment_duration", "run_index", "start_index",
            "end_index", "rank", "births",
        )
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"{job['id']}: unexpected births schema")
        for row in reader:
            key = (int(row["segment_length"]), int(row["run_index"]))
            if key not in expected_trials or key in first_birth:
                raise ValueError(f"{job['id']}: unexpected/duplicate trial {key}")
            births = [float(value) for value in row["births"].split(";") if value]
            if (
                births != sorted(births)
                or len(births) != int(row["rank"])
                or any(not math.isfinite(value) or value < 0 for value in births)
            ):
                raise ValueError(f"{job['id']}: invalid birth vector {key}")
            duration = float(row["segment_duration"])
            wanted_duration = key[0] * float(job["effective_sample_dt"])
            start, stop = int(row["start_index"]), int(row["end_index"])
            if not math.isclose(duration, wanted_duration, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{job['id']}: duration mismatch {key}")
            if start < 1 or stop != start + key[0] - 1 or stop > int(job["analysis_n_samples"]):
                raise ValueError(f"{job['id']}: invalid segment indices {key}")
            first_birth[key] = min(births, default=math.inf)
            windows[key] = (duration, start, stop)
    if set(first_birth) != expected_trials:
        raise ValueError(f"{job['id']}: incomplete births trials")
    seen_starts: set[tuple[int, int]] = set()
    with paths["starts"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_header = (
            "segment_length", "segment_duration", "run_index", "start_index",
            "end_index",
        )
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"{job['id']}: unexpected starts schema")
        for row in reader:
            key = (int(row["segment_length"]), int(row["run_index"]))
            actual = (
                float(row["segment_duration"]), int(row["start_index"]),
                int(row["end_index"]),
            )
            if key in seen_starts or windows.get(key) != actual:
                raise ValueError(f"{job['id']}: starts table mismatch {key}")
            seen_starts.add(key)
    if seen_starts != expected_trials:
        raise ValueError(f"{job['id']}: incomplete starts trials")
    radii = np.linspace(
        float(plan["protocol"]["r_min"]), float(plan["protocol"]["r_max"]),
        int(plan["protocol"]["r_subdivisions"]),
    )
    with paths["rank0"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != ["radius", *[str(value) for value in lengths]]:
        raise ValueError(f"{job['id']}: unexpected rank-zero header")
    if len(rows) != len(radii) + 1:
        raise ValueError(f"{job['id']}: rank-zero radius count mismatch")
    rank0 = np.empty((len(radii), len(lengths)), dtype=int)
    for radius_index, (row, radius) in enumerate(zip(rows[1:], radii)):
        if len(row) != len(lengths) + 1 or not math.isclose(
            float(row[0]), float(radius), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{job['id']}: rank-zero radius mismatch")
        counts = np.asarray([int(value) for value in row[1:]], dtype=int)
        if np.any((counts < 0) | (counts > n_runs)):
            raise ValueError(f"{job['id']}: rank-zero count outside 0..20")
        rank0[radius_index] = counts
    for column, length in enumerate(lengths):
        births = np.asarray([
            first_birth[(length, run)] for run in range(1, n_runs + 1)
        ])
        wanted_counts = np.sum(
            births[np.newaxis, :] > radii[:, np.newaxis], axis=1
        )
        if not np.array_equal(rank0[:, column], wanted_counts):
            raise ValueError(f"{job['id']}: rank-zero counts disagree with births")
    return {
        "schema_version": 2,
        "status": "validated",
        "case_id": job["id"],
        "plan_sha256": sha256(Path(plan["output_root"]) / "plan.json"),
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "positions_sha256": job["positions_sha256"],
        "tangents_sha256": job["tangents_sha256"],
        "display_extract_sha256": job["display_extract"]["sha256"],
        "beta1_Y": beta1,
        "global_curve_bound": float(metadata["global_curve_bound"]),
        "probability_matrix_shape": [len(radii), len(lengths)],
        "probability_matrix_orientation": "rows=radii,columns=segment_durations",
        "raw_results": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
    }


def read_probability_matrix(
    job: dict[str, Any], protocol: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return durations, radii, and P(rank>0) with shape (radii,durations)."""
    lengths = segment_lengths(protocol)
    durations = np.asarray(lengths, dtype=float) * float(job["effective_sample_dt"])
    radii = np.linspace(
        float(protocol["r_min"]), float(protocol["r_max"]),
        int(protocol["r_subdivisions"]),
    )
    rank0_path = result_paths(job)["rank0"]
    with rank0_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != ["radius", *[str(value) for value in lengths]]:
        raise ValueError(f"{job['id']}: invalid rank-zero matrix")
    rank0 = np.asarray([[int(value) for value in row[1:]] for row in rows[1:]], dtype=float)
    if rank0.shape != (len(radii), len(durations)):
        raise ValueError(f"{job['id']}: invalid rank-zero matrix shape")
    probability = 1.0 - rank0 / int(protocol["n_runs"])
    return durations, radii, probability


def execute(plan_path: Path, julia_bin: str, selected_cases: set[str]) -> None:
    _, plan = load_plan(plan_path, julia_bin)
    known = {job["id"] for job in plan["jobs"]}
    if selected_cases - known:
        raise ValueError(f"unknown cases: {sorted(selected_cases - known)}")
    jobs = [job for job in plan["jobs"] if not selected_cases or job["id"] in selected_cases]
    for job in jobs:
        output_dir = Path(job["output_dir"])
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing nonempty result directory: {output_dir}")
        log_path = Path(job["log"])
        if log_path.exists() or log_path.is_symlink():
            raise FileExistsError(f"refusing existing log: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Running Compass v2 shared kernel for {job['id']}", flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            subprocess.run(job["command"], cwd=WORKSPACE_ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, text=True)
        binding = validate_raw_result(job, plan)
        binding["created_utc"] = datetime.now(timezone.utc).isoformat()
        binding_path = output_dir / f"{job['id']}_v2_result.json"
        write_exclusive(binding_path, json.dumps(binding, indent=2, sort_keys=True) + "\n")
        print(f"Validated {job['id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "materialize", "execute"), nargs="?", default="check")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--julia-bin", default="julia")
    parser.add_argument("--python-only", action="store_true", help="Skip Julia --check-only during check")
    parser.add_argument("--case", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "execute":
        if args.plan is None:
            raise ValueError("execute requires --plan")
        execute(args.plan, args.julia_bin, set(args.case))
        return
    configuration = load_configuration(args.cases, args.bundle_manifest)
    bundle = validate_bundle(configuration)
    if args.action == "check":
        check_all(configuration, bundle, args.julia_bin, not args.python_only)
        return
    output_root = args.output_root or suggested_output_root(configuration)
    materialize(configuration, bundle, output_root, args.julia_bin)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
