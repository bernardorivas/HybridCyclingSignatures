#!/usr/bin/env python3
"""Versioned David-family Rössler cycling-probability orchestrator.

The default action is the read-only ``check`` action. Materialization,
signature execution, and rendering are separate explicit actions. This file
does not import or modify the frozen version-1 Python driver or its plans.
It calls the unchanged ``run_shared_probability.jl`` kernel.
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
import tempfile
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
HERE = SCRIPT_PATH.parent
CODE_ROOT = HERE.parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
DEFAULT_CASES_PATH = HERE / "cases.json"
SAFE_OUTPUT_ROOT = (
    CODE_ROOT
    / "experiments_planned"
    / "outputs"
    / "shared_coauthor_protocol"
)
JULIA_KERNEL = (
    CODE_ROOT / "period_doubling" / "julia" / "run_shared_probability.jl"
)
JULIA_PROJECT = CODE_ROOT / "period_doubling" / "julia" / "Project.toml"
JULIA_MANIFEST = CODE_ROOT / "period_doubling" / "julia" / "Manifest.toml"
CYCLING_REPO = CODE_ROOT / "CyclingSignatures.jl"

EXPECTED_CASES = (
    ("period1", "periodic", 2.82, 1, "a = 2.82"),
    ("period2", "periodic", 2.86, 2, "a = 2.86"),
    ("period4", "periodic", 4.10, 4, "a = 4.10"),
    ("period8", "periodic", 4.18, 8, "a = 4.18"),
    ("chaos", "chaos", 4.30, None, "a = 4.30"),
)
EXPECTED_SYSTEM = {
    "name": "roessler",
    "family": "david_hien_period_doubling",
    "state_order": ["x", "y", "z"],
    "equations": {
        "x": "-y-z",
        "y": "x+0.2*y",
        "z": "0.2+z*(x-a)",
    },
    "fixed_parameters": {"b": 0.2},
    "continuation_parameter": "a",
}
EXPECTED_MANIFEST_KEYS = {
    "schema_version",
    "bundle_id",
    "status",
    "created_utc",
    "generator",
    "system",
    "sample_dt",
    "max_supported_segment_length_samples",
    "analysis_duration",
    "cases",
    "summaries",
    "files",
}
EXPECTED_PLAN_KEYS = {
    "schema_version",
    "status",
    "created_utc",
    "analysis_id",
    "figure_filename",
    "output_root",
    "orchestrator",
    "orchestrator_sha256",
    "cases_path",
    "cases_sha256",
    "protocol_path",
    "protocol_sha256",
    "protocol",
    "bundle_root",
    "bundle_manifest",
    "bundle_manifest_sha256",
    "environment",
    "kernel_internal_protocol_label",
    "commands",
    "commands_sha256",
    "jobs",
}
KERNEL_INTERNAL_PROTOCOL_LABEL = "coauthor_roessler_probability_v1"


@dataclass(frozen=True)
class Configuration:
    cases_path: Path
    protocol_path: Path
    cases: dict[str, Any]
    protocol: dict[str, Any]


@dataclass(frozen=True)
class ValidatedBundle:
    root: Path
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


def resolve_under(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or relative in {"", ".", ".."}:
        raise ValueError(f"{label} must be a nonempty relative path: {relative!r}")
    if ".." in candidate.parts:
        raise ValueError(f"{label} escapes its root: {relative!r}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes its root: {relative!r}")
    return resolve_existing_file(resolved, label)


def resolve_workspace_relative(relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or relative in {"", ".", ".."} or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative: {relative!r}")
    resolved = (WORKSPACE_ROOT / path).resolve()
    if not resolved.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"{label} escapes the workspace: {relative!r}")
    return resolve_existing_file(resolved, label)


def command_output(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def python_environment() -> dict[str, str]:
    return {
        "executable": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }


def execution_environment(julia_bin: str) -> dict[str, Any]:
    julia_path = shutil.which(julia_bin)
    if julia_path is None:
        raise ValueError(f"Julia executable not found: {julia_bin}")
    nested_status = command_output(["git", "status", "--porcelain"], CYCLING_REPO)
    if nested_status:
        raise ValueError(
            "CyclingSignatures.jl must be clean before checking or freezing v2"
        )
    return {
        "python": python_environment(),
        "julia_executable": str(Path(julia_path).resolve()),
        "julia_version": command_output([julia_bin, "--version"], CODE_ROOT),
        "julia_project": str(JULIA_PROJECT.resolve()),
        "julia_project_sha256": sha256(JULIA_PROJECT),
        "julia_manifest": str(JULIA_MANIFEST.resolve()),
        "julia_manifest_sha256": sha256(JULIA_MANIFEST),
        "julia_kernel": str(JULIA_KERNEL.resolve()),
        "julia_kernel_sha256": sha256(JULIA_KERNEL),
        "cycling_repo": str(CYCLING_REPO.resolve()),
        "cycling_repo_head": command_output(
            ["git", "rev-parse", "--verify", "HEAD"], CYCLING_REPO
        ),
        "cycling_repo_status": "clean",
    }


def segment_lengths(protocol: dict[str, Any]) -> list[int]:
    spec = protocol["segment_lengths"]
    start = int(spec["start"])
    step = int(spec["step"])
    stop = int(spec["stop"])
    if start < 2 or step <= 0 or stop < start or (stop - start) % step:
        raise ValueError("invalid segment-length grid")
    return list(range(start, stop + 1, step))


def validate_protocol(protocol: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "statistic",
        "segment_lengths",
        "n_runs",
        "sampling",
        "seed",
        "duration_convention",
        "effective_sample_dt",
        "tangent_normalization",
        "position_boxsize",
        "sphere_box_resolution",
        "metric_c",
        "r_min",
        "r_max",
        "r_step",
        "r_subdivisions",
        "field_prime",
        "require_sample_radius_below_r_max",
        "plot_horizontal_radius_guide",
        "plot_vertical_certified_period_guides",
    }
    if set(protocol) != expected_keys:
        raise ValueError(
            f"unexpected v2 protocol keys: {sorted(set(protocol) ^ expected_keys)}"
        )
    if int(protocol.get("schema_version", -1)) != 2:
        raise ValueError("v2 protocol requires schema_version=2")
    if protocol.get("protocol_id") != "roessler_david_probability_linf_extended_v2":
        raise ValueError("unexpected v2 protocol_id")
    if protocol.get("statistic") != "P(rank > 0)":
        raise ValueError("v2 statistic must be P(rank > 0)")
    if segment_lengths(protocol) != list(range(100, 6001, 20)):
        raise ValueError("v2 segment lengths must be 100:20:6000")
    expected = {
        "n_runs": 20,
        "sampling": "independent_uniform_with_replacement_per_length",
        "seed": 20260820,
        "duration_convention": "segment_length_times_effective_dt",
        "effective_sample_dt": 0.01,
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
        "plot_vertical_certified_period_guides": True,
    }
    for key, value in expected.items():
        actual = protocol.get(key)
        if isinstance(value, float):
            try:
                matches = math.isclose(
                    float(actual), value, rel_tol=0.0, abs_tol=1e-12
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(f"protocol {key}={actual!r}, expected {value!r}")
        elif actual != value:
            raise ValueError(f"protocol {key}={actual!r}, expected {value!r}")
    radius_step = (
        float(protocol["r_max"]) - float(protocol["r_min"])
    ) / (int(protocol["r_subdivisions"]) - 1)
    if not math.isclose(radius_step, 0.025, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("r_min/r_max/r_subdivisions do not produce r step 0.025")


def validate_cases_document(document: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "analysis_id",
        "protocol",
        "bundle_root",
        "bundle_manifest",
        "suggested_output_root",
        "figure_filename",
        "cases",
    }
    if set(document) != expected_keys:
        raise ValueError(
            f"unexpected v2 cases keys: {sorted(set(document) ^ expected_keys)}"
        )
    if int(document.get("schema_version", -1)) != 2:
        raise ValueError("v2 cases require schema_version=2")
    if document.get("analysis_id") != (
        "roessler_david_fourier_probability_linf_v2_david_grid"
    ):
        raise ValueError("unexpected analysis_id")
    if document.get("protocol") != "protocol.json":
        raise ValueError("v2 cases must bind protocol.json")
    if document.get("bundle_manifest") != "bundle_manifest.json":
        raise ValueError("unexpected bundle manifest name")
    if document.get("bundle_root") != (
        "experiments_planned/outputs/roessler_david_fourier_continuation_v1"
    ):
        raise ValueError("unexpected orbit bundle root")
    if document.get("suggested_output_root") != (
        "experiments_planned/outputs/shared_coauthor_protocol/"
        "roessler_david_fourier_probability_linf_v2_david_grid"
    ):
        raise ValueError("unexpected suggested v2 output root")
    if document.get("figure_filename") != "roessler_C5p0.pdf":
        raise ValueError("v2 figure must use stable basename roessler_C5p0.pdf")
    actual_cases = document.get("cases")
    if not isinstance(actual_cases, list) or len(actual_cases) != len(EXPECTED_CASES):
        raise ValueError("v2 cases must contain period1/2/4/8 and chaos")
    for actual, expected in zip(actual_cases, EXPECTED_CASES):
        case_id, kind, parameter_a, q, title = expected
        if set(actual) != {"id", "title", "kind", "a", "q"}:
            raise ValueError(f"unexpected case schema for {case_id}")
        if actual["id"] != case_id or actual["kind"] != kind:
            raise ValueError(f"unexpected identity/kind for {case_id}")
        if not math.isclose(
            float(actual["a"]), parameter_a, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"unexpected a for {case_id}")
        if actual["q"] != q or actual["title"] != title:
            raise ValueError(f"unexpected q/title for {case_id}")


def load_configuration(cases_path: Path) -> Configuration:
    cases_path = resolve_existing_file(cases_path, "v2 cases document")
    cases = load_json(cases_path)
    validate_cases_document(cases)
    protocol_path = resolve_existing_file(
        cases_path.parent / str(cases["protocol"]), "v2 protocol"
    )
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    return Configuration(cases_path, protocol_path, cases, protocol)


def file_inventory(manifest: dict[str, Any], bundle_root: Path) -> dict[str, dict[str, Any]]:
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("bundle files inventory must be a nonempty list")
    inventory: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
            raise ValueError("bundle inventory records require path/sha256/bytes")
        relative = str(record["path"])
        if relative in inventory:
            raise ValueError(f"duplicate bundle inventory path: {relative}")
        path = resolve_under(bundle_root, relative, "bundle inventory file")
        expected_bytes = int(record["bytes"])
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"bundle byte count changed: {relative}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"bundle hash changed: {relative}")
        inventory[relative] = record
    symlinks = [path for path in bundle_root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"orbit bundle contains symlinks: {symlinks}")
    actual = {
        str(path.relative_to(bundle_root))
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    if actual != set(inventory):
        missing = sorted(set(inventory) - actual)
        extra = sorted(actual - set(inventory))
        raise ValueError(f"bundle inventory mismatch: missing={missing}, extra={extra}")
    return inventory


def validate_recorded_file(
    record: dict[str, Any],
    bundle_root: Path,
    inventory: dict[str, dict[str, Any]],
    label: str,
) -> Path:
    if "path" not in record or "sha256" not in record:
        raise ValueError(f"{label} lacks path/hash")
    relative = str(record["path"])
    if relative not in inventory:
        raise ValueError(f"{label} is absent from bundle inventory: {relative}")
    if record["sha256"] != inventory[relative]["sha256"]:
        raise ValueError(f"{label} hash disagrees with inventory: {relative}")
    return resolve_under(bundle_root, relative, label)


def load_numeric_matrix(path: Path, expected_rows: int, label: str) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    if values.shape != (expected_rows, 3):
        raise ValueError(
            f"{label} shape {values.shape}, expected {(expected_rows, 3)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains nonfinite values")
    return values


def validate_display_csv(
    path: Path,
    kind: str,
    parameter_a: float,
    expected_rows: int,
    expected_duration: float,
    sample_dt: float,
) -> None:
    periodic = kind == "periodic"
    expected_header = (
        ["phase", "time", "x", "y", "z", "dx", "dy", "dz"]
        if periodic
        else ["time", "x", "y", "z", "dx", "dy", "dz"]
    )
    values = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if list(values.dtype.names or ()) != expected_header:
        raise ValueError(f"unexpected display CSV header: {path}")
    values = np.atleast_1d(values)
    if len(values) != expected_rows:
        raise ValueError(f"display row count changed: {path}")
    matrix = np.column_stack([values[name] for name in expected_header])
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"display CSV contains nonfinite values: {path}")
    positions = np.column_stack([values["x"], values["y"], values["z"]])
    tangents = np.column_stack([values["dx"], values["dy"], values["dz"]])
    if not np.allclose(
        tangents,
        rossler_vector_field(positions, parameter_a),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"display tangents do not match the Rössler field: {path}")
    times = np.asarray(values["time"], dtype=float)
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"display times must increase strictly: {path}")
    duration = float(times[-1] - times[0])
    if not math.isclose(duration, expected_duration, rel_tol=0.0, abs_tol=2e-10):
        raise ValueError(f"display duration changed: {path}")
    if not periodic:
        if expected_rows != 5001 or not math.isclose(
            expected_duration, 50.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("chaos display must be 5001 rows over 50 time units")
        if not np.allclose(np.diff(times), sample_dt, rtol=0.0, atol=2e-10):
            raise ValueError("chaos display is not uniformly sampled at dt=0.01")
    else:
        phase = np.asarray(values["phase"], dtype=float)
        if not (
            np.all(np.diff(phase) > 0)
            and math.isclose(float(phase[0]), 0.0, abs_tol=1e-12)
            and math.isclose(float(phase[-1]), 1.0, abs_tol=1e-12)
        ):
            raise ValueError(f"periodic display phase must run from 0 to 1: {path}")


def validate_fourier_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        expected = [
            "mode",
            "x_real",
            "x_imag",
            "y_real",
            "y_imag",
            "z_real",
            "z_imag",
        ]
        if header != expected:
            raise ValueError(f"unexpected Fourier coefficient header: {path}")
        if not any(True for _ in reader):
            raise ValueError(f"empty Fourier coefficient file: {path}")


def global_curve_bound(positions: np.ndarray, tangents: np.ndarray, metric_c: float) -> float:
    norms = np.max(np.abs(tangents), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("analysis tangents must be finite and nonzero")
    normalized = tangents / norms[:, np.newaxis]
    dx = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dv = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    return float(max(float(np.max(dx)), metric_c * float(np.max(dv))))


def rossler_vector_field(positions: np.ndarray, parameter_a: float) -> np.ndarray:
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    return np.column_stack(
        [-y - z, x + 0.2 * y, 0.2 + z * (x - parameter_a)]
    )


def validate_bundle(configuration: Configuration) -> ValidatedBundle:
    bundle_relative = str(configuration.cases["bundle_root"])
    bundle_root = (CODE_ROOT / bundle_relative).resolve()
    reject_symlink_components(bundle_root, "orbit bundle")
    if not bundle_root.is_dir():
        raise ValueError(
            "certified orbit bundle is not ready: "
            f"expected directory {bundle_root}"
        )
    manifest_path = resolve_existing_file(
        bundle_root / str(configuration.cases["bundle_manifest"]),
        "orbit bundle manifest",
    )
    manifest = load_json(manifest_path)
    if set(manifest) != EXPECTED_MANIFEST_KEYS:
        raise ValueError(
            "unexpected bundle manifest keys: "
            f"{sorted(set(manifest) ^ EXPECTED_MANIFEST_KEYS)}"
        )
    if int(manifest["schema_version"]) != 1:
        raise ValueError("orbit bundle requires schema_version=1")
    if manifest["bundle_id"] != "roessler_david_fourier_continuation_v1":
        raise ValueError("unexpected orbit bundle id")
    if manifest["status"] != "complete":
        raise ValueError(f"orbit bundle is not complete: {manifest['status']!r}")
    if not isinstance(manifest["created_utc"], str) or not manifest["created_utc"]:
        raise ValueError("orbit bundle lacks a creation timestamp")
    if manifest["system"] != EXPECTED_SYSTEM:
        raise ValueError("orbit bundle uses a different Rössler family")
    sample_dt = float(manifest["sample_dt"])
    if not math.isclose(sample_dt, 0.01, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("orbit bundle sample_dt must be 0.01")
    if int(manifest["max_supported_segment_length_samples"]) != 6000:
        raise ValueError("orbit bundle must support 6000-sample segments")
    analysis_duration = float(manifest["analysis_duration"])
    if not math.isclose(analysis_duration, 3000.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("orbit bundle analysis_duration must be 3000")

    generator = manifest["generator"]
    expected_generator_keys = {
        "path",
        "sha256",
        "python_version",
        "numpy_version",
        "scipy_version",
    }
    if not isinstance(generator, dict) or set(generator) != expected_generator_keys:
        raise ValueError("orbit bundle has an unexpected generator schema")
    if generator["path"] != (
        "code/period_doubling/shared_probability/"
        "build_roessler_fourier_bundle.py"
    ):
        raise ValueError("orbit bundle names a different generator")
    for key in ("python_version", "numpy_version", "scipy_version"):
        if not isinstance(generator[key], str) or not generator[key]:
            raise ValueError(f"orbit bundle generator lacks {key}")
    generator_path = resolve_workspace_relative(
        str(generator["path"]), "orbit bundle generator"
    )
    if sha256(generator_path) != generator["sha256"]:
        raise ValueError("orbit bundle generator hash changed")

    inventory = file_inventory(manifest, bundle_root)
    summaries = manifest["summaries"]
    expected_summaries = {
        "branch_points": "summary/branch_points.csv",
        "flips": "summary/flips.csv",
        "representatives": "summary/representatives.csv",
    }
    if not isinstance(summaries, dict) or set(summaries) != set(expected_summaries):
        raise ValueError("orbit bundle has an unexpected summary schema")
    for name, expected_path in expected_summaries.items():
        record = summaries[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"unexpected summary record: {name}")
        if record["path"] != expected_path:
            raise ValueError(f"unexpected summary path: {name}")
        validate_recorded_file(record, bundle_root, inventory, f"{name} summary")
    case_records = manifest["cases"]
    if not isinstance(case_records, list) or len(case_records) != len(EXPECTED_CASES):
        raise ValueError("orbit bundle has an unexpected case count")
    manifest_cases = {str(record.get("id")): record for record in case_records}
    if set(manifest_cases) != {item[0] for item in EXPECTED_CASES}:
        raise ValueError("orbit bundle case ids differ from v2 cases")

    validated_cases: list[dict[str, Any]] = []
    expected_rows = int(round(analysis_duration / sample_dt)) + 1
    if expected_rows != 300001:
        raise ValueError("internal analysis-row expectation failed")
    for configured, expected in zip(configuration.cases["cases"], EXPECTED_CASES):
        case_id, expected_kind, expected_a, expected_q, _ = expected
        record = manifest_cases[case_id]
        required = {"id", "kind", "a", "q", "analysis", "display", "certificate"}
        if expected_kind == "periodic":
            required.add("fourier")
        if set(record) != required:
            raise ValueError(f"unexpected manifest case schema for {case_id}")
        if record["kind"] != expected_kind or record["q"] != expected_q:
            raise ValueError(f"bundle kind/q mismatch for {case_id}")
        if not math.isclose(
            float(record["a"]), expected_a, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"bundle parameter mismatch for {case_id}")
        if configured["kind"] != record["kind"] or configured["q"] != record["q"]:
            raise ValueError(f"case document and bundle disagree for {case_id}")

        analysis = record["analysis"]
        if not isinstance(analysis, dict) or set(analysis) != {
            "positions",
            "tangents",
            "sample_dt",
            "n_samples",
            "positions_sha256",
            "tangents_sha256",
        }:
            raise ValueError(f"unexpected analysis record for {case_id}")
        if int(analysis["n_samples"]) != expected_rows or not math.isclose(
            float(analysis["sample_dt"]), sample_dt, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"analysis cadence/length mismatch for {case_id}")
        expected_case_root = f"cases/{case_id}"
        if analysis["positions"] != f"{expected_case_root}/analysis_positions.csv":
            raise ValueError(f"unexpected positions path for {case_id}")
        if analysis["tangents"] != f"{expected_case_root}/analysis_tangents.csv":
            raise ValueError(f"unexpected tangents path for {case_id}")
        positions_record = {
            "path": analysis["positions"],
            "sha256": analysis["positions_sha256"],
        }
        tangents_record = {
            "path": analysis["tangents"],
            "sha256": analysis["tangents_sha256"],
        }
        positions_path = validate_recorded_file(
            positions_record, bundle_root, inventory, f"{case_id} analysis positions"
        )
        tangents_path = validate_recorded_file(
            tangents_record, bundle_root, inventory, f"{case_id} analysis tangents"
        )
        positions = load_numeric_matrix(
            positions_path, expected_rows, f"{case_id} analysis positions"
        )
        tangents = load_numeric_matrix(
            tangents_path, expected_rows, f"{case_id} analysis tangents"
        )
        if not np.allclose(
            tangents,
            rossler_vector_field(positions, expected_a),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{case_id} analysis tangents do not match the Rössler field"
            )
        curve_bound = global_curve_bound(
            positions, tangents, float(configuration.protocol["metric_c"])
        )
        if not curve_bound < float(configuration.protocol["r_max"]):
            raise ValueError(
                f"{case_id} global curve bound {curve_bound:g} is not below r_max"
            )
        del positions, tangents

        display = record["display"]
        if not isinstance(display, dict) or set(display) != {
            "kind",
            "path",
            "sha256",
            "n_rows",
            "duration",
        }:
            raise ValueError(f"unexpected display record for {case_id}")
        expected_display_kind = (
            "certified_primitive_periodic_orbit"
            if expected_kind == "periodic"
            else "bounded_chaos_segment"
        )
        if display["kind"] != expected_display_kind:
            raise ValueError(f"unexpected display kind for {case_id}")
        expected_display_path = (
            f"{expected_case_root}/orbit_dense.csv"
            if expected_kind == "periodic"
            else f"{expected_case_root}/display_segment.csv"
        )
        if display["path"] != expected_display_path:
            raise ValueError(f"unexpected display path for {case_id}")
        display_path = validate_recorded_file(
            display, bundle_root, inventory, f"{case_id} display"
        )
        display_duration = float(display["duration"])
        if not math.isfinite(display_duration) or display_duration <= 0:
            raise ValueError(f"invalid display duration for {case_id}")
        validate_display_csv(
            display_path,
            expected_kind,
            expected_a,
            int(display["n_rows"]),
            display_duration,
            sample_dt,
        )
        if expected_kind == "periodic" and display_duration > 60.0:
            raise ValueError(f"{case_id} certified period lies outside t<=60")

        certificate = record["certificate"]
        if not isinstance(certificate, dict):
            raise ValueError(f"invalid certificate record for {case_id}")
        if certificate.get("path") != f"{expected_case_root}/certificate.json":
            raise ValueError(f"unexpected certificate path for {case_id}")
        certificate_path = validate_recorded_file(
            certificate, bundle_root, inventory, f"{case_id} certificate"
        )
        certificate_document = load_json(certificate_path)
        common_certificate_values = {
            "schema_version": 1,
            "status": "certified",
            "case_id": case_id,
            "kind": expected_kind,
            "q": expected_q,
            "analysis_n_samples": expected_rows,
        }
        for key, value in common_certificate_values.items():
            if certificate_document.get(key) != value:
                raise ValueError(f"certificate {key} mismatch for {case_id}")
        for key, value in {
            "a": expected_a,
            "sample_dt": sample_dt,
            "analysis_duration": analysis_duration,
        }.items():
            if not math.isclose(
                float(certificate_document.get(key, math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"certificate {key} mismatch for {case_id}")
        certificate_hashes = {
            "analysis_positions_sha256": analysis["positions_sha256"],
            "analysis_tangents_sha256": analysis["tangents_sha256"],
        }
        for key, value in certificate_hashes.items():
            if certificate_document.get(key) != value:
                raise ValueError(f"certificate dependent hash mismatch: {case_id} {key}")

        fourier_record = None
        if expected_kind == "periodic":
            fourier = record["fourier"]
            if not isinstance(fourier, dict) or set(fourier) != {
                "path",
                "sha256",
                "n_modes",
            }:
                raise ValueError(f"unexpected Fourier record for {case_id}")
            if fourier["path"] != f"{expected_case_root}/fourier_coefficients.csv":
                raise ValueError(f"unexpected Fourier path for {case_id}")
            fourier_path = validate_recorded_file(
                fourier, bundle_root, inventory, f"{case_id} Fourier coefficients"
            )
            validate_fourier_csv(fourier_path)
            if int(fourier["n_modes"]) <= 0:
                raise ValueError(f"invalid Fourier mode count for {case_id}")
            if certificate_document.get("stable_transverse") is not True:
                raise ValueError(f"periodic certificate is not stable: {case_id}")
            if not math.isclose(
                float(certificate_document.get("period", math.nan)),
                display_duration,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError(f"certificate/manifest period mismatch for {case_id}")
            periodic_hashes = {
                "orbit_dense_sha256": display["sha256"],
                "fourier_coefficients_sha256": fourier["sha256"],
            }
            for key, value in periodic_hashes.items():
                if certificate_document.get(key) != value:
                    raise ValueError(
                        f"certificate dependent hash mismatch: {case_id} {key}"
                    )
            if int(certificate_document.get("fourier_signed_mode_count", -1)) != int(
                fourier["n_modes"]
            ):
                raise ValueError(f"certificate Fourier mode count mismatch: {case_id}")
            for key in (
                "collocation_residual_infinity_norm",
                "dop853_closure_error_absolute",
                "dop853_closure_error_relative",
                "oversampled_residual_absolute",
                "oversampled_residual_relative",
                "fourier_tail_energy_ratio",
            ):
                value = float(certificate_document.get(key, math.nan))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid periodic certificate metric: {case_id} {key}")
            fourier_record = {
                **fourier,
                "absolute_path": str(fourier_path),
            }
        else:
            if not math.isclose(
                float(certificate_document.get("display_duration", math.nan)),
                50.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("chaos certificate display duration must be 50")
            if certificate_document.get("display_segment_sha256") != display["sha256"]:
                raise ValueError("chaos certificate display hash mismatch")
            if int(
                certificate_document.get(
                    "maximum_supported_segment_length_samples", -1
                )
            ) != int(manifest["max_supported_segment_length_samples"]):
                raise ValueError("chaos certificate segment support mismatch")
            lyapunov = certificate_document.get("largest_lyapunov_validation")
            if not isinstance(lyapunov, dict) or lyapunov.get("positive") is not True:
                raise ValueError("chaos certificate lacks a positive-LLE verdict")
            exponent = float(lyapunov.get("largest_lyapunov_exponent", math.nan))
            threshold = float(lyapunov.get("positive_threshold", math.nan))
            if not (
                math.isfinite(exponent)
                and math.isfinite(threshold)
                and exponent > threshold >= 0
            ):
                raise ValueError("chaos certificate LLE does not clear its threshold")

        validated_cases.append(
            {
                "id": case_id,
                "title": configured["title"],
                "kind": expected_kind,
                "a": expected_a,
                "q": expected_q,
                "positions": str(positions_path),
                "positions_sha256": analysis["positions_sha256"],
                "tangents": str(tangents_path),
                "tangents_sha256": analysis["tangents_sha256"],
                "sample_dt": sample_dt,
                "n_samples": expected_rows,
                "global_curve_bound": curve_bound,
                "display": {
                    **display,
                    "absolute_path": str(display_path),
                },
                "certificate": {
                    **certificate,
                    "absolute_path": str(certificate_path),
                },
                "fourier": fourier_record,
            }
        )

    return ValidatedBundle(
        root=bundle_root,
        manifest_path=manifest_path,
        manifest_sha256=sha256(manifest_path),
        manifest=manifest,
        cases=tuple(validated_cases),
    )


def guard_output_root(path: Path, must_not_exist: bool) -> Path:
    reject_symlink_components(path, "v2 output root")
    resolved = path.resolve(strict=False)
    safe = SAFE_OUTPUT_ROOT.resolve()
    if resolved == safe or not resolved.is_relative_to(safe):
        raise ValueError(f"v2 output must be a named child of {safe}")
    if must_not_exist and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(f"refusing to overwrite output root: {resolved}")
    return resolved


def suggested_output_root(configuration: Configuration) -> Path:
    return guard_output_root(
        CODE_ROOT / str(configuration.cases["suggested_output_root"]),
        must_not_exist=False,
    )


def julia_command(
    case: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
    julia_bin: str,
    check_only: bool = False,
) -> list[str]:
    lengths = segment_lengths(protocol)
    length_spec = f"{lengths[0]}:{lengths[1] - lengths[0]}:{lengths[-1]}"
    command = [
        julia_bin,
        f"--project={JULIA_PROJECT.parent}",
        str(JULIA_KERNEL),
        "--positions",
        case["positions"],
        "--tangents",
        case["tangents"],
        "--stride",
        "1",
        "--sample-dt",
        str(protocol["effective_sample_dt"]),
        "--segment-lengths",
        length_spec,
        "--n-runs",
        str(protocol["n_runs"]),
        "--seed",
        str(protocol["seed"]),
        "--tangent-normalization",
        str(protocol["tangent_normalization"]),
        "--boxsize",
        str(protocol["position_boxsize"]),
        "--sb-radius",
        str(protocol["sphere_box_resolution"]),
        "--metric-c",
        str(protocol["metric_c"]),
        "--r-max",
        str(protocol["r_max"]),
        "--r-subdivisions",
        str(protocol["r_subdivisions"]),
        "--field-prime",
        str(protocol["field_prime"]),
        "--require-sample-radius-below-r-max",
        "true",
        "--parallel-inner",
        "false",
        "--progress",
        "false" if check_only else "true",
        "--out-dir",
        str(output_dir),
        "--out-prefix",
        case["id"],
    ]
    if check_only:
        command.append("--check-only")
    return command


def build_jobs(
    configuration: Configuration,
    bundle: ValidatedBundle,
    output_root: Path,
    julia_bin: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for case in bundle.cases:
        output_dir = output_root / "signatures" / case["id"]
        job = {
            **case,
            "output_dir": str(output_dir),
            "out_prefix": case["id"],
            "log": str(output_root / "logs" / f"{case['id']}.log"),
        }
        job["command"] = julia_command(
            job, configuration.protocol, output_dir, julia_bin
        )
        jobs.append(job)
    return jobs


def commands_text(jobs: list[dict[str, Any]]) -> str:
    return "#!/bin/sh\nset -eu\n" + "\n".join(
        shlex.join(job["command"]) for job in jobs
    ) + "\n"


def write_exclusive(path: Path, data: str | bytes, binary: bool = False) -> None:
    mode = "xb" if binary else "x"
    kwargs: dict[str, Any] = {} if binary else {"encoding": "utf-8"}
    with path.open(mode, **kwargs) as handle:
        handle.write(data)


def materialize(
    configuration: Configuration,
    bundle: ValidatedBundle,
    output_root: Path,
    julia_bin: str,
) -> Path:
    root = guard_output_root(output_root, must_not_exist=True)
    environment = execution_environment(julia_bin)
    resolved_julia = environment["julia_executable"]
    jobs = build_jobs(configuration, bundle, root, resolved_julia)
    root.mkdir(parents=True)
    try:
        commands_path = root / "commands.sh"
        commands = commands_text(jobs)
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
            "bundle_root": str(bundle.root),
            "bundle_manifest": str(bundle.manifest_path),
            "bundle_manifest_sha256": bundle.manifest_sha256,
            "environment": environment,
            "kernel_internal_protocol_label": KERNEL_INTERNAL_PROTOCOL_LABEL,
            "commands": str(commands_path),
            "commands_sha256": sha256_bytes(commands.encode("utf-8")),
            "jobs": jobs,
        }
        plan_path = root / "plan.json"
        plan_hash_path = root / "plan.sha256"
        write_exclusive(
            plan_path,
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
        )
        write_exclusive(
            plan_hash_path,
            f"{sha256(plan_path)}  plan.json\n",
        )
        write_exclusive(commands_path, commands)
        commands_path.chmod(0o755)
    except BaseException:
        marker = root / "MATERIALIZATION_INCOMPLETE"
        if not marker.exists():
            write_exclusive(marker, "v2 materialization did not complete\n")
        raise
    print(f"Materialized {len(jobs)} inert jobs at {root}")
    print("No cycling signatures were executed.")
    return plan_path


def load_plan(
    plan_path: Path, julia_bin: str
) -> tuple[Path, dict[str, Any], Configuration, ValidatedBundle]:
    plan_path = resolve_existing_file(plan_path, "v2 plan")
    root = guard_output_root(plan_path.parent, must_not_exist=False)
    if plan_path != root / "plan.json":
        raise ValueError("plan must be the canonical plan.json in its output root")
    incomplete = root / "MATERIALIZATION_INCOMPLETE"
    if incomplete.exists() or incomplete.is_symlink():
        raise ValueError("refusing an incomplete v2 materialization")
    plan_hash_path = resolve_existing_file(root / "plan.sha256", "v2 plan hash")
    expected_plan_hash_line = f"{sha256(plan_path)}  plan.json\n"
    if plan_hash_path.read_text(encoding="utf-8") != expected_plan_hash_line:
        raise ValueError("v2 plan hash sidecar does not match plan.json")
    plan = load_json(plan_path)
    if set(plan) != EXPECTED_PLAN_KEYS:
        raise ValueError(
            f"unexpected v2 plan keys: {sorted(set(plan) ^ EXPECTED_PLAN_KEYS)}"
        )
    if int(plan.get("schema_version", -1)) != 2:
        raise ValueError("plan requires schema_version=2")
    if plan.get("status") != "materialized_not_executed":
        raise ValueError("unexpected immutable plan status")
    if plan.get("output_root") != str(root):
        raise ValueError("plan output root changed")
    if Path(plan.get("orchestrator", "")).resolve() != SCRIPT_PATH:
        raise ValueError("plan names a different v2 orchestrator")
    if plan.get("orchestrator_sha256") != sha256(SCRIPT_PATH):
        raise ValueError("v2 orchestrator changed after materialization")
    if plan.get("kernel_internal_protocol_label") != KERNEL_INTERNAL_PROTOCOL_LABEL:
        raise ValueError("unexpected legacy protocol label in the immutable plan")
    commands_path = resolve_existing_file(root / "commands.sh", "planned commands")
    if plan.get("commands") != str(commands_path):
        raise ValueError("plan names a different commands file")
    if plan.get("commands_sha256") != sha256(commands_path):
        raise ValueError("planned commands changed after materialization")

    cases_path = resolve_existing_file(Path(plan["cases_path"]), "planned cases")
    protocol_path = resolve_existing_file(
        Path(plan["protocol_path"]), "planned protocol"
    )
    if sha256(cases_path) != plan["cases_sha256"]:
        raise ValueError("v2 cases changed after materialization")
    if sha256(protocol_path) != plan["protocol_sha256"]:
        raise ValueError("v2 protocol changed after materialization")
    configuration = load_configuration(cases_path)
    if configuration.protocol_path != protocol_path:
        raise ValueError("plan protocol path differs from cases binding")
    if configuration.protocol != plan["protocol"]:
        raise ValueError("embedded protocol differs from planned protocol")
    if plan.get("analysis_id") != configuration.cases["analysis_id"]:
        raise ValueError("plan analysis id changed")
    if plan.get("figure_filename") != configuration.cases["figure_filename"]:
        raise ValueError("plan figure filename changed")

    bundle = validate_bundle(configuration)
    if str(bundle.root) != plan["bundle_root"]:
        raise ValueError("plan bundle root changed")
    if str(bundle.manifest_path) != plan["bundle_manifest"]:
        raise ValueError("plan bundle manifest path changed")
    if bundle.manifest_sha256 != plan["bundle_manifest_sha256"]:
        raise ValueError("bundle manifest changed after materialization")

    current_environment = execution_environment(julia_bin)
    if current_environment != plan["environment"]:
        raise ValueError("Python/Julia/CyclingSignatures environment changed")
    expected_jobs = build_jobs(
        configuration,
        bundle,
        root,
        current_environment["julia_executable"],
    )
    if plan.get("jobs") != expected_jobs:
        raise ValueError("planned jobs differ from current hashed inputs/settings")
    if commands_path.read_text(encoding="utf-8") != commands_text(expected_jobs):
        raise ValueError("commands file differs from the reconstructed job commands")
    return root, plan, configuration, bundle


def result_paths(job: dict[str, Any]) -> dict[str, Path]:
    output_dir = Path(job["output_dir"])
    prefix = str(job["out_prefix"])
    return {
        "births": output_dir / f"{prefix}_births.csv",
        "starts": output_dir / f"{prefix}_segment_starts.csv",
        "rank0": output_dir / f"{prefix}_rank0_heatmap.csv",
        "metadata": output_dir / f"{prefix}_metadata.txt",
    }


def result_binding_path(job: dict[str, Any]) -> Path:
    return Path(job["output_dir"]) / f"{job['out_prefix']}_v2_result.json"


def parse_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", maxsplit=1)
            if key in values:
                raise ValueError(f"duplicate metadata key {key!r}: {path}")
            values[key] = value
    return values


def result_binding_document(
    job: dict[str, Any],
    protocol: dict[str, Any],
    plan: dict[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    raw_paths = result_paths(job)
    plan_path = Path(plan["output_root"]) / "plan.json"
    return {
        "schema_version": 2,
        "status": "validated",
        "created_utc": created_utc,
        "analysis_id": plan["analysis_id"],
        "case_id": job["id"],
        "protocol_id": protocol["protocol_id"],
        "kernel_internal_protocol_label": plan[
            "kernel_internal_protocol_label"
        ],
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "bundle_manifest": plan["bundle_manifest"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "certificate": job["certificate"]["absolute_path"],
        "certificate_sha256": job["certificate"]["sha256"],
        "positions": job["positions"],
        "positions_sha256": job["positions_sha256"],
        "tangents": job["tangents"],
        "tangents_sha256": job["tangents_sha256"],
        "raw_results": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in raw_paths.items()
        },
    }


def validate_result_binding(
    job: dict[str, Any], protocol: dict[str, Any], plan: dict[str, Any]
) -> None:
    path = resolve_existing_file(result_binding_path(job), "v2 result binding")
    document = load_json(path)
    created_utc = document.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc:
        raise ValueError(f"v2 result binding lacks created_utc: {job['id']}")
    expected = result_binding_document(job, protocol, plan, created_utc)
    if document != expected:
        raise ValueError(f"v2 result binding changed or is inconsistent: {job['id']}")


def expect_metadata(
    metadata: dict[str, str], key: str, expected: str | int | float, case_id: str
) -> None:
    if key not in metadata:
        raise ValueError(f"{case_id} metadata is missing {key}")
    actual = metadata[key]
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{case_id} metadata {key}={actual}, expected {expected}")
    elif actual != str(expected):
        raise ValueError(
            f"{case_id} metadata {key}={actual!r}, expected {expected!r}"
        )


def validate_result(
    job: dict[str, Any],
    protocol: dict[str, Any],
    plan: dict[str, Any],
    *,
    require_binding: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    case_id = str(job["id"])
    paths = result_paths(job)
    output_dir = Path(job["output_dir"])
    reject_symlink_components(output_dir, f"{case_id} result directory")
    if not output_dir.is_dir():
        raise ValueError(f"missing result directory for {case_id}")
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError(f"incomplete or symlinked result for {case_id}")
    expected_paths = list(paths.values())
    if require_binding:
        expected_paths.append(result_binding_path(job))
    actual_entries = list(output_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in actual_entries):
        raise ValueError(f"non-file or symlinked result entry for {case_id}")
    if {path.name for path in actual_entries} != {
        path.name for path in expected_paths
    } or len(actual_entries) != len(expected_paths):
        raise ValueError(f"unexpected result entries for {case_id}")

    metadata = parse_metadata(paths["metadata"])
    lengths = np.asarray(segment_lengths(protocol), dtype=int)
    length_text = ",".join(str(value) for value in lengths)
    expected_metadata: dict[str, str | int | float] = {
        "protocol": plan["kernel_internal_protocol_label"],
        "positions": str(Path(job["positions"]).resolve()),
        "positions_sha256": job["positions_sha256"],
        "tangents": str(Path(job["tangents"]).resolve()),
        "tangents_sha256": job["tangents_sha256"],
        "driver": str(JULIA_KERNEL.resolve()),
        "driver_sha256": plan["environment"]["julia_kernel_sha256"],
        "dimension": 3,
        "source_samples": int(job["n_samples"]),
        "analysis_samples": int(job["n_samples"]),
        "stride": 1,
        "start_index_space": "post_stride_analysis_samples",
        "raw_sample_dt": float(protocol["effective_sample_dt"]),
        "effective_sample_dt": float(protocol["effective_sample_dt"]),
        "sample_dt_cli_semantics": "raw_source_cadence_before_stride",
        "duration_convention": "segment_length_times_effective_sample_dt",
        "segment_lengths": length_text,
        "n_runs": int(protocol["n_runs"]),
        "seed": int(protocol["seed"]),
        "resample_segment_start": "true",
        "sampling_with_replacement": "true",
        "tangent_normalization": protocol["tangent_normalization"],
        "normalization_applied_after_stride": "true",
        "boxsize": float(protocol["position_boxsize"]),
        "sb_radius": int(protocol["sphere_box_resolution"]),
        "metric_C": float(protocol["metric_c"]),
        "r_max": float(protocol["r_max"]),
        "r_subdivisions": int(protocol["r_subdivisions"]),
        "field_prime": int(protocol["field_prime"]),
        "filtration_threshold": "closed_leq",
        "require_sample_radius_below_r_max": "true",
    }
    for key, value in expected_metadata.items():
        expect_metadata(metadata, key, value, case_id)
    for key in ("post_normalization_min", "post_normalization_max"):
        if not math.isclose(
            float(metadata[key]), 1.0, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(f"{case_id} tangents were not normalized as planned")
    curve_bound = float(metadata["global_curve_bound"])
    if not math.isclose(
        curve_bound,
        float(metadata["sample_radius"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{case_id} curve-bound metadata disagrees")
    if not curve_bound < float(protocol["r_max"]):
        raise ValueError(f"{case_id} curve bound is not below r_max")
    if not math.isclose(
        curve_bound,
        float(job["global_curve_bound"]),
        rel_tol=1e-11,
        abs_tol=1e-11,
    ):
        raise ValueError(f"{case_id} Python/Julia curve bounds disagree")
    for name in ("births", "starts", "rank0"):
        expect_metadata(
            metadata,
            {
                "births": "births_sha256",
                "starts": "segment_starts_sha256",
                "rank0": "rank0_heatmap_sha256",
            }[name],
            sha256(paths[name]),
            case_id,
        )

    n_runs = int(protocol["n_runs"])
    expected_keys = {
        (int(length), run) for length in lengths for run in range(1, n_runs + 1)
    }
    first_birth: dict[tuple[int, int], float] = {}
    windows: dict[tuple[int, int], tuple[float, int, int]] = {}
    with paths["births"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_header = (
            "segment_length",
            "segment_duration",
            "run_index",
            "start_index",
            "end_index",
            "rank",
            "births",
        )
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"unexpected births schema for {case_id}")
        for row in reader:
            key = (int(row["segment_length"]), int(row["run_index"]))
            if key not in expected_keys or key in first_birth:
                raise ValueError(f"unexpected/duplicate trial {case_id} {key}")
            births = [float(value) for value in row["births"].split(";") if value]
            if (
                births != sorted(births)
                or any(not math.isfinite(value) or value < 0 for value in births)
                or len(births) != int(row["rank"])
            ):
                raise ValueError(f"invalid birth vector for {case_id} {key}")
            length = key[0]
            duration = float(row["segment_duration"])
            expected_duration = length * float(protocol["effective_sample_dt"])
            if not math.isclose(
                duration, expected_duration, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"duration mismatch for {case_id} {key}")
            start = int(row["start_index"])
            stop = int(row["end_index"])
            if start < 1 or stop != start + length - 1 or stop > job["n_samples"]:
                raise ValueError(f"invalid segment indices for {case_id} {key}")
            first_birth[key] = min(births, default=np.inf)
            windows[key] = (duration, start, stop)
    if set(first_birth) != expected_keys:
        raise ValueError(f"missing births trials for {case_id}")

    seen_starts: set[tuple[int, int]] = set()
    with paths["starts"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_header = (
            "segment_length",
            "segment_duration",
            "run_index",
            "start_index",
            "end_index",
        )
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"unexpected starts schema for {case_id}")
        for row in reader:
            key = (int(row["segment_length"]), int(row["run_index"]))
            actual = (
                float(row["segment_duration"]),
                int(row["start_index"]),
                int(row["end_index"]),
            )
            if key in seen_starts or windows.get(key) != actual:
                raise ValueError(f"start table mismatch for {case_id} {key}")
            seen_starts.add(key)
    if seen_starts != expected_keys:
        raise ValueError(f"missing starts trials for {case_id}")

    radii = np.linspace(
        float(protocol["r_min"]),
        float(protocol["r_max"]),
        int(protocol["r_subdivisions"]),
    )
    with paths["rank0"].open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["radius", *[str(value) for value in lengths]]:
            raise ValueError(f"unexpected rank-zero header for {case_id}")
        rows = list(reader)
    if len(rows) != len(radii):
        raise ValueError(f"rank-zero radius count mismatch for {case_id}")
    rank0 = np.empty((len(radii), len(lengths)), dtype=int)
    for radius_index, (row, radius) in enumerate(zip(rows, radii)):
        if len(row) != len(lengths) + 1 or not math.isclose(
            float(row[0]), radius, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"rank-zero radius mismatch for {case_id}")
        counts = np.asarray([int(value) for value in row[1:]], dtype=int)
        if np.any((counts < 0) | (counts > n_runs)):
            raise ValueError(f"rank-zero count outside 0..20 for {case_id}")
        rank0[radius_index] = counts
    for column, length in enumerate(lengths):
        births = np.asarray(
            [first_birth[(int(length), run)] for run in range(1, n_runs + 1)]
        )
        expected_counts = np.sum(
            births[np.newaxis, :] > radii[:, np.newaxis], axis=1
        )
        if not np.array_equal(rank0[:, column], expected_counts):
            raise ValueError(f"rank-zero counts disagree with births for {case_id}")
    durations = lengths.astype(float) * float(protocol["effective_sample_dt"])
    probability = 1.0 - rank0.astype(float) / n_runs
    if require_binding:
        validate_result_binding(job, protocol, plan)
    return durations, radii, probability, metadata


def check_all(
    configuration: Configuration,
    bundle: ValidatedBundle,
    julia_bin: str,
    run_kernel_checks: bool,
) -> None:
    environment = execution_environment(julia_bin)
    print(f"protocol={configuration.protocol['protocol_id']}")
    print("cases=" + ",".join(case["id"] for case in bundle.cases))
    print(f"bundle={bundle.root}")
    print(f"bundle_manifest_sha256={bundle.manifest_sha256}")
    print(
        "lengths=100:20:6000 durations=1:0.2:60 n=20 "
        "normalization=linf cover=5x1 C=5 radii=0:0.025:5 F_43"
    )
    print("horizontal_radius_guide=none")
    print("periodic_vertical_guides=manifest-certified full periods")
    print(f"kernel_sha256={environment['julia_kernel_sha256']}")
    print(f"manifest_sha256={environment['julia_manifest_sha256']}")
    print(f"cycling_repo_head={environment['cycling_repo_head']}")
    if not run_kernel_checks:
        print("Python-only check complete; Julia --check-only was explicitly skipped.")
        return
    check_root = SAFE_OUTPUT_ROOT / ".roessler-david-v2-check-only"
    for case in bundle.cases:
        command = julia_command(
            case,
            configuration.protocol,
            check_root / case["id"],
            environment["julia_executable"],
            check_only=True,
        )
        print(f"Julia check-only: {case['id']}")
        subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)
    print("CHECK ONLY complete: no comparison spaces built and no files written.")


def execute(
    plan_path: Path,
    julia_bin: str,
    selected_cases: set[str],
) -> None:
    root, plan, configuration, _ = load_plan(plan_path, julia_bin)
    jobs = plan["jobs"]
    known = {job["id"] for job in jobs}
    unknown = selected_cases - known
    if unknown:
        raise ValueError(f"unknown cases requested: {sorted(unknown)}")
    active = [job for job in jobs if not selected_cases or job["id"] in selected_cases]
    validated: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]
    ] = []
    for job in active:
        paths = result_paths(job)
        output_dir = Path(job["output_dir"])
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"refusing nonempty result directory for {job['id']}: {output_dir}"
            )
        if any(path.exists() or path.is_symlink() for path in paths.values()):
            raise FileExistsError(f"refusing to overwrite result for {job['id']}")
        binding_path = result_binding_path(job)
        if binding_path.exists() or binding_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite binding for {job['id']}")
        log_path = Path(job["log"])
        reject_symlink_components(log_path.parent, f"{job['id']} log directory")
        if log_path.exists() or log_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite log for {job['id']}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = list(job["command"])
        command[0] = plan["environment"]["julia_executable"]
        print(f"Running shared Julia kernel for {job['id']}", flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=WORKSPACE_ROOT,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        validate_result(
            job, configuration.protocol, plan, require_binding=False
        )
        binding = result_binding_document(
            job,
            configuration.protocol,
            plan,
            datetime.now(timezone.utc).isoformat(),
        )
        write_exclusive(
            binding_path,
            json.dumps(binding, indent=2, sort_keys=True) + "\n",
        )
        validated.append(validate_result(job, configuration.protocol, plan))
        print(f"Validated result for {job['id']} (log: {log_path})", flush=True)
    start_hashes = {
        item[3]["segment_starts_sha256"] for item in validated
    }
    if len(start_hashes) > 1:
        raise ValueError(
            "identically seeded cases did not use the same segment windows"
        )


def centers_to_edges(values: np.ndarray, lower: float | None = None) -> np.ndarray:
    if values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("plot centers must increase")
    midpoints = 0.5 * (values[:-1] + values[1:])
    edges = np.r_[
        values[0] - (midpoints[0] - values[0]),
        midpoints,
        values[-1] + (values[-1] - midpoints[-1]),
    ]
    if lower is not None:
        edges[0] = max(lower, edges[0])
    return edges


def display_positions(job: dict[str, Any]) -> np.ndarray:
    display = job["display"]
    values = np.genfromtxt(
        display["absolute_path"], delimiter=",", names=True, dtype=float
    )
    values = np.atleast_1d(values)
    positions = np.column_stack([values["x"], values["y"], values["z"]])
    if positions.shape != (int(display["n_rows"]), 3):
        raise ValueError(f"display shape changed for {job['id']}")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"display contains nonfinite positions for {job['id']}")
    return positions


def figure_bytes(
    plan: dict[str, Any],
    loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]],
) -> tuple[bytes, list[dict[str, Any]]]:
    jobs = plan["jobs"]
    durations = loaded[0][0]
    radii = loaded[0][1]
    if not all(np.array_equal(durations, item[0]) for item in loaded):
        raise ValueError("result cases do not share the duration grid")
    if not all(np.array_equal(radii, item[1]) for item in loaded):
        raise ValueError("result cases do not share the radius grid")
    if not (
        math.isclose(float(durations[0]), 1.0, abs_tol=1e-12)
        and math.isclose(float(durations[-1]), 60.0, abs_tol=1e-12)
        and np.allclose(np.diff(durations), 0.2, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("rendered duration grid must be 1:0.2:60")

    n_cases = len(jobs)
    figure = plt.figure(
        figsize=(3.2 * n_cases + 0.7, 6.8), constrained_layout=True
    )
    grid = figure.add_gridspec(
        2,
        n_cases + 1,
        width_ratios=[1] * n_cases + [0.05],
        height_ratios=[1.0, 1.05],
        hspace=0.12,
    )
    duration_edges = centers_to_edges(durations, lower=0.0)
    radius_edges = centers_to_edges(radii, lower=0.0)
    image = None
    display_records: list[dict[str, Any]] = []
    for column, (job, item) in enumerate(zip(jobs, loaded)):
        top = figure.add_subplot(grid[0, column], projection="3d")
        positions = display_positions(job)
        top.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color="#0072B2",
            linewidth=0.9,
            antialiased=True,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        top.set_xlabel("x")
        top.set_ylabel("y")
        top.set_zlabel("z")
        top.set_title(job["title"], fontsize=9)
        top.view_init(elev=25, azim=-55)

        bottom = figure.add_subplot(grid[1, column])
        image = bottom.pcolormesh(
            duration_edges,
            radius_edges,
            item[2],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            rasterized=True,
        )
        vertical_guide: float | None = None
        if job["kind"] == "periodic":
            vertical_guide = float(job["display"]["duration"])
            bottom.axvline(
                vertical_guide,
                color="white",
                linestyle="--",
                linewidth=0.9,
            )
        bottom.set_title(job["title"], fontsize=9)
        bottom.set_xlabel("segment duration (t)")
        bottom.set_xlim(0.0, 60.0)
        bottom.set_ylim(0.0, 5.0)
        if column == 0:
            bottom.set_ylabel("filtration radius (r)")
        else:
            bottom.set_yticklabels([])
        display_records.append(
            {
                "case_id": job["id"],
                "kind": job["kind"],
                "a": job["a"],
                "q": job["q"],
                "display": job["display"]["absolute_path"],
                "display_sha256": job["display"]["sha256"],
                "display_rows": int(job["display"]["n_rows"]),
                "display_duration": float(job["display"]["duration"]),
                "certificate": job["certificate"]["absolute_path"],
                "certificate_sha256": job["certificate"]["sha256"],
                "vertical_period_guide": vertical_guide,
                "horizontal_radius_guide": None,
            }
        )
    if image is None:
        plt.close(figure)
        raise RuntimeError("no probability image was created")
    color_axis = figure.add_subplot(grid[1, -1])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("probability")
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": plan["analysis_id"],
            "Creator": str(SCRIPT_PATH),
            "Subject": (
                "David-family Rössler certified orbits and validated "
                "cycling-signature probabilities"
            ),
        },
    )
    plt.close(figure)
    return buffer.getvalue(), display_records


def render(plan_path: Path, julia_bin: str) -> Path:
    root, plan, configuration, bundle = load_plan(plan_path, julia_bin)
    figure_path = root / plan["figure_filename"]
    provenance_path = figure_path.with_suffix(".render.json")
    for path in (figure_path, provenance_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite rendered artifact: {path}")
    loaded = [
        validate_result(job, configuration.protocol, plan) for job in plan["jobs"]
    ]
    start_hashes = {
        item[3]["segment_starts_sha256"] for item in loaded
    }
    if len(start_hashes) != 1:
        raise ValueError(
            "identically seeded cases did not use the same segment windows"
        )
    pdf, display_records = figure_bytes(plan, loaded)
    result_records: list[dict[str, Any]] = []
    for job, item in zip(plan["jobs"], loaded):
        raw_paths = result_paths(job)
        binding_path = result_binding_path(job)
        result_records.append(
            {
                "case_id": job["id"],
                "raw_results": {
                    name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in raw_paths.items()
                },
                "v2_result_binding": str(binding_path),
                "v2_result_binding_sha256": sha256(binding_path),
                "segment_starts_sha256": item[3]["segment_starts_sha256"],
                "global_curve_bound": float(item[3]["global_curve_bound"]),
            }
        )
    provenance = {
        "schema_version": 2,
        "analysis_id": plan["analysis_id"],
        "scope": "validated_render_no_signature_recomputation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": sha256(Path(plan_path).resolve()),
        "orchestrator": str(SCRIPT_PATH),
        "orchestrator_sha256": sha256(SCRIPT_PATH),
        "bundle_manifest": str(bundle.manifest_path),
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "julia_kernel": str(JULIA_KERNEL.resolve()),
        "julia_kernel_sha256": plan["environment"]["julia_kernel_sha256"],
        "kernel_internal_protocol_label": plan[
            "kernel_internal_protocol_label"
        ],
        "protocol_id": configuration.protocol["protocol_id"],
        "probability_statistic": "P(rank > 0) = 1 - rank0 / 20",
        "duration_grid": "1:0.2:60",
        "radius_grid": "0:0.025:5",
        "horizontal_radius_guide": None,
        "periodic_vertical_guides": "manifest display.duration",
        "cases": display_records,
        "results": result_records,
        "shared_segment_starts_sha256": next(iter(start_hashes)),
        "output_pdf": str(figure_path),
        "output_pdf_sha256": sha256_bytes(pdf),
    }
    provenance_bytes = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    linked_pdf = False
    with tempfile.TemporaryDirectory(
        dir=root, prefix=".roessler-v2-render-"
    ) as staging_name:
        staging = Path(staging_name)
        staged_pdf = staging / figure_path.name
        staged_provenance = staging / provenance_path.name
        staged_pdf.write_bytes(pdf)
        staged_provenance.write_bytes(provenance_bytes)
        try:
            os.link(staged_pdf, figure_path)
            linked_pdf = True
            os.link(staged_provenance, provenance_path)
        except BaseException:
            if linked_pdf and figure_path.is_file() and not figure_path.is_symlink():
                if sha256(figure_path) == sha256_bytes(pdf):
                    figure_path.unlink()
            raise
    print(f"Wrote {figure_path}")
    print(f"Wrote {provenance_path}")
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Versioned Rössler-only cycling-probability analysis; the default "
            "action is read-only check."
        )
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="check",
        choices=("check", "materialize", "execute", "render"),
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--julia-bin", default="julia")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="new named output root; materialize only",
    )
    parser.add_argument("--plan", type=Path, help="existing v2 plan; execute/render")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="case id to execute; repeat to select more than one",
    )
    parser.add_argument(
        "--skip-kernel-check",
        action="store_true",
        help="check bundle/settings without invoking Julia --check-only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action in {"execute", "render"}:
        if args.plan is None:
            raise ValueError(f"{args.action} requires --plan")
        if args.output_root is not None:
            raise ValueError("--output-root is only valid for materialize")
        if args.skip_kernel_check:
            raise ValueError("--skip-kernel-check is only valid for check")
        if args.action == "render" and args.case:
            raise ValueError("render requires all five completed cases")
        if args.action == "execute":
            execute(args.plan, args.julia_bin, set(args.case))
        else:
            render(args.plan, args.julia_bin)
        return

    if args.plan is not None:
        raise ValueError("--plan is only valid for execute/render")
    if args.case:
        raise ValueError("--case is only valid for execute")
    configuration = load_configuration(args.cases)
    bundle = validate_bundle(configuration)
    if args.action == "check":
        if args.output_root is not None:
            raise ValueError("--output-root is only valid for materialize")
        check_all(
            configuration,
            bundle,
            args.julia_bin,
            run_kernel_checks=not args.skip_kernel_check,
        )
        return
    if args.skip_kernel_check:
        raise ValueError("materialize always performs Julia check-only preflights")
    check_all(configuration, bundle, args.julia_bin, run_kernel_checks=True)
    output_root = (
        guard_output_root(args.output_root, must_not_exist=True)
        if args.output_root is not None
        else suggested_output_root(configuration)
    )
    materialize(configuration, bundle, output_root, args.julia_bin)


if __name__ == "__main__":
    main()
