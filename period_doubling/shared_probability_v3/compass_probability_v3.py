#!/usr/bin/env python3
"""Hash-bound Compass v3 probability analysis for the refined orbit bundle.

The default ``check`` action is read-only. ``materialize`` creates an inert,
immutable plan. Only ``execute`` computes cycling signatures. The frozen v2
driver, protocol, plan, results, summaries, and figure are never modified.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
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
SHARED_V2_HELPER = (
    CODE_ROOT / "period_doubling" / "shared_probability_v2" /
    "compass_probability_v2.py"
)
REFERENCE_V1_MANIFEST = (
    CODE_ROOT / "experiments_planned" / "outputs" /
    "compass_embedded_fourier_orbits_v1" / "bundle_manifest.json"
)
sys.path.insert(0, str(SHARED_V2_HELPER.parent))
import compass_probability_v2 as shared  # noqa: E402

JULIA_KERNEL = CODE_ROOT / "period_doubling" / "julia" / "run_shared_probability.jl"
JULIA_PREFLIGHT = (
    CODE_ROOT / "period_doubling" / "shared_probability_v2" /
    "preflight_compass_bundle_v2.jl"
)
KERNEL_PROTOCOL = shared.KERNEL_INTERNAL_PROTOCOL_LABEL

EXPECTED_CASES = (
    (
        "period1", "phi = 4.00 deg", 4.00, 1,
        0.878246991978623, 0.880, 0.7482409701092134,
    ),
    (
        "period2", "phi = 4.75 deg", 4.75, 2,
        1.762130681818183, 1.760, 1.5021396458781595,
    ),
    (
        "period4", "phi = 5.00 deg", 5.00, 4,
        3.521923462566852, 3.520, 3.0019139378989905,
    ),
    (
        "period8", "phi = 5.02 deg", 5.02, 8,
        7.044294786096257, 7.045, 6.004311980796729,
    ),
    ("chaos", "phi = 5.20 deg", 5.20, None, None, None, None),
)
PERIOD_TOLERANCE = 5e-12
RECURRENCE_TOLERANCE = 1e-12


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
    curve_target_pass: bool


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
            result = json.load(handle)
    except FileNotFoundError as error:
        raise ValueError(f"missing JSON file: {path}") from error
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def resolve_existing_file(path: Path, label: str) -> Path:
    shared.reject_symlink_components(path, label)
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing {label}: {resolved}")
    return resolved


def segment_lengths(protocol: dict[str, Any]) -> list[int]:
    spec = protocol["segment_lengths"]
    start, step, stop = int(spec["start"]), int(spec["step"]), int(spec["stop"])
    if start < 2 or step <= 0 or stop < start or (stop - start) % step:
        raise ValueError("invalid segment-length grid")
    return list(range(start, stop + 1, step))


# The unchanged shared helpers call their module-level parser dynamically.
shared.segment_lengths = segment_lengths


def validate_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "schema_version": 3,
        "protocol_id": "compass_refined_v3_probability_linf_C0p75",
        "statistic": "P(rank > 0)",
        "segment_lengths": {"start": 160, "step": 80, "stop": 9600},
        "n_runs": 20,
        "sampling": "independent_uniform_with_replacement_per_length",
        "paired_starts_across_equal_length_streams": True,
        "seed": 20260820,
        "duration_convention": "segment_length_times_effective_dt",
        "effective_sample_dt": 0.00125,
        "tangent_normalization": "linf",
        "position_boxsize": 0.75,
        "sphere_box_resolution": 1,
        "metric_c": 0.75,
        "r_min": 0.0,
        "r_max": 0.5,
        "r_step": 0.002,
        "r_subdivisions": 251,
        "field_prime": 43,
        "curve_bound_target": 0.18,
        "require_sample_radius_below_r_max": True,
        "require_beta1_y": 1,
        "require_no_exact_full_period_recurrence": True,
        "plot_horizontal_radius_guide": False,
        "plot_vertical_refined_period_guides": True,
    }
    if set(protocol) != set(expected):
        raise ValueError(
            f"unexpected protocol keys: {sorted(set(protocol) ^ set(expected))}"
        )
    for key, wanted in expected.items():
        actual = protocol[key]
        if isinstance(wanted, float):
            if not math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-14):
                raise ValueError(f"protocol {key}={actual!r}, expected {wanted!r}")
        elif actual != wanted:
            raise ValueError(f"protocol {key}={actual!r}, expected {wanted!r}")
    if segment_lengths(protocol) != list(range(160, 9601, 80)):
        raise ValueError("v3 lengths must be 160:80:9600")
    radii = np.linspace(protocol["r_min"], protocol["r_max"], protocol["r_subdivisions"])
    if not np.allclose(np.diff(radii), protocol["r_step"], rtol=0.0, atol=1e-14):
        raise ValueError("radius step/subdivision mismatch")
    if not math.isclose(
        float(protocol["metric_c"]),
        float(protocol["position_boxsize"]) * int(protocol["sphere_box_resolution"]),
        rel_tol=0.0,
        abs_tol=1e-14,
    ):
        raise ValueError("v3 requires C=position_boxsize*sb_radius")


def validate_cases(document: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version", "analysis_id", "protocol", "bundle_manifest",
        "suggested_output_root", "figure_filename", "summary_directory", "cases",
    }
    if set(document) != expected_keys:
        raise ValueError(
            f"unexpected cases keys: {sorted(set(document) ^ expected_keys)}"
        )
    expected_scalars = {
        "schema_version": 3,
        "analysis_id": "compass_refined_v3_probability_linf_C0p75",
        "protocol": "compass_protocol.json",
        "bundle_manifest": (
            "../../experiments_planned/outputs/"
            "compass_embedded_fourier_orbits_v3_refined/bundle_manifest.json"
        ),
        "suggested_output_root": (
            "experiments_planned/outputs/shared_coauthor_protocol/"
            "compass_refined_v3_probability_linf_C0p75"
        ),
        "figure_filename": "compassgait_C0p75.pdf",
        "summary_directory": "compact_summary_v1",
    }
    for key, wanted in expected_scalars.items():
        if document[key] != wanted:
            raise ValueError(f"unexpected cases value for {key}")
    records = document["cases"]
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("cases must contain period1/2/4/8 and chaos")
    keys = {
        "id", "title", "phi_deg", "q", "refined_continuous_period",
        "legacy_nominal_suspension_period", "physical_return_seconds",
    }
    for actual, expected in zip(records, EXPECTED_CASES):
        case_id, title, phi, q, period, legacy, physical = expected
        if set(actual) != keys or actual["id"] != case_id or actual["title"] != title:
            raise ValueError(f"unexpected case record for {case_id}")
        if actual["q"] != q or not math.isclose(
            float(actual["phi_deg"]), phi, rel_tol=0.0, abs_tol=1e-14
        ):
            raise ValueError(f"unexpected phi/q for {case_id}")
        for key, wanted in (
            ("refined_continuous_period", period),
            ("legacy_nominal_suspension_period", legacy),
            ("physical_return_seconds", physical),
        ):
            if wanted is None:
                if actual[key] is not None:
                    raise ValueError(f"chaos must not have {key}")
            elif not math.isclose(
                float(actual[key]), wanted, rel_tol=0.0, abs_tol=PERIOD_TOLERANCE
            ):
                raise ValueError(f"unexpected {key} for {case_id}")


def load_configuration(cases_path: Path, bundle_override: Path | None) -> Configuration:
    cases_path = resolve_existing_file(cases_path, "Compass v3 cases")
    cases = load_json(cases_path)
    validate_cases(cases)
    protocol_path = resolve_existing_file(
        cases_path.parent / str(cases["protocol"]), "Compass v3 protocol"
    )
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    bundle_path = resolve_existing_file(
        bundle_override if bundle_override is not None
        else cases_path.parent / str(cases["bundle_manifest"]),
        "Compass refined v3 bundle",
    )
    return Configuration(cases_path, protocol_path, bundle_path, cases, protocol)


def resolve_bundle_record(
    root: Path, record: Any, label: str, extras: set[str] | None = None
) -> tuple[Path, str]:
    expected = {"path", "sha256"} | (extras or set())
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError(f"{label} has an unexpected record schema")
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
        raise ValueError(f"{label} must be bundle-relative")
    path = resolve_existing_file(root / relative, label)
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the bundle")
    actual = sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"{label} hash changed")
    return path, actual


def validate_inventory(manifest: dict[str, Any], root: Path) -> None:
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
            root, {"path": relative, "sha256": record["sha256"]},
            "bundle inventory file",
        )
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"bundle inventory byte count changed: {relative}")
        expected_paths.add(relative)
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    if actual_paths != expected_paths:
        raise ValueError(
            f"bundle inventory mismatch: missing={sorted(expected_paths-actual_paths)}, "
            f"extra={sorted(actual_paths-expected_paths)}"
        )


def nested_float(record: dict[str, Any], names: tuple[str, ...], label: str) -> float:
    for name in names:
        if name in record:
            value = float(record[name])
            if math.isfinite(value):
                return value
    raise ValueError(f"missing finite {label}; expected one of {names}")


def recurrence_distances(
    positions: np.ndarray, tangents: np.ndarray, period: float, dt: float, metric_c: float
) -> dict[str, float]:
    norms = np.max(np.abs(tangents), axis=1)
    normalized = tangents / norms[:, np.newaxis]
    ratio = period / dt
    result: dict[str, float] = {"period_over_dt": ratio}
    minima: list[float] = []
    for label, lag in (("floor", math.floor(ratio)), ("ceil", math.ceil(ratio))):
        if lag <= 0 or lag >= len(positions):
            raise ValueError("full-period recurrence lag is outside the analysis stream")
        dx = np.linalg.norm(positions[lag:] - positions[:-lag], axis=1)
        dv = np.linalg.norm(normalized[lag:] - normalized[:-lag], axis=1)
        distance = np.maximum(dx, metric_c * dv)
        minimum = float(np.min(distance))
        result[f"{label}_lag"] = int(lag)
        result[f"{label}_minimum_dynamic_distance"] = minimum
        minima.append(minimum)
    result["minimum_dynamic_distance"] = min(minima)
    if result["minimum_dynamic_distance"] <= RECURRENCE_TOLERANCE:
        raise ValueError(
            "exact or numerically zero full-period recurrence remains in refined stream"
        )
    return result


def reduced_sample_recurrence(
    period: float, dt: float, maximum_segment_length: int
) -> dict[str, int | float]:
    ratio = period / dt
    reduced = Fraction(ratio).limit_denominator(1_000_000)
    error = abs(float(reduced) - ratio)
    if error > 1e-10:
        raise ValueError(
            f"period/dt has no stable reduced rational certificate: error={error:.3g}"
        )
    if reduced.numerator <= maximum_segment_length:
        raise ValueError(
            "shortest exact sample recurrence lies inside the maximum segment: "
            f"{reduced.numerator} <= {maximum_segment_length}"
        )
    return {
        "shortest_recurrence_samples": reduced.numerator,
        "cycles_at_recurrence": reduced.denominator,
        "maximum_segment_length": maximum_segment_length,
        "ratio_approximation_error": error,
    }


def validate_bundle(configuration: Configuration) -> ValidatedBundle:
    manifest = load_json(configuration.bundle_manifest_path)
    expected_keys = {
        "schema_version", "bundle_id", "status", "created_utc", "generator",
        "scientific_scope", "analysis_sample_dt", "analysis_duration",
        "analysis_n_samples", "dimension", "metric_c_preflight",
        "maximum_global_curve_bound", "cases", "summary", "files",
        "recurrence_policy",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            f"unexpected refined bundle keys: {sorted(set(manifest) ^ expected_keys)}"
        )
    if (
        int(manifest["schema_version"]) != 3
        or manifest["bundle_id"] != "compass_embedded_fourier_orbits_v3_refined"
        or manifest["status"] != "complete"
    ):
        raise ValueError("unexpected refined bundle identity/status")
    root = configuration.bundle_manifest_path.parent.resolve()
    shared.reject_symlink_components(root, "Compass refined v3 bundle")
    validate_inventory(manifest, root)
    generator = manifest["generator"]
    if not isinstance(generator, dict) or not {"path", "sha256"}.issubset(generator):
        raise ValueError("refined bundle lacks a generator hash binding")
    generator_path = shared.resolve_code_relative(str(generator["path"]), "refined builder")
    if sha256(generator_path) != generator["sha256"]:
        raise ValueError("refined bundle builder changed")
    recurrence_policy = manifest["recurrence_policy"]
    if not isinstance(recurrence_policy, dict) or not recurrence_policy:
        raise ValueError("refined bundle must state its recurrence policy")
    dt = float(manifest["analysis_sample_dt"])
    duration = float(manifest["analysis_duration"])
    n_samples = int(manifest["analysis_n_samples"])
    dimension = int(manifest["dimension"])
    if not math.isclose(dt, 0.00125, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("refined bundle dt must be .00125")
    if not math.isclose(dt, configuration.protocol["effective_sample_dt"], rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("bundle/protocol cadence mismatch")
    if duration <= 12.0 or n_samples <= max(segment_lengths(configuration.protocol)):
        raise ValueError("refined bundle stream is too short")
    if dimension != 11:
        raise ValueError("refined bundle dimension must be 11")
    if not math.isclose(float(manifest["metric_c_preflight"]), 0.75, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("refined bundle preflight C must be .75")
    resolve_bundle_record(root, manifest["summary"], "refined bundle summary")
    reference_manifest_path = resolve_existing_file(
        REFERENCE_V1_MANIFEST, "reference v1 Fourier bundle manifest"
    )
    reference_manifest = load_json(reference_manifest_path)
    if reference_manifest.get("bundle_id") != "compass_embedded_fourier_orbits_v1":
        raise ValueError("unexpected reference v1 Fourier bundle")
    reference_root = reference_manifest_path.parent
    reference_cases = {
        str(item.get("id")): item for item in reference_manifest.get("cases", [])
    }

    records = manifest["cases"]
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("refined bundle must contain five cases")
    by_id = {str(record.get("id")): record for record in records}
    if set(by_id) != {record[0] for record in EXPECTED_CASES}:
        raise ValueError("refined bundle case ids changed")
    case_documents = {case["id"]: case for case in configuration.cases["cases"]}
    validated: list[dict[str, Any]] = []
    maximum_bound = 0.0
    target = float(configuration.protocol["curve_bound_target"])
    for expected in EXPECTED_CASES:
        case_id, _, phi, q, expected_period, legacy, physical = expected
        record = by_id[case_id]
        expected_case_keys = {
            "id", "kind", "phi_deg", "q", "nominal_suspension_period",
            "physical_return_seconds", "dimension", "analysis_sample_dt",
            "n_samples", "duration", "tangent_semantics", "positions",
            "tangents", "display", "fourier", "certificate",
            "global_curve_bound",
        }
        if set(record) != expected_case_keys:
            raise ValueError(
                f"{case_id}: unexpected refined case keys "
                f"{sorted(set(record) ^ expected_case_keys)}"
            )
        if record["id"] != case_id or record["q"] != q:
            raise ValueError(f"{case_id}: identity/q changed")
        expected_kind = (
            "periodic_fourier_closed_continuous_ols_period"
            if q is not None else "chaos_interpolated_frozen_path_quarter_step"
        )
        expected_tangent_semantics = (
            "analytic_fourier_derivative"
            if q is not None
            else "interpolated_learned_flow_direction_on_interpolated_frozen_path"
        )
        if record["kind"] != expected_kind:
            raise ValueError(f"{case_id}: refined case kind changed")
        if record["tangent_semantics"] != expected_tangent_semantics:
            raise ValueError(f"{case_id}: tangent semantics changed")
        if not math.isclose(float(record["phi_deg"]), phi, rel_tol=0.0, abs_tol=1e-14):
            raise ValueError(f"{case_id}: phi changed")
        if int(record["dimension"]) != dimension or int(record["n_samples"]) != n_samples:
            raise ValueError(f"{case_id}: analysis shape metadata changed")
        if not math.isclose(float(record["analysis_sample_dt"]), dt, rel_tol=0.0, abs_tol=1e-14):
            raise ValueError(f"{case_id}: dt changed")
        if not math.isclose(float(record["duration"]), duration, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"{case_id}: duration changed")
        document = case_documents[case_id]
        if q is None:
            if record["nominal_suspension_period"] is not None or record["physical_return_seconds"] is not None:
                raise ValueError("chaos must not have periodic-return values")
            refined_period = None
        else:
            refined_period = float(record["nominal_suspension_period"])
            if not math.isclose(
                refined_period, float(expected_period), rel_tol=0.0,
                abs_tol=PERIOD_TOLERANCE,
            ):
                raise ValueError(f"{case_id}: refined continuous period changed")
            if not math.isclose(
                refined_period, float(document["refined_continuous_period"]),
                rel_tol=0.0, abs_tol=PERIOD_TOLERANCE,
            ):
                raise ValueError(f"{case_id}: cases/bundle period binding changed")
            if not math.isclose(
                float(record["physical_return_seconds"]), float(physical),
                rel_tol=0.0, abs_tol=5e-12,
            ):
                raise ValueError(f"{case_id}: physical-return provenance changed")
        positions_path, positions_hash = resolve_bundle_record(
            root, record["positions"], f"{case_id} positions"
        )
        tangents_path, tangents_hash = resolve_bundle_record(
            root, record["tangents"], f"{case_id} tangents"
        )
        display_path, display_hash = resolve_bundle_record(
            root, record["display"], f"{case_id} display", {"kind", "n_rows"}
        )
        certificate_path, certificate_hash = resolve_bundle_record(
            root, record["certificate"], f"{case_id} certificate"
        )
        certificate = load_json(certificate_path)
        if certificate.get("case_id") != case_id or certificate.get("q") != q:
            raise ValueError(f"{case_id}: certificate identity changed")
        analysis = certificate.get("analysis")
        if not isinstance(analysis, dict) or int(analysis.get("n_samples", -1)) != n_samples:
            raise ValueError(f"{case_id}: certificate analysis metadata changed")
        positions = np.loadtxt(positions_path, dtype=float)
        tangents = np.loadtxt(tangents_path, dtype=float)
        if positions.shape != (n_samples, dimension) or tangents.shape != positions.shape:
            raise ValueError(f"{case_id}: analysis matrix shape changed")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(tangents)):
            raise ValueError(f"{case_id}: nonfinite analysis matrix")
        curve_bound = shared.global_curve_bound(
            positions, tangents, float(configuration.protocol["metric_c"])
        )
        tangent_norms = np.max(np.abs(tangents), axis=1)
        normalized_tangents = tangents / tangent_norms[:, np.newaxis]
        maximum_dx = float(np.max(np.linalg.norm(
            np.diff(positions, axis=0), axis=1
        )))
        maximum_dv = float(np.max(np.linalg.norm(
            np.diff(normalized_tangents, axis=0), axis=1
        )))
        certificate_curve = analysis.get("curve_resolution")
        if not isinstance(certificate_curve, dict):
            raise ValueError(f"{case_id}: missing curve-resolution certificate")
        certified_curve_values = {
            "maximum_consecutive_position_distance": maximum_dx,
            "maximum_consecutive_normalized_tangent_distance": maximum_dv,
            "metric_c": float(configuration.protocol["metric_c"]),
            "global_curve_bound": curve_bound,
        }
        for key, wanted in certified_curve_values.items():
            if not math.isclose(
                float(certificate_curve.get(key, math.nan)), wanted,
                rel_tol=0.0, abs_tol=3e-10,
            ):
                raise ValueError(f"{case_id}: curve certificate {key} changed")
        if not math.isclose(
            curve_bound, float(record["global_curve_bound"]),
            rel_tol=0.0, abs_tol=3e-10,
        ):
            raise ValueError(f"{case_id}: bundle/driver curve bound mismatch")
        if not curve_bound < float(configuration.protocol["r_max"]):
            raise ValueError(f"{case_id}: hard h<r_max gate failed")
        maximum_bound = max(maximum_bound, curve_bound)

        if q is not None:
            estimate = certificate.get("period_estimate")
            recurrence_certificate = certificate.get("full_period_recurrence")
            if not isinstance(estimate, dict) or not isinstance(recurrence_certificate, dict):
                raise ValueError(f"{case_id}: missing refined period/recurrence certificate")
            if analysis.get("full_period_recurrence") != recurrence_certificate:
                raise ValueError(f"{case_id}: duplicated recurrence certificates disagree")
            if (
                recurrence_certificate.get("status") != "passed"
                or int(recurrence_certificate.get(
                    "exact_lifted_pair_count_across_candidates", -1
                )) != 0
                or int(recurrence_certificate.get("max_audited_lag_samples", -1))
                != max(segment_lengths(configuration.protocol))
                or float(recurrence_certificate.get(
                    "minimum_audited_dynamic_distance", -math.inf
                )) <= RECURRENCE_TOLERANCE
            ):
                raise ValueError(f"{case_id}: planned-window recurrence audit failed")
            if estimate.get("canonical_method") != (
                "ordinary_least_squares_slope_on_33_q_spaced_"
                "bridge_to_arc_boundary_indices"
            ):
                raise ValueError(f"{case_id}: period estimator changed")
            boundary_ordinals = estimate.get("boundary_ordinals")
            boundary_indices = estimate.get("boundary_indices_zero_based")
            if (
                int(estimate.get("n_boundary_indices", -1)) != 33
                or int(estimate.get("q_spacing", -1)) != q
                or not isinstance(boundary_ordinals, list)
                or not isinstance(boundary_indices, list)
                or len(boundary_ordinals) != 33
                or len(boundary_indices) != 33
                or any(
                    int(b) - int(a) != q
                    for a, b in zip(boundary_ordinals, boundary_ordinals[1:])
                )
                or any(
                    int(b) <= int(a)
                    for a, b in zip(boundary_indices, boundary_indices[1:])
                )
            ):
                raise ValueError(f"{case_id}: OLS boundary-index certificate changed")
            certificate_period = nested_float(
                estimate,
                ("continuous_period", "ols_continuous_period", "period"),
                f"{case_id} continuous period",
            )
            if not math.isclose(
                certificate_period, refined_period, rel_tol=0.0,
                abs_tol=PERIOD_TOLERANCE,
            ):
                raise ValueError(f"{case_id}: certificate/manifest period mismatch")
            certificate_ratio = nested_float(
                estimate,
                ("period_over_analysis_dt", "period_over_dt"),
                f"{case_id} period/dt",
            )
            if not math.isclose(certificate_ratio, refined_period / dt, rel_tol=0.0, abs_tol=2e-9):
                raise ValueError(f"{case_id}: certificate period/dt mismatch")
            if math.isclose(certificate_ratio, round(certificate_ratio), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{case_id}: refined period is commensurate with dt")
            recurrence = recurrence_distances(
                positions, tangents, refined_period, dt,
                float(configuration.protocol["metric_c"]),
            )
            recurrence.update(reduced_sample_recurrence(
                refined_period, dt, max(segment_lengths(configuration.protocol))
            ))
            for key in (
                "shortest_recurrence_samples", "cycles_at_recurrence",
                "maximum_segment_length",
            ):
                if int(recurrence_certificate.get(key, -1)) != int(recurrence[key]):
                    raise ValueError(
                        f"{case_id}: recurrence certificate {key} changed"
                    )
            if record["fourier"] is None:
                raise ValueError(f"{case_id}: periodic case lacks Fourier coefficients")
            fourier_path, fourier_hash = resolve_bundle_record(
                root, record["fourier"], f"{case_id} Fourier coefficients",
                {"harmonic_cutoff"},
            )
            if int(record["fourier"]["harmonic_cutoff"]) != 6 * q:
                raise ValueError(f"{case_id}: Fourier cutoff must remain H=6q")
            reference_case = reference_cases.get(case_id)
            if reference_case is None or reference_case.get("fourier") is None:
                raise ValueError(f"{case_id}: missing reference v1 Fourier record")
            reference_fourier_path, reference_fourier_hash = resolve_bundle_record(
                reference_root, reference_case["fourier"],
                f"{case_id} reference v1 Fourier coefficients", {"harmonic_cutoff"},
            )
            if fourier_hash != reference_fourier_hash:
                raise ValueError(
                    f"{case_id}: refined Fourier coefficients are not byte-identical to v1"
                )
            fourier: dict[str, Any] | None = {
                **record["fourier"], "absolute_path": str(fourier_path),
                "sha256": fourier_hash,
                "reference_v1_path": str(reference_fourier_path),
                "reference_v1_sha256": reference_fourier_hash,
            }
        else:
            if record["fourier"] is not None:
                raise ValueError("chaos must remain explicitly nonperiodic")
            recurrence = None
            fourier = None
        with display_path.open(encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
            rows = sum(1 for _ in handle)
        if header[:2] != ["nominal_suspension_time", "z0"] or rows != int(record["display"]["n_rows"]):
            raise ValueError(f"{case_id}: display schema/row count changed")
        validated.append({
            "id": case_id,
            "title": document["title"],
            "kind": record["kind"],
            "phi_deg": phi,
            "q": q,
            "refined_continuous_period": refined_period,
            "nominal_suspension_period": refined_period,
            "legacy_nominal_suspension_period": legacy,
            "physical_return_seconds": physical,
            "raw_sample_dt": dt,
            "stride": 1,
            "effective_sample_dt": dt,
            "dimension": dimension,
            "n_samples": n_samples,
            "analysis_n_samples": n_samples,
            "analysis_duration": duration,
            "positions": str(positions_path),
            "positions_sha256": positions_hash,
            "tangents": str(tangents_path),
            "tangents_sha256": tangents_hash,
            "tangent_semantics": record["tangent_semantics"],
            "global_curve_bound": curve_bound,
            "curve_bound_target_pass": curve_bound < target,
            "expected_beta1": int(configuration.protocol["require_beta1_y"]),
            "recurrence_preflight": recurrence,
            "certificate": {
                **record["certificate"], "absolute_path": str(certificate_path),
                "sha256": certificate_hash,
            },
            "fourier": fourier,
            "display_extract": {
                **record["display"], "path": str(display_path),
                "sha256": display_hash,
            },
        })
    if not math.isclose(
        maximum_bound, float(manifest["maximum_global_curve_bound"]),
        rel_tol=0.0, abs_tol=3e-10,
    ):
        raise ValueError("bundle maximum curve bound changed")
    curve_target_pass = maximum_bound < target
    if not curve_target_pass:
        raise ValueError(
            f"refined bundle misses frozen h<{target:g} target: h_max={maximum_bound:.17g}"
        )
    return ValidatedBundle(
        configuration.bundle_manifest_path,
        sha256(configuration.bundle_manifest_path),
        manifest,
        tuple(validated),
        curve_target_pass,
    )


def guard_output_root(path: Path, must_not_exist: bool) -> Path:
    shared.reject_symlink_components(path, "Compass v3 output root")
    resolved = path.resolve(strict=False)
    safe = SAFE_OUTPUT_ROOT.resolve()
    if resolved == safe or resolved.parent != safe:
        raise ValueError(f"output must be a direct named child of {safe}")
    required_tokens = ("compass", "refined", "v3", "c0p75")
    lowered = resolved.name.lower()
    if any(token not in lowered for token in required_tokens):
        raise ValueError("v3 output name must contain compass/refined/v3/C0p75")
    if must_not_exist and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(f"refusing to overwrite output root: {resolved}")
    return resolved


def suggested_output_root(configuration: Configuration) -> Path:
    return guard_output_root(
        CODE_ROOT / str(configuration.cases["suggested_output_root"]), False
    )


def environment(julia_bin: str) -> dict[str, Any]:
    result = shared.execution_environment(julia_bin)
    result.update({
        "v3_orchestrator": str(SCRIPT_PATH),
        "v3_orchestrator_sha256": sha256(SCRIPT_PATH),
        "shared_v2_validation_helper": str(SHARED_V2_HELPER),
        "shared_v2_validation_helper_sha256": sha256(SHARED_V2_HELPER),
    })
    return result


def julia_command(
    case: dict[str, Any], protocol: dict[str, Any], output_dir: Path,
    julia_bin: str, check_only: bool = False,
) -> list[str]:
    return shared.julia_command(case, protocol, output_dir, julia_bin, check_only)


def run_preflights(
    bundle: ValidatedBundle, protocol: dict[str, Any], julia_bin: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in bundle.cases:
        command = shared.beta_preflight_command(case, protocol, julia_bin)
        completed = subprocess.run(
            command, cwd=WORKSPACE_ROOT, check=True, capture_output=True, text=True
        )
        output = completed.stdout
        beta_match = re.search(r"^beta1_Y=(\d+)$", output, re.MULTILINE)
        bound_match = re.search(
            r"^global_curve_bound=([0-9eE+.-]+)$", output, re.MULTILINE
        )
        sample_match = re.search(r"^analysis_samples=(\d+)$", output, re.MULTILINE)
        if not (beta_match and bound_match and sample_match):
            raise ValueError(f"{case['id']}: could not parse Julia preflight")
        beta1 = int(beta_match.group(1))
        bound = float(bound_match.group(1))
        if beta1 != int(protocol["require_beta1_y"]):
            raise ValueError(f"{case['id']}: beta1(Y)={beta1}, expected 1")
        if not math.isclose(bound, case["global_curve_bound"], rel_tol=0.0, abs_tol=3e-10):
            raise ValueError(f"{case['id']}: Python/Julia h mismatch")
        if bound >= float(protocol["r_max"]):
            raise ValueError(f"{case['id']}: Julia hard h<r_max gate failed")
        record = {
            "case_id": case["id"],
            "beta1_Y": beta1,
            "global_curve_bound": bound,
            "curve_bound_target": float(protocol["curve_bound_target"]),
            "curve_bound_target_pass": bound < float(protocol["curve_bound_target"]),
            "analysis_samples": int(sample_match.group(1)),
            "command": command,
        }
        records.append(record)
        print(
            f"{case['id']}: beta1(Y)={beta1} h={bound:.15g} "
            f"h<.18={record['curve_bound_target_pass']} recurrence="
            f"{case['recurrence_preflight'] or 'nonperiodic'}"
        )
    return records


def serializable_case(case: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(case))


def build_jobs(
    bundle: ValidatedBundle, protocol: dict[str, Any], root: Path, julia_bin: str
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for case in bundle.cases:
        job = serializable_case(case)
        job["output_dir"] = str(root / "signatures" / case["id"])
        job["log"] = str(root / "logs" / f"{case['id']}.log")
        job["command"] = julia_command(
            job, protocol, Path(job["output_dir"]), julia_bin, False
        )
        jobs.append(job)
    return jobs


def commands_text(jobs: list[dict[str, Any]]) -> str:
    return "#!/bin/sh\nset -eu\n" + "\n".join(
        shlex.join(job["command"]) for job in jobs
    ) + "\n"


def atomic_write_exclusive(path: Path, value: str | bytes, mode: int | None = None) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing target: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(
    configuration: Configuration, bundle: ValidatedBundle, output_root: Path,
    julia_bin: str,
) -> Path:
    root = guard_output_root(output_root, True)
    env = environment(julia_bin)
    preflight = run_preflights(bundle, configuration.protocol, env["julia_executable"])
    root.mkdir(parents=True)
    try:
        jobs = build_jobs(bundle, configuration.protocol, root, env["julia_executable"])
        command_text = commands_text(jobs)
        commands_path = root / "commands.sh"
        plan = {
            "schema_version": 3,
            "status": "materialized_not_executed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_id": configuration.cases["analysis_id"],
            "figure_filename": configuration.cases["figure_filename"],
            "summary_directory": configuration.cases["summary_directory"],
            "output_root": str(root),
            "orchestrator": str(SCRIPT_PATH),
            "orchestrator_sha256": sha256(SCRIPT_PATH),
            "shared_validation_helper": str(SHARED_V2_HELPER),
            "shared_validation_helper_sha256": sha256(SHARED_V2_HELPER),
            "cases_path": str(configuration.cases_path),
            "cases_sha256": sha256(configuration.cases_path),
            "protocol_path": str(configuration.protocol_path),
            "protocol_sha256": sha256(configuration.protocol_path),
            "protocol": configuration.protocol,
            "bundle_manifest": str(bundle.manifest_path),
            "bundle_manifest_sha256": bundle.manifest_sha256,
            "bundle_id": bundle.manifest["bundle_id"],
            "bundle_generator": bundle.manifest["generator"],
            "reference_v1_bundle_manifest": str(REFERENCE_V1_MANIFEST.resolve()),
            "reference_v1_bundle_manifest_sha256": sha256(REFERENCE_V1_MANIFEST.resolve()),
            "recurrence_policy": bundle.manifest["recurrence_policy"],
            "scientific_scope": bundle.manifest["scientific_scope"],
            "environment": env,
            "kernel_internal_protocol_label": KERNEL_PROTOCOL,
            "preflight": preflight,
            "commands": str(commands_path),
            "commands_sha256": sha256_bytes(command_text.encode("utf-8")),
            "jobs": jobs,
        }
        plan_path = root / "plan.json"
        atomic_write_exclusive(
            commands_path, command_text, 0o755
        )
        atomic_write_exclusive(
            plan_path, json.dumps(plan, indent=2, sort_keys=True) + "\n"
        )
        atomic_write_exclusive(
            root / "plan.sha256", f"{sha256(plan_path)}  plan.json\n"
        )
    except BaseException:
        marker = root / "MATERIALIZATION_INCOMPLETE"
        if not marker.exists():
            atomic_write_exclusive(marker, "Compass v3 materialization did not complete\n")
        raise
    print(f"Materialized {len(jobs)} inert Compass v3 jobs at {root}")
    print("No cycling signatures were executed.")
    return plan_path


def check_all(
    configuration: Configuration, bundle: ValidatedBundle, julia_bin: str,
    python_only: bool,
) -> None:
    env = environment(julia_bin)
    print(f"protocol={configuration.protocol['protocol_id']}")
    print(f"bundle={bundle.manifest['bundle_id']}")
    print(f"bundle_manifest_sha256={bundle.manifest_sha256}")
    print(
        "lengths=160:80:9600 durations=.2:.1:12 n=20 normalization=linf "
        "cover=.75x1 C=.75 radii=0:.002:.5 F_43"
    )
    print("horizontal_radius_guide=none")
    print("periodic_vertical_guides=refined OLS continuous periods")
    for case in bundle.cases:
        guide = case["refined_continuous_period"]
        print(
            f"{case['id']}: samples={case['analysis_n_samples']} "
            f"h={case['global_curve_bound']:.15g} guide="
            f"{'none' if guide is None else f'{guide:.15g}'}"
        )
    print(f"kernel_sha256={env['julia_kernel_sha256']}")
    print(f"shared_validator_sha256={env['shared_v2_validation_helper_sha256']}")
    if python_only:
        print("Python-only v3 check complete; Julia checks explicitly skipped.")
        return
    check_root = SAFE_OUTPUT_ROOT / ".compass-refined-v3-C0p75-check-only"
    for case in bundle.cases:
        subprocess.run(
            julia_command(
                case, configuration.protocol, check_root / case["id"],
                env["julia_executable"], True,
            ),
            cwd=WORKSPACE_ROOT,
            check=True,
        )
    run_preflights(bundle, configuration.protocol, env["julia_executable"])
    print("CHECK ONLY complete: no signatures or plan were written.")


def load_plan(plan_path: Path, julia_bin: str) -> tuple[Path, dict[str, Any]]:
    plan_path = resolve_existing_file(plan_path, "Compass v3 plan")
    root = guard_output_root(plan_path.parent, False)
    if plan_path != root / "plan.json":
        raise ValueError("plan must be canonical plan.json")
    if (root / "MATERIALIZATION_INCOMPLETE").exists():
        raise ValueError("refusing incomplete materialization")
    sidecar = resolve_existing_file(root / "plan.sha256", "plan hash sidecar")
    if sidecar.read_text(encoding="utf-8") != f"{sha256(plan_path)}  plan.json\n":
        raise ValueError("plan hash sidecar mismatch")
    plan = load_json(plan_path)
    expected_keys = {
        "schema_version", "status", "created_utc", "analysis_id",
        "figure_filename", "summary_directory", "output_root", "orchestrator",
        "orchestrator_sha256", "shared_validation_helper",
        "shared_validation_helper_sha256", "cases_path", "cases_sha256",
        "protocol_path", "protocol_sha256", "protocol", "bundle_manifest",
        "bundle_manifest_sha256", "bundle_id", "bundle_generator",
        "reference_v1_bundle_manifest", "reference_v1_bundle_manifest_sha256",
        "recurrence_policy",
        "scientific_scope", "environment", "kernel_internal_protocol_label",
        "preflight", "commands", "commands_sha256", "jobs",
    }
    if set(plan) != expected_keys or int(plan["schema_version"]) != 3:
        raise ValueError("unexpected Compass v3 plan schema")
    if plan["status"] != "materialized_not_executed" or plan["output_root"] != str(root):
        raise ValueError("unexpected Compass v3 plan status/root")
    if Path(plan["orchestrator"]).resolve() != SCRIPT_PATH or plan["orchestrator_sha256"] != sha256(SCRIPT_PATH):
        raise ValueError("Compass v3 orchestrator changed")
    if Path(plan["shared_validation_helper"]).resolve() != SHARED_V2_HELPER or plan["shared_validation_helper_sha256"] != sha256(SHARED_V2_HELPER):
        raise ValueError("shared validation helper changed")
    cases_path = resolve_existing_file(Path(plan["cases_path"]), "planned v3 cases")
    protocol_path = resolve_existing_file(Path(plan["protocol_path"]), "planned v3 protocol")
    bundle_path = resolve_existing_file(Path(plan["bundle_manifest"]), "planned refined bundle")
    for path, key in (
        (cases_path, "cases_sha256"),
        (protocol_path, "protocol_sha256"),
        (bundle_path, "bundle_manifest_sha256"),
    ):
        if sha256(path) != plan[key]:
            raise ValueError(f"planned input changed: {path}")
    configuration = load_configuration(cases_path, bundle_path)
    if configuration.protocol_path != protocol_path or configuration.protocol != plan["protocol"]:
        raise ValueError("planned protocol binding changed")
    bundle = validate_bundle(configuration)
    if bundle.manifest["bundle_id"] != plan["bundle_id"]:
        raise ValueError("planned bundle id changed")
    if bundle.manifest["generator"] != plan["bundle_generator"]:
        raise ValueError("planned refined-bundle generator changed")
    reference_v1_path = resolve_existing_file(
        Path(plan["reference_v1_bundle_manifest"]), "planned reference v1 bundle"
    )
    if (
        reference_v1_path != REFERENCE_V1_MANIFEST.resolve()
        or sha256(reference_v1_path) != plan["reference_v1_bundle_manifest_sha256"]
    ):
        raise ValueError("planned reference v1 bundle changed")
    if bundle.manifest["recurrence_policy"] != plan["recurrence_policy"] or bundle.manifest["scientific_scope"] != plan["scientific_scope"]:
        raise ValueError("planned bundle scientific contract changed")
    current_environment = environment(julia_bin)
    if current_environment != plan["environment"]:
        raise ValueError("Python/Julia/CyclingSignatures environment changed")
    commands_path = resolve_existing_file(root / "commands.sh", "planned commands")
    jobs = build_jobs(
        bundle, configuration.protocol, root, current_environment["julia_executable"]
    )
    if jobs != plan["jobs"]:
        raise ValueError("plan jobs cannot be exactly reconstructed")
    command_text = commands_text(jobs)
    if (
        plan["commands"] != str(commands_path)
        or commands_path.read_text(encoding="utf-8") != command_text
        or sha256(commands_path) != plan["commands_sha256"]
    ):
        raise ValueError("commands cannot be exactly reconstructed")
    expected_preflight = {
        item["case_id"]: item for item in plan["preflight"]
    }
    if set(expected_preflight) != {case["id"] for case in bundle.cases}:
        raise ValueError("plan preflight inventory changed")
    for case in bundle.cases:
        record = expected_preflight[case["id"]]
        if (
            int(record["beta1_Y"]) != 1
            or not math.isclose(
                float(record["global_curve_bound"]), case["global_curve_bound"],
                rel_tol=0.0, abs_tol=3e-10,
            )
            or record["curve_bound_target_pass"] is not True
        ):
            raise ValueError(f"{case['id']}: planned preflight changed")
    return root, plan


def result_binding_path(job: dict[str, Any]) -> Path:
    return Path(job["output_dir"]) / f"{job['id']}_v3_result.json"


def validate_raw_result(job: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    binding = shared.validate_raw_result(job, plan)
    births_path = shared.result_paths(job)["births"]
    zero_birth_trials = 0
    minimum_positive = math.inf
    with births_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            births = [float(value) for value in row["births"].split(";") if value]
            if any(abs(value) <= RECURRENCE_TOLERANCE for value in births):
                zero_birth_trials += 1
            minimum_positive = min(
                minimum_positive,
                min((value for value in births if value > RECURRENCE_TOLERANCE), default=math.inf),
            )
    if plan["protocol"]["require_no_exact_full_period_recurrence"] and zero_birth_trials:
        raise ValueError(
            f"{job['id']}: {zero_birth_trials} trials contain zero/near-zero births"
        )
    binding.update({
        "schema_version": 3,
        "analysis_id": plan["analysis_id"],
        "refined_continuous_period": job["refined_continuous_period"],
        "recurrence_preflight": job["recurrence_preflight"],
        "curve_bound_target_pass": job["curve_bound_target_pass"],
        "zero_or_near_zero_birth_trials": zero_birth_trials,
        "minimum_positive_birth": None if math.isinf(minimum_positive) else minimum_positive,
    })
    return binding


def validate_binding(job: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    path = resolve_existing_file(result_binding_path(job), f"{job['id']} v3 result binding")
    stored = load_json(path)
    current = validate_raw_result(job, plan)
    created = stored.get("created_utc")
    current["created_utc"] = created
    if stored != current:
        raise ValueError(f"{job['id']}: v3 result binding cannot be reconstructed")
    return stored


def execute(plan_path: Path, julia_bin: str, selected_cases: set[str]) -> None:
    _, plan = load_plan(plan_path, julia_bin)
    known = {job["id"] for job in plan["jobs"]}
    if selected_cases - known:
        raise ValueError(f"unknown cases: {sorted(selected_cases-known)}")
    jobs = [
        job for job in plan["jobs"]
        if not selected_cases or job["id"] in selected_cases
    ]
    for job in jobs:
        output_dir = Path(job["output_dir"])
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing nonempty result directory: {output_dir}")
        log_path = Path(job["log"])
        if log_path.exists() or log_path.is_symlink():
            raise FileExistsError(f"refusing existing log: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Running Compass refined v3 kernel for {job['id']}", flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            subprocess.run(
                job["command"], cwd=WORKSPACE_ROOT, check=True, stdout=log,
                stderr=subprocess.STDOUT, text=True,
            )
        binding = validate_raw_result(job, plan)
        binding["created_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_exclusive(
            result_binding_path(job),
            json.dumps(binding, indent=2, sort_keys=True) + "\n",
        )
        print(f"Validated {job['id']}")


def validate_results(plan_path: Path, julia_bin: str) -> None:
    _, plan = load_plan(plan_path, julia_bin)
    starts_hash: str | None = None
    for job in plan["jobs"]:
        binding = validate_binding(job, plan)
        current_starts = binding["raw_results"]["starts"]["sha256"]
        if starts_hash is None:
            starts_hash = current_starts
        elif current_starts != starts_hash:
            raise ValueError("paired-start tables are not byte-identical")
        print(
            f"{job['id']}: validated beta1={binding['beta1_Y']} "
            f"h={binding['global_curve_bound']:.15g} "
            f"zero_birth_trials={binding['zero_or_near_zero_birth_trials']}"
        )
    print(f"paired_segment_starts_sha256={starts_hash}")


def summarize(plan_path: Path, julia_bin: str) -> Path:
    root, plan = load_plan(plan_path, julia_bin)
    output = root / str(plan["summary_directory"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing summary directory: {output}")
    rows: list[dict[str, Any]] = []
    bindings: dict[str, str] = {}
    starts_hash: str | None = None
    radii = np.linspace(
        plan["protocol"]["r_min"], plan["protocol"]["r_max"],
        plan["protocol"]["r_subdivisions"],
    )
    for job in plan["jobs"]:
        binding = validate_binding(job, plan)
        binding_path = result_binding_path(job)
        bindings[job["id"]] = sha256(binding_path)
        current_starts = binding["raw_results"]["starts"]["sha256"]
        if starts_hash is None:
            starts_hash = current_starts
        elif current_starts != starts_hash:
            raise ValueError("paired-start tables are not byte-identical")
        first_resolved_index = int(
            np.searchsorted(radii, float(job["global_curve_bound"]), side="right")
        )
        rows.append({
            "case_id": job["id"],
            "phi_deg": job["phi_deg"],
            "q": job["q"],
            "refined_continuous_period": job["refined_continuous_period"],
            "beta1_Y": binding["beta1_Y"],
            "global_curve_bound": binding["global_curve_bound"],
            "first_strict_curve_resolved_radius": (
                None if first_resolved_index >= len(radii)
                else float(radii[first_resolved_index])
            ),
            "zero_or_near_zero_birth_trials": binding["zero_or_near_zero_birth_trials"],
            "minimum_positive_birth": binding["minimum_positive_birth"],
            "result_binding_sha256": bindings[job["id"]],
        })
    output.mkdir(parents=True)
    try:
        csv_path = output / "case_summary.csv"
        fields = list(rows[0])
        text_lines = [",".join(fields)]
        for row in rows:
            text_lines.append(",".join(
                "" if row[field] is None else str(row[field]) for field in fields
            ))
        atomic_write_exclusive(csv_path, "\n".join(text_lines) + "\n")
        summary = {
            "schema_version": 1,
            "status": "validated_compact_summary",
            "analysis_id": plan["analysis_id"],
            "plan": str(plan_path.resolve()),
            "plan_sha256": sha256(plan_path.resolve()),
            "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
            "statistic": plan["protocol"]["statistic"],
            "paired_segment_starts_sha256": starts_hash,
            "case_summary": str(csv_path),
            "case_summary_sha256": sha256(csv_path),
            "result_bindings": bindings,
        }
        summary_path = output / "summary.json"
        atomic_write_exclusive(
            summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    except BaseException:
        marker = output / "SUMMARY_INCOMPLETE"
        if not marker.exists():
            atomic_write_exclusive(marker, "Compass v3 summary did not complete\n")
        raise
    print(f"Wrote compact validated summary: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("check", "materialize", "execute", "validate", "summarize"),
        nargs="?", default="check",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--julia-bin", default="julia")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action in {"execute", "validate", "summarize"}:
        if args.plan is None:
            raise ValueError(f"{args.action} requires --plan")
        if args.action == "execute":
            execute(args.plan, args.julia_bin, set(args.case))
        elif args.action == "validate":
            validate_results(args.plan, args.julia_bin)
        else:
            summarize(args.plan, args.julia_bin)
        return
    configuration = load_configuration(args.cases, args.bundle_manifest)
    bundle = validate_bundle(configuration)
    if args.action == "check":
        check_all(configuration, bundle, args.julia_bin, args.python_only)
        return
    output_root = args.output_root or suggested_output_root(configuration)
    materialize(configuration, bundle, output_root, args.julia_bin)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
