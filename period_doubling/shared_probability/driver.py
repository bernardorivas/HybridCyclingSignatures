#!/usr/bin/env python3
"""Shared coauthor-protocol cycling-probability analysis.

One protocol file controls sampling, tangent normalization, comparison cover,
metric, coefficient field, filtration grid, probability statistic, and plot
layout.  Case files provide only trajectory sources and display projections.

The default action is read-only validation.  Materialization, signature
execution, and rendering are separate explicit actions.  Full runs write only
below ``code/experiments_planned/outputs/shared_coauthor_protocol``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
SAFE_OUTPUT_ROOT = (
    CODE_ROOT
    / "experiments_planned"
    / "outputs"
    / "shared_coauthor_protocol"
)
JULIA_RUNNER = CODE_ROOT / "period_doubling" / "julia" / "run_shared_probability.jl"
JULIA_PROJECT = CODE_ROOT / "period_doubling" / "julia" / "Project.toml"
JULIA_MANIFEST = CODE_ROOT / "period_doubling" / "julia" / "Manifest.toml"
CYCLING_REPO = CODE_ROOT / "CyclingSignatures.jl"
PROTECTED_ROOTS = (
    CODE_ROOT / "chyll_v2" / "cycling_signature" / "data",
    CODE_ROOT / "period_doubling" / "data",
    CODE_ROOT / "period_doubling" / "data_fine",
    CODE_ROOT / "chyll_v2" / "runs",
    CODE_ROOT / "runs",
    CODE_ROOT / "chyll_v2" / "cycling_signature" / "handoffs",
)


@dataclass(frozen=True)
class AnalysisBundle:
    cases_path: Path
    protocol_path: Path
    cases_document: dict[str, Any]
    protocol: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def environment_snapshot(julia_bin: str = "julia") -> dict[str, Any]:
    nested_status = command_text(
        ["git", "status", "--porcelain"],
        CYCLING_REPO,
    )
    if nested_status:
        raise ValueError(
            "CyclingSignatures.jl must be clean before freezing a shared run"
        )
    return {
        "julia_version": command_text([julia_bin, "--version"], CODE_ROOT),
        "project": str(JULIA_PROJECT.resolve()),
        "project_sha256": sha256(JULIA_PROJECT),
        "manifest": str(JULIA_MANIFEST.resolve()),
        "manifest_sha256": sha256(JULIA_MANIFEST),
        "cycling_repo": str(CYCLING_REPO.resolve()),
        "cycling_repo_head": command_text(
            ["git", "rev-parse", "--verify", "HEAD"],
            CYCLING_REPO,
        ),
        "cycling_repo_status": "clean",
    }


def validate_environment_snapshot(
    expected: dict[str, Any],
    julia_bin: str = "julia",
) -> None:
    current = environment_snapshot(julia_bin)
    if current != expected:
        raise ValueError("Julia/CyclingSignatures environment changed after planning")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    return all(value % divisor for divisor in range(2, math.isqrt(value) + 1))


def segment_lengths(protocol: dict[str, Any]) -> list[int]:
    spec = protocol["segment_lengths"]
    start = int(spec["start"])
    step = int(spec["step"])
    stop = int(spec["stop"])
    if start < 2 or step <= 0 or stop < start or (stop - start) % step:
        raise ValueError("invalid segment-length range")
    return list(range(start, stop + 1, step))


def reject_symlink_components(path: Path, label: str) -> None:
    probe = path.absolute()
    while True:
        if probe.is_symlink():
            raise ValueError(f"{label} contains a symlink: {probe}")
        parent = probe.parent
        if parent == probe:
            return
        probe = parent


def resolve_code_input(path_value: str) -> Path:
    path = Path(path_value)
    candidate = path if path.is_absolute() else CODE_ROOT / path
    reject_symlink_components(candidate, "input path")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing input file: {resolved}")
    return resolved


def load_bundle(
    cases_path: Path,
    protocol_override: Path | None = None,
) -> AnalysisBundle:
    reject_symlink_components(cases_path, "case document")
    cases_path = cases_path.resolve()
    cases_document = load_json(cases_path)
    if int(cases_document.get("schema_version", -1)) != 1:
        raise ValueError("case document must use schema_version=1")
    protocol_ref = cases_document.get("protocol")
    if not isinstance(protocol_ref, str) or not protocol_ref:
        raise ValueError("case document needs a protocol path")
    protocol_candidate = (
        protocol_override
        if protocol_override is not None
        else cases_path.parent / protocol_ref
    )
    reject_symlink_components(protocol_candidate, "protocol path")
    protocol_path = protocol_candidate.resolve()
    protocol = load_json(protocol_path)
    if int(protocol.get("schema_version", -1)) != 1:
        raise ValueError("protocol must use schema_version=1")
    validate_protocol(protocol)
    validate_cases(cases_document, protocol)
    return AnalysisBundle(cases_path, protocol_path, cases_document, protocol)


def validate_protocol(protocol: dict[str, Any]) -> None:
    lengths = segment_lengths(protocol)
    if lengths != list(range(100, 1201, 20)):
        raise ValueError("coauthor protocol requires lengths 100:20:1200")
    if int(protocol["n_runs"]) != 20:
        raise ValueError("coauthor protocol requires 20 runs per length")
    if protocol["sampling"] != "independent_uniform_with_replacement":
        raise ValueError("unexpected segment-sampling rule")
    if protocol["duration_convention"] != "segment_length_times_effective_dt":
        raise ValueError("unexpected duration convention")
    if protocol["tangent_normalization"] not in {"l2", "linf"}:
        raise ValueError("tangent_normalization must be l2 or linf")
    if not is_prime(int(protocol["field_prime"])):
        raise ValueError("field_prime must be prime")
    positive_keys = (
        "effective_sample_dt",
        "position_boxsize",
        "sphere_box_resolution",
        "metric_c",
        "r_max",
        "r_subdivisions",
    )
    for key in positive_keys:
        value = float(protocol[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"protocol {key} must be positive")
    if int(protocol["sphere_box_resolution"]) != protocol["sphere_box_resolution"]:
        raise ValueError("sphere_box_resolution must be integral")
    if int(protocol["r_subdivisions"]) < 2:
        raise ValueError("r_subdivisions must be at least two")
    if protocol.get("require_sample_radius_below_r_max") is not True:
        raise ValueError("shared protocol must preflight sample_radius < r_max")


def validate_flow_tangent_record(
    case_id: str,
    source: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    expected_kind = "fine_compass_learned_flow_tangent_control"
    if (
        int(provenance.get("schema_version", -1)) != 2
        or provenance.get("kind") != expected_kind
    ):
        raise ValueError(f"{case_id}: unexpected tangent-provenance schema")
    records = [
        record for record in provenance.get("regimes", [])
        if record.get("regime") == case_id
    ]
    if len(records) != 1:
        raise ValueError(f"{case_id}: missing unique tangent-provenance record")
    record = records[0]
    positions = resolve_code_input(source["positions"])
    tangents = resolve_code_input(source["tangents"])
    expected_files = (
        ("source_positions_csv", "source_positions_csv_sha256", positions),
        ("learned_flow_tangents_csv", "learned_flow_tangents_csv_sha256", tangents),
        ("source_archive", "source_archive_sha256", None),
        (
            "source_encoder_jvp_tangents_csv",
            "source_encoder_jvp_tangents_csv_sha256",
            None,
        ),
        ("checkpoint", "checkpoint_sha256", None),
        ("config", "config_sha256", None),
    )
    for path_key, hash_key, expected_path in expected_files:
        bound_path = resolve_code_input(str(record[path_key]))
        if expected_path is not None and bound_path != expected_path:
            raise ValueError(f"{case_id}: provenance binds a different {path_key}")
        if sha256(bound_path) != record[hash_key]:
            raise ValueError(f"{case_id}: provenance hash failed for {path_key}")
    exporter = resolve_code_input(str(provenance["exporter_script"]))
    if sha256(exporter) != provenance["exporter_script_sha256"]:
        raise ValueError(f"{case_id}: tangent exporter hash changed")
    if float(record.get("config_w_v", math.nan)) != 0.0:
        raise ValueError(f"{case_id}: learned-flow control requires config_w_v=0")
    binding = record.get("learned_flow_tangent_binding", {})
    generation = record.get("tangent_generation_inputs", {})
    required_binding = {
        "checkpoint_sha256": record["checkpoint_sha256"],
        "config_sha256": record["config_sha256"],
        "config_w_v": 0.0,
        "source_positions_csv_sha256": record["source_positions_csv_sha256"],
        "exporter_script_sha256": provenance["exporter_script_sha256"],
        "code_repo_head": provenance["code_repo_head"],
        "output_csv_sha256": record["learned_flow_tangents_csv_sha256"],
    }
    for key, value in required_binding.items():
        if binding.get(key) != value:
            raise ValueError(f"{case_id}: inconsistent flow-tangent binding {key}")
    for key in (
        "checkpoint_sha256",
        "config_sha256",
        "config_w_v",
        "source_positions_csv_sha256",
    ):
        if generation.get(key) != required_binding[key]:
            raise ValueError(f"{case_id}: inconsistent tangent input binding {key}")
    for manifest in record.get("window_manifests", []):
        manifest_path = resolve_code_input(str(manifest["path"]))
        if sha256(manifest_path) != manifest["sha256"]:
            raise ValueError(f"{case_id}: frozen-window manifest hash changed")


def validate_cases(document: dict[str, Any], protocol: dict[str, Any]) -> None:
    analysis_id = document.get("analysis_id")
    if not isinstance(analysis_id, str) or not analysis_id.isidentifier():
        raise ValueError("case document needs a filename-safe analysis_id")
    figure_filename = document.get("figure_filename")
    if (
        not isinstance(figure_filename, str)
        or Path(figure_filename).name != figure_filename
        or figure_filename in {".", ".."}
        or not figure_filename.endswith(".pdf")
    ):
        raise ValueError("case document needs a PDF basename")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("case document needs a nonempty cases list")
    seen: set[str] = set()
    effective_dt = float(protocol["effective_sample_dt"])
    provenance_cache: dict[Path, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.isidentifier():
            raise ValueError(f"invalid case id: {case_id!r}")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        sample_dt = float(case["sample_dt"])
        stride = int(case["stride"])
        if sample_dt <= 0 or stride <= 0:
            raise ValueError(f"{case_id}: invalid sample_dt/stride")
        if not math.isclose(sample_dt * stride, effective_dt, abs_tol=1e-12):
            raise ValueError(
                f"{case_id}: effective dt {sample_dt * stride:g} does not "
                f"match shared protocol {effective_dt:g}"
            )
        expected_beta1 = case.get("expected_beta1")
        if expected_beta1 is not None and (
            not isinstance(expected_beta1, int) or expected_beta1 < 0
        ):
            raise ValueError(f"{case_id}: expected_beta1 must be nonnegative")
        source = case.get("source", {})
        kind = source.get("kind")
        if kind == "existing_lift":
            resolve_code_input(source["positions"])
            resolve_code_input(source["tangents"])
            provenance = source.get("tangent_provenance")
            if provenance is not None:
                provenance_path = resolve_code_input(provenance)
                provenance_document = provenance_cache.setdefault(
                    provenance_path,
                    load_json(provenance_path),
                )
                if source.get("source_role") != (
                    "learned_flow_direction_on_frozen_encoded_path"
                ):
                    raise ValueError(f"{case_id}: missing learned-flow source role")
                validate_flow_tangent_record(
                    case_id,
                    source,
                    provenance_document,
                )
        elif kind == "roessler_rk4":
            validate_roessler_source(case_id, source, sample_dt)
        else:
            raise ValueError(f"{case_id}: unknown source kind {kind!r}")
        display = case.get("display", {})
        if display.get("kind") == "compass_npz":
            resolve_code_input(display["path"])
            if int(display["n_impacts"]) <= 0:
                raise ValueError(f"{case_id}: n_impacts must be positive")
        elif display.get("kind") in {"positions_3d", "analysis_positions_3d"}:
            if int(display.get("maximum_points", 0)) < 2:
                raise ValueError(f"{case_id}: maximum_points must be at least two")
        else:
            raise ValueError(f"{case_id}: unsupported display adapter")


def validate_roessler_source(case_id: str, source: dict[str, Any], dt: float) -> None:
    required = (
        "linear_y",
        "z_offset",
        "z_control",
        "initial_state",
        "final_time",
        "discard_steps",
    )
    if any(key not in source for key in required):
        raise ValueError(f"{case_id}: incomplete Rössler source")
    initial = np.asarray(source["initial_state"], dtype=float)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError(f"{case_id}: initial_state must contain three values")
    final_time = float(source["final_time"])
    n_steps_float = final_time / dt
    n_steps = int(round(n_steps_float))
    if not math.isclose(n_steps_float, n_steps, abs_tol=1e-9):
        raise ValueError(f"{case_id}: final_time is not an integer number of steps")
    discard = int(source["discard_steps"])
    if discard < 0 or n_steps - discard + 1 < 1200:
        raise ValueError(f"{case_id}: retained trajectory is too short")


def resolved_new_path(path: Path) -> Path:
    return path.resolve(strict=False)


def guard_output_root(path: Path, *, must_not_exist: bool) -> Path:
    resolved = resolved_new_path(path)
    safe = SAFE_OUTPUT_ROOT.resolve(strict=False)
    if resolved == safe or not resolved.is_relative_to(safe):
        raise ValueError(f"output must be a named child of {safe}")
    for protected in PROTECTED_ROOTS:
        protected_resolved = protected.resolve(strict=False)
        if resolved == protected_resolved or resolved.is_relative_to(protected_resolved):
            raise ValueError(f"refusing protected output: {resolved}")
    if path.is_symlink() or resolved.is_symlink():
        raise ValueError(f"refusing symlinked output: {resolved}")
    if must_not_exist and (resolved.exists() or resolved.is_symlink()):
        raise FileExistsError(f"refusing to overwrite: {resolved}")
    return resolved


def roessler_field(
    state: np.ndarray,
    linear_y: float,
    z_offset: float,
    z_control: float,
) -> np.ndarray:
    x, y, z = state
    return np.asarray(
        (-y - z, x + linear_y * y, z_offset + z * (x - z_control)),
        dtype=float,
    )


def generate_roessler(source: dict[str, Any], dt: float) -> tuple[np.ndarray, np.ndarray]:
    linear_y = float(source["linear_y"])
    z_offset = float(source["z_offset"])
    z_control = float(source["z_control"])
    state = np.asarray(source["initial_state"], dtype=float)
    n_steps = int(round(float(source["final_time"]) / dt))
    discard = int(source["discard_steps"])
    retained = np.empty((n_steps - discard + 1, 3), dtype=float)
    retained_index = 0
    if discard == 0:
        retained[retained_index] = state
        retained_index += 1
    for step in range(1, n_steps + 1):
        k1 = roessler_field(state, linear_y, z_offset, z_control)
        k2 = roessler_field(
            state + 0.5 * dt * k1,
            linear_y,
            z_offset,
            z_control,
        )
        k3 = roessler_field(
            state + 0.5 * dt * k2,
            linear_y,
            z_offset,
            z_control,
        )
        k4 = roessler_field(
            state + dt * k3,
            linear_y,
            z_offset,
            z_control,
        )
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if step >= discard:
            retained[retained_index] = state
            retained_index += 1
    if retained_index != len(retained) or not np.all(np.isfinite(retained)):
        raise RuntimeError("Rössler RK4 materialization failed")
    tangents = np.asarray([
        roessler_field(row, linear_y, z_offset, z_control)
        for row in retained
    ])
    if np.any(np.linalg.norm(tangents, axis=1) == 0):
        raise RuntimeError("Rössler trajectory encountered a zero vector field")
    return retained, tangents


def materialize_case_inputs(case: dict[str, Any], input_dir: Path) -> tuple[Path, Path]:
    source = case["source"]
    if source["kind"] == "existing_lift":
        return (
            resolve_code_input(source["positions"]),
            resolve_code_input(source["tangents"]),
        )
    positions, tangents = generate_roessler(source, float(case["sample_dt"]))
    positions_path = input_dir / f"{case['id']}_positions.csv"
    tangents_path = input_dir / f"{case['id']}_tangents.csv"
    np.savetxt(positions_path, positions, fmt="%.17g")
    np.savetxt(tangents_path, tangents, fmt="%.17g")
    return positions_path.resolve(), tangents_path.resolve()


def julia_command(
    case: dict[str, Any],
    protocol: dict[str, Any],
    positions: Path,
    tangents: Path,
    output_dir: Path,
) -> list[str]:
    lengths = protocol["segment_lengths"]
    length_spec = f"{lengths['start']}:{lengths['step']}:{lengths['stop']}"
    return [
        "julia",
        "--startup-file=no",
        f"--project={CODE_ROOT / 'period_doubling' / 'julia'}",
        str(JULIA_RUNNER),
        "--positions", str(positions),
        "--tangents", str(tangents),
        "--stride", str(case["stride"]),
        "--sample-dt", str(case["sample_dt"]),
        "--segment-lengths", length_spec,
        "--n-runs", str(protocol["n_runs"]),
        "--seed", str(protocol["seed"]),
        "--tangent-normalization", str(protocol["tangent_normalization"]),
        "--boxsize", str(protocol["position_boxsize"]),
        "--sb-radius", str(protocol["sphere_box_resolution"]),
        "--metric-c", str(protocol["metric_c"]),
        "--r-max", str(protocol["r_max"]),
        "--r-subdivisions", str(protocol["r_subdivisions"]),
        "--field-prime", str(protocol["field_prime"]),
        "--require-sample-radius-below-r-max", "true",
        "--out-dir", str(output_dir),
        "--out-prefix", str(case["id"]),
    ]


def build_job_record(
    case: dict[str, Any],
    protocol: dict[str, Any],
    positions: Path,
    tangents: Path,
    root: Path,
) -> dict[str, Any]:
    out_dir = root / "signatures" / case["id"]
    display = dict(case["display"])
    if display["kind"] == "compass_npz":
        display["path"] = str(resolve_code_input(display["path"]))
        display["sha256"] = sha256(Path(display["path"]))
    source_provenance = None
    if "tangent_provenance" in case["source"]:
        provenance_path = resolve_code_input(case["source"]["tangent_provenance"])
        source_provenance = {
            "path": str(provenance_path),
            "sha256": sha256(provenance_path),
        }
    return {
        "case_id": case["id"],
        "title": case["title"],
        "sample_dt": float(case["sample_dt"]),
        "stride": int(case["stride"]),
        "effective_dt": float(case["sample_dt"]) * int(case["stride"]),
        "expected_beta1": case.get("expected_beta1"),
        "source": case["source"],
        "positions": str(positions.resolve()),
        "positions_sha256": sha256(positions),
        "tangents": str(tangents.resolve()),
        "tangents_sha256": sha256(tangents),
        "tangent_provenance": source_provenance,
        "display": display,
        "output_dir": str(out_dir),
        "out_prefix": case["id"],
        "command": julia_command(
            case,
            protocol,
            positions.resolve(),
            tangents.resolve(),
            out_dir,
        ),
    }


def materialize(bundle: AnalysisBundle, output_root: Path) -> Path:
    root = guard_output_root(output_root, must_not_exist=True)
    all_inputs: list[tuple[dict[str, Any], Path, Path]] = []
    generated = any(
        case["source"]["kind"] == "roessler_rk4"
        for case in bundle.cases_document["cases"]
    )
    root.mkdir(parents=True)
    input_dir = root / "inputs"
    if generated:
        input_dir.mkdir()
    try:
        for case in bundle.cases_document["cases"]:
            positions, tangents = materialize_case_inputs(case, input_dir)
            all_inputs.append((case, positions, tangents))
        jobs = [
            build_job_record(case, bundle.protocol, positions, tangents, root)
            for case, positions, tangents in all_inputs
        ]
        plan = {
            "schema_version": 1,
            "status": "materialized_not_executed",
            "analysis_id": bundle.cases_document["analysis_id"],
            "figure_filename": bundle.cases_document["figure_filename"],
            "driver": str(Path(__file__).resolve()),
            "driver_sha256": sha256(Path(__file__).resolve()),
            "julia_runner": str(JULIA_RUNNER.resolve()),
            "julia_runner_sha256": sha256(JULIA_RUNNER.resolve()),
            "protocol_path": str(bundle.protocol_path),
            "protocol_sha256": sha256(bundle.protocol_path),
            "cases_path": str(bundle.cases_path),
            "cases_sha256": sha256(bundle.cases_path),
            "protocol": bundle.protocol,
            "environment": environment_snapshot(),
            "jobs": jobs,
        }
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        commands_path = root / "commands.sh"
        commands_path.write_text(
            "#!/bin/sh\nset -eu\n" +
            "\n".join(shlex.join(job["command"]) for job in jobs) + "\n",
            encoding="utf-8",
        )
        commands_path.chmod(0o755)
    except BaseException:
        # Preserve a partially materialized directory for forensic inspection.
        (root / "MATERIALIZATION_INCOMPLETE").touch(exist_ok=True)
        raise
    print(f"Materialized {len(jobs)} jobs at {root}")
    print("No cycling signatures were executed.")
    return plan_path


def load_plan(plan_path: Path) -> tuple[Path, dict[str, Any]]:
    plan_path = plan_path.resolve()
    root = guard_output_root(plan_path.parent, must_not_exist=False)
    if plan_path != root / "plan.json" or not plan_path.is_file():
        raise ValueError("plan must be the canonical plan.json below the safe root")
    plan = load_json(plan_path)
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError("unsupported plan schema")
    if plan.get("status") != "materialized_not_executed":
        raise ValueError("plan status is not its immutable materialization state")
    protocol_path = Path(plan["protocol_path"])
    cases_path = Path(plan["cases_path"])
    driver_path = Path(plan["driver"])
    runner_path = Path(plan["julia_runner"])
    if driver_path.resolve() != Path(__file__).resolve():
        raise ValueError("plan names a noncanonical Python driver")
    if runner_path.resolve() != JULIA_RUNNER.resolve():
        raise ValueError("plan names a noncanonical Julia runner")
    if sha256(protocol_path) != plan["protocol_sha256"]:
        raise ValueError("protocol changed after materialization")
    if sha256(cases_path) != plan["cases_sha256"]:
        raise ValueError("case document changed after materialization")
    if sha256(driver_path) != plan["driver_sha256"]:
        raise ValueError("Python driver changed after materialization")
    if sha256(runner_path) != plan["julia_runner_sha256"]:
        raise ValueError("Julia runner changed after materialization")
    bundle = load_bundle(cases_path, protocol_path)
    if plan["protocol"] != bundle.protocol:
        raise ValueError("embedded protocol differs from its source file")
    if plan.get("analysis_id") != bundle.cases_document["analysis_id"]:
        raise ValueError("plan analysis_id differs from the case document")
    if plan.get("figure_filename") != bundle.cases_document["figure_filename"]:
        raise ValueError("plan figure filename differs from the case document")
    validate_environment_snapshot(plan["environment"])
    expected_jobs = []
    for case in bundle.cases_document["cases"]:
        if case["source"]["kind"] == "existing_lift":
            positions = resolve_code_input(case["source"]["positions"])
            tangents = resolve_code_input(case["source"]["tangents"])
        else:
            positions = root / "inputs" / f"{case['id']}_positions.csv"
            tangents = root / "inputs" / f"{case['id']}_tangents.csv"
            if not positions.is_file() or not tangents.is_file():
                raise ValueError(f"missing materialized inputs for {case['id']}")
        expected_jobs.append(
            build_job_record(case, bundle.protocol, positions, tangents, root)
        )
    if plan.get("jobs") != expected_jobs:
        raise ValueError("plan jobs differ from the hashed case document")
    for job in expected_jobs:
        positions = Path(job["positions"])
        tangents = Path(job["tangents"])
        if sha256(positions) != job["positions_sha256"]:
            raise ValueError(f"positions changed for {job['case_id']}")
        if sha256(tangents) != job["tangents_sha256"]:
            raise ValueError(f"tangents changed for {job['case_id']}")
        provenance = job.get("tangent_provenance")
        if provenance is not None:
            provenance_path = Path(provenance["path"])
            if sha256(provenance_path) != provenance["sha256"]:
                raise ValueError(
                    f"tangent provenance changed for {job['case_id']}"
                )
        display = job["display"]
        if display["kind"] == "compass_npz":
            display_path = Path(display["path"])
            if sha256(display_path) != display["sha256"]:
                raise ValueError(f"display source changed for {job['case_id']}")
    return root, plan


def execute(plan_path: Path, julia_bin: str, selected_cases: set[str]) -> None:
    _, plan = load_plan(plan_path)
    validate_environment_snapshot(plan["environment"], julia_bin)
    known = {job["case_id"] for job in plan["jobs"]}
    unknown = selected_cases - known
    if unknown:
        raise ValueError(f"unknown selected cases: {sorted(unknown)}")
    jobs = [
        job for job in plan["jobs"]
        if not selected_cases or job["case_id"] in selected_cases
    ]
    for job in jobs:
        command = list(job["command"])
        command[0] = julia_bin
        print(f"Running shared protocol for {job['case_id']}")
        subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)


def parse_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", maxsplit=1)
            values[key.strip()] = value.strip()
    return values


def expect_metadata(
    metadata: dict[str, str],
    key: str,
    expected: str | int | float,
    case_id: str,
) -> None:
    if key not in metadata:
        raise ValueError(f"{case_id}: result metadata is missing {key}")
    actual = metadata[key]
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{case_id}: metadata {key}={actual}, expected {expected}"
            )
    elif actual != str(expected):
        raise ValueError(
            f"{case_id}: metadata {key}={actual!r}, expected {expected!r}"
        )


def validate_result_metadata(
    metadata: dict[str, str],
    job: dict[str, Any],
    protocol: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    case_id = job["case_id"]
    lengths = segment_lengths(protocol)
    expected = {
        "positions": str(Path(job["positions"]).resolve()),
        "positions_sha256": job["positions_sha256"],
        "tangents": str(Path(job["tangents"]).resolve()),
        "tangents_sha256": job["tangents_sha256"],
        "driver": str(Path(plan["julia_runner"]).resolve()),
        "driver_sha256": plan["julia_runner_sha256"],
        "stride": int(job["stride"]),
        "raw_sample_dt": float(job["sample_dt"]),
        "effective_sample_dt": float(job["effective_dt"]),
        "duration_convention": "segment_length_times_effective_sample_dt",
        "segment_lengths": ",".join(str(value) for value in lengths),
        "n_runs": int(protocol["n_runs"]),
        "seed": int(protocol["seed"]),
        "resample_segment_start": "true",
        "sampling_with_replacement": "true",
        "tangent_normalization": protocol["tangent_normalization"],
        "boxsize": float(protocol["position_boxsize"]),
        "sb_radius": int(protocol["sphere_box_resolution"]),
        "metric_C": float(protocol["metric_c"]),
        "r_max": float(protocol["r_max"]),
        "r_subdivisions": int(protocol["r_subdivisions"]),
        "field_prime": int(protocol["field_prime"]),
        "filtration_threshold": "closed_leq",
        "require_sample_radius_below_r_max": "true",
    }
    for key, value in expected.items():
        expect_metadata(metadata, key, value, case_id)
    expected_beta1 = job.get("expected_beta1")
    if expected_beta1 is not None:
        expect_metadata(metadata, "beta1_Y", int(expected_beta1), case_id)
    post_min = float(metadata["post_normalization_min"])
    post_max = float(metadata["post_normalization_max"])
    if not (
        math.isclose(post_min, 1.0, rel_tol=1e-10, abs_tol=1e-12)
        and math.isclose(post_max, 1.0, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise ValueError(f"{case_id}: tangents were not normalized as planned")


def load_probability(
    job: dict[str, Any], protocol: dict[str, Any], plan: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    output_dir = Path(job["output_dir"])
    prefix = job["out_prefix"]
    births_path = output_dir / f"{prefix}_births.csv"
    starts_path = output_dir / f"{prefix}_segment_starts.csv"
    rank0_path = output_dir / f"{prefix}_rank0_heatmap.csv"
    metadata_path = output_dir / f"{prefix}_metadata.txt"
    result_paths = (births_path, starts_path, rank0_path, metadata_path)
    if any(path.is_symlink() or not path.is_file() for path in result_paths):
        raise ValueError(f"incomplete or symlinked result for {job['case_id']}")
    metadata = parse_metadata(metadata_path)
    validate_result_metadata(metadata, job, protocol, plan)
    hash_keys = {
        births_path: "births_sha256",
        starts_path: "segment_starts_sha256",
        rank0_path: "rank0_heatmap_sha256",
    }
    for path, key in hash_keys.items():
        expect_metadata(metadata, key, sha256(path), job["case_id"])

    lengths = np.asarray(segment_lengths(protocol), dtype=int)
    n_runs = int(protocol["n_runs"])
    expected_keys = {
        (int(length), run_index)
        for length in lengths
        for run_index in range(1, n_runs + 1)
    }
    first_birth_by_key: dict[tuple[int, int], float] = {}
    window_by_key: dict[tuple[int, int], tuple[float, int, int]] = {}
    with births_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_header = (
            "segment_length", "segment_duration", "run_index", "start_index",
            "end_index", "rank", "births",
        )
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"unexpected births schema for {job['case_id']}")
        for row in reader:
            length = int(row["segment_length"])
            run_index = int(row["run_index"])
            key = (length, run_index)
            if key in first_birth_by_key or key not in expected_keys:
                raise ValueError(f"unexpected or duplicate trial {key}")
            births = [float(value) for value in row["births"].split(";") if value]
            if (
                any(not math.isfinite(value) or value < 0 for value in births)
                or births != sorted(births)
                or len(births) != int(row["rank"])
            ):
                raise ValueError(f"invalid birth vector for {job['case_id']} {key}")
            duration = float(row["segment_duration"])
            expected_duration = length * float(job["effective_dt"])
            if not math.isclose(
                duration, expected_duration, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"duration mismatch for {job['case_id']} {key}")
            start = int(row["start_index"])
            end = int(row["end_index"])
            if start < 1 or end != start + length - 1:
                raise ValueError(f"invalid indices for {job['case_id']} {key}")
            first_birth_by_key[key] = min(births, default=np.inf)
            window_by_key[key] = (duration, start, end)
    if set(first_birth_by_key) != expected_keys:
        raise ValueError(f"missing trials for {job['case_id']}")

    seen_starts: set[tuple[int, int]] = set()
    with starts_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_header = (
            "segment_length", "segment_duration", "run_index", "start_index",
            "end_index",
        )
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"unexpected starts schema for {job['case_id']}")
        for row in reader:
            key = (int(row["segment_length"]), int(row["run_index"]))
            actual = (
                float(row["segment_duration"]),
                int(row["start_index"]),
                int(row["end_index"]),
            )
            if key in seen_starts or window_by_key.get(key) != actual:
                raise ValueError(f"start table mismatch for {job['case_id']} {key}")
            seen_starts.add(key)
    if seen_starts != expected_keys:
        raise ValueError(f"missing start rows for {job['case_id']}")

    radii = np.linspace(0.0, float(protocol["r_max"]),
                        int(protocol["r_subdivisions"]))
    probability = np.empty((len(radii), len(lengths)), dtype=float)
    for column, length in enumerate(lengths):
        first_births = np.asarray([
            first_birth_by_key[(int(length), run_index)]
            for run_index in range(1, n_runs + 1)
        ])
        probability[:, column] = np.mean(
            first_births[np.newaxis, :] <= radii[:, np.newaxis], axis=1
        )
    with rank0_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["radius", *[str(value) for value in lengths]]:
            raise ValueError(f"unexpected rank-zero schema for {job['case_id']}")
        rows = list(reader)
    if len(rows) != len(radii):
        raise ValueError(f"rank-zero radius count mismatch for {job['case_id']}")
    for row_index, (row, radius) in enumerate(zip(rows, radii)):
        if len(row) != len(lengths) + 1 or not math.isclose(
            float(row[0]), radius, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"rank-zero radius mismatch for {job['case_id']}")
        rank0 = np.asarray([int(value) for value in row[1:]], dtype=int)
        expected_rank0 = np.rint(n_runs * (1.0 - probability[row_index])).astype(int)
        if not np.array_equal(rank0, expected_rank0):
            raise ValueError(f"rank-zero counts disagree for {job['case_id']}")

    durations = lengths * float(job["effective_dt"])
    sample_radius = float(metadata["sample_radius"])
    if not math.isfinite(sample_radius) or sample_radius < 0:
        raise ValueError(f"invalid sample radius for {job['case_id']}")
    if sample_radius > float(protocol["r_max"]) + 1e-12:
        raise ValueError(
            f"{job['case_id']}: sample radius {sample_radius:g} lies above "
            f"the plotted r_max={protocol['r_max']}; use a versioned wider "
            "protocol rather than hiding the guide"
        )
    return durations, radii, probability, metadata


def centers_to_edges(values: np.ndarray, lower: float | None = None) -> np.ndarray:
    if values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("plot centers must increase")
    midpoints = 0.5 * (values[:-1] + values[1:])
    edges = np.r_[values[0] - (midpoints[0] - values[0]), midpoints,
                  values[-1] + (values[-1] - midpoints[-1])]
    if lower is not None:
        edges[0] = max(lower, edges[0])
    return edges


def decimate_rows(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return values[indices]


def plot_case_trajectory(axis: Any, job: dict[str, Any]) -> None:
    display = job["display"]
    if display["kind"] in {"positions_3d", "analysis_positions_3d"}:
        positions = np.loadtxt(job["positions"], dtype=float)
        positions = positions[:: int(job["stride"])]
        positions = decimate_rows(positions, int(display["maximum_points"]))
        axis.plot(positions[:, 0], positions[:, 1], positions[:, 2], linewidth=0.8)
        labels = ("x", "y", "z") if display["kind"] == "positions_3d" else (
            r"$z_1$", r"$z_2$", r"$z_3$",
        )
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.set_zlabel(labels[2])
        return
    with np.load(display["path"], allow_pickle=False) as raw:
        t = np.asarray(raw["t"], dtype=float)
        x = np.asarray(raw["x"], dtype=float)
        impact_times = np.asarray(raw["impact_times"], dtype=float)
        jump_minus = np.asarray(raw["jump_minus"], dtype=float)
        jump_plus = np.asarray(raw["jump_plus"], dtype=float)
    n_impacts = int(display["n_impacts"])
    first = len(impact_times) - n_impacts - 1
    if first < 0:
        raise ValueError(f"not enough impacts for {job['case_id']} display")
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_impacts))
    for offset, color in enumerate(colors):
        left = first + offset
        right = left + 1
        interior = (t > impact_times[left]) & (t < impact_times[right])
        arc = np.vstack([jump_plus[left], x[interior], jump_minus[right]])
        axis.plot(arc[:, 0], arc[:, 2], color=color, linewidth=1.0)
    axis.set_xlabel(r"$\theta_{\rm ns}$")
    axis.set_ylabel(r"$\dot\theta_{\rm ns}$")


def render(plan_path: Path) -> Path:
    root, plan = load_plan(plan_path)
    protocol = plan["protocol"]
    figure_path = root / plan["figure_filename"]
    if figure_path.exists() or figure_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {figure_path}")
    loaded = [load_probability(job, protocol, plan) for job in plan["jobs"]]
    reference_duration = loaded[0][0]
    reference_radii = loaded[0][1]
    if not all(np.array_equal(reference_duration, item[0]) for item in loaded):
        raise ValueError("cases do not share the coauthor duration grid")
    if not all(np.array_equal(reference_radii, item[1]) for item in loaded):
        raise ValueError("cases do not share the coauthor radius grid")
    n_cases = len(plan["jobs"])
    all_3d = all(
        job["display"]["kind"] in {"positions_3d", "analysis_positions_3d"}
        for job in plan["jobs"]
    )
    figure = plt.figure(figsize=(3.2 * n_cases + 0.7, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        n_cases + 1,
        width_ratios=[1] * n_cases + [0.05],
        height_ratios=[1.0, 1.05],
        hspace=0.12,
    )
    duration_edges = centers_to_edges(reference_duration, lower=0.0)
    radius_edges = centers_to_edges(reference_radii, lower=0.0)
    image = None
    for column, (job, item) in enumerate(zip(plan["jobs"], loaded)):
        if all_3d:
            top = figure.add_subplot(grid[0, column], projection="3d")
        else:
            top = figure.add_subplot(grid[0, column])
        plot_case_trajectory(top, job)
        top.set_title(job["title"], fontsize=9)
        bottom = figure.add_subplot(grid[1, column])
        image = bottom.pcolormesh(
            duration_edges, radius_edges, item[2], cmap="viridis",
            vmin=0.0, vmax=1.0, shading="flat", rasterized=True,
        )
        guide = protocol.get("horizontal_radius_guide")
        if guide == "sample_radius":
            radius = float(item[3]["sample_radius"])
            bottom.axhline(radius, color="white", linestyle="--", linewidth=0.9)
        elif isinstance(guide, (int, float)):
            bottom.axhline(float(guide), color="white", linestyle="--", linewidth=0.9)
        bottom.set_title(job["title"], fontsize=9)
        bottom.set_xlabel("segment duration (t)")
        if column == 0:
            bottom.set_ylabel("filtration radius (r)")
        else:
            bottom.set_yticklabels([])
    color_axis = figure.add_subplot(grid[1, -1])
    if image is None:
        raise RuntimeError("no probability panels")
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("probability")
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {figure_path}")
    return figure_path


def print_check(bundle: AnalysisBundle) -> None:
    protocol = bundle.protocol
    lengths = segment_lengths(protocol)
    print(f"protocol={protocol['protocol_id']}")
    print(
        f"field=F_{protocol['field_prime']} lengths={lengths[0]}:"
        f"{lengths[1] - lengths[0]}:{lengths[-1]} n={protocol['n_runs']}"
    )
    print(
        f"cover=({protocol['position_boxsize']},"
        f" {protocol['sphere_box_resolution']}) C={protocol['metric_c']} "
        f"r=[0,{protocol['r_max']}]/{protocol['r_subdivisions']}"
    )
    for case in bundle.cases_document["cases"]:
        role = case["source"].get("source_role", "generated_vector_field")
        provenance = case["source"].get("tangent_provenance", "none")
        print(
            f"case={case['id']} source={case['source']['kind']} "
            f"role={role} "
            f"effective_dt={float(case['sample_dt']) * int(case['stride']):g} "
            f"tangent_provenance={provenance}"
        )
    print("Explicit assumptions:")
    for assumption in protocol["explicit_reproducibility_assumptions"]:
        print(f"- {assumption}")
    print("Check-only: no files written and no signatures executed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, execute, and render one coauthor-protocol cycling-space "
            "probability analysis for either Rössler or Compass trajectories."
        )
    )
    parser.add_argument(
        "action", choices=("check", "materialize", "execute", "render"),
        help="check is read-only; the other actions require explicit invocation",
    )
    parser.add_argument("--cases", type=Path, help="case JSON for check/materialize")
    parser.add_argument(
        "--protocol",
        type=Path,
        help="explicit protocol override for a versioned sensitivity run",
    )
    parser.add_argument("--output-root", type=Path, help="new output root for materialize")
    parser.add_argument("--plan", type=Path, help="materialized plan for execute/render")
    parser.add_argument("--julia-bin", default="julia")
    parser.add_argument("--case", action="append", default=[], dest="selected_cases")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action in {"check", "materialize"}:
        if args.cases is None:
            raise ValueError("--cases is required")
        bundle = load_bundle(args.cases, args.protocol)
        if args.action == "check":
            print_check(bundle)
            return
        if args.output_root is None:
            raise ValueError("--output-root is required for materialize")
        materialize(bundle, args.output_root)
        return
    if args.protocol is not None:
        raise ValueError("--protocol is only valid for check/materialize")
    if args.plan is None:
        raise ValueError("--plan is required")
    if args.action == "execute":
        execute(args.plan, args.julia_bin, set(args.selected_cases))
    else:
        if args.selected_cases:
            raise ValueError("render always requires every case")
        render(args.plan)


if __name__ == "__main__":
    main()
