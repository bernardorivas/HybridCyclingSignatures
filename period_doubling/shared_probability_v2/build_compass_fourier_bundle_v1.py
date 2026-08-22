#!/usr/bin/env python3
"""Build the immutable Fourier-closed Compass embedded-orbit bundle v1.

Periodic cases use a frozen rule: align 32 deterministic late q-impact cycles
from one bridge-to-arc phase class, average them on the median stored-row
period, and retain H=6q Fourier harmonics.  Analysis tangents are analytic
derivatives of that closed Fourier curve.  Chaos is explicitly nonperiodic:
its frozen encoded path and stored learned-flow directions are linearly
interpolated from dt=.005 to dt=.0025.  Every case is exactly 469 nominal
suspension-time units long.

The builder writes only a new versioned directory below
``code/experiments_planned/outputs`` and refuses to overwrite it.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
HERE = SCRIPT_PATH.parent
CODE_ROOT = HERE.parents[1]
OUTPUT_PARENT = CODE_ROOT / "experiments_planned" / "outputs"
DEFAULT_OUTPUT = OUTPUT_PARENT / "compass_embedded_fourier_orbits_v1"
SOURCE_LATENT = CODE_ROOT / "period_doubling" / "data_fine" / "compass_gait_latent"
SOURCE_PHYSICAL = CODE_ROOT / "period_doubling" / "data_fine" / "compass_gait"
FLOW_ROOT = CODE_ROOT / "experiments_planned" / "outputs" / "fine_compass_learned_flow_tangents_v2"
FLOW_PROVENANCE = FLOW_ROOT / "provenance.json"

RAW_DT = 0.005
ANALYSIS_DT = 0.0025
ANALYSIS_DURATION = 469.0
ANALYSIS_ROWS = 187_601
N_CYCLES = 32
HARMONICS_PER_IMPACT = 6
METRIC_C = 5.0

CASES = (
    ("period1", 4.00, 1, 0.880, 0.7482409701092134),
    ("period2", 4.75, 2, 1.760, 1.5021396458781595),
    ("period4", 5.00, 4, 3.520, 3.0019139378989905),
    ("period8", 5.02, 8, 7.045, 6.004311980796729),
    ("chaos", 5.20, None, None, None),
)

SOURCE_HASHES = {
    "period1": {
        "latent": "ada66f1d152244c6d6bb1b2ee2ff31678a3ca1475f67662773da255f6baad6ca",
        "physical": "26578d3bf95a8aa8a7be72c71e205b3ccf5903afe20988a97017de5991eff699",
    },
    "period2": {
        "latent": "336bd81e4092a4cdf4e03a9f0fdf0dcddab3b0aa9da3b500527a18eecbcb145e",
        "physical": "0aeb3322b3a9f080581414eb460788b60ac8e4a3abae3a40468c5e6885b12c90",
    },
    "period4": {
        "latent": "7987e25f3405d6dd3b1d08ca8e432ded3c6191c4b9618b9fa94b4b042096840a",
        "physical": "977472b31feb0003805738a94fa6035cab5a809fadde98088d371c61dce24d1e",
    },
    "period8": {
        "latent": "7dbca72b638d67ff5fd0cba4695868976d7a162ae79d360c921a13a1ac67c771",
        "physical": "12abf7119b5c5c48e3b51576afa1fa4b6439b0f53e31f39809fa7e67d01ec8a8",
    },
    "chaos": {
        "latent": "582d5eff34c4472561aab292e134a7090028b4714a527fdbca4b80350983669a",
        "physical": "d46fb03697ec319f5fb1810de9de3d9eeecd7127791ba1813f5587456db58cff",
        "flow": "d74cfe4d35cf6a039c01e86a9cda2f1c8629ef412e126754e16a6e51eab67d91",
    },
}

EXPECTED_PERIODIC_DIAGNOSTICS = {
    "period1": (0.871703826163954, 0.005496282856307443, 0.08177458123508727),
    "period2": (1.0090682242143334, 0.007256738811469752, 0.10949691586629753),
    "period4": (1.4506535796801079, 0.007051925407054866, 0.09536494056279919),
    "period8": (1.0890650025503106, 0.007286623602861267, 0.11035331662911861),
}
EXPECTED_CHAOS_BOUND = 1.6962586626119316


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def code_relative(path: Path) -> str:
    return str(path.resolve().relative_to(CODE_ROOT))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_source(path: Path, expected_hash: str, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked {label}: {path}")
    actual = sha256(path)
    if actual != expected_hash:
        raise ValueError(f"{label} hash changed: {actual}")


def load_latent(case_id: str, phi_deg: float, q: int | None) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    path = SOURCE_LATENT / f"compass_{case_id}.npz"
    validate_source(path, SOURCE_HASHES[case_id]["latent"], f"{case_id} latent archive")
    with np.load(path, allow_pickle=False) as source:
        required = {"t", "x", "piece_kind", "meta_json"}
        if not required.issubset(source.files):
            raise ValueError(f"{case_id}: latent archive lacks required arrays")
        t = np.asarray(source["t"], dtype=float).copy()
        x = np.asarray(source["x"], dtype=float).copy()
        piece_kind = np.asarray(source["piece_kind"], dtype=np.uint8).copy()
        meta = json.loads(str(source["meta_json"].item()))
    if x.ndim != 2 or x.shape != (len(t), 11) or len(piece_kind) != len(t):
        raise ValueError(f"{case_id}: unexpected latent array shape")
    if not np.all(np.isfinite(x)) or not np.allclose(np.diff(t), RAW_DT, rtol=0.0, atol=1e-12):
        raise ValueError(f"{case_id}: invalid latent values/cadence")
    if meta.get("label") != case_id or meta.get("expected_period") != q:
        raise ValueError(f"{case_id}: latent metadata changed")
    if not math.isclose(float(meta["phi_deg"]), phi_deg, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{case_id}: latent phi changed")
    return path, meta, t, x, piece_kind


def validate_physical_return(case_id: str, phi_deg: float, q: int, expected: float) -> tuple[Path, dict[str, Any]]:
    path = SOURCE_PHYSICAL / f"compass_{case_id}.npz"
    validate_source(path, SOURCE_HASHES[case_id]["physical"], f"{case_id} physical archive")
    with np.load(path, allow_pickle=False) as source:
        impact_times = np.asarray(source["impact_times"], dtype=float)
        jump_plus = np.asarray(source["jump_plus"], dtype=float)
        meta = json.loads(str(source["meta_json"].item()))
    if meta.get("label") != case_id or int(meta.get("expected_period", -1)) != q:
        raise ValueError(f"{case_id}: physical metadata changed")
    if not math.isclose(float(meta["phi_deg"]), phi_deg, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{case_id}: physical phi changed")
    trim = 10
    durations = impact_times[trim + q :] - impact_times[trim:-q]
    measured = float(np.median(durations))
    if not math.isclose(measured, expected, rel_tol=0.0, abs_tol=5e-13):
        raise ValueError(f"{case_id}: physical return changed")
    closure = np.linalg.norm(jump_plus[trim + q :] - jump_plus[trim:-q], axis=1)
    return path, {
        "method": "median_event_time_q_impact_return_after_ten_impact_trim",
        "full_return_lag_impacts": q,
        "median_return_seconds": measured,
        "median_recurrence_residual": float(np.median(closure)),
        "maximum_recurrence_residual": float(np.max(closure)),
    }


def bridge_to_arc_starts(piece_kind: np.ndarray) -> np.ndarray:
    return np.flatnonzero((piece_kind[:-1] == 1) & (piece_kind[1:] == 0)) + 1


def fourier_evaluate(
    coefficients: np.ndarray, frequencies: np.ndarray, period: float, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    phase = np.mod(times / period, 1.0)
    exponentials = np.exp(2j * np.pi * np.outer(phase, frequencies))
    positions = (exponentials @ coefficients).real
    derivative_coefficients = (
        (2j * np.pi * frequencies / period)[:, np.newaxis] * coefficients
    )
    tangents = (exponentials @ derivative_coefficients).real
    return positions, tangents


def global_curve_bound(positions: np.ndarray, tangents: np.ndarray) -> dict[str, float]:
    tangent_norms = np.max(np.abs(tangents), axis=1)
    if np.any(~np.isfinite(tangent_norms)) or np.any(tangent_norms <= 0):
        raise ValueError("Fourier/resampled tangents must be finite and nonzero")
    normalized = tangents / tangent_norms[:, np.newaxis]
    max_dx = float(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    max_dv = float(np.max(np.linalg.norm(np.diff(normalized, axis=0), axis=1)))
    return {
        "maximum_consecutive_position_distance": max_dx,
        "maximum_consecutive_normalized_tangent_distance": max_dv,
        "metric_c": METRIC_C,
        "global_curve_bound": max(max_dx, METRIC_C * max_dv),
    }


def save_matrix(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, fmt="%.17g")


def build_periodic(
    stage: Path, case_id: str, phi_deg: float, q: int,
    expected_nominal_period: float, physical_period: float,
) -> dict[str, Any]:
    latent_path, latent_meta, _, source_positions, piece_kind = load_latent(case_id, phi_deg, q)
    physical_path, physical_certificate = validate_physical_return(
        case_id, phi_deg, q, physical_period
    )
    starts = bridge_to_arc_starts(piece_kind)
    indices = list(
        range(
            len(starts) - q - 1,
            max(-1, len(starts) - q - 1 - N_CYCLES * q),
            -q,
        )
    )[::-1]
    if len(indices) != N_CYCLES:
        raise ValueError(f"{case_id}: failed to select exactly {N_CYCLES} cycles")
    spans = np.asarray([starts[index + q] - starts[index] for index in indices], dtype=int)
    template_rows = int(round(float(np.median(spans))))
    nominal_period = template_rows * RAW_DT
    if not math.isclose(nominal_period, expected_nominal_period, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{case_id}: nominal suspension period changed")
    target_phase = np.arange(template_rows, dtype=float) / template_rows
    cycles = np.empty((N_CYCLES, template_rows, source_positions.shape[1]), dtype=float)
    source_blocks: list[dict[str, Any]] = []
    for cycle_index, start_index in enumerate(indices):
        first = int(starts[start_index])
        last = int(starts[start_index + q])
        block = source_positions[first : last + 1]
        old_phase = np.linspace(0.0, 1.0, len(block))
        cycles[cycle_index] = np.column_stack([
            np.interp(target_phase, old_phase, block[:, coordinate])
            for coordinate in range(block.shape[1])
        ])
        source_blocks.append({
            "phase_class_start_ordinal": int(start_index),
            "source_start_index_zero_based": first,
            "source_end_index_zero_based_inclusive": last,
            "source_row_span": last - first,
        })
    template = np.mean(cycles, axis=0)
    spectrum = np.fft.fft(template, axis=0) / template_rows
    integer_frequencies = np.fft.fftfreq(template_rows) * template_rows
    harmonic_cutoff = HARMONICS_PER_IMPACT * q
    keep = np.abs(integer_frequencies) <= harmonic_cutoff
    frequencies = integer_frequencies[keep].astype(int)
    coefficients = spectrum[keep]
    reconstructed = np.fft.ifft(
        np.where(keep[:, np.newaxis], spectrum, 0.0) * template_rows, axis=0
    ).real

    reconstruction_error = reconstructed - template
    cycle_deviation = cycles - template[np.newaxis, :, :]
    fourier_vs_cycles = cycles - reconstructed[np.newaxis, :, :]
    fit_rms = float(np.sqrt(np.mean(reconstruction_error * reconstruction_error)))
    fit_max_row = float(np.max(np.linalg.norm(reconstruction_error, axis=1)))

    analysis_times = np.arange(ANALYSIS_ROWS, dtype=float) * ANALYSIS_DT
    if not math.isclose(float(analysis_times[-1]), ANALYSIS_DURATION, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("analysis time grid drifted")
    analysis_positions, analysis_tangents = fourier_evaluate(
        coefficients, frequencies, nominal_period, analysis_times
    )
    curve = global_curve_bound(analysis_positions, analysis_tangents)
    expected_bound, expected_rms, expected_max = EXPECTED_PERIODIC_DIAGNOSTICS[case_id]
    for label, actual, expected in (
        ("curve bound", curve["global_curve_bound"], expected_bound),
        ("fit RMS", fit_rms, expected_rms),
        ("fit max row", fit_max_row, expected_max),
    ):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-8):
            raise ValueError(f"{case_id}: {label} drifted ({actual:.17g} != {expected:.17g})")
    if not curve["global_curve_bound"] < 5.0:
        raise ValueError(f"{case_id}: curve bound does not fit r_max=5")

    start_position, start_tangent = fourier_evaluate(
        coefficients, frequencies, nominal_period, np.asarray([0.0])
    )
    end_position, end_tangent = fourier_evaluate(
        coefficients, frequencies, nominal_period, np.asarray([nominal_period])
    )
    dense_phase = np.arange(max(4096, template_rows * 4), dtype=float)
    dense_phase /= len(dense_phase)
    dense_times = dense_phase * nominal_period
    dense_positions, dense_tangents = fourier_evaluate(
        coefficients, frequencies, nominal_period, dense_times
    )
    divisor_separation: dict[str, Any] = {}
    for divisor in range(1, q):
        if q % divisor:
            continue
        shifted, _ = fourier_evaluate(
            coefficients, frequencies, nominal_period,
            np.mod(dense_times + nominal_period * divisor / q, nominal_period),
        )
        distances = np.linalg.norm(dense_positions - shifted, axis=1)
        divisor_separation[str(divisor)] = {
            "phase_shift_fraction": divisor / q,
            "minimum": float(np.min(distances)),
            "median": float(np.median(distances)),
            "maximum": float(np.max(distances)),
        }

    case_root = stage / "cases" / case_id
    case_root.mkdir(parents=True)
    positions_path = case_root / "analysis_positions.csv"
    tangents_path = case_root / "analysis_tangents.csv"
    fourier_path = case_root / "fourier_coefficients.csv"
    display_path = case_root / "display_orbit.csv"
    certificate_path = case_root / "certificate.json"
    save_matrix(positions_path, analysis_positions)
    save_matrix(tangents_path, analysis_tangents)
    with fourier_path.open("w", encoding="utf-8") as handle:
        header = ["mode"] + [
            f"z{coordinate}_{part}"
            for coordinate in range(source_positions.shape[1])
            for part in ("real", "imag")
        ]
        handle.write(",".join(header) + "\n")
        for frequency, coefficient in zip(frequencies, coefficients):
            values = [str(int(frequency))]
            for value in coefficient:
                values.extend((f"{value.real:.17g}", f"{value.imag:.17g}"))
            handle.write(",".join(values) + "\n")
    with display_path.open("w", encoding="utf-8") as handle:
        header = ["nominal_suspension_time"] + [f"z{i}" for i in range(11)] + [f"dz{i}" for i in range(11)]
        handle.write(",".join(header) + "\n")
        display_times = np.linspace(0.0, nominal_period, 2001)
        display_positions, display_tangents = fourier_evaluate(
            coefficients, frequencies, nominal_period, display_times
        )
        for time, position, tangent in zip(display_times, display_positions, display_tangents):
            handle.write(",".join([
                f"{time:.17g}", *[f"{v:.17g}" for v in position],
                *[f"{v:.17g}" for v in tangent],
            ]) + "\n")

    certificate = {
        "schema_version": 1,
        "status": "certified_derived_control",
        "case_id": case_id,
        "kind": "fourier_closed_periodic_embedded_orbit",
        "phi_deg": phi_deg,
        "q": q,
        "source_latent_archive": code_relative(latent_path),
        "source_latent_archive_sha256": sha256(latent_path),
        "source_physical_archive": code_relative(physical_path),
        "source_physical_archive_sha256": sha256(physical_path),
        "source_model_run": latent_meta["model_run"],
        "source_tangent_semantics": "positions_only; source tangents are not used for periodic cases",
        "analysis_tangent_semantics": "analytic derivative of the fitted Fourier-closed embedded curve",
        "selection_rule": {
            "bridge_to_arc_definition": "flatnonzero((piece_kind[:-1]==1)&(piece_kind[1:]==0))+1",
            "same_phase_class_rule": "reverse(range(len(starts)-q-1,max(-1,len(starts)-q-1-32*q),-q))",
            "n_cycles": N_CYCLES,
            "source_blocks": source_blocks,
            "source_row_spans": spans.tolist(),
        },
        "template": {
            "row_count": template_rows,
            "raw_sample_dt": RAW_DT,
            "nominal_suspension_period": nominal_period,
            "target_phase": "arange(N)/N; endpoint excluded",
            "source_block_phase": "linspace(0,1,len(block)); endpoints included",
        },
        "fourier": {
            "harmonics_per_impact": HARMONICS_PER_IMPACT,
            "harmonic_cutoff": harmonic_cutoff,
            "signed_mode_count": int(len(frequencies)),
            "template_fit_coordinate_rms": fit_rms,
            "template_fit_maximum_row_l2": fit_max_row,
            "cycle_spread_coordinate_rms": float(np.sqrt(np.mean(cycle_deviation * cycle_deviation))),
            "cycle_spread_maximum_row_l2": float(np.max(np.linalg.norm(cycle_deviation, axis=2))),
            "fourier_vs_all_cycles_coordinate_rms": float(np.sqrt(np.mean(fourier_vs_cycles * fourier_vs_cycles))),
            "fourier_vs_all_cycles_maximum_row_l2": float(np.max(np.linalg.norm(fourier_vs_cycles, axis=2))),
            "position_closure_l2": float(np.linalg.norm(end_position - start_position)),
            "tangent_closure_l2": float(np.linalg.norm(end_tangent - start_tangent)),
            "proper_divisor_separation": divisor_separation,
        },
        "physical_return_context": physical_certificate,
        "analysis": {
            "sample_dt": ANALYSIS_DT,
            "duration": ANALYSIS_DURATION,
            "n_samples": ANALYSIS_ROWS,
            "curve_resolution": curve,
        },
    }
    write_json(certificate_path, certificate)
    return {
        "id": case_id,
        "kind": "periodic_fourier_closed",
        "phi_deg": phi_deg,
        "q": q,
        "nominal_suspension_period": nominal_period,
        "physical_return_seconds": physical_period,
        "dimension": 11,
        "analysis_sample_dt": ANALYSIS_DT,
        "analysis_n_samples": ANALYSIS_ROWS,
        "analysis_duration": ANALYSIS_DURATION,
        "tangent_semantics": "analytic_fourier_derivative",
        "positions": {"path": str(positions_path.relative_to(stage)), "sha256": sha256(positions_path)},
        "tangents": {"path": str(tangents_path.relative_to(stage)), "sha256": sha256(tangents_path)},
        "display": {"kind": "one_full_fourier_closed_orbit", "path": str(display_path.relative_to(stage)), "sha256": sha256(display_path), "n_rows": 2001},
        "fourier": {"path": str(fourier_path.relative_to(stage)), "sha256": sha256(fourier_path), "harmonic_cutoff": harmonic_cutoff},
        "certificate": {"path": str(certificate_path.relative_to(stage)), "sha256": sha256(certificate_path)},
        "global_curve_bound": curve["global_curve_bound"],
    }


def build_chaos(stage: Path, phi_deg: float) -> dict[str, Any]:
    case_id = "chaos"
    latent_path, latent_meta, source_times, source_positions, _ = load_latent(case_id, phi_deg, None)
    physical_path = SOURCE_PHYSICAL / "compass_chaos.npz"
    validate_source(physical_path, SOURCE_HASHES[case_id]["physical"], "chaos physical archive")
    flow_path = FLOW_ROOT / "compass_chaos_tangents.csv"
    validate_source(flow_path, SOURCE_HASHES[case_id]["flow"], "chaos learned-flow tangents")
    validate_source(
        FLOW_PROVENANCE,
        "dd246ce54116225ca962d7f5f27c04e19ed04ff986bc2c9308ae48b64350dce3",
        "learned-flow tangent provenance",
    )
    flow = np.loadtxt(flow_path, dtype=float)
    if flow.shape != source_positions.shape:
        raise ValueError("chaos learned-flow tangent shape changed")
    target_times = np.arange(ANALYSIS_ROWS, dtype=float) * ANALYSIS_DT
    if target_times[-1] > source_times[-1]:
        raise ValueError("chaos source is too short for 469-unit stream")
    positions = np.column_stack([
        np.interp(target_times, source_times, source_positions[:, coordinate])
        for coordinate in range(source_positions.shape[1])
    ])
    tangents = np.column_stack([
        np.interp(target_times, source_times, flow[:, coordinate])
        for coordinate in range(flow.shape[1])
    ])
    curve = global_curve_bound(positions, tangents)
    if not math.isclose(
        curve["global_curve_bound"], EXPECTED_CHAOS_BOUND,
        rel_tol=0.0, abs_tol=5e-8,
    ):
        raise ValueError("chaos curve bound drifted")
    if not curve["global_curve_bound"] < 5.0:
        raise ValueError("chaos curve bound does not fit r_max=5")
    midpoint_rows = source_times / ANALYSIS_DT
    exact_source_rows = (
        (source_times <= target_times[-1])
        & np.isclose(midpoint_rows, np.round(midpoint_rows), rtol=0.0, atol=1e-10)
    )
    reconstructed_source = positions[np.round(midpoint_rows[exact_source_rows]).astype(int)]
    interpolation_exactness = float(np.max(np.abs(reconstructed_source - source_positions[exact_source_rows])))

    case_root = stage / "cases" / case_id
    case_root.mkdir(parents=True)
    positions_path = case_root / "analysis_positions.csv"
    tangents_path = case_root / "analysis_tangents.csv"
    display_path = case_root / "display_segment.csv"
    certificate_path = case_root / "certificate.json"
    save_matrix(positions_path, positions)
    save_matrix(tangents_path, tangents)
    display_rows = int(round(50.0 / ANALYSIS_DT)) + 1
    display_start = len(target_times) - display_rows
    with display_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(["nominal_suspension_time", *[f"z{i}" for i in range(11)], *[f"vtheta{i}" for i in range(11)]]) + "\n")
        for time, position, tangent in zip(
            target_times[display_start:] - target_times[display_start],
            positions[display_start:], tangents[display_start:],
        ):
            handle.write(",".join([
                f"{time:.17g}", *[f"{v:.17g}" for v in position],
                *[f"{v:.17g}" for v in tangent],
            ]) + "\n")
    certificate = {
        "schema_version": 1,
        "status": "certified_derived_control",
        "case_id": case_id,
        "kind": "nonperiodic_interpolated_frozen_encoded_path",
        "phi_deg": phi_deg,
        "q": None,
        "source_latent_archive": code_relative(latent_path),
        "source_latent_archive_sha256": sha256(latent_path),
        "source_physical_archive": code_relative(physical_path),
        "source_physical_archive_sha256": sha256(physical_path),
        "source_learned_flow_tangents": code_relative(flow_path),
        "source_learned_flow_tangents_sha256": sha256(flow_path),
        "source_tangent_provenance": code_relative(FLOW_PROVENANCE),
        "source_tangent_provenance_sha256": sha256(FLOW_PROVENANCE),
        "source_model_run": latent_meta["model_run"],
        "analysis_tangent_semantics": "linearly interpolated V_theta(z) directions on a linearly interpolated frozen encoded simulator path; mixed semantics relative to periodic analytic derivatives",
        "interpolation": {
            "kind": "coordinatewise_linear_half_step",
            "source_sample_dt": RAW_DT,
            "analysis_sample_dt": ANALYSIS_DT,
            "source_grid_reconstruction_maximum_absolute_error": interpolation_exactness,
        },
        "analysis": {
            "sample_dt": ANALYSIS_DT,
            "duration": ANALYSIS_DURATION,
            "n_samples": ANALYSIS_ROWS,
            "curve_resolution": curve,
        },
    }
    write_json(certificate_path, certificate)
    return {
        "id": case_id,
        "kind": "chaos_interpolated_frozen_path",
        "phi_deg": phi_deg,
        "q": None,
        "nominal_suspension_period": None,
        "physical_return_seconds": None,
        "dimension": 11,
        "analysis_sample_dt": ANALYSIS_DT,
        "analysis_n_samples": ANALYSIS_ROWS,
        "analysis_duration": ANALYSIS_DURATION,
        "tangent_semantics": "interpolated_learned_flow_direction_on_interpolated_frozen_path",
        "positions": {"path": str(positions_path.relative_to(stage)), "sha256": sha256(positions_path)},
        "tangents": {"path": str(tangents_path.relative_to(stage)), "sha256": sha256(tangents_path)},
        "display": {"kind": "late_50_unit_nonperiodic_segment", "path": str(display_path.relative_to(stage)), "sha256": sha256(display_path), "n_rows": display_rows},
        "fourier": None,
        "certificate": {"path": str(certificate_path.relative_to(stage)), "sha256": sha256(certificate_path)},
        "global_curve_bound": curve["global_curve_bound"],
    }


def build(output_root: Path) -> None:
    output_root = output_root.resolve(strict=False)
    if output_root.parent != OUTPUT_PARENT.resolve():
        raise ValueError(f"output must be a direct named child of {OUTPUT_PARENT}")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output_root}")
    if sha256(SCRIPT_PATH) == "":
        raise RuntimeError("unreachable builder hash failure")
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".compass-fourier-v1-stage-", dir=OUTPUT_PARENT))
    try:
        case_records: list[dict[str, Any]] = []
        for case_id, phi_deg, q, nominal_period, physical_period in CASES:
            print(f"Building {case_id}", flush=True)
            if q is None:
                record = build_chaos(stage, phi_deg)
            else:
                record = build_periodic(
                    stage, case_id, phi_deg, q, nominal_period, physical_period
                )
            case_records.append(record)
        summary_dir = stage / "summary"
        summary_dir.mkdir()
        summary_path = summary_dir / "cases.csv"
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write("case_id,kind,phi_deg,q,nominal_suspension_period,physical_return_seconds,harmonic_cutoff,global_curve_bound,tangent_semantics\n")
            for record in case_records:
                harmonic = "" if record["fourier"] is None else str(record["fourier"]["harmonic_cutoff"])
                handle.write(",".join([
                    record["id"], record["kind"], f"{record['phi_deg']:.17g}",
                    "" if record["q"] is None else str(record["q"]),
                    "" if record["nominal_suspension_period"] is None else f"{record['nominal_suspension_period']:.17g}",
                    "" if record["physical_return_seconds"] is None else f"{record['physical_return_seconds']:.17g}",
                    harmonic, f"{record['global_curve_bound']:.17g}", record["tangent_semantics"],
                ]) + "\n")
        files = [
            file_record(path, stage)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        ]
        manifest = {
            "schema_version": 1,
            "bundle_id": "compass_embedded_fourier_orbits_v1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generator": {
                "path": code_relative(SCRIPT_PATH),
                "sha256": sha256(SCRIPT_PATH),
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
            },
            "scientific_scope": {
                "periodic_positions": "32-cycle same-phase average of frozen encoded simulator paths, Fourier closed at H=6q",
                "periodic_tangents": "analytic derivative of the fitted Fourier curve",
                "chaos_positions": "linear half-step interpolation of the frozen encoded simulator path",
                "chaos_tangents": "linear half-step interpolation of V_theta(z) directions; explicitly mixed tangent semantics",
                "not_a_learned_rollout": True,
            },
            "analysis_sample_dt": ANALYSIS_DT,
            "analysis_duration": ANALYSIS_DURATION,
            "analysis_n_samples": ANALYSIS_ROWS,
            "dimension": 11,
            "metric_c_preflight": METRIC_C,
            "maximum_global_curve_bound": max(record["global_curve_bound"] for record in case_records),
            "cases": case_records,
            "summary": {"path": str(summary_path.relative_to(stage)), "sha256": sha256(summary_path)},
            "files": files,
        }
        write_json(stage / "bundle_manifest.json", manifest)
        stage.rename(output_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"Wrote immutable bundle: {output_root}")
    print(f"manifest_sha256={sha256(output_root / 'bundle_manifest.json')}")


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_OUTPUT
    if len(sys.argv) > 2:
        raise SystemExit(f"usage: {SCRIPT_PATH.name} [OUTPUT_ROOT]")
    build(output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"ERROR: {error}") from error
