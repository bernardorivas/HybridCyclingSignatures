#!/usr/bin/env python3
"""Build the refined, continuously timed Compass Fourier input bundle v3.

Periodic cases retain the v1 frozen selection and Fourier rule: 32 late,
q-impact, bridge-to-arc aligned cycles are averaged on a common phase grid,
then truncated to H=6q harmonics.  The period is no longer rounded to an
integer number of stored rows.  It is the OLS slope through the 33 q-spaced
bridge-to-arc boundary indices that delimit those 32 cycles, converted to the
nominal suspension clock.

The output cadence is dt=.00125 for 469 nominal units (375201 rows), and the
curve preflight uses C=.75 after row-wise l-infinity tangent normalization.
No cycling signatures are computed here.  ``check`` is read-only;
``materialize`` writes a new immutable directory and refuses overwrite.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import numpy as np


SCRIPT_PATH = Path(__file__).absolute()
HERE = SCRIPT_PATH.parent
CODE_ROOT = HERE.parents[1]
OUTPUT_PARENT = CODE_ROOT / "experiments_planned" / "outputs"
OUTPUT_ROOT = OUTPUT_PARENT / "compass_embedded_fourier_orbits_v3_refined"
SOURCE_LATENT = CODE_ROOT / "period_doubling" / "data_fine" / "compass_gait_latent"
SOURCE_PHYSICAL = CODE_ROOT / "period_doubling" / "data_fine" / "compass_gait"
FLOW_ROOT = OUTPUT_PARENT / "fine_compass_learned_flow_tangents_v2"
FLOW_PROVENANCE = FLOW_ROOT / "provenance.json"

RAW_DT = 0.005
ANALYSIS_DT = 0.00125
ANALYSIS_DURATION = 469.0
ANALYSIS_ROWS = 375_201
N_CYCLES = 32
HARMONICS_PER_IMPACT = 6
METRIC_C = 0.75
MAX_PLANNED_WINDOW_DURATION = 12.0
MAX_PLANNED_WINDOW_LAG = 9_600
MACHINE_ZERO_THRESHOLD = 1e-10
EVALUATION_CHUNK_ROWS = 20_000
DIAGNOSTIC_CHUNK_ROWS = 50_000

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

FLOW_PROVENANCE_HASH = "dd246ce54116225ca962d7f5f27c04e19ed04ff986bc2c9308ae48b64350dce3"

EXPECTED_OLS_SLOPES = {
    "period1": Fraction(525_543, 2_992),
    "period2": Fraction(62_027, 176),
    "period4": Fraction(2_107_519, 2_992),
    "period8": Fraction(2_107_653, 1_496),
}

EXPECTED_TEMPLATE_DIAGNOSTICS = {
    "period1": (0.005496282856307443, 0.08177458123508727),
    "period2": (0.007256738811469752, 0.10949691586629753),
    "period4": (0.007051925407054866, 0.09536494056279919),
    "period8": (0.007286623602861267, 0.11035331662911861),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def code_relative(path: Path) -> str:
    return str(path.absolute().relative_to(CODE_ROOT.absolute()))


def reject_symlink_components(path: Path, stop: Path) -> None:
    probe = path.absolute()
    stop = stop.absolute()
    while True:
        if probe.is_symlink():
            raise ValueError(f"symlink component is forbidden: {probe}")
        if probe == stop:
            return
        parent = probe.parent
        if parent == probe or stop not in (probe, *probe.parents):
            raise ValueError(f"path escapes required root {stop}: {path}")
        probe = parent


def validate_real_file(path: Path, expected_hash: str, label: str) -> None:
    reject_symlink_components(path, CODE_ROOT)
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected_hash:
        raise ValueError(f"{label} hash changed: {actual}")


def validate_output_absent() -> None:
    reject_symlink_components(OUTPUT_PARENT, CODE_ROOT)
    if OUTPUT_ROOT.parent.absolute() != OUTPUT_PARENT.absolute():
        raise ValueError("v3 output must be a direct child of the output parent")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT_ROOT}")


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def binding_record(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
    }


def save_matrix(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, fmt="%.17g")


def load_latent(
    case_id: str, phi_deg: float, q: int | None,
) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    path = SOURCE_LATENT / f"compass_{case_id}.npz"
    validate_real_file(path, SOURCE_HASHES[case_id]["latent"], f"{case_id} latent archive")
    with np.load(path, allow_pickle=False) as source:
        required = {"t", "x", "piece_kind", "meta_json"}
        if not required.issubset(source.files):
            raise ValueError(f"{case_id}: latent archive lacks required arrays")
        times = np.asarray(source["t"], dtype=float).copy()
        positions = np.asarray(source["x"], dtype=float).copy()
        piece_kind = np.asarray(source["piece_kind"], dtype=np.uint8).copy()
        meta = json.loads(str(source["meta_json"].item()))
    if positions.shape != (len(times), 11) or len(piece_kind) != len(times):
        raise ValueError(f"{case_id}: unexpected latent shape")
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{case_id}: nonfinite latent values")
    if not np.allclose(np.diff(times), RAW_DT, rtol=0.0, atol=1e-12):
        raise ValueError(f"{case_id}: latent cadence changed")
    if meta.get("label") != case_id or meta.get("expected_period") != q:
        raise ValueError(f"{case_id}: latent metadata changed")
    if not math.isclose(float(meta["phi_deg"]), phi_deg, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{case_id}: latent parameter changed")
    return path, meta, times, positions, piece_kind


def validate_physical_return(
    case_id: str, phi_deg: float, q: int, expected: float,
) -> tuple[Path, dict[str, Any]]:
    path = SOURCE_PHYSICAL / f"compass_{case_id}.npz"
    validate_real_file(path, SOURCE_HASHES[case_id]["physical"], f"{case_id} physical archive")
    with np.load(path, allow_pickle=False) as source:
        impact_times = np.asarray(source["impact_times"], dtype=float)
        jump_plus = np.asarray(source["jump_plus"], dtype=float)
        meta = json.loads(str(source["meta_json"].item()))
    if meta.get("label") != case_id or int(meta.get("expected_period", -1)) != q:
        raise ValueError(f"{case_id}: physical metadata changed")
    if not math.isclose(float(meta["phi_deg"]), phi_deg, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{case_id}: physical parameter changed")
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


def exact_ols(boundary_indices: Iterable[int]) -> dict[str, Any]:
    ys = [int(value) for value in boundary_indices]
    xs = list(range(len(ys)))
    if len(xs) != N_CYCLES + 1:
        raise ValueError("OLS period fit requires exactly 33 boundaries")
    xbar = Fraction(sum(xs), len(xs))
    ybar = Fraction(sum(ys), len(ys))
    sxx = sum((Fraction(x) - xbar) ** 2 for x in xs)
    sxy = sum(
        (Fraction(x) - xbar) * (Fraction(y) - ybar)
        for x, y in zip(xs, ys)
    )
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    residuals = [Fraction(y) - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(value * value for value in residuals)
    sst = sum((Fraction(y) - ybar) ** 2 for y in ys)
    sigma2 = float(sse / (len(xs) - 2))
    slope_standard_error = math.sqrt(sigma2 / float(sxx))
    return {
        "slope": slope,
        "intercept": intercept,
        "residuals": residuals,
        "r_squared": 1.0 - float(sse / sst),
        "residual_rms_rows": math.sqrt(float(sse) / len(xs)),
        "maximum_absolute_residual_rows": max(abs(float(value)) for value in residuals),
        "slope_standard_error_rows_per_cycle": slope_standard_error,
    }


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def fourier_evaluate(
    coefficients: np.ndarray,
    frequencies: np.ndarray,
    period: float,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((len(times), coefficients.shape[1]), dtype=float)
    tangents = np.empty_like(positions)
    derivative_coefficients = (
        (2j * np.pi * frequencies / period)[:, np.newaxis] * coefficients
    )
    for start in range(0, len(times), EVALUATION_CHUNK_ROWS):
        stop = min(len(times), start + EVALUATION_CHUNK_ROWS)
        phase = np.mod(times[start:stop] / period, 1.0)
        exponentials = np.exp(2j * np.pi * np.outer(phase, frequencies))
        positions[start:stop] = (exponentials @ coefficients).real
        tangents[start:stop] = (exponentials @ derivative_coefficients).real
    return positions, tangents


def normalize_linf(tangents: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    norms = np.max(np.abs(tangents), axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("tangents must be finite and nonzero")
    normalized = tangents / norms[:, np.newaxis]
    return normalized, {
        "pre_normalization_minimum_linf": float(np.min(norms)),
        "pre_normalization_maximum_linf": float(np.max(norms)),
        "post_normalization_minimum_linf": float(np.min(np.max(np.abs(normalized), axis=1))),
        "post_normalization_maximum_linf": float(np.max(np.max(np.abs(normalized), axis=1))),
    }


def curve_resolution(
    positions: np.ndarray, normalized_tangents: np.ndarray,
) -> dict[str, Any]:
    dx = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dv = np.linalg.norm(np.diff(normalized_tangents, axis=0), axis=1)
    max_dx_index = int(np.argmax(dx))
    max_dv_index = int(np.argmax(dv))
    max_dx = float(dx[max_dx_index])
    max_dv = float(dv[max_dv_index])
    scaled_dv = METRIC_C * max_dv
    return {
        "maximum_consecutive_position_distance": max_dx,
        "position_component_maximum_l2": max_dx,
        "position_component_argmax_zero_based": max_dx_index,
        "maximum_consecutive_normalized_tangent_distance": max_dv,
        "normalized_tangent_component_maximum_l2": max_dv,
        "normalized_tangent_component_argmax_zero_based": max_dv_index,
        "metric_c": METRIC_C,
        "metric_scaled_tangent_component_maximum": scaled_dv,
        "global_curve_bound": max(max_dx, scaled_dv),
        "dynamic_metric": "max(l2(position difference), C*l2(linf-normalized tangent difference))",
    }


def recurrence_for_lag(
    positions: np.ndarray,
    normalized_tangents: np.ndarray,
    lag: int,
) -> dict[str, Any]:
    if not 0 < lag < len(positions):
        raise ValueError(f"invalid recurrence lag: {lag}")
    minimum = math.inf
    argmin = -1
    exact_count = 0
    for start in range(0, len(positions) - lag, DIAGNOSTIC_CHUNK_ROWS):
        stop = min(len(positions) - lag, start + DIAGNOSTIC_CHUNK_ROWS)
        left_p = positions[start:stop]
        right_p = positions[start + lag : stop + lag]
        left_v = normalized_tangents[start:stop]
        right_v = normalized_tangents[start + lag : stop + lag]
        dx = np.linalg.norm(right_p - left_p, axis=1)
        dv = METRIC_C * np.linalg.norm(right_v - left_v, axis=1)
        dynamic = np.maximum(dx, dv)
        local = int(np.argmin(dynamic))
        if float(dynamic[local]) < minimum:
            minimum = float(dynamic[local])
            argmin = start + local
        exact_count += int(np.count_nonzero(
            np.all(right_p == left_p, axis=1)
            & np.all(right_v == left_v, axis=1)
        ))
    return {
        "lag_samples": lag,
        "lag_duration": lag * ANALYSIS_DT,
        "minimum_dynamic_distance": minimum,
        "argmin_start_index_zero_based": argmin,
        "exact_lifted_pair_count": exact_count,
    }


def recurrence_certificate(
    positions: np.ndarray,
    normalized_tangents: np.ndarray,
    period_ratio: Fraction,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    multiple = 1
    while True:
        exact_lag = period_ratio * multiple
        if math.floor(float(exact_lag)) > MAX_PLANNED_WINDOW_LAG:
            break
        lags = sorted({math.floor(float(exact_lag)), math.ceil(float(exact_lag))})
        for lag in lags:
            if lag <= 0 or lag > MAX_PLANNED_WINDOW_LAG:
                continue
            row = recurrence_for_lag(positions, normalized_tangents, lag)
            row.update({
                "period_multiple": multiple,
                "ideal_lag_samples": float(exact_lag),
                "signed_timing_error": lag * ANALYSIS_DT
                - multiple * float(period_ratio) * ANALYSIS_DT,
            })
            candidates.append(row)
        multiple += 1
    if not candidates:
        raise ValueError("periodic recurrence audit produced no candidates")
    min_row = min(candidates, key=lambda item: item["minimum_dynamic_distance"])
    exact_total = sum(int(item["exact_lifted_pair_count"]) for item in candidates)
    passed = (
        exact_total == 0
        and float(min_row["minimum_dynamic_distance"]) > MACHINE_ZERO_THRESHOLD
    )
    if not passed:
        raise ValueError(
            "machine-zero recurrence inside planned window: "
            f"lag={min_row['lag_samples']} distance={min_row['minimum_dynamic_distance']}"
        )
    alignment_multiple = period_ratio.denominator
    alignment_lag = period_ratio.numerator
    return {
        "status": "passed",
        "scope": (
            "nearest floor/ceil sample lags to every positive integer full-period "
            "multiple with lag <= 9600; this is the planned 12-unit window, not a "
            "claim of global stream noncommensurability"
        ),
        "machine_zero_threshold_dynamic_distance": MACHINE_ZERO_THRESHOLD,
        "max_audited_lag_samples": MAX_PLANNED_WINDOW_LAG,
        "max_audited_lag_duration": MAX_PLANNED_WINDOW_DURATION,
        "audited_period_multiples": multiple - 1,
        "audited_candidate_count": len(candidates),
        "minimum_audited_dynamic_distance": float(min_row["minimum_dynamic_distance"]),
        "minimum_audited_candidate": min_row,
        "exact_lifted_pair_count_across_candidates": exact_total,
        "shortest_recurrence_samples": alignment_lag,
        "cycles_at_recurrence": alignment_multiple,
        "maximum_segment_length": MAX_PLANNED_WINDOW_LAG,
        "candidates": candidates,
        "exact_rational_commensurability": {
            "period_over_dt": fraction_record(period_ratio),
            "first_alignment_period_multiple": alignment_multiple,
            "first_alignment_lag_samples": alignment_lag,
            "first_alignment_duration": alignment_lag * ANALYSIS_DT,
            "within_planned_12_unit_window": alignment_lag <= MAX_PLANNED_WINDOW_LAG,
            "within_469_unit_analysis_stream": alignment_lag < ANALYSIS_ROWS,
        },
    }


def selected_boundaries(starts: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    ordinals = np.arange(len(starts) - 1 - N_CYCLES * q, len(starts), q, dtype=int)
    if len(ordinals) != N_CYCLES + 1 or ordinals[0] < 0:
        raise ValueError("failed to select 33 q-spaced late boundaries")
    return ordinals, starts[ordinals].astype(int)


def prepare_periodic(
    case_id: str,
    phi_deg: float,
    q: int,
    legacy_nominal_period: float,
    physical_period: float,
) -> dict[str, Any]:
    latent_path, latent_meta, _, source_positions, piece_kind = load_latent(
        case_id, phi_deg, q
    )
    physical_path, physical_context = validate_physical_return(
        case_id, phi_deg, q, physical_period
    )
    starts = bridge_to_arc_starts(piece_kind)
    boundary_ordinals, boundary_indices = selected_boundaries(starts, q)
    cycle_start_ordinals = boundary_ordinals[:-1]
    spans = np.diff(boundary_indices)
    ols = exact_ols(boundary_indices)
    slope = ols["slope"]
    if slope != EXPECTED_OLS_SLOPES[case_id]:
        raise ValueError(f"{case_id}: OLS slope changed ({slope})")
    continuous_period_fraction = slope * Fraction(1, 200)
    continuous_period = float(continuous_period_fraction)
    period_ratio = continuous_period_fraction / Fraction(1, 800)
    template_rows = int(round(float(np.median(spans))))
    legacy_reconstructed_period = template_rows * RAW_DT
    if not math.isclose(
        legacy_reconstructed_period, legacy_nominal_period, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{case_id}: legacy template period changed")

    target_phase = np.arange(template_rows, dtype=float) / template_rows
    cycles = np.empty((N_CYCLES, template_rows, source_positions.shape[1]), dtype=float)
    source_blocks: list[dict[str, Any]] = []
    for cycle_index, start_ordinal in enumerate(cycle_start_ordinals):
        first = int(starts[start_ordinal])
        last = int(starts[start_ordinal + q])
        block = source_positions[first : last + 1]
        old_phase = np.linspace(0.0, 1.0, len(block))
        cycles[cycle_index] = np.column_stack([
            np.interp(target_phase, old_phase, block[:, coordinate])
            for coordinate in range(block.shape[1])
        ])
        source_blocks.append({
            "phase_class_start_ordinal": int(start_ordinal),
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
        np.where(keep[:, np.newaxis], spectrum, 0.0) * template_rows,
        axis=0,
    ).real
    reconstruction_error = reconstructed - template
    cycle_deviation = cycles - template[np.newaxis, :, :]
    fourier_vs_cycles = cycles - reconstructed[np.newaxis, :, :]
    fit_rms = float(np.sqrt(np.mean(reconstruction_error * reconstruction_error)))
    fit_max = float(np.max(np.linalg.norm(reconstruction_error, axis=1)))
    expected_rms, expected_max = EXPECTED_TEMPLATE_DIAGNOSTICS[case_id]
    if not math.isclose(fit_rms, expected_rms, rel_tol=0.0, abs_tol=5e-12):
        raise ValueError(f"{case_id}: Fourier fit RMS changed")
    if not math.isclose(fit_max, expected_max, rel_tol=0.0, abs_tol=5e-12):
        raise ValueError(f"{case_id}: Fourier fit maximum changed")

    analysis_times = np.arange(ANALYSIS_ROWS, dtype=float) * ANALYSIS_DT
    if not math.isclose(analysis_times[-1], ANALYSIS_DURATION, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("analysis time grid drifted")
    positions, tangents = fourier_evaluate(
        coefficients, frequencies, continuous_period, analysis_times
    )
    normalized_tangents, normalization = normalize_linf(tangents)
    curve = curve_resolution(positions, normalized_tangents)
    recurrence = recurrence_certificate(positions, normalized_tangents, period_ratio)

    at_zero_p, at_zero_v = fourier_evaluate(
        coefficients, frequencies, continuous_period, np.asarray([0.0])
    )
    at_period_p, at_period_v = fourier_evaluate(
        coefficients, frequencies, continuous_period, np.asarray([continuous_period])
    )
    dense_count = max(4096, template_rows * 4)
    dense_times = np.arange(dense_count, dtype=float) / dense_count * continuous_period
    dense_positions, dense_tangents = fourier_evaluate(
        coefficients, frequencies, continuous_period, dense_times
    )
    dense_normalized, _ = normalize_linf(dense_tangents)
    divisor_separation: dict[str, Any] = {}
    for divisor in range(1, q):
        if q % divisor:
            continue
        shifted_p, shifted_v = fourier_evaluate(
            coefficients,
            frequencies,
            continuous_period,
            np.mod(
                dense_times + continuous_period * divisor / q,
                continuous_period,
            ),
        )
        shifted_normalized, _ = normalize_linf(shifted_v)
        dx = np.linalg.norm(dense_positions - shifted_p, axis=1)
        dv = METRIC_C * np.linalg.norm(dense_normalized - shifted_normalized, axis=1)
        dynamic = np.maximum(dx, dv)
        divisor_separation[str(divisor)] = {
            "phase_shift_fraction": divisor / q,
            "minimum_position_l2": float(np.min(dx)),
            "minimum_dynamic_distance": float(np.min(dynamic)),
            "median_dynamic_distance": float(np.median(dynamic)),
            "maximum_dynamic_distance": float(np.max(dynamic)),
        }

    span_values, span_counts = np.unique(spans, return_counts=True)
    certificate = {
        "schema_version": 3,
        "status": "certified_derived_control",
        "case_id": case_id,
        "kind": "fourier_closed_periodic_embedded_orbit_continuous_ols_period",
        "phi_deg": phi_deg,
        "q": q,
        "source_latent_archive": code_relative(latent_path),
        "source_latent_archive_sha256": sha256(latent_path),
        "source_physical_archive": code_relative(physical_path),
        "source_physical_archive_sha256": sha256(physical_path),
        "source_model_run": latent_meta["model_run"],
        "source_tangent_semantics": "positions_only; source tangents are not used",
        "analysis_tangent_semantics": "analytic derivative of fitted Fourier-closed curve",
        "selection_rule": {
            "bridge_to_arc_definition": "flatnonzero((piece_kind[:-1]==1)&(piece_kind[1:]==0))+1",
            "same_phase_class": True,
            "q_spacing": q,
            "n_cycles": N_CYCLES,
            "n_boundary_indices": len(boundary_indices),
            "boundary_ordinals": boundary_ordinals.tolist(),
            "boundary_indices_zero_based": boundary_indices.tolist(),
            "source_blocks": source_blocks,
            "source_row_spans": spans.tolist(),
            "source_row_span_histogram": {
                str(int(value)): int(count)
                for value, count in zip(span_values, span_counts)
            },
        },
        "period_estimate": {
            "canonical_method": "ordinary_least_squares_slope_on_33_q_spaced_bridge_to_arc_boundary_indices",
            "regressor": "cycle_number_0_through_32",
            "response": "zero_based_stored_row_boundary_index",
            "n_boundary_indices": len(boundary_indices),
            "q_spacing": q,
            "boundary_ordinals": boundary_ordinals.tolist(),
            "boundary_indices_zero_based": boundary_indices.tolist(),
            "continuous_period": continuous_period,
            "period_over_analysis_dt": float(period_ratio),
            "ols_slope_rows_per_q_impact_cycle": fraction_record(slope),
            "ols_intercept_rows": fraction_record(ols["intercept"]),
            "continuous_nominal_suspension_period": fraction_record(continuous_period_fraction),
            "continuous_period_over_analysis_dt": fraction_record(period_ratio),
            "r_squared": ols["r_squared"],
            "residual_rms_rows": ols["residual_rms_rows"],
            "maximum_absolute_residual_rows": ols["maximum_absolute_residual_rows"],
            "slope_standard_error_rows_per_cycle": ols["slope_standard_error_rows_per_cycle"],
            "residuals_rows": [float(value) for value in ols["residuals"]],
            "legacy_template_row_count": template_rows,
            "legacy_rounded_median_period": legacy_reconstructed_period,
            "legacy_period_is_provenance_only": True,
        },
        "template": {
            "row_count": template_rows,
            "raw_sample_dt": RAW_DT,
            "target_phase": "arange(N)/N; endpoint excluded",
            "source_block_phase": "linspace(0,1,len(block)); endpoints included",
        },
        "fourier": {
            "harmonics_per_impact": HARMONICS_PER_IMPACT,
            "harmonic_cutoff": harmonic_cutoff,
            "signed_mode_count": int(len(frequencies)),
            "template_fit_coordinate_rms": fit_rms,
            "template_fit_maximum_row_l2": fit_max,
            "cycle_spread_coordinate_rms": float(np.sqrt(np.mean(cycle_deviation**2))),
            "cycle_spread_maximum_row_l2": float(
                np.max(np.linalg.norm(cycle_deviation, axis=2))
            ),
            "fourier_vs_all_cycles_coordinate_rms": float(
                np.sqrt(np.mean(fourier_vs_cycles**2))
            ),
            "fourier_vs_all_cycles_maximum_row_l2": float(
                np.max(np.linalg.norm(fourier_vs_cycles, axis=2))
            ),
            "analytic_position_closure_l2": float(np.linalg.norm(at_period_p - at_zero_p)),
            "analytic_tangent_closure_l2": float(np.linalg.norm(at_period_v - at_zero_v)),
            "proper_divisor_separation": divisor_separation,
        },
        "physical_return_context": physical_context,
        "full_period_recurrence": recurrence,
        "analysis": {
            "sample_dt": ANALYSIS_DT,
            "duration": ANALYSIS_DURATION,
            "n_samples": ANALYSIS_ROWS,
            "tangent_normalization": "linf",
            "normalization": normalization,
            "curve_resolution": curve,
            "full_period_recurrence": recurrence,
        },
    }
    return {
        "id": case_id,
        "kind": "periodic_fourier_closed_continuous_ols_period",
        "phi_deg": phi_deg,
        "q": q,
        "nominal_suspension_period": continuous_period,
        "physical_return_seconds": physical_period,
        "dimension": 11,
        "analysis_sample_dt": ANALYSIS_DT,
        "analysis_n_samples": ANALYSIS_ROWS,
        "analysis_duration": ANALYSIS_DURATION,
        "tangent_semantics": "analytic_fourier_derivative",
        "harmonic_cutoff": harmonic_cutoff,
        "global_curve_bound": curve["global_curve_bound"],
        "position_curve_component": curve["position_component_maximum_l2"],
        "normalized_tangent_curve_component": curve[
            "normalized_tangent_component_maximum_l2"
        ],
        "metric_scaled_tangent_curve_component": curve[
            "metric_scaled_tangent_component_maximum"
        ],
        "minimum_planned_window_recurrence_distance": recurrence[
            "minimum_audited_dynamic_distance"
        ],
        "exact_alignment_multiple": period_ratio.denominator,
        "exact_alignment_lag_samples": period_ratio.numerator,
        "positions_array": positions,
        "tangents_array": tangents,
        "frequencies_array": frequencies,
        "coefficients_array": coefficients,
        "certificate_document": certificate,
    }


def prepare_chaos(phi_deg: float) -> dict[str, Any]:
    case_id = "chaos"
    latent_path, latent_meta, source_times, source_positions, _ = load_latent(
        case_id, phi_deg, None
    )
    physical_path = SOURCE_PHYSICAL / "compass_chaos.npz"
    validate_real_file(
        physical_path, SOURCE_HASHES[case_id]["physical"], "chaos physical archive"
    )
    flow_path = FLOW_ROOT / "compass_chaos_tangents.csv"
    validate_real_file(flow_path, SOURCE_HASHES[case_id]["flow"], "chaos learned-flow tangents")
    validate_real_file(FLOW_PROVENANCE, FLOW_PROVENANCE_HASH, "learned-flow provenance")
    flow = np.loadtxt(flow_path, dtype=float)
    if flow.shape != source_positions.shape:
        raise ValueError("chaos flow tangent shape changed")
    target_times = np.arange(ANALYSIS_ROWS, dtype=float) * ANALYSIS_DT
    if target_times[-1] > source_times[-1]:
        raise ValueError("chaos source is too short")
    positions = np.column_stack([
        np.interp(target_times, source_times, source_positions[:, coordinate])
        for coordinate in range(source_positions.shape[1])
    ])
    tangents = np.column_stack([
        np.interp(target_times, source_times, flow[:, coordinate])
        for coordinate in range(flow.shape[1])
    ])
    normalized_tangents, normalization = normalize_linf(tangents)
    curve = curve_resolution(positions, normalized_tangents)
    source_rows = np.rint(source_times / ANALYSIS_DT).astype(int)
    valid = source_rows <= ANALYSIS_ROWS - 1
    exactness = float(
        np.max(np.abs(positions[source_rows[valid]] - source_positions[valid]))
    )
    certificate = {
        "schema_version": 3,
        "status": "certified_derived_control",
        "case_id": case_id,
        "kind": "nonperiodic_quarter_step_interpolated_frozen_encoded_path",
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
        "analysis_tangent_semantics": (
            "coordinatewise-linear interpolation of V_theta(z) directions on a "
            "coordinatewise-linear frozen encoded simulator path; mixed semantics "
            "relative to periodic analytic derivatives"
        ),
        "interpolation": {
            "kind": "coordinatewise_linear_quarter_step",
            "source_sample_dt": RAW_DT,
            "analysis_sample_dt": ANALYSIS_DT,
            "source_grid_reconstruction_maximum_absolute_error": exactness,
        },
        "analysis": {
            "sample_dt": ANALYSIS_DT,
            "duration": ANALYSIS_DURATION,
            "n_samples": ANALYSIS_ROWS,
            "tangent_normalization": "linf",
            "normalization": normalization,
            "curve_resolution": curve,
            "full_period_recurrence": None,
        },
    }
    return {
        "id": case_id,
        "kind": "chaos_interpolated_frozen_path_quarter_step",
        "phi_deg": phi_deg,
        "q": None,
        "nominal_suspension_period": None,
        "physical_return_seconds": None,
        "dimension": 11,
        "analysis_sample_dt": ANALYSIS_DT,
        "analysis_n_samples": ANALYSIS_ROWS,
        "analysis_duration": ANALYSIS_DURATION,
        "tangent_semantics": "interpolated_learned_flow_direction_on_interpolated_frozen_path",
        "harmonic_cutoff": None,
        "global_curve_bound": curve["global_curve_bound"],
        "position_curve_component": curve["position_component_maximum_l2"],
        "normalized_tangent_curve_component": curve[
            "normalized_tangent_component_maximum_l2"
        ],
        "metric_scaled_tangent_curve_component": curve[
            "metric_scaled_tangent_component_maximum"
        ],
        "minimum_planned_window_recurrence_distance": None,
        "exact_alignment_multiple": None,
        "exact_alignment_lag_samples": None,
        "positions_array": positions,
        "tangents_array": tangents,
        "frequencies_array": None,
        "coefficients_array": None,
        "certificate_document": certificate,
    }


def prepare_all() -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for case_id, phi_deg, q, legacy_period, physical_period in CASES:
        print(f"Preparing {case_id}", flush=True)
        if q is None:
            prepared.append(prepare_chaos(phi_deg))
        else:
            prepared.append(prepare_periodic(
                case_id,
                phi_deg,
                q,
                float(legacy_period),
                float(physical_period),
            ))
    return prepared


def public_diagnostics(prepared: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bundle_id": "compass_embedded_fourier_orbits_v3_refined",
        "status": "check_passed",
        "analysis_sample_dt": ANALYSIS_DT,
        "analysis_duration": ANALYSIS_DURATION,
        "analysis_n_samples": ANALYSIS_ROWS,
        "metric_c_preflight": METRIC_C,
        "recurrence_scope_max_lag_samples": MAX_PLANNED_WINDOW_LAG,
        "recurrence_machine_zero_threshold": MACHINE_ZERO_THRESHOLD,
        "cases": [{
            key: case[key]
            for key in (
                "id",
                "q",
                "nominal_suspension_period",
                "harmonic_cutoff",
                "position_curve_component",
                "normalized_tangent_curve_component",
                "metric_scaled_tangent_curve_component",
                "global_curve_bound",
                "minimum_planned_window_recurrence_distance",
                "exact_alignment_multiple",
                "exact_alignment_lag_samples",
            )
        } for case in prepared],
    }


def write_periodic_files(case_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    positions_path = case_root / "analysis_positions.csv"
    tangents_path = case_root / "analysis_tangents.csv"
    coefficients_path = case_root / "fourier_coefficients.csv"
    display_path = case_root / "display_orbit.csv"
    certificate_path = case_root / "certificate.json"
    save_matrix(positions_path, case["positions_array"])
    save_matrix(tangents_path, case["tangents_array"])
    with coefficients_path.open("w", encoding="utf-8") as handle:
        header = ["mode"] + [
            f"z{coordinate}_{part}"
            for coordinate in range(11)
            for part in ("real", "imag")
        ]
        handle.write(",".join(header) + "\n")
        for frequency, coefficient in zip(
            case["frequencies_array"], case["coefficients_array"]
        ):
            values = [str(int(frequency))]
            for value in coefficient:
                values.extend((f"{value.real:.17g}", f"{value.imag:.17g}"))
            handle.write(",".join(values) + "\n")
    display_times = np.linspace(0.0, case["nominal_suspension_period"], 4001)
    display_positions, display_tangents = fourier_evaluate(
        case["coefficients_array"],
        case["frequencies_array"],
        case["nominal_suspension_period"],
        display_times,
    )
    with display_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(
            ["nominal_suspension_time"]
            + [f"z{i}" for i in range(11)]
            + [f"dz{i}" for i in range(11)]
        ) + "\n")
        for time, position, tangent in zip(
            display_times, display_positions, display_tangents
        ):
            handle.write(",".join([
                f"{time:.17g}",
                *[f"{value:.17g}" for value in position],
                *[f"{value:.17g}" for value in tangent],
            ]) + "\n")
    write_json(certificate_path, case["certificate_document"])
    return {
        "positions": binding_record(positions_path, case_root.parents[1]),
        "tangents": binding_record(tangents_path, case_root.parents[1]),
        "display": {
            "kind": "one_full_fourier_closed_orbit",
            **binding_record(display_path, case_root.parents[1]),
            "n_rows": len(display_times),
        },
        "fourier": {
            **binding_record(coefficients_path, case_root.parents[1]),
            "harmonic_cutoff": case["harmonic_cutoff"],
        },
        "certificate": binding_record(certificate_path, case_root.parents[1]),
    }


def write_chaos_files(case_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    positions_path = case_root / "analysis_positions.csv"
    tangents_path = case_root / "analysis_tangents.csv"
    display_path = case_root / "display_segment.csv"
    certificate_path = case_root / "certificate.json"
    save_matrix(positions_path, case["positions_array"])
    save_matrix(tangents_path, case["tangents_array"])
    display_rows = int(round(50.0 / ANALYSIS_DT)) + 1
    start = len(case["positions_array"]) - display_rows
    display_times = np.arange(display_rows, dtype=float) * ANALYSIS_DT
    with display_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(
            ["nominal_suspension_time"]
            + [f"z{i}" for i in range(11)]
            + [f"vtheta{i}" for i in range(11)]
        ) + "\n")
        for time, position, tangent in zip(
            display_times,
            case["positions_array"][start:],
            case["tangents_array"][start:],
        ):
            handle.write(",".join([
                f"{time:.17g}",
                *[f"{value:.17g}" for value in position],
                *[f"{value:.17g}" for value in tangent],
            ]) + "\n")
    write_json(certificate_path, case["certificate_document"])
    return {
        "positions": binding_record(positions_path, case_root.parents[1]),
        "tangents": binding_record(tangents_path, case_root.parents[1]),
        "display": {
            "kind": "late_50_unit_nonperiodic_segment",
            **binding_record(display_path, case_root.parents[1]),
            "n_rows": display_rows,
        },
        "fourier": None,
        "certificate": binding_record(certificate_path, case_root.parents[1]),
    }


def materialize(prepared: list[dict[str, Any]]) -> Path:
    validate_output_absent()
    stage = Path(tempfile.mkdtemp(prefix=".compass-v3-refined-stage-", dir=OUTPUT_PARENT))
    try:
        case_records: list[dict[str, Any]] = []
        for case in prepared:
            print(f"Writing {case['id']}", flush=True)
            case_root = stage / "cases" / case["id"]
            case_root.mkdir(parents=True)
            if case["q"] is None:
                files = write_chaos_files(case_root, case)
            else:
                files = write_periodic_files(case_root, case)
            case_records.append({
                key: case[key]
                for key in (
                    "id",
                    "kind",
                    "phi_deg",
                    "q",
                    "nominal_suspension_period",
                    "physical_return_seconds",
                    "dimension",
                    "analysis_sample_dt",
                    "tangent_semantics",
                    "global_curve_bound",
                )
            } | {
                "n_samples": case["analysis_n_samples"],
                "duration": case["analysis_duration"],
            } | files)

        summary_root = stage / "summary"
        summary_root.mkdir()
        summary_path = summary_root / "cases.csv"
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write(
                "case_id,kind,phi_deg,q,continuous_nominal_period,physical_return_seconds,"
                "harmonic_cutoff,position_h,tangent_direction_h,C_times_tangent_h,global_h,"
                "minimum_window_recurrence,first_exact_alignment_multiple,"
                "first_exact_alignment_lag_samples,tangent_semantics\n"
            )
            for case in prepared:
                handle.write(",".join([
                    case["id"],
                    case["kind"],
                    f"{case['phi_deg']:.17g}",
                    "" if case["q"] is None else str(case["q"]),
                    "" if case["nominal_suspension_period"] is None
                    else f"{case['nominal_suspension_period']:.17g}",
                    "" if case["physical_return_seconds"] is None
                    else f"{case['physical_return_seconds']:.17g}",
                    "" if case["harmonic_cutoff"] is None else str(case["harmonic_cutoff"]),
                    f"{case['position_curve_component']:.17g}",
                    f"{case['normalized_tangent_curve_component']:.17g}",
                    f"{case['metric_scaled_tangent_curve_component']:.17g}",
                    f"{case['global_curve_bound']:.17g}",
                    "" if case["minimum_planned_window_recurrence_distance"] is None
                    else f"{case['minimum_planned_window_recurrence_distance']:.17g}",
                    "" if case["exact_alignment_multiple"] is None
                    else str(case["exact_alignment_multiple"]),
                    "" if case["exact_alignment_lag_samples"] is None
                    else str(case["exact_alignment_lag_samples"]),
                    case["tangent_semantics"],
                ]) + "\n")

        inventory = [
            file_record(path, stage)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        ]
        manifest = {
            "schema_version": 3,
            "bundle_id": "compass_embedded_fourier_orbits_v3_refined",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generator": {
                "path": code_relative(SCRIPT_PATH),
                "sha256": sha256(SCRIPT_PATH),
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
            },
            "scientific_scope": {
                "periodic_selection": "32 late q-impact cycles in one bridge-to-arc phase class",
                "periodic_period": "OLS slope through the 33 bounding q-spaced start indices",
                "periodic_positions": "same-phase average, Fourier closed at H=6q",
                "periodic_tangents": "analytic derivative of fitted Fourier curve",
                "chaos_positions": "linear quarter-step interpolation of frozen encoded path",
                "chaos_tangents": "linear quarter-step interpolation of V_theta(z); mixed semantics",
                "not_a_learned_rollout": True,
                "no_signatures_in_bundle": True,
            },
            "analysis_sample_dt": ANALYSIS_DT,
            "analysis_duration": ANALYSIS_DURATION,
            "analysis_n_samples": ANALYSIS_ROWS,
            "dimension": 11,
            "metric_c_preflight": METRIC_C,
            "maximum_global_curve_bound": max(
                case["global_curve_bound"] for case in prepared
            ),
            "recurrence_policy": {
                "machine_zero_threshold_dynamic_distance": MACHINE_ZERO_THRESHOLD,
                "max_planned_window_duration": MAX_PLANNED_WINDOW_DURATION,
                "max_planned_window_lag_samples": MAX_PLANNED_WINDOW_LAG,
                "scope": "full-period-multiple recurrence candidates inside planned windows",
                "global_469_unit_noncommensurability_claim": False,
            },
            "cases": case_records,
            "summary": binding_record(summary_path, stage),
            "files": inventory,
        }
        write_json(stage / "bundle_manifest.json", manifest)
        stage.rename(OUTPUT_ROOT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return OUTPUT_ROOT / "bundle_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "materialize"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_output_absent()
    prepared = prepare_all()
    diagnostics = public_diagnostics(prepared)
    if args.command == "check":
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
        print("CHECK ONLY: no files written")
        return
    manifest_path = materialize(prepared)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(f"Wrote immutable bundle: {OUTPUT_ROOT}")
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
