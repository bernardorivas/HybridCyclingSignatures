#!/usr/bin/env python3
"""Scientific acceptance summary for the completed refined Compass v3 run.

This is a read-only postprocessor.  It validates the immutable v3 plan and
raw results, reconstructs the same ``P(rank > 0)`` statistic from the stored
birth vectors, and writes one new versioned summary directory.  It never
recomputes a cycling signature and never modifies v1/v2 summaries or results.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
HERE = SCRIPT_PATH.parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import compass_probability_v3 as driver  # noqa: E402


EXPECTED_ANALYSIS_ID = "compass_refined_v3_probability_linf_C0p75"
EXPECTED_PLAN_SHA256 = (
    "3245bbd941debf5eebda6ba0ddf2ae618154095757563d644166d914f8eab45a"
)
EXPECTED_BUNDLE_SHA256 = (
    "371cfb4bf751ab6f4b04226dece7e1049fceb516db638fee44f111ada352b442"
)
EXPECTED_CASE_IDS = ("period1", "period2", "period4", "period8", "chaos")
PERIODIC_CASE_IDS = EXPECTED_CASE_IDS[:4]
DEFAULT_ROOT = (
    CODE_ROOT
    / "experiments_planned"
    / "outputs"
    / "shared_coauthor_protocol"
    / EXPECTED_ANALYSIS_ID
)
DEFAULT_PLAN = DEFAULT_ROOT / "plan.json"
SUMMARY_DIRNAME = "compact_summary_v2"
SUMMARY_ID = "compass_refined_v3_scientific_acceptance_summary_v2"
SUMMARY_SCHEMA_VERSION = 2
THRESHOLDS = (0.25, 0.50, 0.75)
SPLITS: dict[str, tuple[int, ...]] = {
    "pooled": tuple(range(1, 21)),
    "tune": tuple(range(1, 11)),
    "validate": tuple(range(11, 21)),
}
LOW_R_RADII = np.linspace(0.0, 0.02, 201)
ORDERING_RULE = "finite_strict_period1_lt_period2_lt_period4_lt_period8"


@dataclass(frozen=True)
class CaseData:
    job: dict[str, Any]
    binding: dict[str, Any]
    durations: np.ndarray
    radii: np.ndarray
    first_births: np.ndarray
    birth_vectors: tuple[tuple[tuple[float, ...], ...], ...]


@dataclass(frozen=True)
class LoadedAnalysis:
    root: Path
    plan_path: Path
    plan: dict[str, Any]
    cases: tuple[CaseData, ...]
    common_valid_mask: np.ndarray
    first_common_valid_radius: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("CSV values must be finite or absent")
        rounded = round(number, 14)
        if math.isclose(number, rounded, rel_tol=0.0, abs_tol=5e-15):
            number = rounded
        return repr(number)
    return value


def csv_bytes(fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> bytes:
    names = list(fieldnames)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=names, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(names):
            raise ValueError(
                "summary row schema mismatch: "
                f"missing={sorted(set(names)-set(row))}, "
                f"extra={sorted(set(row)-set(names))}"
            )
        writer.writerow({name: csv_scalar(row[name]) for name in names})
    return buffer.getvalue().encode("utf-8")


def matrix_csv_bytes(
    radii: np.ndarray, durations: np.ndarray, probability: np.ndarray
) -> bytes:
    if probability.shape != (len(radii), len(durations)):
        raise ValueError("probability matrix shape does not match grids")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["radius", *[csv_scalar(value) for value in durations]])
    for radius, row in zip(radii, probability):
        writer.writerow([csv_scalar(radius), *[csv_scalar(value) for value in row]])
    return buffer.getvalue().encode("utf-8")


def onset(
    durations: np.ndarray, probability: np.ndarray, threshold: float
) -> tuple[float | None, float | None]:
    if durations.ndim != 1 or probability.shape != durations.shape:
        raise ValueError("onset arrays must be aligned and one dimensional")
    if len(durations) == 0 or np.any(np.diff(durations) <= 0.0):
        raise ValueError("duration grid must be strictly increasing")
    if np.any(~np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("probabilities must lie in [0,1]")
    above = probability >= threshold - 1e-12
    first_indices = np.flatnonzero(above)
    sustained_flags = np.logical_and.accumulate(above[::-1])[::-1]
    sustained_indices = np.flatnonzero(sustained_flags)
    return (
        None if not len(first_indices) else float(durations[first_indices[0]]),
        None
        if not len(sustained_indices)
        else float(durations[sustained_indices[0]]),
    )


def finite_quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"p25": None, "p50": None, "p75": None}
    result = np.quantile(array, THRESHOLDS, method="linear")
    return {
        "p25": float(result[0]),
        "p50": float(result[1]),
        "p75": float(result[2]),
    }


def segment_lengths(protocol: dict[str, Any]) -> np.ndarray:
    specification = protocol["segment_lengths"]
    return np.arange(
        int(specification["start"]),
        int(specification["stop"]) + 1,
        int(specification["step"]),
        dtype=int,
    )


def load_births(
    job: dict[str, Any], plan: dict[str, Any]
) -> tuple[np.ndarray, tuple[tuple[tuple[float, ...], ...], ...]]:
    protocol = plan["protocol"]
    lengths = segment_lengths(protocol)
    n_runs = int(protocol["n_runs"])
    index_by_length = {int(value): index for index, value in enumerate(lengths)}
    vectors: list[list[tuple[float, ...] | None]] = [
        [None for _ in range(n_runs)] for _ in lengths
    ]
    path = driver.shared.result_paths(job)["births"]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "segment_length",
            "segment_duration",
            "run_index",
            "start_index",
            "end_index",
            "rank",
            "births",
        ]:
            raise ValueError(f"{job['id']}: unexpected births schema")
        for row in reader:
            length = int(row["segment_length"])
            run_index = int(row["run_index"])
            if length not in index_by_length or not 1 <= run_index <= n_runs:
                raise ValueError(f"{job['id']}: unexpected birth trial key")
            i = index_by_length[length]
            j = run_index - 1
            if vectors[i][j] is not None:
                raise ValueError(f"{job['id']}: duplicate birth trial")
            births = tuple(float(value) for value in row["births"].split(";") if value)
            if (
                len(births) != int(row["rank"])
                or births != tuple(sorted(births))
                or any(not math.isfinite(value) or value < 0.0 for value in births)
            ):
                raise ValueError(f"{job['id']}: invalid birth vector")
            expected_duration = length * float(protocol["effective_sample_dt"])
            if not math.isclose(
                float(row["segment_duration"]),
                expected_duration,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{job['id']}: duration disagrees with protocol")
            vectors[i][j] = births
    if any(item is None for row in vectors for item in row):
        raise ValueError(f"{job['id']}: incomplete birth table")
    frozen_vectors = tuple(
        tuple(item if item is not None else () for item in row) for row in vectors
    )
    first = np.asarray(
        [
            [min(item, default=math.inf) for item in row]
            for row in frozen_vectors
        ],
        dtype=float,
    )
    return first, frozen_vectors


def probability_matrix(
    case: CaseData, radii: np.ndarray, run_indices: tuple[int, ...]
) -> np.ndarray:
    columns = np.asarray(run_indices, dtype=int) - 1
    if np.any(columns < 0) or np.any(columns >= case.first_births.shape[1]):
        raise ValueError("split run indices are outside the stored trials")
    return np.mean(
        case.first_births[:, columns][np.newaxis, :, :]
        <= radii[:, np.newaxis, np.newaxis],
        axis=2,
    )


def verify_plan(plan_path: Path, julia_bin: str) -> LoadedAnalysis:
    plan_path = plan_path.resolve()
    if sha256(plan_path) != EXPECTED_PLAN_SHA256:
        raise ValueError("summarizer is bound to another immutable plan")
    root, plan = driver.load_plan(plan_path, julia_bin)
    if plan.get("analysis_id") != EXPECTED_ANALYSIS_ID:
        raise ValueError("refusing an unrelated analysis")
    if plan.get("bundle_manifest_sha256") != EXPECTED_BUNDLE_SHA256:
        raise ValueError("refined bundle binding changed")
    if plan_path != root / "plan.json" or root != DEFAULT_ROOT:
        raise ValueError("plan is outside the canonical analysis root")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or tuple(job.get("id") for job in jobs) != EXPECTED_CASE_IDS:
        raise ValueError("expected ordered period1/2/4/8/chaos jobs")
    protocol = plan["protocol"]
    if (
        protocol.get("statistic") != "P(rank > 0)"
        or int(protocol.get("n_runs")) != 20
        or not math.isclose(float(protocol.get("r_min")), 0.0, abs_tol=1e-15)
        or not math.isclose(float(protocol.get("r_max")), 0.5, abs_tol=1e-15)
        or int(protocol.get("r_subdivisions")) != 251
    ):
        raise ValueError("scientific summary protocol changed")
    lengths = segment_lengths(protocol)
    durations = lengths.astype(float) * float(protocol["effective_sample_dt"])
    radii = np.linspace(
        float(protocol["r_min"]),
        float(protocol["r_max"]),
        int(protocol["r_subdivisions"]),
    )
    cases: list[CaseData] = []
    starts_hash: str | None = None
    for job in jobs:
        binding = driver.validate_binding(job, plan)
        current_starts = binding["raw_results"]["starts"]["sha256"]
        if starts_hash is None:
            starts_hash = current_starts
        elif starts_hash != current_starts:
            raise ValueError("paired-start files are not byte-identical")
        first_births, vectors = load_births(job, plan)
        case = CaseData(
            job=job,
            binding=binding,
            durations=durations.copy(),
            radii=radii.copy(),
            first_births=first_births,
            birth_vectors=vectors,
        )
        pooled = probability_matrix(case, radii, SPLITS["pooled"])
        stored_durations, stored_radii, stored_probability = (
            driver.shared.read_probability_matrix(job, protocol)
        )
        if (
            not np.array_equal(durations, stored_durations)
            or not np.array_equal(radii, stored_radii)
            or not np.allclose(pooled, stored_probability, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"{job['id']}: exact births disagree with stored heatmap")
        cases.append(case)
    maximum_bound = max(float(case.job["global_curve_bound"]) for case in cases)
    metric_c = float(protocol["metric_c"])
    common_mask = (radii > maximum_bound) & (radii < metric_c)
    common_indices = np.flatnonzero(common_mask)
    if not len(common_indices):
        raise ValueError("no sampled common curve-resolved r<C radius")
    return LoadedAnalysis(
        root=root,
        plan_path=plan_path,
        plan=plan,
        cases=tuple(cases),
        common_valid_mask=common_mask,
        first_common_valid_radius=float(radii[common_indices[0]]),
    )


def split_birth_values(
    case: CaseData, run_indices: tuple[int, ...]
) -> tuple[list[float], list[float], int, int]:
    columns = [index - 1 for index in run_indices]
    first_values: list[float] = []
    all_values: list[float] = []
    zero_trials = 0
    zero_entries = 0
    for row in case.birth_vectors:
        for column in columns:
            vector = row[column]
            all_values.extend(vector)
            if vector:
                first_values.append(vector[0])
            if any(value <= driver.RECURRENCE_TOLERANCE for value in vector):
                zero_trials += 1
            zero_entries += sum(
                value <= driver.RECURRENCE_TOLERANCE for value in vector
            )
    return first_values, all_values, zero_trials, zero_entries


def case_split_rows(loaded: LoadedAnalysis) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in loaded.cases:
        valid_indices = np.flatnonzero(
            (case.radii > float(case.job["global_curve_bound"]))
            & (case.radii < float(loaded.plan["protocol"]["metric_c"]))
        )
        if not len(valid_indices):
            raise ValueError(f"{case.job['id']}: no per-case valid radius")
        valid_index = int(valid_indices[0])
        for split, run_indices in SPLITS.items():
            probability = probability_matrix(case, case.radii, run_indices)
            first_values, all_values, zero_trials, zero_entries = split_birth_values(
                case, run_indices
            )
            first_quantiles = finite_quantiles(first_values)
            all_quantiles = finite_quantiles(all_values)
            row: dict[str, Any] = {
                "case_id": case.job["id"],
                "split": split,
                "run_indices": ";".join(str(value) for value in run_indices),
                "n_runs": len(run_indices),
                "phi_deg": float(case.job["phi_deg"]),
                "q": case.job["q"],
                "refined_continuous_period": case.job["refined_continuous_period"],
                "global_curve_bound_h": float(case.job["global_curve_bound"]),
                "first_case_valid_radius": float(case.radii[valid_index]),
                "first_common_valid_radius": loaded.first_common_valid_radius,
                "trial_count": len(case.durations) * len(run_indices),
                "finite_first_birth_count": len(first_values),
                "empty_birth_trial_count": len(case.durations) * len(run_indices)
                - len(first_values),
                "all_birth_entry_count": len(all_values),
                "zero_or_near_zero_birth_trial_count": zero_trials,
                "zero_or_near_zero_birth_entry_count": zero_entries,
                "finite_first_birth_p25": first_quantiles["p25"],
                "finite_first_birth_p50": first_quantiles["p50"],
                "finite_first_birth_p75": first_quantiles["p75"],
                "all_finite_birth_p25": all_quantiles["p25"],
                "all_finite_birth_p50": all_quantiles["p50"],
                "all_finite_birth_p75": all_quantiles["p75"],
            }
            for label, index in (("r0", 0), ("first_valid", valid_index)):
                for threshold in THRESHOLDS:
                    threshold_label = f"p{int(round(100 * threshold))}"
                    first, sustained = onset(
                        case.durations, probability[index], threshold
                    )
                    row[f"{label}_{threshold_label}_first"] = first
                    row[f"{label}_{threshold_label}_sustained"] = sustained
            rows.append(row)
    return rows


def onset_rows(
    loaded: LoadedAnalysis,
    probabilities: dict[tuple[str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_c = float(loaded.plan["protocol"]["metric_c"])
    for case in loaded.cases:
        first_case_radius = float(
            case.radii[
                np.flatnonzero(
                    (case.radii > float(case.job["global_curve_bound"]))
                    & (case.radii < metric_c)
                )[0]
            ]
        )
        for split in SPLITS:
            probability = probabilities[(split, case.job["id"])]
            for radius_index, radius in enumerate(case.radii):
                for threshold in THRESHOLDS:
                    first, sustained = onset(
                        case.durations, probability[radius_index], threshold
                    )
                    rows.append(
                        {
                            "case_id": case.job["id"],
                            "split": split,
                            "radius_index": radius_index,
                            "radius": float(radius),
                            "at_r0": radius_index == 0,
                            "case_curve_resolved_r_lt_c": bool(
                                radius > float(case.job["global_curve_bound"])
                                and radius < metric_c
                            ),
                            "at_first_case_valid_radius": math.isclose(
                                float(radius), first_case_radius, abs_tol=1e-14
                            ),
                            "common_curve_resolved_r_lt_c": bool(
                                loaded.common_valid_mask[radius_index]
                            ),
                            "probability_threshold": threshold,
                            "first_crossing_duration": first,
                            "sustained_onset_duration": sustained,
                        }
                    )
    return rows


def birth_quantile_rows(loaded: LoadedAnalysis) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in loaded.cases:
        for split, run_indices in SPLITS.items():
            columns = [value - 1 for value in run_indices]
            for duration_index, duration in enumerate(case.durations):
                selected = [case.birth_vectors[duration_index][column] for column in columns]
                first_values = [vector[0] for vector in selected if vector]
                all_values = [value for vector in selected for value in vector]
                first_quantiles = finite_quantiles(first_values)
                all_quantiles = finite_quantiles(all_values)
                rows.append(
                    {
                        "case_id": case.job["id"],
                        "split": split,
                        "segment_duration": float(duration),
                        "n_runs": len(run_indices),
                        "finite_first_birth_count": len(first_values),
                        "empty_birth_trial_count": len(run_indices) - len(first_values),
                        "all_birth_entry_count": len(all_values),
                        "zero_or_near_zero_birth_trial_count": sum(
                            any(value <= driver.RECURRENCE_TOLERANCE for value in vector)
                            for vector in selected
                        ),
                        "zero_or_near_zero_birth_entry_count": sum(
                            value <= driver.RECURRENCE_TOLERANCE
                            for vector in selected
                            for value in vector
                        ),
                        "finite_first_birth_p25": first_quantiles["p25"],
                        "finite_first_birth_p50": first_quantiles["p50"],
                        "finite_first_birth_p75": first_quantiles["p75"],
                        "all_finite_birth_p25": all_quantiles["p25"],
                        "all_finite_birth_p50": all_quantiles["p50"],
                        "all_finite_birth_p75": all_quantiles["p75"],
                    }
                )
    return rows


def contiguous_bands(mask: np.ndarray, radii: np.ndarray) -> list[dict[str, Any]]:
    if mask.shape != radii.shape:
        raise ValueError("band mask and radius grid differ")
    bands: list[dict[str, Any]] = []
    start: int | None = None
    for index, selected in enumerate(mask):
        if selected and start is None:
            start = index
        at_end = index == len(mask) - 1
        if start is not None and ((not selected) or at_end):
            end = index if selected and at_end else index - 1
            bands.append(
                {
                    "start_radius_index": start,
                    "end_radius_index": end,
                    "start_radius_center": float(radii[start]),
                    "end_radius_center": float(radii[end]),
                    "radius_count": end - start + 1,
                }
            )
            start = None
    return bands


def ordering_rows_and_bands(
    loaded: LoadedAnalysis,
    probabilities: dict[tuple[str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case_by_id = {case.job["id"]: case for case in loaded.cases}
    radii = loaded.cases[0].radii
    flag_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    masks: dict[tuple[str, float, str], np.ndarray] = {}
    for split in SPLITS:
        for threshold in THRESHOLDS:
            for onset_kind in ("first", "sustained"):
                strict_mask = np.zeros(len(radii), dtype=bool)
                for radius_index in np.flatnonzero(loaded.common_valid_mask):
                    values: list[float | None] = []
                    for case_id in PERIODIC_CASE_IDS:
                        case = case_by_id[case_id]
                        first, sustained = onset(
                            case.durations,
                            probabilities[(split, case_id)][radius_index],
                            threshold,
                        )
                        values.append(first if onset_kind == "first" else sustained)
                    finite = all(value is not None for value in values)
                    weak = bool(
                        finite
                        and all(
                            float(left) <= float(right)
                            for left, right in zip(values, values[1:])
                        )
                    )
                    strict = bool(
                        finite
                        and all(
                            float(left) < float(right)
                            for left, right in zip(values, values[1:])
                        )
                    )
                    strict_mask[radius_index] = strict
                    flag_rows.append(
                        {
                            "split": split,
                            "probability_threshold": threshold,
                            "onset_kind": onset_kind,
                            "radius_index": int(radius_index),
                            "radius": float(radii[radius_index]),
                            "period1_onset": values[0],
                            "period2_onset": values[1],
                            "period4_onset": values[2],
                            "period8_onset": values[3],
                            "all_finite": finite,
                            "weakly_ordered": weak,
                            "strictly_ordered": strict,
                        }
                    )
                masks[(split, threshold, onset_kind)] = strict_mask
                bands = contiguous_bands(strict_mask, radii)
                if not bands:
                    band_rows.append(
                        {
                            "scope": split,
                            "probability_threshold": threshold,
                            "onset_kind": onset_kind,
                            "ordering_rule": ORDERING_RULE,
                            "band_index": 0,
                            "start_radius_index": None,
                            "end_radius_index": None,
                            "start_radius_center": None,
                            "end_radius_center": None,
                            "radius_count": 0,
                        }
                    )
                for band_index, band in enumerate(bands, 1):
                    band_rows.append(
                        {
                            "scope": split,
                            "probability_threshold": threshold,
                            "onset_kind": onset_kind,
                            "ordering_rule": ORDERING_RULE,
                            "band_index": band_index,
                            **band,
                        }
                    )
    intersections: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        for onset_kind in ("first", "sustained"):
            for scope, members in (
                ("tune_and_validate", ("tune", "validate")),
                ("pooled_tune_and_validate", tuple(SPLITS)),
            ):
                mask = loaded.common_valid_mask.copy()
                for split in members:
                    mask &= masks[(split, threshold, onset_kind)]
                key = f"{scope}_p{int(threshold*100)}_{onset_kind}"
                bands = contiguous_bands(mask, radii)
                intersections[key] = bands
                if not bands:
                    band_rows.append(
                        {
                            "scope": scope,
                            "probability_threshold": threshold,
                            "onset_kind": onset_kind,
                            "ordering_rule": ORDERING_RULE,
                            "band_index": 0,
                            "start_radius_index": None,
                            "end_radius_index": None,
                            "start_radius_center": None,
                            "end_radius_center": None,
                            "radius_count": 0,
                        }
                    )
                for band_index, band in enumerate(bands, 1):
                    band_rows.append(
                        {
                            "scope": scope,
                            "probability_threshold": threshold,
                            "onset_kind": onset_kind,
                            "ordering_rule": ORDERING_RULE,
                            "band_index": band_index,
                            **band,
                        }
                    )
    for split in SPLITS:
        mask = loaded.common_valid_mask.copy()
        for threshold in THRESHOLDS:
            for onset_kind in ("first", "sustained"):
                mask &= masks[(split, threshold, onset_kind)]
        intersections[f"{split}_all_six_metrics"] = contiguous_bands(mask, radii)
    tune_validate_all = loaded.common_valid_mask.copy()
    for split in ("tune", "validate"):
        for threshold in THRESHOLDS:
            for onset_kind in ("first", "sustained"):
                tune_validate_all &= masks[(split, threshold, onset_kind)]
    intersections["tune_and_validate_all_six_metrics"] = contiguous_bands(
        tune_validate_all, radii
    )
    return flag_rows, band_rows, intersections


def build_files(loaded: LoadedAnalysis) -> tuple[dict[str, bytes], dict[str, Any]]:
    probabilities: dict[tuple[str, str], np.ndarray] = {}
    files: dict[str, bytes] = {}
    for split, run_indices in SPLITS.items():
        for case in loaded.cases:
            probability = probability_matrix(case, case.radii, run_indices)
            probabilities[(split, case.job["id"])] = probability
            files[f"probability_{split}_{case.job['id']}.csv"] = matrix_csv_bytes(
                case.radii, case.durations, probability
            )
    for case in loaded.cases:
        low_r_probability = probability_matrix(case, LOW_R_RADII, SPLITS["pooled"])
        files[f"probability_lowr_pooled_{case.job['id']}.csv"] = matrix_csv_bytes(
            LOW_R_RADII, case.durations, low_r_probability
        )

    case_rows = case_split_rows(loaded)
    case_fields = list(case_rows[0])
    files["case_split_metrics.csv"] = csv_bytes(case_fields, case_rows)

    detailed_onsets = onset_rows(loaded, probabilities)
    files["onsets_by_radius.csv"] = csv_bytes(
        list(detailed_onsets[0]), detailed_onsets
    )

    quantile_rows = birth_quantile_rows(loaded)
    files["birth_quantiles_by_duration.csv"] = csv_bytes(
        list(quantile_rows[0]), quantile_rows
    )

    common_rows = [
        {
            "radius_index": int(index),
            "radius": float(loaded.cases[0].radii[index]),
            "strictly_above_maximum_global_curve_bound": True,
            "strictly_below_metric_c": True,
        }
        for index in np.flatnonzero(loaded.common_valid_mask)
    ]
    files["common_valid_radii.csv"] = csv_bytes(list(common_rows[0]), common_rows)

    flag_rows, band_rows, intersections = ordering_rows_and_bands(
        loaded, probabilities
    )
    files["ordered_radius_flags.csv"] = csv_bytes(list(flag_rows[0]), flag_rows)
    files["ordered_bands.csv"] = csv_bytes(list(band_rows[0]), band_rows)
    return files, intersections


def input_records(loaded: LoadedAnalysis) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in loaded.cases:
        raw = driver.shared.result_paths(case.job)
        binding_path = driver.result_binding_path(case.job)
        records.append(
            {
                "case_id": case.job["id"],
                "result_binding": {
                    "path": str(binding_path.resolve()),
                    "sha256": sha256(binding_path),
                },
                "raw_results": {
                    name: {"path": str(path.resolve()), "sha256": sha256(path)}
                    for name, path in sorted(raw.items())
                },
                "display_extract": case.job["display_extract"],
                "global_curve_bound_h": float(case.job["global_curve_bound"]),
                "first_case_valid_radius": float(
                    case.radii[
                        np.flatnonzero(
                            (case.radii > float(case.job["global_curve_bound"]))
                            & (
                                case.radii
                                < float(loaded.plan["protocol"]["metric_c"])
                            )
                        )[0]
                    ]
                ),
            }
        )
    return records


def manifest_bytes(
    loaded: LoadedAnalysis,
    files: dict[str, bytes],
    intersections: dict[str, Any],
) -> bytes:
    inventory = [
        {
            "path": name,
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for name, value in sorted(files.items())
    ]
    maximum_bound = max(float(case.job["global_curve_bound"]) for case in loaded.cases)
    common_indices = np.flatnonzero(loaded.common_valid_mask)
    document = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "summary_id": SUMMARY_ID,
        "status": "complete",
        "scope": "validated_read_only_postprocessing_of_existing_v3_births",
        "analysis_id": loaded.plan["analysis_id"],
        "statistic": "P(rank > 0) = fraction of selected trials with first birth <= r",
        "no_signature_recomputation": True,
        "plan": {
            "path": str(loaded.plan_path),
            "sha256": sha256(loaded.plan_path),
        },
        "bundle_manifest": {
            "path": loaded.plan["bundle_manifest"],
            "sha256": loaded.plan["bundle_manifest_sha256"],
        },
        "frozen_v3_driver": {
            "path": str(driver.SCRIPT_PATH),
            "sha256": sha256(driver.SCRIPT_PATH),
        },
        "summarizer": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256(SCRIPT_PATH),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "splits": {
            name: {
                "run_indices_one_based": list(indices),
                "n_runs": len(indices),
            }
            for name, indices in SPLITS.items()
        },
        "split_policy": (
            "paired one-based run indices; tune=1..10, validate=11..20, "
            "pooled=1..20; fixed before inspecting case outcomes"
        ),
        "grids": {
            "computed": {
                "duration": {"minimum": 0.2, "maximum": 12.0, "step": 0.1, "count": 119},
                "radius": {"minimum": 0.0, "maximum": 0.5, "step": 0.002, "count": 251},
                "provenance": "exact grid stored by the signature kernel",
            },
            "low_radius_diagnostic": {
                "radius": {"minimum": 0.0, "maximum": 0.02, "step": 0.0001, "count": 201},
                "provenance": (
                    "postprocessed by exact thresholding of stored birth values; "
                    "no interpolation, smoothing, or signature rerun"
                ),
                "entirely_below_every_global_curve_bound_h": bool(
                    LOW_R_RADII[-1]
                    < min(float(case.job["global_curve_bound"]) for case in loaded.cases)
                ),
            },
        },
        "curve_resolution": {
            "rule": "strict h < r < metric C on sampled radii",
            "metric_c": float(loaded.plan["protocol"]["metric_c"]),
            "maximum_global_curve_bound_h": maximum_bound,
            "first_common_valid_radius": loaded.first_common_valid_radius,
            "last_common_valid_radius": float(loaded.cases[0].radii[common_indices[-1]]),
            "common_valid_radius_count": int(len(common_indices)),
            "common_valid_radii_csv": "common_valid_radii.csv",
        },
        "onset_definitions": {
            "thresholds": list(THRESHOLDS),
            "first": "earliest sampled duration with probability >= threshold",
            "sustained": (
                "earliest sampled duration with probability >= threshold at every "
                "remaining sampled duration through t=12; no extrapolation"
            ),
        },
        "birth_quantiles": {
            "method": "NumPy linear quantile over finite stored birth values",
            "first_birth": "minimum birth in each nonempty trial",
            "all_births": "every finite birth entry, including higher-rank births",
            "empty_trials": "reported separately and excluded from finite quantiles",
            "zero_or_near_zero_threshold": driver.RECURRENCE_TOLERANCE,
        },
        "ordering": {
            "cases": list(PERIODIC_CASE_IDS),
            "rule": ORDERING_RULE,
            "domain": (
                "sampled common-valid radii strictly above every case h and "
                "strictly below metric C"
            ),
            "bands": (
                "maximal runs of adjacent sampled radius centers satisfying the rule"
            ),
            "strict_band_intersections": intersections,
            "chaos_excluded_from_period_doubling_order": True,
        },
        "inputs": input_records(loaded),
        "files": inventory,
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build(loaded: LoadedAnalysis) -> tuple[dict[str, bytes], bytes]:
    files, intersections = build_files(loaded)
    manifest = manifest_bytes(loaded, files, intersections)
    return files, manifest


def check(loaded: LoadedAnalysis) -> None:
    files, manifest = build(loaded)
    print(f"summary_id={SUMMARY_ID}")
    print(f"plan_sha256={sha256(loaded.plan_path)}")
    print(f"file_count={len(files) + 2}")
    print(f"summary_manifest_sha256={hashlib.sha256(manifest).hexdigest()}")
    print(f"first_common_valid_radius={loaded.first_common_valid_radius}")
    print("status=complete_no_files_written")


def write(loaded: LoadedAnalysis) -> Path:
    output = loaded.root / SUMMARY_DIRNAME
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite summary: {output}")
    files, manifest = build(loaded)
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    files["summary_manifest.json"] = manifest
    files["summary_manifest.sha256"] = (
        f"{manifest_hash}  summary_manifest.json\n".encode("ascii")
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{SUMMARY_DIRNAME}.", dir=loaded.root))
    try:
        for name, content in sorted(files.items()):
            path = staging / name
            with path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    print(f"output={output}")
    print(f"summary_manifest_sha256={manifest_hash}")
    return output


def self_test() -> None:
    durations = np.asarray([0.2, 0.3, 0.4, 0.5])
    first, sustained = onset(durations, np.asarray([0.0, 0.5, 0.0, 0.5]), 0.5)
    if first != 0.3 or sustained != 0.5:
        raise AssertionError((first, sustained))
    first, sustained = onset(durations, np.asarray([0.0, 0.5, 0.5, 0.5]), 0.5)
    if first != 0.3 or sustained != 0.3:
        raise AssertionError((first, sustained))
    mask = np.asarray([False, True, True, False, True])
    bands = contiguous_bands(mask, np.arange(5, dtype=float))
    if bands != [
        {
            "start_radius_index": 1,
            "end_radius_index": 2,
            "start_radius_center": 1.0,
            "end_radius_center": 2.0,
            "radius_count": 2,
        },
        {
            "start_radius_index": 4,
            "end_radius_index": 4,
            "start_radius_center": 4.0,
            "end_radius_center": 4.0,
            "radius_count": 1,
        },
    ]:
        raise AssertionError(bands)
    print("Self-test passed; no files were written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "write", "self-test"), nargs="?", default="check")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--julia-bin", default="julia")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "self-test":
        self_test()
        return
    loaded = verify_plan(args.plan, args.julia_bin)
    if args.action == "check":
        check(loaded)
    else:
        write(loaded)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"ERROR: {error}") from error
