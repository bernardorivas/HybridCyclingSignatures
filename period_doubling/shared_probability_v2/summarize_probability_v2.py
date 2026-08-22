#!/usr/bin/env python3
"""Reproducible summaries of the validated David-family v2 probability grid.

The default ``check`` action is read-only.  It validates the immutable plan,
the certified orbit bundle, and every available result using the v2 analysis
driver.  ``write`` requires all five cases and creates one new, versioned
summary directory; it never modifies plans, raw signatures, figures, or the
manuscript.

The statistic summarized here is exactly the statistic produced by the
shared Julia kernel::

    P(rank > 0) = 1 - rank0 / 20

No smoothing, interpolation, aggregation over starts, or replacement
statistic is applied.
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

import roessler_probability_v2 as analysis


SCRIPT_PATH = Path(__file__).resolve()
HERE = SCRIPT_PATH.parent
CODE_ROOT = HERE.parents[1]
DEFAULT_ANALYSIS_ROOT = (
    CODE_ROOT
    / "experiments_planned"
    / "outputs"
    / "shared_coauthor_protocol"
    / "roessler_david_fourier_probability_linf_v2_david_grid"
)
DEFAULT_PLAN = DEFAULT_ANALYSIS_ROOT / "plan.json"
SUMMARY_DIRNAME = "probability_summary_v1"
SUMMARY_SCHEMA_VERSION = 1
SUMMARY_ID = "roessler_david_probability_summary_v1"
THRESHOLDS = (0.25, 0.50, 0.75)
EXPECTED_CASE_IDS = ("period1", "period2", "period4", "period8", "chaos")


@dataclass(frozen=True)
class CaseData:
    job: dict[str, Any]
    durations: np.ndarray
    radii: np.ndarray
    probability: np.ndarray
    metadata: dict[str, str]
    beta1: int
    curve_bound: float
    certified_period: float | None
    certificate_sha256: str


@dataclass(frozen=True)
class LoadedAnalysis:
    root: Path
    plan_path: Path
    plan: dict[str, Any]
    protocol: dict[str, Any]
    cases: tuple[CaseData, ...]
    missing_case_ids: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def optional_float(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        raise ValueError("summary values must be finite or absent")
    # Normalize only floating-point noise on the exact 0.025/0.2 protocol
    # grids, while retaining full precision for bounds, periods, and errors.
    rounded = round(float(value), 12)
    if math.isclose(float(value), rounded, rel_tol=0.0, abs_tol=5e-15):
        value = rounded
    # Python's shortest round-trip representation is deterministic.
    return repr(float(value))


def csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.floating, float)):
        return optional_float(float(value))
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return value


def csv_bytes(fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> bytes:
    names = list(fieldnames)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=names,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(names):
            raise ValueError(
                "summary row schema mismatch: "
                f"missing={sorted(set(names) - set(row))}, "
                f"extra={sorted(set(row) - set(names))}"
            )
        writer.writerow({name: csv_scalar(row[name]) for name in names})
    return buffer.getvalue().encode("utf-8")


def onset(
    durations: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[float | None, float | None]:
    """Return first crossing and observed-suffix sustained onset.

    ``sustained`` means that the probability remains at or above the
    threshold through the end of the analyzed duration grid.  It is not an
    extrapolation beyond the 60-time-unit horizon.
    """

    durations = np.asarray(durations, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if durations.ndim != 1 or probabilities.shape != durations.shape:
        raise ValueError("onset inputs must be aligned one-dimensional arrays")
    if len(durations) == 0 or not np.all(np.diff(durations) > 0):
        raise ValueError("duration grid must be nonempty and strictly increasing")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probabilities must lie in [0,1]")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0,1]")

    above = probabilities >= threshold
    indices = np.flatnonzero(above)
    first = float(durations[indices[0]]) if len(indices) else None
    suffix_above = np.logical_and.accumulate(above[::-1])[::-1]
    sustained_indices = np.flatnonzero(suffix_above)
    sustained = (
        float(durations[sustained_indices[0]])
        if len(sustained_indices)
        else None
    )
    return first, sustained


def onset_difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else right - left


def period_error(value: float | None, period: float | None) -> float | None:
    return None if value is None or period is None else value - period


def period_ratio(value: float | None, period: float | None) -> float | None:
    return None if value is None or period is None else value / period


def result_is_complete(job: dict[str, Any]) -> bool:
    paths = analysis.result_paths(job)
    binding = analysis.result_binding_path(job)
    return all(path.is_file() and not path.is_symlink() for path in paths.values()) and (
        binding.is_file() and not binding.is_symlink()
    )


def load_certificate(job: dict[str, Any]) -> tuple[float | None, str]:
    certificate_path = Path(job["certificate"]["absolute_path"])
    expected_hash = str(job["certificate"]["sha256"])
    if sha256(certificate_path) != expected_hash:
        raise ValueError(f"certificate hash changed for {job['id']}")
    certificate = load_json(certificate_path)
    if certificate.get("case_id") != job["id"]:
        raise ValueError(f"certificate case mismatch for {job['id']}")
    if certificate.get("status") != "certified":
        raise ValueError(f"certificate is not certified for {job['id']}")
    if job["kind"] == "periodic":
        period = finite_float(certificate.get("period"), f"{job['id']} period")
        if period <= 0.0:
            raise ValueError(f"certified period must be positive for {job['id']}")
        if int(certificate.get("q")) != int(job["q"]):
            raise ValueError(f"certificate q mismatch for {job['id']}")
        return period, expected_hash
    if "period" in certificate:
        raise ValueError("chaos certificate unexpectedly defines a period")
    return None, expected_hash


def load_validated_analysis(plan_path: Path, julia_bin: str) -> LoadedAnalysis:
    root, plan, configuration, _ = analysis.load_plan(plan_path, julia_bin)
    plan_path = Path(plan_path).resolve()
    case_ids = tuple(str(job["id"]) for job in plan["jobs"])
    if case_ids != EXPECTED_CASE_IDS:
        raise ValueError(f"unexpected case order: {case_ids}")
    if configuration.protocol.get("statistic") != "P(rank > 0)":
        raise ValueError("summarizer only accepts the P(rank > 0) protocol")

    loaded: list[CaseData] = []
    missing: list[str] = []
    reference_durations: np.ndarray | None = None
    reference_radii: np.ndarray | None = None
    reference_starts_hash: str | None = None
    for job in plan["jobs"]:
        case_id = str(job["id"])
        if not result_is_complete(job):
            missing.append(case_id)
            continue
        durations, radii, probability, metadata = analysis.validate_result(
            job, configuration.protocol, plan
        )
        if probability.shape != (len(radii), len(durations)):
            raise ValueError(f"probability shape mismatch for {case_id}")
        if np.any(~np.isfinite(probability)) or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise ValueError(f"probability outside [0,1] for {case_id}")
        if reference_durations is None:
            reference_durations = durations
            reference_radii = radii
            reference_starts_hash = metadata["segment_starts_sha256"]
        else:
            if not np.array_equal(durations, reference_durations):
                raise ValueError(f"duration grid differs for {case_id}")
            if not np.array_equal(radii, reference_radii):
                raise ValueError(f"radius grid differs for {case_id}")
            if metadata["segment_starts_sha256"] != reference_starts_hash:
                raise ValueError(f"segment starts differ for {case_id}")

        beta1 = int(metadata["beta1_Y"])
        if beta1 < 0:
            raise ValueError(f"negative beta1_Y for {case_id}")
        curve_bound = finite_float(
            metadata["global_curve_bound"], f"{case_id} curve bound"
        )
        certified_period, certificate_sha = load_certificate(job)
        loaded.append(
            CaseData(
                job=job,
                durations=durations,
                radii=radii,
                probability=probability,
                metadata=metadata,
                beta1=beta1,
                curve_bound=curve_bound,
                certified_period=certified_period,
                certificate_sha256=certificate_sha,
            )
        )
    return LoadedAnalysis(
        root=root,
        plan_path=plan_path,
        plan=plan,
        protocol=configuration.protocol,
        cases=tuple(loaded),
        missing_case_ids=tuple(missing),
    )


def curve_resolved(case: CaseData, radius: float) -> bool:
    """Whether the filtration radius is strictly above the curve bound h."""

    return bool(radius > case.curve_bound)


def r_lt_metric_c(radius: float, metric_c: float) -> bool:
    return bool(radius < metric_c)


def curve_resolved_candidate(
    case: CaseData, radius: float, metric_c: float
) -> bool:
    """Numerical/theory candidate flag, not an r0 certificate."""

    return curve_resolved(case, radius) and r_lt_metric_c(radius, metric_c)


def first_candidate_radius(case: CaseData, metric_c: float) -> float:
    mask = (case.radii > case.curve_bound) & (case.radii < metric_c)
    candidates = case.radii[mask]
    if len(candidates) == 0:
        raise ValueError(f"no curve-resolved r<C candidate for {case.job['id']}")
    return float(candidates[0])


def case_summary_rows(loaded: LoadedAnalysis) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_c = float(loaded.protocol["metric_c"])
    for case in loaded.cases:
        resolved = case.radii > case.curve_bound
        below_c = case.radii < metric_c
        candidates = resolved & below_c
        nearest_duration: float | None = None
        nearest_error: float | None = None
        if case.certified_period is not None:
            index = int(np.argmin(np.abs(case.durations - case.certified_period)))
            nearest_duration = float(case.durations[index])
            nearest_error = nearest_duration - case.certified_period
        rows.append(
            {
                "case_id": case.job["id"],
                "kind": case.job["kind"],
                "a": float(case.job["a"]),
                "q": case.job["q"],
                "beta1_Y": case.beta1,
                "curve_bound": case.curve_bound,
                "metric_c": metric_c,
                "first_curve_resolved_radius": float(case.radii[resolved][0]),
                "last_curve_resolved_radius": float(case.radii[resolved][-1]),
                "curve_resolved_radius_count": int(np.sum(resolved)),
                "first_curve_resolved_r_lt_c_candidate_radius": float(
                    case.radii[candidates][0]
                ),
                "last_curve_resolved_r_lt_c_candidate_radius": float(
                    case.radii[candidates][-1]
                ),
                "curve_resolved_r_lt_c_candidate_radius_count": int(
                    np.sum(candidates)
                ),
                "r_lt_metric_c_radius_count": int(np.sum(below_c)),
                "radius_count": len(case.radii),
                "certified_r0_available": False,
                "certified_period": case.certified_period,
                "nearest_duration_to_period": nearest_duration,
                "nearest_duration_minus_period": nearest_error,
                "certificate_sha256": case.certificate_sha256,
            }
        )
    return rows


def onset_rows(loaded: LoadedAnalysis) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_c = float(loaded.protocol["metric_c"])
    for case in loaded.cases:
        for radius_index, radius in enumerate(case.radii):
            for threshold in THRESHOLDS:
                first, sustained = onset(
                    case.durations, case.probability[radius_index], threshold
                )
                rows.append(
                    {
                        "case_id": case.job["id"],
                        "kind": case.job["kind"],
                        "a": float(case.job["a"]),
                        "q": case.job["q"],
                        "beta1_Y": case.beta1,
                        "certified_period": case.certified_period,
                        "curve_bound": case.curve_bound,
                        "metric_c": metric_c,
                        "radius": float(radius),
                        "curve_resolved": curve_resolved(case, float(radius)),
                        "r_lt_metric_c": r_lt_metric_c(float(radius), metric_c),
                        "curve_resolved_r_lt_c_candidate": curve_resolved_candidate(
                            case, float(radius), metric_c
                        ),
                        "certified_r0_available": False,
                        "probability_threshold": threshold,
                        "first_crossing_duration": first,
                        "sustained_onset_duration": sustained,
                        "first_minus_certified_period": period_error(
                            first, case.certified_period
                        ),
                        "first_over_certified_period": period_ratio(
                            first, case.certified_period
                        ),
                        "sustained_minus_certified_period": period_error(
                            sustained, case.certified_period
                        ),
                        "sustained_over_certified_period": period_ratio(
                            sustained, case.certified_period
                        ),
                        "observed_duration_max": float(case.durations[-1]),
                    }
                )
    return rows


def expected_period_rows(loaded: LoadedAnalysis) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_c = float(loaded.protocol["metric_c"])
    for case in loaded.cases:
        if case.certified_period is None:
            continue
        nearest_index = int(np.argmin(np.abs(case.durations - case.certified_period)))
        nearest_duration = float(case.durations[nearest_index])
        for radius_index, radius in enumerate(case.radii):
            probability_at_nearest = float(case.probability[radius_index, nearest_index])
            for threshold in THRESHOLDS:
                first, sustained = onset(
                    case.durations, case.probability[radius_index], threshold
                )
                rows.append(
                    {
                        "case_id": case.job["id"],
                        "a": float(case.job["a"]),
                        "q": case.job["q"],
                        "beta1_Y": case.beta1,
                        "curve_bound": case.curve_bound,
                        "metric_c": metric_c,
                        "radius": float(radius),
                        "curve_resolved": curve_resolved(case, float(radius)),
                        "r_lt_metric_c": r_lt_metric_c(float(radius), metric_c),
                        "curve_resolved_r_lt_c_candidate": curve_resolved_candidate(
                            case, float(radius), metric_c
                        ),
                        "certified_r0_available": False,
                        "certified_period": case.certified_period,
                        "nearest_grid_duration": nearest_duration,
                        "nearest_grid_minus_period": nearest_duration
                        - case.certified_period,
                        "probability_at_nearest_grid_duration": probability_at_nearest,
                        "probability_threshold": threshold,
                        "first_crossing_duration": first,
                        "first_minus_period": period_error(
                            first, case.certified_period
                        ),
                        "first_over_period": period_ratio(first, case.certified_period),
                        "sustained_onset_duration": sustained,
                        "sustained_minus_period": period_error(
                            sustained, case.certified_period
                        ),
                        "sustained_over_period": period_ratio(
                            sustained, case.certified_period
                        ),
                    }
                )
    return rows


def pairwise_probability_rows(loaded: LoadedAnalysis) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_c = float(loaded.protocol["metric_c"])
    for left_index, left in enumerate(loaded.cases):
        for right in loaded.cases[left_index + 1 :]:
            if not np.array_equal(left.durations, right.durations) or not np.array_equal(
                left.radii, right.radii
            ):
                raise ValueError("pairwise grids are not aligned")
            for radius_index, radius in enumerate(left.radii):
                delta = right.probability[radius_index] - left.probability[radius_index]
                absolute = np.abs(delta)
                maximum_index = int(np.argmax(absolute))
                rows.append(
                    {
                        "case_from": left.job["id"],
                        "case_to": right.job["id"],
                        "radius": float(radius),
                        "from_curve_resolved": curve_resolved(left, float(radius)),
                        "to_curve_resolved": curve_resolved(right, float(radius)),
                        "both_curve_resolved": curve_resolved(left, float(radius))
                        and curve_resolved(right, float(radius)),
                        "r_lt_metric_c": r_lt_metric_c(float(radius), metric_c),
                        "both_curve_resolved_r_lt_c_candidate": (
                            curve_resolved(left, float(radius))
                            and curve_resolved(right, float(radius))
                            and r_lt_metric_c(float(radius), metric_c)
                        ),
                        "certified_r0_available": False,
                        "mean_signed_probability_delta_to_minus_from": float(
                            np.mean(delta)
                        ),
                        "mean_absolute_probability_difference": float(
                            np.mean(absolute)
                        ),
                        "rms_probability_difference": float(
                            np.sqrt(np.mean(delta * delta))
                        ),
                        "maximum_absolute_probability_difference": float(
                            absolute[maximum_index]
                        ),
                        "duration_at_maximum_absolute_difference": float(
                            left.durations[maximum_index]
                        ),
                        "signed_delta_at_maximum_absolute_difference": float(
                            delta[maximum_index]
                        ),
                    }
                )
    return rows


def pairwise_onset_rows(loaded: LoadedAnalysis) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_c = float(loaded.protocol["metric_c"])
    for left_index, left in enumerate(loaded.cases):
        for right in loaded.cases[left_index + 1 :]:
            for radius_index, radius in enumerate(left.radii):
                for threshold in THRESHOLDS:
                    left_first, left_sustained = onset(
                        left.durations, left.probability[radius_index], threshold
                    )
                    right_first, right_sustained = onset(
                        right.durations, right.probability[radius_index], threshold
                    )
                    rows.append(
                        {
                            "case_from": left.job["id"],
                            "case_to": right.job["id"],
                            "radius": float(radius),
                            "from_curve_resolved": curve_resolved(
                                left, float(radius)
                            ),
                            "to_curve_resolved": curve_resolved(
                                right, float(radius)
                            ),
                            "both_curve_resolved": curve_resolved(
                                left, float(radius)
                            )
                            and curve_resolved(right, float(radius)),
                            "r_lt_metric_c": r_lt_metric_c(
                                float(radius), metric_c
                            ),
                            "both_curve_resolved_r_lt_c_candidate": (
                                curve_resolved(left, float(radius))
                                and curve_resolved(right, float(radius))
                                and r_lt_metric_c(float(radius), metric_c)
                            ),
                            "certified_r0_available": False,
                            "probability_threshold": threshold,
                            "from_first_crossing_duration": left_first,
                            "to_first_crossing_duration": right_first,
                            "first_crossing_delta_to_minus_from": onset_difference(
                                left_first, right_first
                            ),
                            "from_sustained_onset_duration": left_sustained,
                            "to_sustained_onset_duration": right_sustained,
                            "sustained_onset_delta_to_minus_from": onset_difference(
                                left_sustained, right_sustained
                            ),
                        }
                    )
    return rows


def best_period_match(
    case: CaseData, *, candidate_only: bool, sustained: bool, metric_c: float
) -> dict[str, Any] | None:
    if case.certified_period is None:
        return None
    candidates: list[tuple[float, float, float]] = []
    for radius_index, radius in enumerate(case.radii):
        if candidate_only and not curve_resolved_candidate(
            case, float(radius), metric_c
        ):
            continue
        first, sustained_value = onset(
            case.durations, case.probability[radius_index], 0.50
        )
        value = sustained_value if sustained else first
        if value is not None:
            candidates.append(
                (abs(value - case.certified_period), float(radius), value)
            )
    if not candidates:
        return None
    _, radius, value = min(candidates)
    return {
        "radius": radius,
        "curve_resolved": curve_resolved(case, radius),
        "r_lt_metric_c": r_lt_metric_c(radius, metric_c),
        "curve_resolved_r_lt_c_candidate": curve_resolved_candidate(
            case, radius, metric_c
        ),
        "certified_r0_available": False,
        "onset_duration": value,
        "minus_certified_period": value - case.certified_period,
        "over_certified_period": value / case.certified_period,
    }


def selected_findings(loaded: LoadedAnalysis) -> dict[str, Any]:
    findings: dict[str, Any] = {}
    metric_c = float(loaded.protocol["metric_c"])
    for case in loaded.cases:
        first_radius = first_candidate_radius(case, metric_c)
        first_candidate_index = int(
            np.flatnonzero(
                (case.radii > case.curve_bound) & (case.radii < metric_c)
            )[0]
        )
        p50_first, p50_sustained = onset(
            case.durations, case.probability[first_candidate_index], 0.50
        )
        findings[str(case.job["id"])] = {
            "beta1_Y": case.beta1,
            "curve_bound": case.curve_bound,
            "metric_c": metric_c,
            "certified_r0_available": False,
            "first_curve_resolved_r_lt_c_candidate_radius": first_radius,
            "p50_at_first_curve_resolved_r_lt_c_candidate_radius": {
                "first_crossing_duration": p50_first,
                "sustained_onset_duration": p50_sustained,
            },
            "p50_best_period_match_all_radii_first_crossing": best_period_match(
                case,
                candidate_only=False,
                sustained=False,
                metric_c=metric_c,
            ),
            "p50_best_period_match_candidate_radii_first_crossing": best_period_match(
                case,
                candidate_only=True,
                sustained=False,
                metric_c=metric_c,
            ),
            "p50_best_period_match_all_radii_sustained_onset": best_period_match(
                case,
                candidate_only=False,
                sustained=True,
                metric_c=metric_c,
            ),
            "p50_best_period_match_candidate_radii_sustained_onset": best_period_match(
                case,
                candidate_only=True,
                sustained=True,
                metric_c=metric_c,
            ),
        }
    return findings


def build_summary_files(loaded: LoadedAnalysis) -> dict[str, bytes]:
    if loaded.missing_case_ids:
        raise ValueError(
            "cannot summarize incomplete analysis; missing "
            + ",".join(loaded.missing_case_ids)
        )
    if tuple(case.job["id"] for case in loaded.cases) != EXPECTED_CASE_IDS:
        raise ValueError("all five ordered cases are required")

    tables: dict[str, tuple[list[str], list[dict[str, Any]]]] = {
        "case_summary.csv": (
            [
                "case_id",
                "kind",
                "a",
                "q",
                "beta1_Y",
                "curve_bound",
                "metric_c",
                "first_curve_resolved_radius",
                "last_curve_resolved_radius",
                "curve_resolved_radius_count",
                "first_curve_resolved_r_lt_c_candidate_radius",
                "last_curve_resolved_r_lt_c_candidate_radius",
                "curve_resolved_r_lt_c_candidate_radius_count",
                "r_lt_metric_c_radius_count",
                "radius_count",
                "certified_r0_available",
                "certified_period",
                "nearest_duration_to_period",
                "nearest_duration_minus_period",
                "certificate_sha256",
            ],
            case_summary_rows(loaded),
        ),
        "onsets_by_radius.csv": (
            [
                "case_id",
                "kind",
                "a",
                "q",
                "beta1_Y",
                "certified_period",
                "curve_bound",
                "metric_c",
                "radius",
                "curve_resolved",
                "r_lt_metric_c",
                "curve_resolved_r_lt_c_candidate",
                "certified_r0_available",
                "probability_threshold",
                "first_crossing_duration",
                "sustained_onset_duration",
                "first_minus_certified_period",
                "first_over_certified_period",
                "sustained_minus_certified_period",
                "sustained_over_certified_period",
                "observed_duration_max",
            ],
            onset_rows(loaded),
        ),
        "expected_period_comparisons_by_radius.csv": (
            [
                "case_id",
                "a",
                "q",
                "beta1_Y",
                "curve_bound",
                "metric_c",
                "radius",
                "curve_resolved",
                "r_lt_metric_c",
                "curve_resolved_r_lt_c_candidate",
                "certified_r0_available",
                "certified_period",
                "nearest_grid_duration",
                "nearest_grid_minus_period",
                "probability_at_nearest_grid_duration",
                "probability_threshold",
                "first_crossing_duration",
                "first_minus_period",
                "first_over_period",
                "sustained_onset_duration",
                "sustained_minus_period",
                "sustained_over_period",
            ],
            expected_period_rows(loaded),
        ),
        "pairwise_probability_differences_by_radius.csv": (
            [
                "case_from",
                "case_to",
                "radius",
                "from_curve_resolved",
                "to_curve_resolved",
                "both_curve_resolved",
                "r_lt_metric_c",
                "both_curve_resolved_r_lt_c_candidate",
                "certified_r0_available",
                "mean_signed_probability_delta_to_minus_from",
                "mean_absolute_probability_difference",
                "rms_probability_difference",
                "maximum_absolute_probability_difference",
                "duration_at_maximum_absolute_difference",
                "signed_delta_at_maximum_absolute_difference",
            ],
            pairwise_probability_rows(loaded),
        ),
        "pairwise_onset_differences_by_radius.csv": (
            [
                "case_from",
                "case_to",
                "radius",
                "from_curve_resolved",
                "to_curve_resolved",
                "both_curve_resolved",
                "r_lt_metric_c",
                "both_curve_resolved_r_lt_c_candidate",
                "certified_r0_available",
                "probability_threshold",
                "from_first_crossing_duration",
                "to_first_crossing_duration",
                "first_crossing_delta_to_minus_from",
                "from_sustained_onset_duration",
                "to_sustained_onset_duration",
                "sustained_onset_delta_to_minus_from",
            ],
            pairwise_onset_rows(loaded),
        ),
    }
    return {
        name: csv_bytes(fieldnames, rows)
        for name, (fieldnames, rows) in tables.items()
    }


def input_records(loaded: LoadedAnalysis) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in loaded.cases:
        raw_paths = analysis.result_paths(case.job)
        binding = analysis.result_binding_path(case.job)
        records.append(
            {
                "case_id": case.job["id"],
                "raw_results": {
                    name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in sorted(raw_paths.items())
                },
                "result_binding": {
                    "path": str(binding),
                    "sha256": sha256(binding),
                },
                "certificate": {
                    "path": case.job["certificate"]["absolute_path"],
                    "sha256": case.certificate_sha256,
                },
            }
        )
    return records


def manifest_bytes(
    loaded: LoadedAnalysis, files: dict[str, bytes]
) -> bytes:
    inventory = [
        {
            "path": name,
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for name, value in sorted(files.items())
    ]
    document = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "summary_id": SUMMARY_ID,
        "status": "complete",
        "scope": "validated_read_only_summary_of_existing_signatures",
        "analysis_id": loaded.plan["analysis_id"],
        "statistic": "P(rank > 0) = 1 - rank0 / 20",
        "transformations": {
            "smoothing": False,
            "interpolation": False,
            "probability_thresholds": list(THRESHOLDS),
            "filtration_threshold": (
                "closed birth <= r, inherited from validated kernel metadata"
            ),
            "first_crossing": (
                "earliest sampled duration with probability >= threshold"
            ),
            "sustained_onset": (
                "earliest sampled duration at or above threshold for every "
                "remaining sampled duration through t=60; no extrapolation"
            ),
            "curve_resolved": (
                "strict radius > per-case global_curve_bound h"
            ),
            "r_lt_metric_c": "strict radius < metric C=5",
            "curve_resolved_r_lt_c_candidate": (
                "curve_resolved and r_lt_metric_c; numerical/theory candidate "
                "only, not a certified admissible interval"
            ),
            "pairwise_orientation": "case_to minus case_from",
        },
        "certification": {
            "certified_r0_available": False,
            "note": (
                "No certified r0 is available. No sampled radius is labeled "
                "certified or admissible by this summary."
            ),
        },
        "plan": {
            "path": str(loaded.plan_path),
            "sha256": sha256(loaded.plan_path),
        },
        "summarizer": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256(SCRIPT_PATH),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "duration_grid": {
            "minimum": float(loaded.cases[0].durations[0]),
            "maximum": float(loaded.cases[0].durations[-1]),
            "count": len(loaded.cases[0].durations),
        },
        "radius_grid": {
            "minimum": float(loaded.cases[0].radii[0]),
            "maximum": float(loaded.cases[0].radii[-1]),
            "count": len(loaded.cases[0].radii),
        },
        "inputs": input_records(loaded),
        "selected_findings": selected_findings(loaded),
        "files": inventory,
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def check(loaded: LoadedAnalysis) -> int:
    print(f"summary_id={SUMMARY_ID}")
    print(f"plan={loaded.plan_path}")
    print(f"plan_sha256={sha256(loaded.plan_path)}")
    print("statistic=P(rank > 0) = 1 - rank0 / 20")
    print("validated_cases=" + ",".join(case.job["id"] for case in loaded.cases))
    if loaded.missing_case_ids:
        print("status=incomplete")
        print("missing_cases=" + ",".join(loaded.missing_case_ids))
        print("No summary files were written.")
        return 2
    files = build_summary_files(loaded)
    manifest = manifest_bytes(loaded, files)
    print("status=complete")
    print(f"summary_file_count={len(files) + 1}")
    print(f"summary_manifest_sha256={hashlib.sha256(manifest).hexdigest()}")
    print("No summary files were written.")
    return 0


def write(loaded: LoadedAnalysis) -> Path:
    if loaded.missing_case_ids:
        raise ValueError(
            "cannot write summary; missing cases: "
            + ",".join(loaded.missing_case_ids)
        )
    output_dir = loaded.root / SUMMARY_DIRNAME
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite summary: {output_dir}")
    files = build_summary_files(loaded)
    files["summary_manifest.json"] = manifest_bytes(loaded, files)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{SUMMARY_DIRNAME}-", dir=loaded.root)
    )
    try:
        for name, value in sorted(files.items()):
            path = staging / name
            with path.open("xb") as handle:
                handle.write(value)
        os.rename(staging, output_dir)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    print(f"Wrote {output_dir}")
    return output_dir


def self_test() -> None:
    durations = np.asarray([1.0, 2.0, 3.0, 4.0])
    first, sustained = onset(
        durations, np.asarray([0.0, 0.5, 0.25, 0.75]), 0.5
    )
    if first != 2.0 or sustained != 4.0:
        raise AssertionError((first, sustained))
    first, sustained = onset(durations, np.asarray([0.0, 0.5, 0.5, 0.5]), 0.5)
    if first != 2.0 or sustained != 2.0:
        raise AssertionError((first, sustained))
    first, sustained = onset(durations, np.asarray([0.0, 0.0, 0.0, 0.0]), 0.25)
    if first is not None or sustained is not None:
        raise AssertionError((first, sustained))
    if onset_difference(2.0, 5.0) != 3.0:
        raise AssertionError("pairwise onset orientation changed")
    if onset_difference(None, 5.0) is not None:
        raise AssertionError("missing onset must propagate")
    boundary_case = CaseData(
        job={"id": "boundary"},
        durations=np.asarray([1.0]),
        radii=np.asarray([0.5, 0.6, 5.0]),
        probability=np.asarray([[0.0], [0.0], [0.0]]),
        metadata={},
        beta1=0,
        curve_bound=0.5,
        certified_period=None,
        certificate_sha256="",
    )
    if curve_resolved(boundary_case, 0.5):
        raise AssertionError("curve resolution must be strict r > h")
    if not curve_resolved_candidate(boundary_case, 0.6, 5.0):
        raise AssertionError("h < r < C must be a candidate")
    if r_lt_metric_c(5.0, 5.0) or curve_resolved_candidate(
        boundary_case, 5.0, 5.0
    ):
        raise AssertionError("r=C must never be a candidate")
    if optional_float(4.9750000000000005) != "4.975":
        raise AssertionError("protocol-grid float normalization changed")
    print("Self-test passed; no files were written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("check", "write", "self-test"),
        default="check",
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--julia-bin", default="julia")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "self-test":
        self_test()
        return 0
    loaded = load_validated_analysis(args.plan, args.julia_bin)
    if args.action == "check":
        return check(loaded)
    write(loaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
