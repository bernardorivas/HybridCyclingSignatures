#!/usr/bin/env python3
"""Write a hash-bound compact summary of the completed Compass v2 run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
HERE = SCRIPT_PATH.parent
sys.path.insert(0, str(HERE))

import compass_probability_v2 as driver  # noqa: E402


DEFAULT_PLAN = (
    driver.SAFE_OUTPUT_ROOT
    / "compass_fourier_embedded_probability_linf_v2_david_grid"
    / "plan.json"
)
SUMMARY_DIRNAME = "compass_probability_summary_v1"
CLASSIFICATION = "low_r_fourier_closure_empirical_not_curve_resolved"


def onset(
    durations: np.ndarray, probability: np.ndarray, threshold: float, sustained: bool
) -> float | None:
    meets = probability >= threshold
    if sustained:
        meets = np.logical_and.accumulate(meets[::-1])[::-1]
    indices = np.flatnonzero(meets)
    return None if len(indices) == 0 else float(durations[indices[0]])


def binding_record(job: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    expected = driver.validate_raw_result(job, plan)
    path = Path(job["output_dir"]) / f"{job['id']}_v2_result.json"
    actual = driver.load_json(driver.resolve_existing_file(path, "v2 result binding"))
    created_utc = actual.pop("created_utc", None)
    if not isinstance(created_utc, str) or actual != expected:
        raise ValueError(f"{job['id']}: result binding does not match fresh validation")
    return {
        "path": str(path),
        "sha256": driver.sha256(path),
        "created_utc": created_utc,
    }


def case_summary(job: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    binding = binding_record(job, plan)
    durations, radii, probability = driver.read_probability_matrix(job, plan["protocol"])
    if not math.isclose(float(radii[0]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{job['id']}: radius grid does not begin at zero")
    metadata_path = driver.result_paths(job)["metadata"]
    metadata = driver.parse_metadata(metadata_path)
    beta1 = int(metadata["beta1_Y"])
    curve_bound = float(metadata["global_curve_bound"])
    valid_indices = np.flatnonzero(radii > curve_bound)
    if len(valid_indices) == 0:
        raise ValueError(f"{job['id']}: no strict curve-resolved radius")
    valid_index = int(valid_indices[0])
    r0_record: dict[str, Any] = {}
    for threshold in (0.25, 0.50, 0.75):
        label = f"p{int(round(100 * threshold))}"
        r0_record[f"{label}_first"] = onset(
            durations, probability[0], threshold, False
        )
        r0_record[f"{label}_sustained"] = onset(
            durations, probability[0], threshold, True
        )
    return {
        "case_id": job["id"],
        "phi_deg": float(job["phi_deg"]),
        "q": job["q"],
        "nominal_suspension_period": job["nominal_suspension_period"],
        "physical_return_seconds_context_only": job["physical_return_seconds"],
        "tangent_semantics": job["tangent_semantics"],
        "beta1_Y": beta1,
        "global_curve_bound_h": curve_bound,
        "first_strict_curve_resolved_radius": float(radii[valid_index]),
        "r0_onsets": r0_record,
        "first_valid_radius_p50_first": onset(
            durations, probability[valid_index], 0.5, False
        ),
        "first_valid_radius_p50_sustained": onset(
            durations, probability[valid_index], 0.5, True
        ),
        "probability_minimum": float(np.min(probability)),
        "probability_maximum": float(np.max(probability)),
        "comparison_space_seconds": float(metadata["comparison_space_seconds"]),
        "experiment_seconds": float(metadata["experiment_seconds"]),
        "segment_starts": {
            "path": str(driver.result_paths(job)["starts"]),
            "sha256": driver.sha256(driver.result_paths(job)["starts"]),
        },
        "result_binding": binding,
        "classification": CLASSIFICATION,
    }


CSV_FIELDS = (
    "case_id", "phi_deg", "q", "nominal_suspension_period",
    "physical_return_seconds_context_only", "beta1_Y", "global_curve_bound_h",
    "first_strict_curve_resolved_radius", "r0_p25_first", "r0_p25_sustained",
    "r0_p50_first", "r0_p50_sustained", "r0_p75_first", "r0_p75_sustained",
    "first_valid_radius_p50_first", "first_valid_radius_p50_sustained",
    "comparison_space_seconds", "experiment_seconds", "paired_start_sha256",
    "classification",
)


def csv_row(case: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: case.get(key) for key in CSV_FIELDS
    }
    for label in ("p25", "p50", "p75"):
        row[f"r0_{label}_first"] = case["r0_onsets"][f"{label}_first"]
        row[f"r0_{label}_sustained"] = case["r0_onsets"][f"{label}_sustained"]
    row["paired_start_sha256"] = case["segment_starts"]["sha256"]
    return row


def write_summary(plan_path: Path, julia_bin: str) -> Path:
    root, plan = driver.load_plan(plan_path, julia_bin)
    if plan["analysis_id"] != "compass_fourier_embedded_probability_linf_v2_david_grid":
        raise ValueError("summary requires the final Compass Fourier plan")
    output_dir = root / SUMMARY_DIRNAME
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite summary: {output_dir}")
    cases = [case_summary(job, plan) for job in plan["jobs"]]
    start_hashes = {case["segment_starts"]["sha256"] for case in cases}
    if len(start_hashes) != 1:
        raise ValueError("paired-start files are not byte-identical")
    paired_start_hash = next(iter(start_hashes))

    output_dir.mkdir()
    try:
        csv_path = output_dir / "case_summary.csv"
        with csv_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for case in cases:
                writer.writerow(csv_row(case))
        summary = {
            "schema_version": 1,
            "status": "validated_summary",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_id": plan["analysis_id"],
            "classification": CLASSIFICATION,
            "probability_statistic": "P(rank > 0) = 1 - rank0/20",
            "probability_processing": "none; no smoothing, interpolation, or scale selection",
            "curve_resolution_rule": "strict r > global_curve_bound_h",
            "interpretation": (
                "The r=0 Fourier-closure staircase is an empirical discrete "
                "control and is not curve resolved."
            ),
            "plan": {"path": str(plan_path.resolve()), "sha256": driver.sha256(plan_path)},
            "bundle_manifest": {
                "path": plan["bundle_manifest"],
                "sha256": plan["bundle_manifest_sha256"],
            },
            "driver": {
                "path": plan["orchestrator"],
                "sha256": plan["orchestrator_sha256"],
            },
            "summary_script": {
                "path": str(SCRIPT_PATH),
                "sha256": driver.sha256(SCRIPT_PATH),
            },
            "paired_start_sha256": paired_start_hash,
            "case_summary_csv": {
                "path": str(csv_path),
                "sha256": driver.sha256(csv_path),
            },
            "cases": cases,
        }
        json_path = output_dir / "summary.json"
        driver.write_exclusive(
            json_path, json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    except BaseException:
        marker = output_dir / "SUMMARY_INCOMPLETE"
        if not marker.exists():
            driver.write_exclusive(marker, "Compass v2 summary failed\n")
        raise
    print(f"Wrote {json_path}")
    print(f"summary_sha256={driver.sha256(json_path)}")
    print(f"csv_sha256={driver.sha256(csv_path)}")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--julia-bin", default="julia")
    args = parser.parse_args()
    write_summary(args.plan, args.julia_bin)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"ERROR: {error}") from error
