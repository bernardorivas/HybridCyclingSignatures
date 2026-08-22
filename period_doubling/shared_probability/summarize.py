#!/usr/bin/env python3
"""Summarize completed shared-protocol probability results.

This script does not recompute cycling signatures.  It reloads a frozen plan
through the production validator, reconstructs the documented probability
from the stored birth vectors, and reports fixed-radius threshold summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
DRIVER_PATH = SCRIPT_PATH.with_name("driver.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_driver() -> Any:
    spec = importlib.util.spec_from_file_location(
        "shared_probability_driver_for_summary", DRIVER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the shared probability driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def first_threshold(
    probability: np.ndarray,
    durations: np.ndarray,
    threshold: float,
    *,
    sustained: bool,
) -> float | None:
    selected = probability >= threshold - 1e-12
    if sustained:
        selected = np.logical_and.accumulate(selected[::-1])[::-1]
    indices = np.flatnonzero(selected)
    if len(indices) == 0:
        return None
    return float(durations[indices[0]])


def finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def summarize(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    driver = load_driver()
    root, plan = driver.load_plan(plan_path)
    protocol = plan["protocol"]
    loaded = [
        driver.load_probability(job, protocol, plan) for job in plan["jobs"]
    ]
    durations = loaded[0][0]
    radii = loaded[0][1]
    if not all(np.array_equal(durations, item[0]) for item in loaded):
        raise ValueError("cases do not share one duration grid")
    if not all(np.array_equal(radii, item[1]) for item in loaded):
        raise ValueError("cases do not share one radius grid")

    case_ids = [job["case_id"] for job in plan["jobs"]]
    sample_radii = [float(item[3]["sample_radius"]) for item in loaded]
    common_index = int(np.searchsorted(radii, max(sample_radii) - 1e-12))
    common_radius = float(radii[common_index])
    metric_c = float(protocol["metric_c"])
    upper_index = int(np.searchsorted(radii, metric_c - 1e-12))
    if common_index >= upper_index:
        raise ValueError("no shared curve-resolved grid row satisfies r < C")
    thresholds = (0.25, 0.5, 0.75)
    onset_rows: list[dict[str, Any]] = []
    for radius_index in range(common_index, upper_index):
        for case_id, item in zip(case_ids, loaded):
            curve = item[2][radius_index]
            for threshold in thresholds:
                onset_rows.append(
                    {
                        "radius": float(radii[radius_index]),
                        "case_id": case_id,
                        "probability_threshold": threshold,
                        "first_duration": finite_or_none(
                            first_threshold(
                                curve,
                                durations,
                                threshold,
                                sustained=False,
                            )
                        ),
                        "sustained_duration": finite_or_none(
                            first_threshold(
                                curve,
                                durations,
                                threshold,
                                sustained=True,
                            )
                        ),
                    }
                )

    first_common: dict[str, dict[str, list[float | None]]] = {}
    for threshold in thresholds:
        label = f"q{threshold:g}"
        first_common[label] = {
            "first_duration": [
                finite_or_none(
                    first_threshold(
                        item[2][common_index],
                        durations,
                        threshold,
                        sustained=False,
                    )
                )
                for item in loaded
            ],
            "sustained_duration": [
                finite_or_none(
                    first_threshold(
                        item[2][common_index],
                        durations,
                        threshold,
                        sustained=True,
                    )
                )
                for item in loaded
            ],
        }

    radius_monotonicity_violations = {
        case_id: int(np.count_nonzero(np.diff(item[2], axis=0) < -1e-12))
        for case_id, item in zip(case_ids, loaded)
    }
    valid_band_all_one = {
        case_id: bool(
            np.all(item[2][common_index:upper_index] >= 1.0 - 1e-12)
        )
        for case_id, item in zip(case_ids, loaded)
    }

    pair_summaries: list[dict[str, Any]] = []
    for left_index in range(len(loaded)):
        for right_index in range(left_index + 1, len(loaded)):
            left = loaded[left_index][2][common_index:upper_index]
            right = loaded[right_index][2][common_index:upper_index]
            pair_summaries.append(
                {
                    "left": case_ids[left_index],
                    "right": case_ids[right_index],
                    "equal_cells": int(np.count_nonzero(np.isclose(left, right))),
                    "total_cells": int(left.size),
                    "mean_absolute_difference": float(np.mean(np.abs(left - right))),
                    "maximum_absolute_difference": float(np.max(np.abs(left - right))),
                }
            )

    summary = {
        "schema_version": 1,
        "analysis_id": plan["analysis_id"],
        "plan": str((root / "plan.json").resolve()),
        "plan_sha256": sha256(root / "plan.json"),
        "summarizer": str(SCRIPT_PATH),
        "summarizer_sha256": sha256(SCRIPT_PATH),
        "statistic": "P(rank > 0) = 1 - rank0 / n_runs",
        "case_order": case_ids,
        "n_runs_per_duration": int(protocol["n_runs"]),
        "duration_grid": [float(value) for value in durations],
        "radius_grid": [float(value) for value in radii],
        "sample_radius_by_case": dict(zip(case_ids, sample_radii)),
        "beta1_Y_by_case": {
            case_id: int(item[3]["beta1_Y"])
            for case_id, item in zip(case_ids, loaded)
        },
        "first_common_curve_resolved_radius": common_radius,
        "last_displayed_radius_below_metric_C": float(radii[upper_index - 1]),
        "common_band_definition": (
            "sample_radius <= r < metric_C; this checks consecutive-sample "
            "curve resolution and the protocol's r<C restriction, but does "
            "not certify the separate upper bound r<r0(Y;Gamma)"
        ),
        "first_common_radius_onsets": first_common,
        "radius_monotonicity_violations": radius_monotonicity_violations,
        "entire_common_curve_resolved_band_has_probability_one": (
            valid_band_all_one
        ),
        "pairwise_common_curve_resolved_band": pair_summaries,
        "onset_definition": {
            "first_duration": (
                "first duration-grid cell with probability at least q"
            ),
            "sustained_duration": (
                "first duration-grid cell from which all later cells have "
                "probability at least q"
            ),
            "warning": (
                "starts are resampled independently at each duration, so "
                "sustained_duration is descriptive rather than a nested-window "
                "monotonicity test"
            ),
        },
    }
    return summary, onset_rows


def write_summary(plan_path: Path) -> Path:
    summary, onset_rows = summarize(plan_path)
    output_dir = plan_path.resolve().parent / "summary"
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    output_dir.mkdir()
    try:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        onset_path = output_dir / "common_curve_resolved_onsets.csv"
        with onset_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(onset_rows[0]))
            writer.writeheader()
            writer.writerows(onset_rows)
        hashes = {
            summary_path.name: sha256(summary_path),
            onset_path.name: sha256(onset_path),
        }
        (output_dir / "output_hashes.json").write_text(
            json.dumps(hashes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        (output_dir / "SUMMARY_INCOMPLETE").touch(exist_ok=True)
        raise
    print(f"Wrote {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize one completed shared-protocol analysis."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_summary(args.plan)


if __name__ == "__main__":
    main()
