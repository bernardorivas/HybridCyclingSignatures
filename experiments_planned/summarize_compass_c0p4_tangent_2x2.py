#!/usr/bin/env python3
"""Summarize the completed fine-Compass C=.4 tangent/cover 2x2.

This is a read-only analysis of materialized cycling-signature results.  It
does not simulate, train, or compute new signatures.  Three arms are loaded
from the diagnostic plan's jobs file; the fourth (the pre-existing JVP/sb2
arm) is loaded through the hash-pinned external comparator in ``plan.json``.

Probabilities use every fixed manifest start in an exact target-duration
cell.  ``pooled`` contains all 20 starts, while ``tune`` and ``validate``
contain their frozen 10-start halves.  A curve is resolved only when its
stored strict bound satisfies ``curve_bound < raw_radius`` (with the same
numerical epsilon used by the existing analyzer).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Iterator

import analyze_fine_compass_c_radius_sweep as analysis


REGIMES = analysis.REGIMES
SPLITS = ("pooled", "tune", "validate")
ARM_ORDER = ("jvp_sb1", "jvp_sb2_existing", "flow_sb1", "flow_sb2")
LOCAL_ARMS = ("jvp_sb1", "flow_sb1", "flow_sb2")
JOB_KIND = "fine_compass_duration_c_radius_jobs"
SCHEMA_VERSION = 1
OUTPUT_NAMES = (
    "beta_vectors.csv",
    "curve_resolution.csv",
    "curve_resolved_regime_evaluations.csv",
    "curve_resolved_arm_evaluations.csv",
    "probability_matrices.csv",
    "p50.csv",
    "period4_vs_period8.csv",
    "pairwise_arm_differences.csv",
    "pairwise_arm_difference_summary.csv",
    "input_hashes.json",
    "summary.json",
)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"missing real {label}: {path}")
    return resolved


def _assert_hash(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} lacks a SHA-256 digest")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, observed {actual}"
        )


def _require_equal(actual: object, expected: object, label: str) -> None:
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        if not _close(float(actual), float(expected)):
            raise ValueError(f"{label}: expected {expected}, observed {actual}")
    elif actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, observed {actual!r}")


def _relative_hash_key(path: Path) -> str:
    code_root = Path(__file__).resolve().parents[1]
    resolved = path.resolve()
    if resolved.is_relative_to(code_root):
        return str(resolved.relative_to(code_root))
    return str(resolved)


def _duration_field(duration: float) -> str:
    value = f"{duration:.12g}".replace("-", "m").replace(".", "p")
    return f"t_{value}_s"


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.17g}"


def _probability_number(value: float) -> str:
    return f"{value:.12g}"


def _load_jobs(plan_root: Path, code_root: Path) -> tuple[dict, dict[tuple[str, str], dict]]:
    path = _require_file(plan_root / "jobs.json", "jobs document")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("kind") != JOB_KIND:
        raise ValueError(f"unsupported jobs document: {path}")
    if Path(str(document.get("plan_root", ""))).resolve() != plan_root:
        raise ValueError("jobs document names a different plan root")
    if Path(str(document.get("working_directory", ""))).resolve() != code_root:
        raise ValueError("jobs document names a different code root")
    raw_jobs = document.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("jobs document lacks its jobs list")
    jobs: dict[tuple[str, str], dict] = {}
    for job in raw_jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be an object")
        key = (str(job.get("arm", "")), str(job.get("regime", "")))
        if key in jobs:
            raise ValueError(f"duplicate job for {key}")
        jobs[key] = job
    expected = {(arm, regime) for arm in LOCAL_ARMS for regime in REGIMES}
    if set(jobs) != expected:
        raise ValueError(
            "jobs do not form the expected three-arm by five-regime grid; "
            f"missing={sorted(expected - set(jobs))}, "
            f"extra={sorted(set(jobs) - expected)}"
        )
    return document, jobs


def _validate_local_record(
    record: analysis.ResultRecord,
    job: dict,
    arm: dict,
    regime: str,
    plan_root: Path,
    durations: list[float],
    rho_grid: list[float],
    n_starts: int,
) -> list[Path]:
    for key, expected in (
        ("C", arm["C"]),
        ("boxsize", arm["boxsize"]),
        ("sb_radius", arm["sb_radius"]),
        ("metric_mode", arm["metric_mode"]),
        ("cover_default_C", arm["cover_default_C"]),
    ):
        _require_equal(job.get(key), expected, f"{arm['label']}/{regime} job {key}")
    _require_equal(job.get("rho_max"), rho_grid[-1], f"{arm['label']}/{regime} rho_max")
    _require_equal(record.regime, regime, f"{arm['label']}/{regime} record regime")
    _require_equal(record.c_scale, arm["C"], f"{arm['label']}/{regime} result C")
    _require_equal(record.boxsize, arm["boxsize"], f"{arm['label']}/{regime} boxsize")
    _require_equal(record.sb_radius, arm["sb_radius"], f"{arm['label']}/{regime} sb")
    _require_equal(record.metric_mode, arm["metric_mode"], f"{arm['label']}/{regime} metric mode")
    _require_equal(
        record.cover_default_c,
        arm["cover_default_C"],
        f"{arm['label']}/{regime} cover-default C",
    )
    _require_equal(record.rho_max, rho_grid[-1], f"{arm['label']}/{regime} result rho_max")
    _require_equal(
        record.n_windows,
        len(durations) * n_starts,
        f"{arm['label']}/{regime} result window count",
    )
    metadata = analysis._parse_metadata(record.metadata_path)
    if metadata.get("split") != "all":
        raise ValueError(f"{arm['label']}/{regime}: result does not include both splits")

    manifest = (plan_root / "manifests" / f"{regime}.csv").resolve()
    _require_equal(
        Path(str(job.get("manifest", ""))).resolve(),
        manifest,
        f"{arm['label']}/{regime} manifest path",
    )
    _require_equal(
        Path(metadata.get("manifest", "")).resolve(),
        manifest,
        f"{arm['label']}/{regime} result manifest path",
    )
    expected_out = Path(str(job.get("out_dir", ""))).resolve()
    signature_root = (plan_root / "signatures").resolve()
    if not expected_out.is_relative_to(signature_root):
        raise ValueError(f"{arm['label']}/{regime}: job output leaves plan signatures")
    prefix = str(job.get("out_prefix", ""))
    expected_metadata = expected_out / f"{prefix}_metadata.txt"
    expected_births = expected_out / f"{prefix}_births.csv"
    _require_equal(record.metadata_path, expected_metadata, f"{arm['label']}/{regime} metadata path")
    _require_equal(record.births_path, expected_births, f"{arm['label']}/{regime} births path")

    positions = _require_file(Path(str(job.get("positions", ""))), "positions")
    tangents = _require_file(Path(str(job.get("tangents", ""))), "tangents")
    _require_equal(record.positions_path, positions, f"{arm['label']}/{regime} positions")
    _require_equal(record.tangents_path, tangents, f"{arm['label']}/{regime} tangents")
    for path, field in (
        (positions, "positions_sha256"),
        (tangents, "tangents_sha256"),
        (manifest, "manifest_sha256"),
    ):
        _assert_hash(path, job.get(field), f"{arm['label']}/{regime} {field}")

    tangent_source = str(arm.get("tangent_source", "default"))
    _require_equal(job.get("tangent_source"), tangent_source, f"{arm['label']}/{regime} tangent source")
    if tangent_source == "override":
        provenance = arm.get("tangent_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{arm['label']} lacks tangent provenance")
        provenance_path = _require_file(Path(str(provenance.get("path", ""))), "tangent provenance")
        _assert_hash(provenance_path, provenance.get("sha256"), f"{arm['label']} tangent provenance")
        _require_equal(record.tangent_provenance_path, provenance_path, f"{arm['label']}/{regime} provenance")
        return [record.metadata_path, record.births_path, positions, tangents, manifest, provenance_path]
    if record.tangent_provenance_path is not None:
        raise ValueError(f"{arm['label']}/{regime}: default tangents unexpectedly name provenance")
    return [record.metadata_path, record.births_path, positions, tangents, manifest]


def _load_local_records(
    plan: dict,
    jobs: dict[tuple[str, str], dict],
    plan_root: Path,
    manifests: dict,
    durations: list[float],
    rho_grid: list[float],
) -> tuple[dict[str, dict[str, analysis.ResultRecord]], list[Path], dict[str, dict]]:
    arm_entries = plan.get("experiment_arms")
    if not isinstance(arm_entries, list):
        raise ValueError("plan lacks experiment arms")
    arms = {str(entry.get("label", "")): entry for entry in arm_entries if isinstance(entry, dict)}
    if set(arms) != set(LOCAL_ARMS):
        raise ValueError(f"unexpected local experiment arms: {sorted(arms)}")
    records: dict[str, dict[str, analysis.ResultRecord]] = {}
    source_paths: list[Path] = []
    for label in LOCAL_ARMS:
        records[label] = {}
        for regime in REGIMES:
            job = jobs[(label, regime)]
            out_dir = Path(str(job.get("out_dir", ""))).resolve()
            prefix = str(job.get("out_prefix", ""))
            metadata_path = _require_file(out_dir / f"{prefix}_metadata.txt", "runner metadata")
            record = analysis._load_result_record(metadata_path, manifests, durations)
            source_paths.extend(
                _validate_local_record(
                    record,
                    job,
                    arms[label],
                    regime,
                    plan_root,
                    durations,
                    rho_grid,
                    int(plan["n_starts_per_duration"]),
                )
            )
            records[label][regime] = record
    return records, source_paths, arms


def _load_external_records(
    plan: dict,
    plan_root: Path,
    manifests: dict,
    durations: list[float],
    rho_grid: list[float],
) -> tuple[str, dict[str, analysis.ResultRecord], list[Path], dict]:
    entries = plan.get("external_comparators")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ValueError("plan must contain exactly one external comparator")
    entry = entries[0]
    label = str(entry.get("label", ""))
    if label != "jvp_sb2_existing":
        raise ValueError(f"unexpected external comparator: {label!r}")
    results = entry.get("results")
    if not isinstance(results, dict) or set(results) != set(REGIMES):
        raise ValueError("external comparator does not cover all five regimes")
    external_root = Path(str(entry.get("plan_root", ""))).resolve()
    safe_root = (Path(__file__).resolve().parent / "outputs").resolve()
    if not external_root.is_relative_to(safe_root) or not external_root.is_dir():
        raise ValueError("external comparator plan root is not a materialized planned output")

    records: dict[str, analysis.ResultRecord] = {}
    source_paths = [_require_file(external_root / "plan.json", "external plan")]
    for regime in REGIMES:
        specification = results[regime]
        if not isinstance(specification, dict):
            raise ValueError(f"malformed external comparator row for {regime}")
        named: dict[str, Path] = {}
        for kind in ("metadata", "births", "manifest", "positions", "tangents"):
            path = _require_file(Path(str(specification.get(kind, ""))), f"external {kind}")
            _assert_hash(path, specification.get(f"{kind}_sha256"), f"external {regime} {kind}")
            named[kind] = path
            source_paths.append(path)
        current_manifest = (plan_root / "manifests" / f"{regime}.csv").resolve()
        if _sha256(named["manifest"]) != _sha256(current_manifest):
            raise ValueError(f"external {regime} manifest is not the plan's frozen manifest")
        record = analysis._load_result_record(named["metadata"], manifests, durations)
        _require_equal(record.births_path, named["births"], f"external {regime} births path")
        _require_equal(record.positions_path, named["positions"], f"external {regime} positions")
        _require_equal(record.tangents_path, named["tangents"], f"external {regime} tangents")
        for key, observed in (
            ("C", record.c_scale),
            ("boxsize", record.boxsize),
            ("sb_radius", record.sb_radius),
            ("metric_mode", record.metric_mode),
            ("cover_default_C", record.cover_default_c),
        ):
            _require_equal(observed, entry[key], f"external {regime} {key}")
        _require_equal(record.rho_max, rho_grid[-1], f"external {regime} rho_max")
        _require_equal(
            record.n_windows,
            len(durations) * int(plan["n_starts_per_duration"]),
            f"external {regime} window count",
        )
        if analysis._parse_metadata(record.metadata_path).get("split") != "all":
            raise ValueError(f"external {regime} result does not include both splits")
        records[regime] = record
    return label, records, source_paths, entry


def _validate_curve_invariants(
    plan: dict,
    records: dict[str, dict[str, analysis.ResultRecord]],
) -> list[dict[str, object]]:
    invariants = plan.get("curve_bound_invariants")
    if not isinstance(invariants, list) or not invariants:
        raise ValueError("plan lacks curve-bound invariants")
    audit: list[dict[str, object]] = []
    for invariant in invariants:
        if not isinstance(invariant, dict) or invariant.get("comparison") != "exact_rowwise":
            raise ValueError(f"unsupported curve-bound invariant: {invariant!r}")
        left = str(invariant.get("left", ""))
        right = str(invariant.get("right", ""))
        if left not in records or right not in records:
            raise ValueError(f"curve-bound invariant names unknown arms: {left}, {right}")
        compared = 0
        for regime in REGIMES:
            left_rows = records[left][regime].rows
            right_rows = records[right][regime].rows
            if left_rows.keys() != right_rows.keys():
                raise ValueError(f"curve-bound invariant row keys differ: {left}, {right}, {regime}")
            for key in left_rows:
                if left_rows[key].curve_bound != right_rows[key].curve_bound:
                    raise ValueError(
                        f"curve-bound invariant failed at {left}, {right}, {regime}, {key}"
                    )
                compared += 1
        audit.append({"left": left, "right": right, "comparison": "exact_rowwise", "rows": compared})
    return audit


def _rows_for(
    record: analysis.ResultRecord,
    split: str,
    duration: float,
) -> tuple[analysis.ResultRow, ...]:
    if split == "pooled":
        rows = tuple(
            row
            for key, row in sorted(record.rows.items())
            if _close(key[0], duration)
        )
    else:
        rows = record.grouped_rows[(split, duration)]
    if not rows:
        raise ValueError(f"empty {split} cell for {record.regime}, duration={duration}")
    return rows


def _compute_probabilities(
    records: dict[str, dict[str, analysis.ResultRecord]],
    durations: list[float],
    rho_grid: list[float],
    c_scale: float,
) -> dict[tuple[str, str, str, int, int], tuple[int, int, float]]:
    probabilities: dict[tuple[str, str, str, int, int], tuple[int, int, float]] = {}
    for arm in ARM_ORDER:
        for split in SPLITS:
            for regime in REGIMES:
                record = records[arm][regime]
                for duration_index, duration in enumerate(durations):
                    rows = _rows_for(record, split, duration)
                    for radius_index, rho in enumerate(rho_grid):
                        radius = c_scale * rho
                        nontrivial = sum(
                            analysis._rank_at(row, radius, c_scale) > 0 for row in rows
                        )
                        probabilities[(arm, split, regime, radius_index, duration_index)] = (
                            nontrivial,
                            len(rows),
                            nontrivial / len(rows),
                        )
    return probabilities


def _curve_pass(bound: float, radius: float) -> bool:
    return bound < radius - analysis._numeric_epsilon(bound, radius)


def _first_resolved_index(maximum_bound: float, raw_radii: list[float]) -> int | None:
    return next(
        (index for index, radius in enumerate(raw_radii) if _curve_pass(maximum_bound, radius)),
        None,
    )


def _curve_resolution_rows(
    records: dict[str, dict[str, analysis.ResultRecord]],
    rho_grid: list[float],
    raw_radii: list[float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(scope: str, arm: str, split: str, regime: str, bounds: list[float]) -> dict[str, object]:
        maximum = max(bounds)
        index = _first_resolved_index(maximum, raw_radii)
        row: dict[str, object] = {
            "scope": scope,
            "arm": arm,
            "split": split,
            "regime": regime,
            "n_windows": len(bounds),
            "max_curve_bound": maximum,
            "first_resolved_index": index,
            "first_resolved_rho": None if index is None else rho_grid[index],
            "first_resolved_raw_radius": None if index is None else raw_radii[index],
            "rmax_resolved": _curve_pass(maximum, raw_radii[-1]),
        }
        rows.append(row)
        return row

    global_bounds: list[float] = []
    for arm in ARM_ORDER:
        arm_bounds: list[float] = []
        for split in SPLITS:
            split_bounds: list[float] = []
            for regime in REGIMES:
                record = records[arm][regime]
                if split == "pooled":
                    bounds = [row.curve_bound for row in record.rows.values()]
                else:
                    bounds = [
                        row.curve_bound
                        for (duration, row_split, _), row in record.rows.items()
                        if row_split == split
                    ]
                add("arm_split_regime", arm, split, regime, bounds)
                split_bounds.extend(bounds)
                if split == "pooled":
                    arm_bounds.extend(bounds)
            add("arm_split", arm, split, "", split_bounds)
        add("arm", arm, "pooled", "", arm_bounds)
        global_bounds.extend(arm_bounds)
    global_row = add("global", "", "pooled", "", global_bounds)
    return rows, global_row


def _p50(
    values: list[float],
    durations: list[float],
) -> tuple[float | None, str, float | None]:
    index = next((index for index, probability in enumerate(values) if probability >= 0.5), None)
    if index is None:
        return None, "right", None
    censoring = "left" if index == 0 else "none"
    return durations[index], censoring, values[index]


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _beta_rows(
    records: dict[str, dict[str, analysis.ResultRecord]],
    arm_details: dict[str, dict],
) -> Iterator[dict[str, object]]:
    for arm in ARM_ORDER:
        detail = arm_details[arm]
        row: dict[str, object] = {
            "arm": arm,
            "tangent_source": "jvp" if arm.startswith("jvp") else "learned_flow",
            "C": detail["C"],
            "metric_mode": detail["metric_mode"],
            "boxsize": detail["boxsize"],
            "sb_radius": detail["sb_radius"],
            "cover_default_C": detail["cover_default_C"],
        }
        row.update({f"beta1_{regime}": records[arm][regime].beta1 for regime in REGIMES})
        yield row


def _probability_matrix_rows(
    probabilities: dict,
    durations: list[float],
    rho_grid: list[float],
    raw_radii: list[float],
) -> Iterator[dict[str, object]]:
    for arm in ARM_ORDER:
        for split in SPLITS:
            for regime in REGIMES:
                for radius_index, (rho, radius) in enumerate(zip(rho_grid, raw_radii)):
                    values = [
                        probabilities[(arm, split, regime, radius_index, duration_index)]
                        for duration_index in range(len(durations))
                    ]
                    row: dict[str, object] = {
                        "arm": arm,
                        "split": split,
                        "regime": regime,
                        "rho": _number(rho),
                        "raw_radius": _number(radius),
                        "n_per_duration": values[0][1],
                    }
                    row.update(
                        {
                            _duration_field(duration): _probability_number(values[index][2])
                            for index, duration in enumerate(durations)
                        }
                    )
                    yield row


def _p50_rows(
    probabilities: dict,
    durations: list[float],
    rho_grid: list[float],
    raw_radii: list[float],
    evaluation_indices: dict[str, int],
) -> Iterator[dict[str, object]]:
    for evaluation, radius_index in evaluation_indices.items():
        for arm in ARM_ORDER:
            for split in SPLITS:
                for regime in REGIMES:
                    values = [
                        probabilities[(arm, split, regime, radius_index, index)][2]
                        for index in range(len(durations))
                    ]
                    duration, censoring, crossing_probability = _p50(values, durations)
                    yield {
                        "evaluation": evaluation,
                        "arm": arm,
                        "split": split,
                        "regime": regime,
                        "rho": _number(rho_grid[radius_index]),
                        "raw_radius": _number(raw_radii[radius_index]),
                        "probability_threshold": "0.5",
                        "p50_target_duration": _number(duration),
                        "censoring": censoring,
                        "probability_at_p50": _number(crossing_probability),
                        "probability_at_min_duration": _probability_number(values[0]),
                        "probability_at_max_duration": _probability_number(values[-1]),
                    }


def _period4_period8_rows(
    probabilities: dict,
    durations: list[float],
    rho_grid: list[float],
    raw_radii: list[float],
) -> Iterator[dict[str, object]]:
    for arm in ARM_ORDER:
        for split in SPLITS:
            for radius_index, (rho, radius) in enumerate(zip(rho_grid, raw_radii)):
                differences = [
                    probabilities[(arm, split, "period4", radius_index, index)][2]
                    - probabilities[(arm, split, "period8", radius_index, index)][2]
                    for index in range(len(durations))
                ]
                differing = [
                    duration for duration, difference in zip(durations, differences)
                    if not _close(difference, 0.0)
                ]
                yield {
                    "arm": arm,
                    "split": split,
                    "rho": _number(rho),
                    "raw_radius": _number(radius),
                    "n_duration_cells": len(durations),
                    "agreement_count": len(durations) - len(differing),
                    "agreement_fraction": _probability_number(
                        (len(durations) - len(differing)) / len(durations)
                    ),
                    "p4_greater_count": sum(value > 0 for value in differences),
                    "p8_greater_count": sum(value < 0 for value in differences),
                    "mean_difference_p4_minus_p8": _probability_number(
                        sum(differences) / len(differences)
                    ),
                    "mean_absolute_difference": _probability_number(
                        sum(abs(value) for value in differences) / len(differences)
                    ),
                    "max_absolute_difference": _probability_number(
                        max(abs(value) for value in differences)
                    ),
                    "differing_target_durations": ";".join(f"{value:.12g}" for value in differing),
                }


def _pairwise_difference_rows(
    probabilities: dict,
    durations: list[float],
    rho_grid: list[float],
    raw_radii: list[float],
) -> Iterator[dict[str, object]]:
    for left, right in itertools.combinations(ARM_ORDER, 2):
        pair = f"{left}__minus__{right}"
        for split in SPLITS:
            for regime in REGIMES:
                for radius_index, (rho, radius) in enumerate(zip(rho_grid, raw_radii)):
                    for duration_index, duration in enumerate(durations):
                        left_count, left_n, left_probability = probabilities[
                            (left, split, regime, radius_index, duration_index)
                        ]
                        right_count, right_n, right_probability = probabilities[
                            (right, split, regime, radius_index, duration_index)
                        ]
                        difference = left_probability - right_probability
                        yield {
                            "pair": pair,
                            "left_arm": left,
                            "right_arm": right,
                            "split": split,
                            "regime": regime,
                            "rho": _number(rho),
                            "raw_radius": _number(radius),
                            "target_duration": _number(duration),
                            "left_nontrivial": left_count,
                            "left_n": left_n,
                            "left_probability": _probability_number(left_probability),
                            "right_nontrivial": right_count,
                            "right_n": right_n,
                            "right_probability": _probability_number(right_probability),
                            "difference_left_minus_right": _probability_number(difference),
                            "absolute_difference": _probability_number(abs(difference)),
                        }


def _pairwise_summary_rows(
    probabilities: dict,
    durations: list[float],
    rho_grid: list[float],
    raw_radii: list[float],
) -> Iterator[dict[str, object]]:
    for left, right in itertools.combinations(ARM_ORDER, 2):
        pair = f"{left}__minus__{right}"
        for split in SPLITS:
            for regime in REGIMES:
                for radius_index, (rho, radius) in enumerate(zip(rho_grid, raw_radii)):
                    differences = [
                        probabilities[(left, split, regime, radius_index, index)][2]
                        - probabilities[(right, split, regime, radius_index, index)][2]
                        for index in range(len(durations))
                    ]
                    differing = sum(not _close(value, 0.0) for value in differences)
                    yield {
                        "pair": pair,
                        "left_arm": left,
                        "right_arm": right,
                        "split": split,
                        "regime": regime,
                        "rho": _number(rho),
                        "raw_radius": _number(radius),
                        "n_duration_cells": len(durations),
                        "agreement_count": len(durations) - differing,
                        "agreement_fraction": _probability_number(
                            (len(durations) - differing) / len(durations)
                        ),
                        "left_greater_count": sum(value > 0 for value in differences),
                        "right_greater_count": sum(value < 0 for value in differences),
                        "mean_difference_left_minus_right": _probability_number(
                            sum(differences) / len(differences)
                        ),
                        "mean_absolute_difference": _probability_number(
                            sum(abs(value) for value in differences) / len(differences)
                        ),
                        "max_absolute_difference": _probability_number(
                            max(abs(value) for value in differences)
                        ),
                    }


def _grid_index(value: float, grid: list[float], label: str) -> int:
    matches = [index for index, candidate in enumerate(grid) if _close(value, candidate)]
    if len(matches) != 1:
        raise ValueError(f"{label}={value:g} is not a unique planned grid row")
    return matches[0]


def _tail_saturation_onset(
    values: list[float],
    durations: list[float],
) -> tuple[float | None, str]:
    index = next(
        (
            index
            for index in range(len(values))
            if all(_close(value, 1.0) for value in values[index:])
        ),
        None,
    )
    if index is None:
        return None, "right"
    return durations[index], "left" if index == 0 else "none"


def _nested_rank_diagnostics(
    record: analysis.ResultRecord,
    split: str,
    durations: list[float],
    radius: float,
    c_scale: float,
) -> dict[str, int]:
    ranks_by_run: dict[int, list[tuple[float, int]]] = {}
    for duration in durations:
        for row in _rows_for(record, split, duration):
            ranks_by_run.setdefault(row.manifest.run_index, []).append(
                (duration, analysis._rank_at(row, radius, c_scale))
            )
    decreases = 0
    runs_with_decrease = 0
    comparisons = 0
    for run_index, entries in ranks_by_run.items():
        ordered = sorted(entries)
        if len(ordered) != len(durations):
            raise ValueError(
                f"nested-rank audit lacks durations for {record.regime}, "
                f"{split}, run {run_index}"
            )
        run_decreases = sum(
            later_rank < earlier_rank
            for (_, earlier_rank), (_, later_rank) in zip(ordered, ordered[1:])
        )
        decreases += run_decreases
        runs_with_decrease += run_decreases > 0
        comparisons += len(ordered) - 1
    return {
        "nested_rank_decrease_count": decreases,
        "runs_with_nested_rank_decrease": runs_with_decrease,
        "nested_rank_adjacent_comparisons": comparisons,
    }


def _p4_p8_metrics(
    probabilities: dict,
    arm: str,
    split: str,
    radius_index: int,
    durations: list[float],
) -> dict[str, object]:
    differences = [
        probabilities[(arm, split, "period4", radius_index, index)][2]
        - probabilities[(arm, split, "period8", radius_index, index)][2]
        for index in range(len(durations))
    ]
    differing = [
        duration
        for duration, difference in zip(durations, differences)
        if not _close(difference, 0.0)
    ]
    return {
        "p4_p8_agreement_count": len(durations) - len(differing),
        "p4_p8_agreement_fraction": (len(durations) - len(differing)) / len(durations),
        "p4_greater_count": sum(value > 0 for value in differences),
        "p8_greater_count": sum(value < 0 for value in differences),
        "p4_p8_mean_difference": sum(differences) / len(differences),
        "p4_p8_mean_absolute_difference": (
            sum(abs(value) for value in differences) / len(differences)
        ),
        "p4_p8_max_absolute_difference": max(abs(value) for value in differences),
        "p4_p8_differing_target_durations": differing,
    }


def _focused_evaluation_points(
    curve_rows: list[dict[str, object]],
    raw_radii: list[float],
) -> list[dict[str, object]]:
    arm_curve_rows = {
        str(row["arm"]): row for row in curve_rows if row["scope"] == "arm"
    }
    if set(arm_curve_rows) != set(ARM_ORDER):
        raise ValueError("curve-resolution audit lacks exactly one row per arm")
    points: list[dict[str, object]] = []
    for arm in ARM_ORDER:
        index = arm_curve_rows[arm]["first_resolved_index"]
        if index is None:
            raise ValueError(f"{arm} has no curve-resolved grid row")
        points.append(
            {
                "evaluation": "arm_first_curve_resolved",
                "arm": arm,
                "radius_index": int(index),
                "flow_clean_band": False,
            }
        )

    for radius in (0.19, 0.20, 0.21, 0.22):
        index = _grid_index(radius, raw_radii, "flow clean-band raw radius")
        for arm in ("flow_sb1", "flow_sb2"):
            maximum_bound = float(arm_curve_rows[arm]["max_curve_bound"])
            if not _curve_pass(maximum_bound, raw_radii[index]):
                raise ValueError(f"flow clean-band row r={radius:g} is not resolved for {arm}")
            points.append(
                {
                    "evaluation": "flow_clean_band",
                    "arm": arm,
                    "radius_index": index,
                    "flow_clean_band": True,
                }
            )
    return points


def _focused_evaluation_rows(
    records: dict[str, dict[str, analysis.ResultRecord]],
    probabilities: dict,
    durations: list[float],
    rho_grid: list[float],
    raw_radii: list[float],
    points: list[dict[str, object]],
    c_scale: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    regime_rows: list[dict[str, object]] = []
    arm_rows: list[dict[str, object]] = []
    json_points: list[dict[str, object]] = []
    for point in points:
        arm = str(point["arm"])
        radius_index = int(point["radius_index"])
        radius = raw_radii[radius_index]
        json_point: dict[str, object] = {
            "evaluation": point["evaluation"],
            "arm": arm,
            "flow_clean_band": point["flow_clean_band"],
            "rho": rho_grid[radius_index],
            "raw_radius": radius,
            "splits": {},
        }
        for split in SPLITS:
            p50_vector: dict[str, object] = {}
            saturation_vector: dict[str, object] = {}
            decrease_vector: dict[str, object] = {}
            probability_curves: dict[str, list[float]] = {}
            total_decreases = 0
            total_runs_with_decrease = 0
            total_comparisons = 0
            for regime in REGIMES:
                values = [
                    probabilities[(arm, split, regime, radius_index, index)][2]
                    for index in range(len(durations))
                ]
                probability_curves[regime] = values
                p50_duration, p50_censoring, crossing_probability = _p50(
                    values, durations
                )
                saturation_duration, saturation_censoring = _tail_saturation_onset(
                    values, durations
                )
                nested = _nested_rank_diagnostics(
                    records[arm][regime], split, durations, radius, c_scale
                )
                total_decreases += nested["nested_rank_decrease_count"]
                total_runs_with_decrease += nested["runs_with_nested_rank_decrease"]
                total_comparisons += nested["nested_rank_adjacent_comparisons"]
                p50_vector[regime] = {
                    "target_duration_seconds": p50_duration,
                    "censoring": p50_censoring,
                    "probability_at_crossing": crossing_probability,
                }
                saturation_vector[regime] = {
                    "target_duration_seconds": saturation_duration,
                    "censoring": saturation_censoring,
                }
                decrease_vector[regime] = nested
                regime_rows.append(
                    {
                        "evaluation": point["evaluation"],
                        "flow_clean_band": point["flow_clean_band"],
                        "arm": arm,
                        "split": split,
                        "regime": regime,
                        "rho": rho_grid[radius_index],
                        "raw_radius": radius,
                        "beta1": records[arm][regime].beta1,
                        "p50_target_duration": p50_duration,
                        "p50_censoring": p50_censoring,
                        "probability_at_p50": crossing_probability,
                        "probability_at_min_duration": values[0],
                        "probability_at_max_duration": values[-1],
                        "regime_sustained_saturation_onset": saturation_duration,
                        "regime_saturation_censoring": saturation_censoring,
                        **nested,
                    }
                )
            all_regime_curve = [
                min(probability_curves[regime][index] for regime in REGIMES)
                for index in range(len(durations))
            ]
            all_saturation, all_saturation_censoring = _tail_saturation_onset(
                all_regime_curve, durations
            )
            p4_p8 = _p4_p8_metrics(
                probabilities, arm, split, radius_index, durations
            )
            arm_rows.append(
                {
                    "evaluation": point["evaluation"],
                    "flow_clean_band": point["flow_clean_band"],
                    "arm": arm,
                    "split": split,
                    "rho": rho_grid[radius_index],
                    "raw_radius": radius,
                    "all_regime_sustained_saturation_onset": all_saturation,
                    "all_regime_saturation_censoring": all_saturation_censoring,
                    "nested_rank_decrease_count_all_regimes": total_decreases,
                    "runs_with_nested_rank_decrease_sum": total_runs_with_decrease,
                    "nested_rank_adjacent_comparisons_all_regimes": total_comparisons,
                    "p50_vector": json.dumps(p50_vector, sort_keys=True),
                    "nested_rank_decrease_vector": json.dumps(
                        decrease_vector, sort_keys=True
                    ),
                    **{
                        key: (
                            ";".join(f"{value:.12g}" for value in metric)
                            if key == "p4_p8_differing_target_durations"
                            else metric
                        )
                        for key, metric in p4_p8.items()
                    },
                }
            )
            json_point["splits"][split] = {
                "p50": p50_vector,
                "sustained_saturation_onset": saturation_vector,
                "nested_rank_diagnostics": decrease_vector,
                "all_regime_sustained_saturation_onset": {
                    "target_duration_seconds": all_saturation,
                    "censoring": all_saturation_censoring,
                },
                "p4_vs_p8": p4_p8,
            }
        json_points.append(json_point)
    return regime_rows, arm_rows, json_points


def _evaluation_summary(
    probabilities: dict,
    durations: list[float],
    rho_grid: list[float],
    raw_radii: list[float],
    evaluation_indices: dict[str, int],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for evaluation, radius_index in evaluation_indices.items():
        p50: dict[str, object] = {}
        p4_p8: dict[str, object] = {}
        pairwise: dict[str, object] = {}
        for arm in ARM_ORDER:
            p50[arm] = {}
            p4_p8[arm] = {}
            for split in SPLITS:
                p50[arm][split] = {}
                for regime in REGIMES:
                    values = [
                        probabilities[(arm, split, regime, radius_index, index)][2]
                        for index in range(len(durations))
                    ]
                    duration, censoring, crossing = _p50(values, durations)
                    p50[arm][split][regime] = {
                        "target_duration_seconds": duration,
                        "censoring": censoring,
                        "probability_at_crossing": crossing,
                    }
                differences = [
                    probabilities[(arm, split, "period4", radius_index, index)][2]
                    - probabilities[(arm, split, "period8", radius_index, index)][2]
                    for index in range(len(durations))
                ]
                differing = sum(not _close(value, 0.0) for value in differences)
                p4_p8[arm][split] = {
                    "agreement_fraction": (len(durations) - differing) / len(durations),
                    "mean_absolute_difference": sum(abs(value) for value in differences) / len(differences),
                    "max_absolute_difference": max(abs(value) for value in differences),
                }
        for left, right in itertools.combinations(ARM_ORDER, 2):
            pair = f"{left}__minus__{right}"
            pairwise[pair] = {}
            for split in SPLITS:
                pairwise[pair][split] = {}
                for regime in REGIMES:
                    differences = [
                        probabilities[(left, split, regime, radius_index, index)][2]
                        - probabilities[(right, split, regime, radius_index, index)][2]
                        for index in range(len(durations))
                    ]
                    pairwise[pair][split][regime] = {
                        "mean_difference_left_minus_right": sum(differences) / len(differences),
                        "mean_absolute_difference": sum(abs(value) for value in differences) / len(differences),
                        "max_absolute_difference": max(abs(value) for value in differences),
                    }
        result[evaluation] = {
            "rho": rho_grid[radius_index],
            "raw_radius": raw_radii[radius_index],
            "p50": p50,
            "period4_vs_period8": p4_p8,
            "pairwise_arm_differences": pairwise,
        }
    return result


def _prepare_output_dir(plan_root: Path, requested: Path | None, overwrite: bool) -> Path:
    output = (requested or plan_root / "summary").expanduser().absolute().resolve()
    if output == plan_root or not output.is_relative_to(plan_root):
        raise ValueError(f"output directory must be a strict child of {plan_root}")
    if output.is_symlink():
        raise ValueError(f"output directory may not be a symlink: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite summaries; pass --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize the completed C=.4 fine-Compass "
            "JVP/learned-flow by sb-radius 1/2 diagnostic."
        )
    )
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Summary destination (default: PLAN_ROOT/summary).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of this script's named summary files.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    code_root = Path(__file__).resolve().parents[1]
    safe_root = (code_root / "experiments_planned" / "outputs").resolve()
    requested_plan = args.plan_root.expanduser().absolute()
    if requested_plan.is_symlink() or not requested_plan.is_dir():
        raise FileNotFoundError(f"missing real plan root: {requested_plan}")
    plan_root = requested_plan.resolve()
    if not plan_root.is_relative_to(safe_root):
        raise ValueError(f"plan root must stay below {safe_root}")

    plan, plan_path, c_grid, rho_grid, durations = analysis._load_plan(plan_root)
    if len(c_grid) != 1 or not _close(c_grid[0], 0.4):
        raise ValueError(f"this focused summarizer requires the one-cell C=.4 plan: {c_grid}")
    c_scale = c_grid[0]
    raw_radii = [c_scale * rho for rho in rho_grid]
    jobs_document, jobs = _load_jobs(plan_root, code_root)
    manifests, manifest_paths = analysis._load_manifests(plan_root, plan, durations)
    local_records, local_paths, local_details = _load_local_records(
        plan, jobs, plan_root, manifests, durations, rho_grid
    )
    external_label, external_records, external_paths, external_detail = _load_external_records(
        plan, plan_root, manifests, durations, rho_grid
    )
    records = {**local_records, external_label: external_records}
    if set(records) != set(ARM_ORDER):
        raise ValueError(f"materialized arms differ from the required 2x2: {sorted(records)}")
    arm_details = {**local_details, external_label: external_detail}
    invariant_audit = _validate_curve_invariants(plan, records)

    probabilities = _compute_probabilities(records, durations, rho_grid, c_scale)
    curve_rows, common_curve = _curve_resolution_rows(records, rho_grid, raw_radii)
    first_common_index = common_curve["first_resolved_index"]
    if first_common_index is None:
        raise ValueError("the planned raw-radius grid has no common curve-resolved row")
    evaluation_indices = {
        "first_common_curve_resolved": int(first_common_index),
        "rmax": len(rho_grid) - 1,
    }
    focused_points = _focused_evaluation_points(curve_rows, raw_radii)
    focused_regime_rows, focused_arm_rows, focused_json = _focused_evaluation_rows(
        records,
        probabilities,
        durations,
        rho_grid,
        raw_radii,
        focused_points,
        c_scale,
    )

    output = _prepare_output_dir(plan_root, args.output_dir, args.overwrite)
    row_counts: dict[str, int] = {}
    beta_fields = (
        "arm", "tangent_source", "C", "metric_mode", "boxsize", "sb_radius",
        "cover_default_C", *(f"beta1_{regime}" for regime in REGIMES),
    )
    row_counts["beta_vectors.csv"] = _write_csv(
        output / "beta_vectors.csv", beta_fields, _beta_rows(records, arm_details)
    )
    curve_fields = (
        "scope", "arm", "split", "regime", "n_windows", "max_curve_bound",
        "first_resolved_index", "first_resolved_rho", "first_resolved_raw_radius",
        "rmax_resolved",
    )
    row_counts["curve_resolution.csv"] = _write_csv(
        output / "curve_resolution.csv", curve_fields, curve_rows
    )
    focused_regime_fields = (
        "evaluation", "flow_clean_band", "arm", "split", "regime", "rho",
        "raw_radius", "beta1", "p50_target_duration", "p50_censoring",
        "probability_at_p50", "probability_at_min_duration",
        "probability_at_max_duration", "regime_sustained_saturation_onset",
        "regime_saturation_censoring", "nested_rank_decrease_count",
        "runs_with_nested_rank_decrease", "nested_rank_adjacent_comparisons",
    )
    row_counts["curve_resolved_regime_evaluations.csv"] = _write_csv(
        output / "curve_resolved_regime_evaluations.csv",
        focused_regime_fields,
        focused_regime_rows,
    )
    focused_arm_fields = (
        "evaluation", "flow_clean_band", "arm", "split", "rho", "raw_radius",
        "all_regime_sustained_saturation_onset",
        "all_regime_saturation_censoring",
        "nested_rank_decrease_count_all_regimes",
        "runs_with_nested_rank_decrease_sum",
        "nested_rank_adjacent_comparisons_all_regimes", "p50_vector",
        "nested_rank_decrease_vector", "p4_p8_agreement_count",
        "p4_p8_agreement_fraction", "p4_greater_count", "p8_greater_count",
        "p4_p8_mean_difference", "p4_p8_mean_absolute_difference",
        "p4_p8_max_absolute_difference", "p4_p8_differing_target_durations",
    )
    row_counts["curve_resolved_arm_evaluations.csv"] = _write_csv(
        output / "curve_resolved_arm_evaluations.csv",
        focused_arm_fields,
        focused_arm_rows,
    )
    probability_fields = (
        "arm", "split", "regime", "rho", "raw_radius", "n_per_duration",
        *(_duration_field(duration) for duration in durations),
    )
    row_counts["probability_matrices.csv"] = _write_csv(
        output / "probability_matrices.csv",
        probability_fields,
        _probability_matrix_rows(probabilities, durations, rho_grid, raw_radii),
    )
    p50_fields = (
        "evaluation", "arm", "split", "regime", "rho", "raw_radius",
        "probability_threshold", "p50_target_duration", "censoring",
        "probability_at_p50", "probability_at_min_duration",
        "probability_at_max_duration",
    )
    row_counts["p50.csv"] = _write_csv(
        output / "p50.csv",
        p50_fields,
        _p50_rows(probabilities, durations, rho_grid, raw_radii, evaluation_indices),
    )
    p4p8_fields = (
        "arm", "split", "rho", "raw_radius", "n_duration_cells",
        "agreement_count", "agreement_fraction", "p4_greater_count",
        "p8_greater_count", "mean_difference_p4_minus_p8",
        "mean_absolute_difference", "max_absolute_difference",
        "differing_target_durations",
    )
    row_counts["period4_vs_period8.csv"] = _write_csv(
        output / "period4_vs_period8.csv",
        p4p8_fields,
        _period4_period8_rows(probabilities, durations, rho_grid, raw_radii),
    )
    pair_fields = (
        "pair", "left_arm", "right_arm", "split", "regime", "rho",
        "raw_radius", "target_duration", "left_nontrivial", "left_n",
        "left_probability", "right_nontrivial", "right_n", "right_probability",
        "difference_left_minus_right", "absolute_difference",
    )
    row_counts["pairwise_arm_differences.csv"] = _write_csv(
        output / "pairwise_arm_differences.csv",
        pair_fields,
        _pairwise_difference_rows(probabilities, durations, rho_grid, raw_radii),
    )
    pair_summary_fields = (
        "pair", "left_arm", "right_arm", "split", "regime", "rho",
        "raw_radius", "n_duration_cells", "agreement_count", "agreement_fraction",
        "left_greater_count", "right_greater_count",
        "mean_difference_left_minus_right", "mean_absolute_difference",
        "max_absolute_difference",
    )
    row_counts["pairwise_arm_difference_summary.csv"] = _write_csv(
        output / "pairwise_arm_difference_summary.csv",
        pair_summary_fields,
        _pairwise_summary_rows(probabilities, durations, rho_grid, raw_radii),
    )

    source_paths = {
        Path(__file__).resolve(),
        Path(analysis.__file__).resolve(),
        plan_path.resolve(),
        (plan_root / "jobs.json").resolve(),
        *[path.resolve() for path in manifest_paths],
        *[path.resolve() for path in local_paths],
        *[path.resolve() for path in external_paths],
    }
    input_hashes = {
        _relative_hash_key(path): _sha256(path) for path in sorted(source_paths)
    }
    (output / "input_hashes.json").write_text(
        json.dumps(input_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    row_counts["input_hashes.json"] = len(input_hashes)

    beta_vectors = {
        arm: {regime: records[arm][regime].beta1 for regime in REGIMES}
        for arm in ARM_ORDER
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fine_compass_c0p4_tangent_2x2_summary",
        "plan_root": str(plan_root),
        "plan_declared_status": plan.get("status"),
        "jobs_declared_status": jobs_document.get("status"),
        "materialized_result_status": "validated_complete",
        "analysis_policy": {
            "probability": "count(rank > 0) / every fixed manifest start",
            "splits": {"pooled": 20, "tune": 10, "validate": 10},
            "duration_axis": "exact target_duration values; no rebinning or median span",
            "radius_axis": "raw_radius = C * rho",
            "curve_resolution": "strict curve_bound < raw_radius with numerical epsilon",
            "p50": "first discrete target duration with P(rank > 0) >= 0.5; no interpolation",
            "nested_rank_decrease": (
                "count of adjacent target-duration steps with decreasing full rank "
                "within each fixed-start nested window sequence"
            ),
            "sustained_saturation_onset": (
                "first target duration from which P(rank > 0)=1 at every "
                "remaining duration; all-regime onset requires this jointly "
                "for all five regimes"
            ),
            "period4_vs_period8": "aggregate probability grids; individual starts are not paired across regimes",
        },
        "C": c_scale,
        "duration_grid_seconds": durations,
        "rho_grid": rho_grid,
        "raw_radius_grid": raw_radii,
        "arms": list(ARM_ORDER),
        "regimes": list(REGIMES),
        "beta_vectors": beta_vectors,
        "curve_bound_invariant_audit": invariant_audit,
        "common_curve_resolution": common_curve,
        "evaluation_rows": _evaluation_summary(
            probabilities, durations, rho_grid, raw_radii, evaluation_indices
        ),
        "arm_specific_curve_resolved_evaluations": focused_json,
        "input_hashes_file": "input_hashes.json",
        "output_row_counts": {**row_counts, "summary.json": 1},
        "output_files": list(OUTPUT_NAMES),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"validated four arms and wrote {len(OUTPUT_NAMES)} summaries beneath {output}")
    print(
        "common curve-resolved row: "
        f"rho={common_curve['first_resolved_rho']}, "
        f"raw_r={common_curve['first_resolved_raw_radius']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
