#!/usr/bin/env python3
"""Analyze a prepared fine-Compass duration/C/radius sweep in two stages.

This script never simulates a system, trains a model, or computes a cycling
signature.  ``tune`` reads only rows labelled ``tune`` when selecting a
connected (C, rho) plateau and freezes one discrete medoid.  An explicitly
one-C plan may instead request a plateau along rho.  ``validate`` reads only
rows labelled ``validate`` and evaluates that frozen cell.  Julia commands
for the recorded same-C cover-factorization controls are written as text by
``tune`` and are never launched here.

All analyzer outputs must be new paths below ``experiments_planned/outputs``.
The current runner stores tune and validation rows in one CSV; stage isolation
is therefore enforced logically and recorded in the output, rather than by
separate source files.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import shlex
import statistics
from collections import deque
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable


PERIODIC_REGIMES = ("period1", "period2", "period4", "period8")
REGIMES = PERIODIC_REGIMES + ("chaos",)
ADJACENT_REGIMES = (("period1", "period2"), ("period2", "period4"),
                    ("period4", "period8"))
REFERENCE_PERIODS = {
    "period1": 0.748241,
    "period2": 1.502140,
    "period4": 3.001914,
    "period8": 6.004312,
}
MANIFEST_FIELDS = (
    "target_duration",
    "split",
    "run_index",
    "start_index",
    "end_index",
    "realized_duration",
    "duration_error",
)
RESULT_FIELDS = MANIFEST_FIELDS + ("curve_bound", "rank", "births")
R0_FIELDS = (
    "regime",
    "C",
    "boxsize",
    "sb_radius",
    "r0_lower",
    "certified",
    "provenance",
)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ManifestRow:
    target_duration: float
    split: str
    run_index: int
    start_index: int
    end_index: int
    realized_duration: float
    duration_error: float


@dataclass(frozen=True)
class ResultRow:
    manifest: ManifestRow
    curve_bound: float
    declared_rank: int
    births: tuple[float, ...]


@dataclass(frozen=True)
class ResultRecord:
    regime: str
    c_scale: float
    metric_mode: str
    cover_default_c: float
    boxsize: float
    sb_radius: int
    rho_max: float
    r_max: float
    beta1: int
    n_windows: int
    metadata_path: Path
    births_path: Path
    positions_path: Path
    tangents_path: Path
    tangent_provenance_path: Path | None
    tangent_provenance_sha256: str | None
    rows: dict[tuple[float, str, int], ResultRow]
    grouped_rows: dict[tuple[str, float], tuple[ResultRow, ...]]


@dataclass(frozen=True)
class R0Entry:
    regime: str
    c_scale: float
    boxsize: float
    sb_radius: int
    lower_bound: float
    provenance: str


@dataclass
class ProbabilityRow:
    split: str
    regime: str
    c_scale: float
    rho: float
    radius: float
    target_duration: float
    realized_duration_min: float
    realized_duration_max: float
    n: int
    nontrivial: int
    probability: float
    curve_resolved: int
    curve_coverage: float
    beta1: int


@dataclass
class CellMetric:
    c_index: int
    rho_index: int
    c_scale: float
    rho: float
    radius: float
    beta_vector: tuple[int, ...]
    min_curve_coverage: float
    monotonicity_violations: int
    r0_status: str
    r0_margin_min: float | None
    onsets: dict[str, float | None]
    onset_censoring: dict[str, str]
    max_onset_error: float | None
    contrast_durations: dict[str, float]
    contrasts: dict[str, float]
    minimum_contrast: float | None
    candidate: bool
    rejection_reasons: list[str]
    pareto: bool = False
    near_optimal: bool = False
    plateau_id: int | None = None
    selected: bool = False


def _code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_root() -> Path:
    return (_code_root() / "experiments_planned" / "outputs").resolve()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _require_under(path: Path, parent: Path, label: str) -> Path:
    resolved = _resolved(path)
    allowed = _resolved(parent)
    if resolved != allowed and not resolved.is_relative_to(allowed):
        raise ValueError(f"{label} must stay below {allowed}: {resolved}")
    return resolved


def _existing_input(path: Path, label: str, *, directory: bool = False) -> Path:
    resolved = _require_under(path, _safe_root(), label)
    if not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    if directory and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def _new_output(path: Path) -> Path:
    resolved = _require_under(path, _safe_root(), "output directory")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite output: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_key(path: Path) -> str:
    resolved = path.resolve()
    root = _code_root().resolve()
    if resolved.is_relative_to(root):
        return str(resolved.relative_to(root))
    return str(resolved)


def _hash_files(paths: Iterable[Path]) -> dict[str, str]:
    return {_hash_key(path): _sha256(path) for path in sorted(set(paths))}


def _finite_float(value: str | float | int, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_float(value: str | float | int, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _metric_mode(value: object) -> str:
    mode = str(value if value is not None else "cover_default")
    if mode not in ("cover_default", "explicit"):
        raise ValueError(f"unsupported metric mode: {mode!r}")
    return mode


def _numeric_epsilon(*values: float) -> float:
    return 1e-12 * max(1.0, *(abs(value) for value in values))


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-10 * max(1.0, abs(left), abs(right))


def _format_number(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace(".", "p")


def _expand_grid(spec: object, label: str) -> list[float]:
    if not isinstance(spec, list) or len(spec) != 3:
        raise ValueError(f"{label} must be [start, step, stop]")
    start, step, stop = (Decimal(str(value)) for value in spec)
    if step <= 0 or stop < start:
        raise ValueError(f"invalid {label}")
    quotient = (stop - start) / step
    if quotient != quotient.to_integral_value():
        raise ValueError(f"{label} stop must lie exactly on its step grid")
    return [float(start + index * step) for index in range(int(quotient) + 1)]


def _load_plan(plan_root: Path) -> tuple[dict, Path, list[float], list[float], list[float]]:
    plan_path = plan_root / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"missing plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for key in (
        "c_grid",
        "rho_grid",
        "duration_grid_seconds",
        "n_starts_per_duration",
        "tune_starts",
        "validation_starts",
    ):
        if key not in plan:
            raise ValueError(f"plan lacks {key}")

    c_grid = [_positive_float(value, "C grid value") for value in plan["c_grid"]]
    durations = [
        _positive_float(value, "duration grid value")
        for value in plan["duration_grid_seconds"]
    ]
    rho_grid = _expand_grid(plan["rho_grid"], "rho_grid")
    if c_grid != sorted(set(c_grid)):
        raise ValueError("C grid must be strictly increasing and unique")
    if durations != sorted(set(durations)):
        raise ValueError("duration grid must be strictly increasing and unique")
    if rho_grid != sorted(set(rho_grid)) or rho_grid[0] < 0:
        raise ValueError("rho grid must be nonnegative and strictly increasing")

    primary_entries = plan.get("primary_constructions")
    if primary_entries is not None:
        if not isinstance(primary_entries, list) or len(primary_entries) != len(c_grid):
            raise ValueError("plan must provide one primary construction per C")
        seen_primary_c: set[float] = set()
        for entry in primary_entries:
            if not isinstance(entry, dict):
                raise ValueError("primary construction must be an object")
            c_scale = _match_grid(
                _positive_float(entry.get("C"), "primary C"), c_grid, "primary C"
            )
            boxsize = _positive_float(entry.get("boxsize"), "primary boxsize")
            sb_radius = int(entry.get("sb_radius", 0))
            metric_mode = _metric_mode(entry.get("metric_mode"))
            label = entry.get("label")
            if (
                sb_radius <= 0
                or not isinstance(label, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None
            ):
                raise ValueError("invalid primary construction")
            cover_default_c = _positive_float(
                entry.get("cover_default_C", boxsize * sb_radius),
                "primary cover-default C",
            )
            if not _close(cover_default_c, boxsize * sb_radius):
                raise ValueError("primary cover-default C is inconsistent")
            if metric_mode == "cover_default" and not _close(
                c_scale, cover_default_c
            ):
                raise ValueError("primary construction does not satisfy C=boxsize*sb")
            if c_scale in seen_primary_c:
                raise ValueError(f"duplicate primary construction for C={c_scale}")
            seen_primary_c.add(c_scale)

    experiment_entries = plan.get("experiment_arms")
    if experiment_entries is not None:
        if not isinstance(experiment_entries, list) or not experiment_entries:
            raise ValueError("experiment arms must be a nonempty list")
        seen_arms: set[tuple[str, float]] = set()
        for entry in experiment_entries:
            if not isinstance(entry, dict):
                raise ValueError("experiment arm must be an object")
            label = entry.get("label")
            if (
                not isinstance(label, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None
            ):
                raise ValueError("invalid experiment arm label")
            c_scale = _match_grid(
                _positive_float(entry.get("C"), "arm C"), c_grid, "arm C"
            )
            key = (label, c_scale)
            if key in seen_arms:
                raise ValueError(f"duplicate experiment arm: {key}")
            seen_arms.add(key)
            boxsize = _positive_float(entry.get("boxsize"), "arm boxsize")
            sb_radius = int(entry.get("sb_radius", 0))
            if sb_radius <= 0:
                raise ValueError("arm sb-radius must be positive")
            mode = _metric_mode(entry.get("metric_mode"))
            cover_default_c = _positive_float(
                entry.get("cover_default_C", boxsize * sb_radius),
                "arm cover-default C",
            )
            if not _close(cover_default_c, boxsize * sb_radius):
                raise ValueError("arm cover-default C is inconsistent")
            if mode == "cover_default" and not _close(c_scale, cover_default_c):
                raise ValueError("cover-default arm has an independent C")
            tangent_source = str(entry.get("tangent_source", "default"))
            if tangent_source not in ("default", "override"):
                raise ValueError("unsupported tangent source")
            tangents_dir = entry.get("tangents_dir")
            if tangent_source == "default" and tangents_dir is not None:
                raise ValueError("default tangent arm names an override directory")
            if tangent_source == "override" and not isinstance(tangents_dir, str):
                raise ValueError("override tangent arm lacks its directory")

    control_entries = plan.get("control_constructions")
    if control_entries is not None:
        if not isinstance(control_entries, list):
            raise ValueError("control constructions must be a list")
        seen_labels: set[str] = set()
        for entry in control_entries:
            if not isinstance(entry, dict):
                raise ValueError("control construction must be an object")
            label = entry.get("label")
            factor = _positive_float(
                entry.get("boxsize_over_C"), "control boxsize/C"
            )
            sb_radius = int(entry.get("sb_radius", 0))
            if (
                sb_radius <= 0
                or not isinstance(label, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None
                or label == "primary"
                or label in seen_labels
            ):
                raise ValueError("invalid or duplicate control construction")
            if not _close(factor * sb_radius, 1.0):
                raise ValueError("control construction must preserve C")
            seen_labels.add(label)

    n_starts = int(plan["n_starts_per_duration"])
    tune_starts = int(plan["tune_starts"])
    validation_starts = int(plan["validation_starts"])
    if min(n_starts, tune_starts, validation_starts) <= 0:
        raise ValueError("plan start counts must be positive")
    if tune_starts + validation_starts != n_starts:
        raise ValueError("tune and validation starts do not sum to n_starts")
    return plan, plan_path, c_grid, rho_grid, durations


def _primary_construction(plan: dict, c_scale: float) -> tuple[str, float, int]:
    entries = plan.get("primary_constructions")
    if entries is None:
        return "primary", c_scale, 1
    matches = [entry for entry in entries if _close(float(entry["C"]), c_scale)]
    if len(matches) != 1:
        raise ValueError(f"expected one primary construction for C={c_scale}")
    entry = matches[0]
    return str(entry["label"]), float(entry["boxsize"]), int(entry["sb_radius"])


def _experiment_arm(plan: dict, label: str, c_scale: float) -> dict | None:
    entries = plan.get("experiment_arms")
    if entries is None:
        return None
    matches = [
        entry
        for entry in entries
        if str(entry["label"]) == label and _close(float(entry["C"]), c_scale)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one experiment arm {label!r} at C={c_scale}")
    return matches[0]


def _primary_details(plan: dict, c_scale: float) -> dict[str, object]:
    entries = plan.get("primary_constructions")
    if entries is None:
        return {
            "label": "primary",
            "C": c_scale,
            "boxsize": c_scale,
            "sb_radius": 1,
            "metric_mode": "cover_default",
            "tangent_source": "default",
            "tangents_dir": None,
        }
    matches = [entry for entry in entries if _close(float(entry["C"]), c_scale)]
    if len(matches) != 1:
        raise ValueError(f"expected one primary construction for C={c_scale}")
    return matches[0]


def _details_tangents_path(details: dict[str, object], regime: str) -> Path:
    tangents_dir = details.get("tangents_dir")
    if tangents_dir is None:
        root = (
            _code_root()
            / "period_doubling"
            / "data_fine"
            / "compass_gait_latent"
        )
    else:
        root = Path(str(tangents_dir))
    return (root / f"compass_{regime}_tangents.csv").resolve()


def _control_constructions(plan: dict, c_scale: float) -> list[tuple[str, float, int]]:
    entries = plan.get("control_constructions")
    if entries is None:
        return [("factorization", c_scale / 2.0, 2)]
    _, primary_boxsize, primary_sb_radius = _primary_construction(
        plan, c_scale
    )
    controls = [
        (
            str(entry["label"]),
            c_scale * float(entry["boxsize_over_C"]),
            int(entry["sb_radius"]),
        )
        for entry in entries
    ]
    return [
        (label, boxsize, sb_radius)
        for label, boxsize, sb_radius in controls
        if not (
            _close(boxsize, primary_boxsize)
            and sb_radius == primary_sb_radius
        )
    ]


def _match_grid(value: float, grid: list[float], label: str) -> float:
    match = min(grid, key=lambda candidate: abs(candidate - value))
    if not _close(value, match):
        raise ValueError(f"{label} {value} is not on the planned grid")
    return match


def _manifest_key(row: ManifestRow) -> tuple[float, str, int]:
    return (row.target_duration, row.split, row.run_index)


def _parse_manifest_row(raw: dict[str, str], durations: list[float]) -> ManifestRow:
    duration = _match_grid(
        _finite_float(raw["target_duration"], "target_duration"),
        durations,
        "target_duration",
    )
    split = raw["split"]
    if split not in ("tune", "validate"):
        raise ValueError(f"invalid split: {split}")
    row = ManifestRow(
        target_duration=duration,
        split=split,
        run_index=int(raw["run_index"]),
        start_index=int(raw["start_index"]),
        end_index=int(raw["end_index"]),
        realized_duration=_finite_float(raw["realized_duration"], "realized_duration"),
        duration_error=_finite_float(raw["duration_error"], "duration_error"),
    )
    if row.run_index <= 0 or row.start_index <= 0 or row.end_index <= row.start_index:
        raise ValueError(f"invalid manifest indices: {row}")
    if row.realized_duration + _numeric_epsilon(row.realized_duration) < duration:
        raise ValueError(f"physical endpoint falls before target: {row}")
    if not _close(row.duration_error, row.realized_duration - duration):
        raise ValueError(f"duration error is inconsistent: {row}")
    return row


def _load_manifests(
    plan_root: Path,
    plan: dict,
    durations: list[float],
) -> tuple[dict[str, dict[tuple[float, str, int], ManifestRow]], list[Path]]:
    manifests: dict[str, dict[tuple[float, str, int], ManifestRow]] = {}
    paths: list[Path] = []
    n_starts = int(plan["n_starts_per_duration"])
    tune_starts = int(plan["tune_starts"])
    for regime in REGIMES:
        path = plan_root / "manifests" / f"{regime}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing manifest: {path}")
        paths.append(path)
        rows: dict[tuple[float, str, int], ManifestRow] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise ValueError(f"unexpected manifest header: {path}")
            for raw in reader:
                row = _parse_manifest_row(raw, durations)
                if row.run_index > n_starts:
                    raise ValueError(f"run index exceeds plan in {path}")
                expected_split = "tune" if row.run_index <= tune_starts else "validate"
                if row.split != expected_split:
                    raise ValueError(f"split/run mismatch in {path}: {row}")
                key = _manifest_key(row)
                if key in rows:
                    raise ValueError(f"duplicate manifest key in {path}: {key}")
                rows[key] = row

        expected = len(durations) * n_starts
        if len(rows) != expected:
            raise ValueError(f"{path} has {len(rows)} rows; expected {expected}")
        for duration in durations:
            found = [row for row in rows.values() if row.target_duration == duration]
            if len(found) != n_starts:
                raise ValueError(f"incomplete duration {duration} in {path}")
        for run_index in range(1, n_starts + 1):
            run_rows = sorted(
                (row for row in rows.values() if row.run_index == run_index),
                key=lambda row: row.target_duration,
            )
            if len({row.start_index for row in run_rows}) != 1:
                raise ValueError(f"start is not paired across durations: {path}, run {run_index}")
            if any(
                later.end_index < earlier.end_index
                for earlier, later in zip(run_rows, run_rows[1:])
            ):
                raise ValueError(f"endpoints decrease with duration: {path}, run {run_index}")
        manifests[regime] = rows
    return manifests, paths


def _parse_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in metadata:
            raise ValueError(f"duplicate metadata key {key}: {path}")
        metadata[key] = value
    required = {
        "base", "boxsize", "sb_radius", "C", "rho_max", "r_max",
        "beta1_Y", "n_windows",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"metadata lacks {sorted(missing)}: {path}")
    return metadata


def _parse_births(value: str, path: Path) -> tuple[float, ...]:
    if not value.strip():
        return ()
    births = tuple(_finite_float(item, "birth") for item in value.split(";") if item)
    if any(birth < 0 for birth in births):
        raise ValueError(f"negative birth in {path}")
    if list(births) != sorted(births):
        raise ValueError(f"unsorted birth vector in {path}")
    return births


def _load_result_record(
    metadata_path: Path,
    manifests: dict[str, dict[tuple[float, str, int], ManifestRow]],
    durations: list[float],
) -> ResultRecord:
    metadata = _parse_metadata(metadata_path)
    base = metadata["base"]
    if not base.startswith("compass_"):
        raise ValueError(f"unexpected base in {metadata_path}: {base}")
    regime = base.removeprefix("compass_")
    if regime not in REGIMES:
        raise ValueError(f"unexpected regime in {metadata_path}: {regime}")
    births_path = metadata_path.with_name(
        metadata_path.name.removesuffix("_metadata.txt") + "_births.csv"
    )
    if not births_path.is_file():
        raise FileNotFoundError(f"missing births file: {births_path}")

    c_scale = _positive_float(metadata["C"], "C")
    boxsize = _positive_float(metadata["boxsize"], "boxsize")
    sb_radius = int(metadata["sb_radius"])
    metric_mode = _metric_mode(metadata.get("metric_mode"))
    cover_default_c = _positive_float(
        metadata.get("cover_default_C", boxsize * sb_radius),
        "cover-default C",
    )
    if "metric_C" in metadata and not _close(
        _positive_float(metadata["metric_C"], "metric C"), c_scale
    ):
        raise ValueError(f"metric_C != C: {metadata_path}")
    rho_max = _positive_float(metadata["rho_max"], "rho_max")
    r_max = _positive_float(metadata["r_max"], "r_max")
    beta1 = int(metadata["beta1_Y"])
    n_windows = int(metadata["n_windows"])
    if sb_radius <= 0 or beta1 < 0 or n_windows <= 0:
        raise ValueError(f"invalid metadata values: {metadata_path}")
    if not _close(cover_default_c, boxsize * sb_radius):
        raise ValueError(f"cover_default_C != boxsize * sb_radius: {metadata_path}")
    if metric_mode == "cover_default" and not _close(c_scale, cover_default_c):
        raise ValueError(f"cover-default C != boxsize * sb_radius: {metadata_path}")
    if not _close(r_max, c_scale * rho_max):
        raise ValueError(f"r_max != C * rho_max: {metadata_path}")

    manifest_rows = manifests[regime]
    rows: dict[tuple[float, str, int], ResultRow] = {}
    with births_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise ValueError(f"unexpected result header: {births_path}")
        for raw in reader:
            manifest_row = _parse_manifest_row(raw, durations)
            key = _manifest_key(manifest_row)
            expected = manifest_rows.get(key)
            if expected is None or manifest_row != expected:
                raise ValueError(f"result/manifest mismatch in {births_path}: {key}")
            births = _parse_births(raw["births"], births_path)
            declared_rank = int(raw["rank"])
            curve_bound = _finite_float(raw["curve_bound"], "curve_bound")
            if curve_bound < 0:
                raise ValueError(f"negative curve bound in {births_path}")
            if declared_rank != len(births):
                raise ValueError(f"rank/birth-vector mismatch in {births_path}: {key}")
            if declared_rank > beta1:
                raise ValueError(f"rank exceeds beta1(Y) in {births_path}: {key}")
            if births and births[-1] > r_max + _numeric_epsilon(r_max):
                raise ValueError(f"birth exceeds r_max in {births_path}: {key}")
            if key in rows:
                raise ValueError(f"duplicate result key in {births_path}: {key}")
            rows[key] = ResultRow(manifest_row, curve_bound, declared_rank, births)

    if len(rows) != n_windows or rows.keys() != manifest_rows.keys():
        raise ValueError(f"result rows do not exactly match manifest: {births_path}")
    grouped_rows: dict[tuple[str, float], tuple[ResultRow, ...]] = {}
    for split in ("tune", "validate"):
        for duration in durations:
            group = tuple(
                row for key, row in sorted(rows.items())
                if key[0] == duration and key[1] == split
            )
            if not group:
                raise ValueError(
                    f"missing {split} result group at duration {duration}: {births_path}"
                )
            grouped_rows[(split, duration)] = group

    positions_path = Path(metadata.get("positions", "")).resolve()
    tangents_path = Path(metadata.get("tangents", "")).resolve()
    if not positions_path.is_file() or not tangents_path.is_file():
        raise FileNotFoundError(f"result metadata names missing lift inputs: {metadata_path}")
    for key, input_path in (
        ("positions_sha256", positions_path),
        ("tangents_sha256", tangents_path),
    ):
        if key in metadata and metadata[key] != _sha256(input_path):
            raise ValueError(f"{key} mismatch: {metadata_path}")
    manifest_path = Path(metadata.get("manifest", "")).resolve()
    if "manifest_sha256" in metadata and (
        not manifest_path.is_file()
        or metadata["manifest_sha256"] != _sha256(manifest_path)
    ):
        raise ValueError(f"manifest_sha256 mismatch: {metadata_path}")
    tangent_provenance_path: Path | None = None
    tangent_provenance_sha256: str | None = None
    if "tangent_provenance" in metadata:
        tangent_provenance_path = Path(metadata["tangent_provenance"]).resolve()
        if not tangent_provenance_path.is_file():
            raise FileNotFoundError(
                f"missing tangent provenance named by {metadata_path}"
            )
        tangent_provenance_sha256 = metadata.get("tangent_provenance_sha256")
        if tangent_provenance_sha256 != _sha256(tangent_provenance_path):
            raise ValueError(f"tangent provenance hash mismatch: {metadata_path}")

    return ResultRecord(
        regime=regime,
        c_scale=c_scale,
        metric_mode=metric_mode,
        cover_default_c=cover_default_c,
        boxsize=boxsize,
        sb_radius=sb_radius,
        rho_max=rho_max,
        r_max=r_max,
        beta1=beta1,
        n_windows=n_windows,
        metadata_path=metadata_path,
        births_path=births_path,
        positions_path=positions_path,
        tangents_path=tangents_path,
        tangent_provenance_path=tangent_provenance_path,
        tangent_provenance_sha256=tangent_provenance_sha256,
        rows=rows,
        grouped_rows=grouped_rows,
    )


def _scan_results(
    results_root: Path,
    manifests: dict[str, dict[tuple[float, str, int], ManifestRow]],
    durations: list[float],
    construction: tuple[float, float, int, float] | None = None,
) -> list[ResultRecord]:
    metadata_paths = sorted(results_root.rglob("*_metadata.txt"))
    if not metadata_paths:
        raise FileNotFoundError(f"no runner metadata below {results_root}")
    if construction is not None:
        c_scale, boxsize, sb_radius, required_rho = construction
        selected_paths: list[Path] = []
        for path in metadata_paths:
            metadata = _parse_metadata(path)
            base = metadata["base"]
            if not base.startswith("compass_"):
                continue
            regime = base.removeprefix("compass_")
            if (
                regime in REGIMES
                and _close(float(metadata["C"]), c_scale)
                and _close(float(metadata["boxsize"]), boxsize)
                and int(metadata["sb_radius"]) == sb_radius
                and float(metadata["rho_max"])
                + _numeric_epsilon(float(metadata["rho_max"])) >= required_rho
            ):
                selected_paths.append(path)
        metadata_paths = selected_paths
        if not metadata_paths:
            raise FileNotFoundError(
                "no runner metadata matches the frozen validation construction"
            )
    records = [
        _load_result_record(path, manifests, durations) for path in metadata_paths
    ]
    fingerprints: set[tuple[str, float, str, float, int, str]] = set()
    for record in records:
        fingerprint = (
            record.regime,
            round(record.c_scale, 12),
            record.metric_mode,
            round(record.boxsize, 12),
            record.sb_radius,
            str(record.tangents_path),
        )
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate result construction: {fingerprint}")
        fingerprints.add(fingerprint)
    return records


def _select_record(
    records: list[ResultRecord],
    regime: str,
    c_scale: float,
    boxsize: float,
    sb_radius: int,
    required_rho: float,
    metric_mode: str | None = None,
    tangents_path: Path | None = None,
) -> ResultRecord:
    matches = [
        record for record in records
        if record.regime == regime
        and _close(record.c_scale, c_scale)
        and _close(record.boxsize, boxsize)
        and record.sb_radius == sb_radius
        and (metric_mode is None or record.metric_mode == metric_mode)
        and (
            tangents_path is None
            or record.tangents_path == tangents_path.resolve()
        )
        and record.rho_max + _numeric_epsilon(record.rho_max) >= required_rho
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected one result for "
            f"{regime}, C={c_scale}, boxsize={boxsize}, sb_radius={sb_radius}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_r0(path: Path | None) -> tuple[dict[tuple[str, float, float, int], R0Entry], Path | None]:
    if path is None:
        return {}, None
    resolved = path.expanduser().resolve(strict=True)
    entries: dict[tuple[str, float, float, int], R0Entry] = {}
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != R0_FIELDS:
            raise ValueError(
                f"r0 CSV header must be {','.join(R0_FIELDS)}: {resolved}"
            )
        for raw in reader:
            regime = raw["regime"]
            if regime not in REGIMES:
                raise ValueError(f"invalid r0 regime: {regime}")
            if raw["certified"].strip().lower() not in ("true", "1", "yes"):
                raise ValueError(f"r0 row is not certified: {raw}")
            provenance = raw["provenance"].strip()
            if not provenance:
                raise ValueError(f"r0 row lacks provenance: {raw}")
            entry = R0Entry(
                regime=regime,
                c_scale=_positive_float(raw["C"], "r0 C"),
                boxsize=_positive_float(raw["boxsize"], "r0 boxsize"),
                sb_radius=int(raw["sb_radius"]),
                lower_bound=_positive_float(raw["r0_lower"], "r0 lower bound"),
                provenance=provenance,
            )
            if entry.sb_radius <= 0 or not _close(
                entry.c_scale, entry.boxsize * entry.sb_radius
            ):
                raise ValueError(f"invalid r0 construction: {raw}")
            key = (
                regime,
                round(entry.c_scale, 12),
                round(entry.boxsize, 12),
                entry.sb_radius,
            )
            if key in entries:
                raise ValueError(f"duplicate r0 row: {key}")
            entries[key] = entry
    if not entries:
        raise ValueError(f"r0 CSV is empty: {resolved}")
    return entries, resolved


def _r0_entry(
    entries: dict[tuple[str, float, float, int], R0Entry],
    regime: str,
    c_scale: float,
    boxsize: float,
    sb_radius: int,
) -> R0Entry | None:
    return entries.get(
        (regime, round(c_scale, 12), round(boxsize, 12), sb_radius)
    )


def _rank_at(row: ResultRow, radius: float, c_scale: float) -> int:
    epsilon = _numeric_epsilon(radius, c_scale)
    return bisect.bisect_right(row.births, radius + epsilon)


def _compute_cell(
    records: dict[str, ResultRecord],
    split: str,
    c_index: int,
    rho_index: int,
    c_scale: float,
    rho: float,
    durations: list[float],
    onset_probability: float,
    onset_tolerance: float,
    contrast_factor: float,
    minimum_contrast: float,
    r0_entries: dict[tuple[str, float, float, int], R0Entry],
    r0_required: bool,
) -> tuple[CellMetric, list[ProbabilityRow]]:
    radius = c_scale * rho
    probabilities: list[ProbabilityRow] = []
    by_regime: dict[str, dict[float, ProbabilityRow]] = {}
    monotonicity_violations = 0
    min_coverage = 1.0
    beta_vector = tuple(records[regime].beta1 for regime in REGIMES)

    for regime in REGIMES:
        record = records[regime]
        regime_rows: dict[float, ProbabilityRow] = {}
        ranks_by_run: dict[int, list[int]] = {}
        for duration in durations:
            rows = record.grouped_rows[(split, duration)]
            ranks = [_rank_at(row, radius, c_scale) for row in rows]
            curve_ok = [
                row.curve_bound < radius - _numeric_epsilon(radius, row.curve_bound)
                for row in rows
            ]
            for row, rank in zip(rows, ranks):
                ranks_by_run.setdefault(row.manifest.run_index, []).append(rank)
            nontrivial = sum(rank > 0 for rank in ranks)
            curve_count = sum(curve_ok)
            aggregate = ProbabilityRow(
                split=split,
                regime=regime,
                c_scale=c_scale,
                rho=rho,
                radius=radius,
                target_duration=duration,
                realized_duration_min=min(
                    row.manifest.realized_duration for row in rows
                ),
                realized_duration_max=max(
                    row.manifest.realized_duration for row in rows
                ),
                n=len(rows),
                nontrivial=nontrivial,
                probability=nontrivial / len(rows),
                curve_resolved=curve_count,
                curve_coverage=curve_count / len(rows),
                beta1=record.beta1,
            )
            probabilities.append(aggregate)
            regime_rows[duration] = aggregate
            min_coverage = min(min_coverage, aggregate.curve_coverage)
        for ranks in ranks_by_run.values():
            monotonicity_violations += sum(
                later < earlier for earlier, later in zip(ranks, ranks[1:])
            )
        by_regime[regime] = regime_rows

    onsets: dict[str, float | None] = {}
    censoring: dict[str, str] = {}
    for regime in PERIODIC_REGIMES:
        curve = [by_regime[regime][duration].probability for duration in durations]
        if curve[0] >= onset_probability:
            onsets[regime] = None
            censoring[regime] = "left"
            continue
        crossing = next(
            (duration for duration, probability in zip(durations, curve)
             if probability >= onset_probability),
            None,
        )
        onsets[regime] = crossing
        censoring[regime] = "right" if crossing is None else "none"

    contrast_durations: dict[str, float] = {}
    contrasts: dict[str, float] = {}
    for shorter, longer in ADJACENT_REGIMES:
        target = contrast_factor * REFERENCE_PERIODS[shorter]
        query = next((duration for duration in durations if duration >= target), None)
        if query is None:
            raise ValueError(
                f"duration grid does not reach {contrast_factor} * T_{shorter}"
            )
        contrast_durations[f"{shorter}_vs_{longer}"] = query
        contrasts[f"{shorter}_vs_{longer}"] = (
            by_regime[shorter][query].probability
            - by_regime[longer][query].probability
        )

    rejection_reasons: list[str] = []
    if any(beta <= 0 for beta in beta_vector):
        rejection_reasons.append("beta1_zero")
    if min_coverage < 1.0:
        rejection_reasons.append("curve_coverage_below_one")
    if monotonicity_violations:
        rejection_reasons.append("nested_rank_decrease")
    if any(censoring[regime] != "none" for regime in PERIODIC_REGIMES):
        rejection_reasons.append("onset_censored")

    max_onset_error: float | None = None
    if all(onsets[regime] is not None for regime in PERIODIC_REGIMES):
        max_onset_error = max(
            abs(float(onsets[regime]) - REFERENCE_PERIODS[regime])
            for regime in PERIODIC_REGIMES
        )
        if max_onset_error > onset_tolerance + _numeric_epsilon(onset_tolerance):
            rejection_reasons.append("onset_outside_tolerance")

    minimum_observed_contrast = min(contrasts.values())
    if minimum_observed_contrast + _numeric_epsilon(minimum_observed_contrast) < minimum_contrast:
        rejection_reasons.append("contrast_below_threshold")

    r0_status = "unverified"
    r0_margin_min: float | None = None
    if r0_entries:
        margins: list[float] = []
        missing = False
        for regime in REGIMES:
            record = records[regime]
            entry = _r0_entry(
                r0_entries, regime, c_scale, record.boxsize, record.sb_radius
            )
            if entry is None:
                missing = True
                continue
            margins.append(entry.lower_bound - radius)
        if missing:
            r0_status = "missing_certified_bound"
            rejection_reasons.append("r0_bound_missing")
        else:
            r0_margin_min = min(margins)
            if all(margin > _numeric_epsilon(radius, margin) for margin in margins):
                r0_status = "certified_pass"
            else:
                r0_status = "certified_fail"
                rejection_reasons.append("r0_bound_failed")
    elif r0_required:
        rejection_reasons.append("r0_bound_required")

    metric = CellMetric(
        c_index=c_index,
        rho_index=rho_index,
        c_scale=c_scale,
        rho=rho,
        radius=radius,
        beta_vector=beta_vector,
        min_curve_coverage=min_coverage,
        monotonicity_violations=monotonicity_violations,
        r0_status=r0_status,
        r0_margin_min=r0_margin_min,
        onsets=onsets,
        onset_censoring=censoring,
        max_onset_error=max_onset_error,
        contrast_durations=contrast_durations,
        contrasts=contrasts,
        minimum_contrast=minimum_observed_contrast,
        candidate=not rejection_reasons,
        rejection_reasons=rejection_reasons,
    )
    return metric, probabilities


def _components(metrics: dict[tuple[int, int], CellMetric]) -> list[list[CellMetric]]:
    remaining = set(metrics)
    components: list[list[CellMetric]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue = deque([seed])
        component: list[CellMetric] = []
        beta_vector = metrics[seed].beta_vector
        while queue:
            node = queue.popleft()
            component.append(metrics[node])
            for neighbor in (
                (node[0] - 1, node[1]),
                (node[0] + 1, node[1]),
                (node[0], node[1] - 1),
                (node[0], node[1] + 1),
            ):
                if neighbor in remaining and metrics[neighbor].beta_vector == beta_vector:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _component_key(component: list[CellMetric]) -> tuple:
    return (
        -len(component),
        -statistics.median(float(metric.minimum_contrast) for metric in component),
        max(float(metric.max_onset_error) for metric in component),
        min(metric.c_scale for metric in component),
        min(metric.rho for metric in component),
    )


def _medoid(component: list[CellMetric]) -> CellMetric:
    def key(candidate: CellMetric) -> tuple:
        distance = sum(
            abs(candidate.c_index - other.c_index)
            + abs(candidate.rho_index - other.rho_index)
            for other in component
        )
        return (
            distance,
            -float(candidate.minimum_contrast),
            float(candidate.max_onset_error),
            candidate.c_scale,
            candidate.rho,
        )

    return min(component, key=key)


def _metric_dict(metric: CellMetric) -> dict:
    result = asdict(metric)
    result["beta_vector"] = dict(zip(REGIMES, metric.beta_vector))
    return result


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_probability_csv(path: Path, rows: list[ProbabilityRow]) -> None:
    fields = tuple(ProbabilityRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_cell_csv(path: Path, metrics: list[CellMetric]) -> None:
    fields = (
        "C", "rho", "radius", "beta_vector", "min_curve_coverage",
        "monotonicity_violations", "r0_status", "r0_margin_min", "onsets",
        "onset_censoring", "max_onset_error", "contrast_durations", "contrasts",
        "minimum_contrast", "candidate", "rejection_reasons", "pareto", "near_optimal",
        "plateau_id", "selected",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in sorted(metrics, key=lambda item: (item.c_index, item.rho_index)):
            writer.writerow({
                "C": metric.c_scale,
                "rho": metric.rho,
                "radius": metric.radius,
                "beta_vector": json.dumps(dict(zip(REGIMES, metric.beta_vector)), sort_keys=True),
                "min_curve_coverage": metric.min_curve_coverage,
                "monotonicity_violations": metric.monotonicity_violations,
                "r0_status": metric.r0_status,
                "r0_margin_min": metric.r0_margin_min,
                "onsets": json.dumps(metric.onsets, sort_keys=True),
                "onset_censoring": json.dumps(metric.onset_censoring, sort_keys=True),
                "max_onset_error": metric.max_onset_error,
                "contrast_durations": json.dumps(metric.contrast_durations, sort_keys=True),
                "contrasts": json.dumps(metric.contrasts, sort_keys=True),
                "minimum_contrast": metric.minimum_contrast,
                "candidate": metric.candidate,
                "rejection_reasons": ";".join(metric.rejection_reasons),
                "pareto": metric.pareto,
                "near_optimal": metric.near_optimal,
                "plateau_id": metric.plateau_id,
                "selected": metric.selected,
            })


def _validate_thresholds(args: argparse.Namespace) -> dict:
    probability = _finite_float(args.onset_probability, "onset probability")
    if not 0 < probability < 1:
        raise ValueError("--onset-probability must lie strictly between 0 and 1")
    onset_tolerance = _positive_float(
        args.onset_tolerance_seconds, "onset tolerance"
    )
    contrast_factor = _positive_float(args.contrast_time_factor, "contrast factor")
    minimum_contrast = _finite_float(args.minimum_contrast, "minimum contrast")
    if not -1 <= minimum_contrast <= 1:
        raise ValueError("--minimum-contrast must lie in [-1,1]")
    slack = _finite_float(
        args.near_optimal_contrast_slack, "near-optimal contrast slack"
    )
    if not 0 <= slack <= 2:
        raise ValueError("--near-optimal-contrast-slack must lie in [0,2]")
    if args.minimum_plateau_c_values < 1:
        raise ValueError("the C plateau span must be at least one grid value")
    if args.minimum_plateau_rho_values < 2:
        raise ValueError("the rho plateau span must be at least two grid values")
    return {
        "onset_probability": probability,
        "onset_tolerance_seconds": onset_tolerance,
        "contrast_time_factor": contrast_factor,
        "minimum_contrast": minimum_contrast,
        "near_optimal_contrast_slack": slack,
        "minimum_plateau_c_values": args.minimum_plateau_c_values,
        "minimum_plateau_rho_values": args.minimum_plateau_rho_values,
        "curve_coverage_required": 1.0,
        "reference_periods_seconds": REFERENCE_PERIODS,
    }


def _generated_commands(
    plan: dict,
    plan_root: Path,
    results_root: Path,
    selection_path: Path,
    selected: CellMetric,
    tag: str,
    r0_path: Path | None,
) -> tuple[list[str], dict[str, object], list[dict[str, object]]]:
    code_root = _code_root().resolve()
    python = code_root / "venv" / "bin" / "python"
    analyzer = Path(__file__).resolve()
    runner = code_root / "experiments_planned" / "run_duration_c_radius.jl"
    julia_project = code_root / "period_doubling" / "julia"
    data_dir = code_root / "period_doubling" / "data_fine" / "compass_gait_latent"
    primary_validation_output = plan_root / "analysis" / f"{tag}_primary_validation"

    primary_args = [
        str(python), str(analyzer), "validate",
        "--plan-root", str(plan_root),
        "--results-root", str(results_root),
        "--selection", str(selection_path),
        "--output-dir", str(primary_validation_output),
        "--arm", "primary",
    ]
    if r0_path is not None:
        primary_args.extend(["--r0-csv", str(r0_path)])
    commands = [shlex.join(primary_args)]
    paths: dict[str, object] = {
        "primary_validation_output": str(primary_validation_output),
        "controls": {},
    }
    control_jobs: list[dict[str, object]] = []

    selected_c = selected.c_scale
    selected_rho = selected.rho
    c_suffix = _format_number(selected_c)
    control_root = plan_root / "factorization" / tag
    for label, boxsize, sb_radius in _control_constructions(plan, selected_c):
        results_dir = control_root / label
        validation_output = plan_root / "analysis" / f"{tag}_{label}_validation"
        box_suffix = _format_number(boxsize)
        for regime in REGIMES:
            out_dir = results_dir / regime
            prefix = (
                f"fine_{regime}_C_{c_suffix}_bs_{box_suffix}_sb{sb_radius}"
            )
            command = [
                "julia", f"--project={julia_project}", str(runner),
                "--data-dir", str(data_dir),
                "--base", f"compass_{regime}",
                "--manifest", str(plan_root / "manifests" / f"{regime}.csv"),
                "--boxsize", f"{boxsize:.12g}",
                "--sb-radius", str(sb_radius),
                "--rho-max", f"{selected_rho:.12g}",
                "--out-dir", str(out_dir),
                "--out-prefix", prefix,
            ]
            commands.append(shlex.join(command))
            control_jobs.append(
                {
                    "job_id": f"{label}__{regime}__C_{c_suffix}",
                    "arm": label,
                    "regime": regime,
                    "C": selected_c,
                    "boxsize": boxsize,
                    "sb_radius": sb_radius,
                    "rho_max": selected_rho,
                    "data_dir": str(data_dir),
                    "manifest": str(
                        plan_root / "manifests" / f"{regime}.csv"
                    ),
                    "out_dir": str(out_dir),
                    "out_prefix": prefix,
                    "argv": command,
                }
            )

        control_args = [
            str(python), str(analyzer), "validate",
            "--plan-root", str(plan_root),
            "--results-root", str(results_dir),
            "--selection", str(selection_path),
            "--output-dir", str(validation_output),
            "--arm", label,
        ]
        if r0_path is not None:
            control_args.extend(["--r0-csv", str(r0_path)])
        commands.append(shlex.join(control_args))
        paths["controls"][label] = {
            "results_root": str(results_dir),
            "validation_output": str(validation_output),
            "boxsize": boxsize,
            "sb_radius": sb_radius,
        }
    return commands, paths, control_jobs


def tune(args: argparse.Namespace) -> int:
    thresholds = _validate_thresholds(args)
    if args.require_certified_r0 and args.r0_csv is None:
        raise ValueError("--require-certified-r0 also requires --r0-csv")
    plan_root = _existing_input(args.plan_root, "plan root", directory=True)
    results_root = _existing_input(
        args.results_root or plan_root / "signatures", "results root", directory=True
    )
    output_dir = _new_output(args.output_dir)
    plan, plan_path, c_grid, rho_grid, durations = _load_plan(plan_root)
    if plan.get("experiment_arms") is not None:
        raise ValueError(
            "factorial diagnostic plans are audit-only; primary-only tune "
            "selection is intentionally disabled"
        )
    if args.minimum_plateau_c_values == 1 and len(c_grid) != 1:
        raise ValueError(
            "--minimum-plateau-c-values=1 is only valid for an explicit one-C plan"
        )
    if args.minimum_plateau_c_values > len(c_grid):
        raise ValueError("the requested C plateau span exceeds the planned C grid")
    manifests, manifest_paths = _load_manifests(plan_root, plan, durations)
    records = _scan_results(results_root, manifests, durations)
    r0_entries, r0_path = _load_r0(args.r0_csv)

    selected_records: dict[tuple[str, float], ResultRecord] = {}
    for c_scale in c_grid:
        _, boxsize, sb_radius = _primary_construction(plan, c_scale)
        details = _primary_details(plan, c_scale)
        metric_mode = _metric_mode(details.get("metric_mode"))
        for regime in REGIMES:
            selected_records[(regime, c_scale)] = _select_record(
                records,
                regime,
                c_scale,
                boxsize,
                sb_radius,
                rho_grid[-1],
                metric_mode,
                _details_tangents_path(details, regime),
            )

    source_paths = [plan_path, *manifest_paths, Path(__file__).resolve()]
    for record in selected_records.values():
        source_paths.extend([record.metadata_path, record.births_path])
    if r0_path is not None:
        source_paths.append(r0_path)
    source_hashes = _hash_files(source_paths)

    metrics: list[CellMetric] = []
    probability_rows: list[ProbabilityRow] = []
    for c_index, c_scale in enumerate(c_grid):
        regime_records = {
            regime: selected_records[(regime, c_scale)] for regime in REGIMES
        }
        for rho_index, rho in enumerate(rho_grid):
            metric, aggregates = _compute_cell(
                regime_records,
                "tune",
                c_index,
                rho_index,
                c_scale,
                rho,
                durations,
                thresholds["onset_probability"],
                thresholds["onset_tolerance_seconds"],
                thresholds["contrast_time_factor"],
                thresholds["minimum_contrast"],
                r0_entries,
                bool(args.require_certified_r0),
            )
            metrics.append(metric)
            probability_rows.extend(aggregates)

    threshold_only_reasons = {
        "onset_outside_tolerance", "contrast_below_threshold"
    }
    scoreable = [
        metric for metric in metrics
        if metric.max_onset_error is not None
        and metric.minimum_contrast is not None
        and set(metric.rejection_reasons).issubset(threshold_only_reasons)
    ]
    for metric in scoreable:
        dominated = any(
            other is not metric
            and float(other.max_onset_error) <= float(metric.max_onset_error)
            and float(other.minimum_contrast) >= float(metric.minimum_contrast)
            and (
                float(other.max_onset_error) < float(metric.max_onset_error)
                or float(other.minimum_contrast) > float(metric.minimum_contrast)
            )
            for other in scoreable
        )
        metric.pareto = not dominated

    candidates = [metric for metric in metrics if metric.candidate]
    selection: CellMetric | None = None
    selected_component: list[CellMetric] = []
    if candidates:
        best_contrast = max(float(metric.minimum_contrast) for metric in candidates)
        near = {
            (metric.c_index, metric.rho_index): metric
            for metric in candidates
            if float(metric.minimum_contrast)
            >= best_contrast - thresholds["near_optimal_contrast_slack"]
            - _numeric_epsilon(best_contrast)
        }
        for metric in near.values():
            metric.near_optimal = True
        components = _components(near)
        eligible_components = [
            component for component in components
            if len({metric.c_index for metric in component})
            >= thresholds["minimum_plateau_c_values"]
            and len({metric.rho_index for metric in component})
            >= thresholds["minimum_plateau_rho_values"]
        ]
        eligible_components.sort(key=_component_key)
        for plateau_id, component in enumerate(eligible_components, start=1):
            for metric in component:
                metric.plateau_id = plateau_id
        if eligible_components:
            selected_component = eligible_components[0]
            selection = _medoid(selected_component)
            selection.selected = True

    status = "frozen" if selection is not None else "no_qualifying_plateau"
    audit = {
        "schema_version": SCHEMA_VERSION,
        "stage": "tune",
        "status": status,
        "stage_isolation": (
            "Only rows with split=tune were aggregated or scored. Validation rows "
            "share the source CSVs but were not passed to the selection computation."
        ),
        "thresholds": thresholds,
        "r0_policy": {
            "required": bool(args.require_certified_r0),
            "source": str(r0_path) if r0_path else None,
            "status_without_source": "provisional_curve_resolved_only",
        },
        "formulae": {
            "radius": "r = C * rho",
            "rank": "count(birth <= r + numerical_epsilon)",
            "curve_gate": "curve_bound < r - numerical_epsilon",
            "probability": "count(rank > 0) / all fixed starts; never conditioned on curve_gate",
            "onset": "first target physical duration with probability >= onset_probability",
            "contrast": "P_short(d) - P_long(d) at first d >= contrast_time_factor * T_short",
            "connectivity": "4-neighbor C/rho grid, broken when the beta1 vector changes",
            "medoid": "minimum summed Manhattan grid distance within the selected plateau",
        },
        "regime_roles": {
            "alignment_and_contrast_score": list(PERIODIC_REGIMES),
            "diagnostic_probability_only": ["chaos"],
            "curve_beta1_r0_and_plateau_provenance": list(REGIMES),
        },
        "metric_scale_policy": (
            "One common numerical C is applied to separately trained latent encoders. "
            "The analyzer does not assert that their raw latent units are intrinsically "
            "commensurate."
        ),
        "primary_constructions": plan.get(
            "primary_constructions",
            [
                {
                    "C": c_scale,
                    "boxsize": c_scale,
                    "sb_radius": 1,
                    "label": "primary",
                }
                for c_scale in c_grid
            ],
        ),
        "reference_periods_seconds": REFERENCE_PERIODS,
        "input_hashes": source_hashes,
        "n_cells": len(metrics),
        "n_candidates": len(candidates),
        "n_pareto_diagnostic_cells": sum(metric.pareto for metric in metrics),
        "selected_plateau_size": len(selected_component),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_probability_csv(output_dir / "tune_probabilities.csv", probability_rows)
    _write_cell_csv(output_dir / "tune_cells.csv", metrics)
    _write_json(output_dir / "input_hashes.json", source_hashes)

    if selection is None:
        _write_json(output_dir / "tune_audit.json", audit)
        print(f"no qualifying plateau; diagnostics written beneath {output_dir}")
        return 2

    config_digest = hashlib.sha256(
        json.dumps(
            {"thresholds": thresholds, "cell": _metric_dict(selection)},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    tag = (
        f"C_{_format_number(selection.c_scale)}_rho_{_format_number(selection.rho)}_"
        f"{config_digest}"
    )
    selection_path = output_dir / "frozen_selection.json"
    commands, generated_paths, control_jobs = _generated_commands(
        plan,
        plan_root,
        results_root,
        selection_path,
        selection,
        tag,
        r0_path,
    )
    control_jobs_path = output_dir / "control_jobs.json"
    generated_paths["control_jobs_file"] = str(control_jobs_path)
    frozen = {
        **audit,
        "selection_tag": tag,
        "plan_root": str(plan_root),
        "primary_results_root": str(results_root),
        "selected_cell": _metric_dict(selection),
        "selected_plateau": [_metric_dict(metric) for metric in selected_component],
        "generated_paths": generated_paths,
        "r0_source_hash": _sha256(r0_path) if r0_path else None,
    }
    _write_json(selection_path, frozen)
    _write_json(output_dir / "tune_audit.json", audit)
    (output_dir / "commands.sh").write_text(
        "# Prepared commands only; review and run manually from code/.\n"
        + "\n".join(commands)
        + "\n",
        encoding="utf-8",
    )
    control_jobs_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "fine_compass_duration_c_radius_jobs",
                "status": "prepared_not_executed",
                "plan_root": str(plan_root),
                "working_directory": str(_code_root().resolve()),
                "selection": str(selection_path),
                "selection_tag": tag,
                "jobs": control_jobs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"froze C={selection.c_scale:g}, rho={selection.rho:g}; "
        f"outputs written beneath {output_dir}; no generated command was run"
    )
    return 0


def _verify_frozen_hashes(
    frozen: dict,
    paths: Iterable[Path],
    *,
    require_all: bool,
) -> dict[str, str]:
    current = _hash_files(paths)
    expected = frozen.get("input_hashes", {})
    for key, digest in current.items():
        if key not in expected:
            if require_all:
                raise ValueError(f"frozen selection lacks hash for {key}")
            continue
        if expected[key] != digest:
            raise ValueError(f"input changed since tuning: {key}")
    return current


def validate(args: argparse.Namespace) -> int:
    plan_root = _existing_input(args.plan_root, "plan root", directory=True)
    results_root = _existing_input(args.results_root, "results root", directory=True)
    selection_path = _existing_input(args.selection, "frozen selection")
    output_dir = _new_output(args.output_dir)
    frozen = json.loads(selection_path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != SCHEMA_VERSION or frozen.get("status") != "frozen":
        raise ValueError("selection is not a frozen analyzer selection")
    if _resolved(Path(frozen["plan_root"])) != plan_root:
        raise ValueError("selection belongs to a different plan root")

    thresholds = frozen["thresholds"]
    selected = frozen["selected_cell"]
    c_scale = _positive_float(selected["c_scale"], "selected C")
    rho = _finite_float(selected["rho"], "selected rho")
    if rho < 0:
        raise ValueError("selected rho is negative")
    plan, plan_path, c_grid, rho_grid, durations = _load_plan(plan_root)
    _match_grid(c_scale, c_grid, "selected C")
    _match_grid(rho, rho_grid, "selected rho")
    manifests, manifest_paths = _load_manifests(plan_root, plan, durations)
    _verify_frozen_hashes(
        frozen,
        [plan_path, *manifest_paths, Path(__file__).resolve()],
        require_all=True,
    )

    if args.arm == "primary":
        _, boxsize, sb_radius = _primary_construction(plan, c_scale)
        arm_details = _primary_details(plan, c_scale)
    else:
        experiment_arm = _experiment_arm(plan, args.arm, c_scale)
        if experiment_arm is not None:
            arm_details = experiment_arm
            boxsize = float(experiment_arm["boxsize"])
            sb_radius = int(experiment_arm["sb_radius"])
        else:
            controls = {
                label: (candidate_boxsize, candidate_sb_radius)
                for label, candidate_boxsize, candidate_sb_radius
                in _control_constructions(plan, c_scale)
            }
            if args.arm not in controls:
                raise ValueError(
                    f"unknown control arm {args.arm!r}; expected one of "
                    f"{', '.join(sorted(controls))}"
                )
            boxsize, sb_radius = controls[args.arm]
            arm_details = {
                "metric_mode": "cover_default",
                "tangents_dir": None,
            }
    records = _scan_results(
        results_root,
        manifests,
        durations,
        construction=(c_scale, boxsize, sb_radius, rho),
    )
    selected_records = {
        regime: _select_record(
            records,
            regime,
            c_scale,
            boxsize,
            sb_radius,
            rho,
            _metric_mode(arm_details.get("metric_mode")),
            _details_tangents_path(arm_details, regime),
        )
        for regime in REGIMES
    }
    result_paths: list[Path] = []
    for record in selected_records.values():
        result_paths.extend([record.metadata_path, record.births_path])
    if args.arm == "primary":
        _verify_frozen_hashes(frozen, result_paths, require_all=True)

    r0_entries, r0_path = _load_r0(args.r0_csv)
    tune_had_r0 = frozen.get("r0_policy", {}).get("source") is not None
    if args.arm == "primary":
        if tune_had_r0 and r0_path is None:
            raise ValueError("primary validation requires the frozen certified r0 CSV")
        if not tune_had_r0 and r0_path is not None:
            raise ValueError("cannot add an r0 gate after tuning")
        if r0_path is not None and _sha256(r0_path) != frozen.get("r0_source_hash"):
            raise ValueError("r0 CSV changed since tuning")

    metric, probabilities = _compute_cell(
        selected_records,
        "validate",
        int(selected["c_index"]),
        int(selected["rho_index"]),
        c_scale,
        rho,
        durations,
        float(thresholds["onset_probability"]),
        float(thresholds["onset_tolerance_seconds"]),
        float(thresholds["contrast_time_factor"]),
        float(thresholds["minimum_contrast"]),
        r0_entries,
        bool(frozen.get("r0_policy", {}).get("required", False)),
    )
    frozen_beta = tuple(int(selected["beta_vector"][regime]) for regime in REGIMES)
    beta_stable = metric.beta_vector == frozen_beta
    if not beta_stable:
        metric.rejection_reasons.append("beta1_vector_changed_from_frozen_cell")
        metric.candidate = False

    empirical_pass = metric.candidate
    rigorous_pass = empirical_pass and metric.r0_status == "certified_pass"
    if metric.r0_status == "unverified":
        interpretation = "curve_resolved_provisional_r0_unverified"
    elif rigorous_pass:
        interpretation = "validated_with_certified_r0_lower_bound"
    else:
        interpretation = "validation_failed"

    paths_to_hash = [plan_path, *manifest_paths, *result_paths, selection_path]
    if r0_path is not None:
        paths_to_hash.append(r0_path)
    validation = {
        "schema_version": SCHEMA_VERSION,
        "stage": "validate",
        "arm": args.arm,
        "status": interpretation,
        "stage_isolation": "Only rows with split=validate were aggregated.",
        "selection_tag": frozen["selection_tag"],
        "frozen_cell": selected,
        "validation_metric": _metric_dict(metric),
        "beta_vector_stable": beta_stable,
        "empirical_pass": empirical_pass,
        "rigorous_pass": rigorous_pass,
        "thresholds": thresholds,
        "input_hashes": _hash_files(paths_to_hash),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_probability_csv(output_dir / "validation_probabilities.csv", probabilities)
    _write_json(output_dir / "validation.json", validation)
    _write_json(output_dir / "input_hashes.json", validation["input_hashes"])
    print(
        f"{args.arm} validation: {interpretation}; outputs written beneath "
        f"{output_dir}; no experiment was run"
    )
    return 0 if empirical_pass else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Staged, prepared-only analysis of fine Compass C/r outputs.",
        epilog=(
            "A certified r0 CSV must have the exact header: "
            + ",".join(R0_FIELDS)
            + ". Every certified cell must provide a positive lower bound and "
            "nonempty provenance."
        ),
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    tune_parser = subparsers.add_parser(
        "tune", help="Select and freeze a plateau using tune rows only."
    )
    tune_parser.add_argument("--plan-root", type=Path, required=True)
    tune_parser.add_argument(
        "--results-root", type=Path,
        help="Primary runner outputs; defaults to PLAN_ROOT/signatures.",
    )
    tune_parser.add_argument("--output-dir", type=Path, required=True)
    tune_parser.add_argument("--r0-csv", type=Path)
    tune_parser.add_argument(
        "--require-certified-r0", action="store_true",
        help="Reject every cell unless a certified r0 lower bound is supplied and passed.",
    )
    tune_parser.add_argument("--onset-probability", type=float, required=True)
    tune_parser.add_argument(
        "--onset-tolerance-seconds", type=float, required=True
    )
    tune_parser.add_argument("--contrast-time-factor", type=float, required=True)
    tune_parser.add_argument("--minimum-contrast", type=float, required=True)
    tune_parser.add_argument(
        "--near-optimal-contrast-slack", type=float, required=True
    )
    tune_parser.add_argument(
        "--minimum-plateau-c-values", type=int, required=True
    )
    tune_parser.add_argument(
        "--minimum-plateau-rho-values", type=int, required=True
    )
    tune_parser.set_defaults(func=tune)

    validate_parser = subparsers.add_parser(
        "validate", help="Evaluate only the frozen cell using validation rows."
    )
    validate_parser.add_argument("--plan-root", type=Path, required=True)
    validate_parser.add_argument("--results-root", type=Path, required=True)
    validate_parser.add_argument("--selection", type=Path, required=True)
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument(
        "--arm",
        required=True,
        help=(
            "Use 'primary' or a control-construction label recorded in plan.json."
        ),
    )
    validate_parser.add_argument("--r0-csv", type=Path)
    validate_parser.set_defaults(func=validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
