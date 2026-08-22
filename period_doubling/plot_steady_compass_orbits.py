#!/usr/bin/env python
"""Extract and plot steady compass-gait sampled doubling returns.

This script is deliberately read-only with respect to the source archives.  It
loads the stored reference-simulator NPZ files, validates their post-impact return
periods, selects one deterministic late orbit per periodic regime, and writes
small derived extracts, diagnostics, and figures to a separate output
directory.  It never simulates, integrates, or trains a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REGIMES = (
    ("period1", 1),
    ("period2", 2),
    ("period4", 4),
    ("period8", 8),
)


class ValidationError(RuntimeError):
    """Raised when a source archive or periodicity check fails."""


@dataclass(frozen=True)
class CompassArchive:
    """Arrays and metadata loaded from one stored compass-gait archive."""

    path: Path
    source_name: str
    source_sha256: str
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    impact_times: np.ndarray
    jump_minus: np.ndarray
    jump_plus: np.ndarray
    meta: dict


@dataclass
class ExtractedOrbit:
    """One k-impact orbit represented as separate continuous arcs."""

    start_impact_index: int
    end_impact_index: int
    t_abs: np.ndarray
    t_rel: np.ndarray
    tau_p1: np.ndarray | None
    x: np.ndarray
    arc_offsets: np.ndarray
    event_indices: np.ndarray
    event_times_abs: np.ndarray
    event_times_rel: np.ndarray
    event_tau_p1: np.ndarray | None
    jump_minus: np.ndarray
    jump_plus: np.ndarray


@dataclass
class RegimeResult:
    """Validation metrics and extracted orbit for one regime."""

    regime: str
    expected_period: int
    archive: CompassArchive
    orbit: ExtractedOrbit
    metrics: dict
    checks: dict


def _is_within(path: Path, root: Path) -> bool:
    """Return whether *path* equals or is below *root*."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_output_dir(output_dir: Path, package_root: Path, data_dir: Path) -> Path:
    """Reject output locations in protected source-data and run directories."""

    output = output_dir.expanduser().resolve()
    package = package_root.resolve()
    code_root = package.parent

    try:
        relative_to_package = output.relative_to(package)
    except ValueError:
        relative_to_package = None

    if (
        relative_to_package is not None
        and relative_to_package.parts
        and relative_to_package.parts[0].startswith("data")
    ):
        raise ValidationError(
            f"refusing output below protected period_doubling/data*: {output}"
        )

    protected_roots = (
        data_dir.expanduser().resolve(),
        code_root / "runs",
        code_root / "chyll_v2" / "runs",
        code_root / "chyll_v2" / "cycling_signature" / "data",
        code_root / "chyll_v2" / "cycling_signature" / "handoffs",
    )
    for protected in protected_roots:
        protected = protected.resolve()
        if _is_within(output, protected):
            raise ValidationError(
                f"refusing output below protected directory: {protected}"
            )

    return output


def sha256_file(path: Path) -> str:
    """Compute a source-file checksum without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path, repo_root: Path) -> str:
    """Prefer a repository-relative path in reports when available."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def load_archive(path: Path, repo_root: Path) -> CompassArchive:
    """Load and structurally validate a stored hybrid-timeseries archive."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValidationError(f"missing source archive: {path}")

    required = {
        "t",
        "x",
        "v",
        "impact_times",
        "jump_minus",
        "jump_plus",
        "meta_json",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValidationError(f"{path} is missing arrays: {', '.join(missing)}")

        t = np.asarray(source["t"], dtype=float).copy()
        x = np.asarray(source["x"], dtype=float).copy()
        v = np.asarray(source["v"], dtype=float).copy()
        impact_times = np.asarray(source["impact_times"], dtype=float).copy()
        jump_minus = np.asarray(source["jump_minus"], dtype=float).copy()
        jump_plus = np.asarray(source["jump_plus"], dtype=float).copy()
        meta_raw = source["meta_json"].item()

    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode("utf-8")
    try:
        meta = json.loads(str(meta_raw))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid meta_json in {path}: {exc}") from exc

    if t.ndim != 1 or len(t) < 2:
        raise ValidationError(f"{path}: t must be a one-dimensional nontrivial array")
    if x.shape != (len(t), 4) or v.shape != (len(t), 4):
        raise ValidationError(
            f"{path}: expected x and v shapes ({len(t)}, 4), got {x.shape} and {v.shape}"
        )
    if impact_times.ndim != 1:
        raise ValidationError(f"{path}: impact_times must be one-dimensional")
    expected_jump_shape = (len(impact_times), 4)
    if jump_minus.shape != expected_jump_shape or jump_plus.shape != expected_jump_shape:
        raise ValidationError(
            f"{path}: expected jump arrays {expected_jump_shape}, got "
            f"{jump_minus.shape} and {jump_plus.shape}"
        )
    for name, array in (
        ("t", t),
        ("x", x),
        ("v", v),
        ("impact_times", impact_times),
        ("jump_minus", jump_minus),
        ("jump_plus", jump_plus),
    ):
        if not np.all(np.isfinite(array)):
            raise ValidationError(f"{path}: {name} contains non-finite values")
    if not np.all(np.diff(t) > 0.0):
        raise ValidationError(f"{path}: sample times are not strictly increasing")
    if not np.all(np.diff(impact_times) > 0.0):
        raise ValidationError(f"{path}: impact times are not strictly increasing")
    if impact_times[0] < t[0] or impact_times[-1] > t[-1]:
        raise ValidationError(f"{path}: impact times lie outside the sampled time span")

    return CompassArchive(
        path=path,
        source_name=display_path(path, repo_root),
        source_sha256=sha256_file(path),
        t=t,
        x=x,
        v=v,
        impact_times=impact_times,
        jump_minus=jump_minus,
        jump_plus=jump_plus,
        meta=meta,
    )


def cluster_jump_states(points: np.ndarray, link_tol: float) -> dict:
    """Single-linkage components used only as a return-map diagnostic."""

    if len(points) == 0:
        return {
            "n_clusters": 0,
            "cluster_sizes": [],
            "min_intercluster_distance": None,
            "n_points": 0,
        }
    if len(points) > 2000:
        points = points[-2000:]

    squared_norm = np.sum(points * points, axis=1)
    distance_squared = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * (points @ points.T)
    )
    np.maximum(distance_squared, 0.0, out=distance_squared)

    parent = np.arange(len(points))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    edge_i, edge_j = np.nonzero(
        np.triu(distance_squared <= link_tol * link_tol, k=1)
    )
    for left, right in zip(edge_i, edge_j):
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left

    roots = np.array([find(index) for index in range(len(points))])
    unique_roots = np.unique(roots)
    members = [np.flatnonzero(roots == root) for root in unique_roots]
    centroids = [np.mean(points[indices], axis=0) for indices in members]

    min_intercluster = None
    if len(centroids) > 1:
        min_intercluster = min(
            float(np.linalg.norm(centroids[left] - centroids[right]))
            for left in range(len(centroids))
            for right in range(left + 1, len(centroids))
        )

    return {
        "n_clusters": len(members),
        "cluster_sizes": sorted((len(indices) for indices in members), reverse=True),
        "min_intercluster_distance": min_intercluster,
        "n_points": len(points),
    }


def proper_divisors(period: int) -> list[int]:
    """Return positive proper divisors of an expected period."""

    return [lag for lag in range(1, period) if period % lag == 0]


def select_start_impact(
    archive: CompassArchive,
    period: int,
    trim_impacts: int,
    tail_cycles: int,
) -> tuple[int, int, dict[int, float]]:
    """Choose a deterministic late phase anchor without optimizing closure."""

    n_impacts = len(archive.impact_times)
    valid_starts = np.arange(trim_impacts, n_impacts - period, dtype=int)
    if len(valid_starts) == 0:
        raise ValidationError(
            f"{archive.path}: no complete period-{period} return after trimming"
        )

    tail_floor = max(trim_impacts, n_impacts - period - tail_cycles * period)
    tail_impacts = np.arange(tail_floor, n_impacts, dtype=int)
    phase_medians = {}
    for phase in range(period):
        phase_indices = tail_impacts[(tail_impacts - trim_impacts) % period == phase]
        if len(phase_indices):
            phase_medians[phase] = float(
                np.median(archive.jump_plus[phase_indices, 0])
            )
    if len(phase_medians) != period:
        raise ValidationError(
            f"{archive.path}: tail window does not contain all {period} phase classes"
        )

    anchor_phase = min(phase_medians, key=lambda phase: (phase_medians[phase], phase))
    candidates = valid_starts[
        ((valid_starts - trim_impacts) % period == anchor_phase)
        & (valid_starts >= tail_floor)
    ]
    if len(candidates) == 0:
        candidates = valid_starts[
            (valid_starts - trim_impacts) % period == anchor_phase
        ]
    if len(candidates) == 0:
        raise ValidationError(f"{archive.path}: could not realize the phase anchor")

    return int(candidates[-1]), int(anchor_phase), phase_medians


def extract_orbit(
    archive: CompassArchive,
    start_impact: int,
    period: int,
) -> ExtractedOrbit:
    """Extract k continuous arcs with recorded event endpoint states."""

    pieces_t = []
    pieces_x = []
    arc_offsets = [0]

    for phase in range(period):
        left_index = start_impact + phase
        right_index = left_index + 1
        left_time = archive.impact_times[left_index]
        right_time = archive.impact_times[right_index]
        interior = (archive.t > left_time) & (archive.t < right_time)
        if not np.any(interior):
            raise ValidationError(
                f"{archive.path}: no raw samples inside impact interval {left_index}"
            )

        arc_t = np.concatenate(
            ([left_time], archive.t[interior], [right_time])
        )
        arc_x = np.vstack(
            (
                archive.jump_plus[left_index],
                archive.x[interior],
                archive.jump_minus[right_index],
            )
        )
        if not np.all(np.diff(arc_t) > 0.0):
            raise ValidationError(
                f"{archive.path}: extracted arc {phase + 1} has nonmonotone times"
            )

        pieces_t.append(arc_t)
        pieces_x.append(arc_x)
        arc_offsets.append(arc_offsets[-1] + len(arc_t))

    t_abs = np.concatenate(pieces_t)
    x = np.vstack(pieces_x)
    start_time = archive.impact_times[start_impact]
    event_indices = np.arange(start_impact, start_impact + period + 1, dtype=int)
    event_times = archive.impact_times[event_indices]

    return ExtractedOrbit(
        start_impact_index=start_impact,
        end_impact_index=start_impact + period,
        t_abs=t_abs,
        t_rel=t_abs - start_time,
        tau_p1=None,
        x=x,
        arc_offsets=np.asarray(arc_offsets, dtype=int),
        event_indices=event_indices,
        event_times_abs=event_times,
        event_times_rel=event_times - start_time,
        event_tau_p1=None,
        jump_minus=archive.jump_minus[event_indices].copy(),
        jump_plus=archive.jump_plus[event_indices].copy(),
    )


def analyze_regime(
    regime: str,
    period: int,
    archive: CompassArchive,
    trim_impacts: int,
    tail_cycles: int,
    link_tol: float,
    closure_atol: float,
) -> RegimeResult:
    """Validate recurrence and extract one representative steady orbit."""

    meta = archive.meta
    meta_period = meta.get("expected_period")
    meta_label = meta.get("label")
    burn_in = meta.get("burn_in_strides")
    n_impacts = len(archive.impact_times)

    if meta_label != regime:
        raise ValidationError(
            f"{archive.path}: metadata label {meta_label!r} does not match {regime!r}"
        )
    if meta_period is None or int(meta_period) != period:
        raise ValidationError(
            f"{archive.path}: metadata expected_period {meta_period!r} does not match {period}"
        )
    if burn_in is None or int(burn_in) < 80:
        raise ValidationError(
            f"{archive.path}: expected at least 80 stored burn-in strides, got {burn_in!r}"
        )
    if trim_impacts < 0:
        raise ValidationError("trim_impacts must be nonnegative")
    if n_impacts <= trim_impacts + 2 * period:
        raise ValidationError(
            f"{archive.path}: insufficient impacts for trim={trim_impacts}, period={period}"
        )

    active_start = archive.jump_plus[trim_impacts : n_impacts - period]
    active_return = archive.jump_plus[trim_impacts + period :]
    closure_errors = np.linalg.norm(active_return - active_start, axis=1)
    cycle_times = (
        archive.impact_times[trim_impacts + period :]
        - archive.impact_times[trim_impacts : n_impacts - period]
    )

    recurrence_lag_stats = {}
    for lag in range(1, min(16, n_impacts - trim_impacts - 1) + 1):
        lag_errors = np.linalg.norm(
            archive.jump_plus[trim_impacts + lag :]
            - archive.jump_plus[trim_impacts : n_impacts - lag],
            axis=1,
        )
        recurrence_lag_stats[str(lag)] = {
            "min": float(np.min(lag_errors)),
            "median": float(np.median(lag_errors)),
            "rms": float(np.sqrt(np.mean(lag_errors * lag_errors))),
            "q95": float(np.quantile(lag_errors, 0.95)),
            "max": float(np.max(lag_errors)),
        }

    divisor_distance_stats = {}
    for divisor in proper_divisors(period):
        divisor_errors = np.linalg.norm(
            archive.jump_plus[trim_impacts + divisor :]
            - archive.jump_plus[trim_impacts : n_impacts - divisor],
            axis=1,
        )
        divisor_distance_stats[divisor] = {
            "min": float(np.min(divisor_errors)),
            "median": float(np.median(divisor_errors)),
            "rms": float(np.sqrt(np.mean(divisor_errors * divisor_errors))),
            "q95": float(np.quantile(divisor_errors, 0.95)),
            "max": float(np.max(divisor_errors)),
        }

    cluster_result = cluster_jump_states(
        archive.jump_plus[trim_impacts:], link_tol=link_tol
    )
    start_impact, anchor_phase, phase_medians = select_start_impact(
        archive,
        period=period,
        trim_impacts=trim_impacts,
        tail_cycles=tail_cycles,
    )
    orbit = extract_orbit(archive, start_impact=start_impact, period=period)
    selected_closure = float(
        np.linalg.norm(
            archive.jump_plus[start_impact + period]
            - archive.jump_plus[start_impact]
        )
    )
    selected_cycle_time = float(
        archive.impact_times[start_impact + period]
        - archive.impact_times[start_impact]
    )

    checks = {
        "metadata_matches": True,
        "burn_in_at_least_80": True,
        "recurrence_closure": bool(np.max(closure_errors) <= closure_atol),
        "selected_endpoint_closure": bool(selected_closure <= closure_atol),
        "proper_divisors_do_not_close": bool(
            all(
                stats["min"] > link_tol
                for stats in divisor_distance_stats.values()
            )
        ),
        "exactly_k_extracted_arcs": bool(len(orbit.arc_offsets) - 1 == period),
    }
    metrics = {
        "source_dt": float(np.median(np.diff(archive.t))),
        "n_samples": len(archive.t),
        "n_impacts": n_impacts,
        "post_trim_impacts": n_impacts - trim_impacts,
        "burn_in_strides": int(burn_in),
        "phi_deg": float(meta["phi_deg"]),
        "detected_clusters": int(cluster_result["n_clusters"]),
        "cluster_count_matches_expected": bool(
            cluster_result["n_clusters"] == period
        ),
        "cluster_sizes": cluster_result["cluster_sizes"],
        "min_intercluster_distance": cluster_result[
            "min_intercluster_distance"
        ],
        "median_cycle_time": float(np.median(cycle_times)),
        "selected_cycle_time": selected_cycle_time,
        "adjacent_impact_dt_min": float(np.min(np.diff(archive.impact_times[trim_impacts:]))),
        "adjacent_impact_dt_max": float(np.max(np.diff(archive.impact_times[trim_impacts:]))),
        "closure_median": float(np.median(closure_errors)),
        "closure_rms": float(np.sqrt(np.mean(closure_errors * closure_errors))),
        "closure_q95": float(np.quantile(closure_errors, 0.95)),
        "closure_max": float(np.max(closure_errors)),
        "selected_closure": selected_closure,
        "proper_divisor_min_distances": {
            str(lag): stats["min"]
            for lag, stats in divisor_distance_stats.items()
        },
        "proper_divisor_distance_stats": {
            str(lag): stats for lag, stats in divisor_distance_stats.items()
        },
        "recurrence_lag_stats": recurrence_lag_stats,
        "anchor_phase": anchor_phase,
        "phase_median_theta_ns_plus": {
            str(phase): value for phase, value in phase_medians.items()
        },
        "start_impact_index": orbit.start_impact_index,
        "end_impact_index": orbit.end_impact_index,
        "start_time": float(orbit.event_times_abs[0]),
        "end_time": float(orbit.event_times_abs[-1]),
    }

    return RegimeResult(
        regime=regime,
        expected_period=period,
        archive=archive,
        orbit=orbit,
        metrics=metrics,
        checks=checks,
    )


def add_period_ratios(
    results: list[RegimeResult], ratio_rtol: float
) -> tuple[float, list[str]]:
    """Describe k-impact return times relative to the period-one regime."""

    reference = next(
        result.metrics["median_cycle_time"]
        for result in results
        if result.expected_period == 1
    )
    failures = []
    ordered = sorted(results, key=lambda result: result.expected_period)
    previous_cycle_time = None
    for result in ordered:
        ratio = result.metrics["median_cycle_time"] / reference
        expected = float(result.expected_period)
        relative_error = abs(ratio - expected) / expected
        if previous_cycle_time is None:
            successive_ratio = 1.0
            expected_successive_ratio = 1.0
        else:
            successive_ratio = result.metrics["median_cycle_time"] / previous_cycle_time
            expected_successive_ratio = 2.0
        successive_relative_error = (
            abs(successive_ratio - expected_successive_ratio)
            / expected_successive_ratio
        )
        result.metrics["cycle_time_ratio_to_period1"] = float(ratio)
        result.metrics["expected_cycle_time_ratio"] = expected
        result.metrics["cycle_time_ratio_relative_error"] = float(relative_error)
        result.metrics["cycle_time_ratio_to_preceding"] = float(successive_ratio)
        result.metrics["expected_successive_ratio"] = expected_successive_ratio
        result.metrics["successive_ratio_relative_error"] = float(
            successive_relative_error
        )
        result.metrics["cycle_time_ratio_within_tolerance"] = bool(
            relative_error <= ratio_rtol
        )
        result.orbit.tau_p1 = result.orbit.t_rel / reference
        result.orbit.event_tau_p1 = result.orbit.event_times_rel / reference
        if not result.metrics["cycle_time_ratio_within_tolerance"]:
            failures.append(
                f"{result.regime}: ratio {ratio:.9f} differs from {expected:g} "
                f"by {100.0 * relative_error:.3f}%"
            )
        previous_cycle_time = result.metrics["median_cycle_time"]
    return float(reference), failures


def validation_failures(results: list[RegimeResult]) -> list[str]:
    """Format all failed boolean checks."""

    return [
        f"{result.regime}: {check_name}"
        for result in results
        for check_name, passed in result.checks.items()
        if not passed
    ]


def physical_coordinates(state: np.ndarray, phase: int) -> tuple[np.ndarray, ...]:
    """Map stance/nonstance slots to physical legs relative to the anchor."""

    if phase % 2 == 0:
        return state[..., 0], state[..., 2], state[..., 1], state[..., 3]
    return state[..., 1], state[..., 3], state[..., 0], state[..., 2]


def arc_slice(orbit: ExtractedOrbit, phase: int) -> slice:
    """Return the concatenated-array slice for one continuous arc."""

    return slice(int(orbit.arc_offsets[phase]), int(orbit.arc_offsets[phase + 1]))


def padded_limits(values: list[np.ndarray], fraction: float = 0.05) -> tuple[float, float]:
    """Compute finite shared limits with a small deterministic margin."""

    combined = np.concatenate([np.ravel(value) for value in values])
    low = float(np.min(combined))
    high = float(np.max(combined))
    span = high - low
    padding = fraction * span if span > 0.0 else max(1.0, abs(low)) * fraction
    return low - padding, high + padding


def plot_limits(results: list[RegimeResult]) -> dict:
    """Derive shared axes from the union of all four extracted orbits."""

    theta_values = []
    velocity_values = []
    section_theta = []
    section_velocity = []
    for result in results:
        orbit = result.orbit
        for phase in range(result.expected_period):
            state = orbit.x[arc_slice(orbit, phase)]
            theta_a, velocity_a, theta_b, velocity_b = physical_coordinates(
                state, phase
            )
            theta_values.extend((theta_a, theta_b))
            velocity_values.extend((velocity_a, velocity_b))
        section = result.archive.jump_plus[
            orbit.start_impact_index : orbit.start_impact_index
            + result.expected_period
        ]
        section_theta.append(section[:, 0])
        section_velocity.append(section[:, 2])

    return {
        "theta": padded_limits(theta_values),
        "velocity": padded_limits(velocity_values),
        "section_theta": padded_limits(section_theta, fraction=0.08),
        "section_velocity": padded_limits(section_velocity, fraction=0.08),
    }


def event_leg_state(
    orbit: ExtractedOrbit, event_offset: int, post_impact: bool
) -> tuple[np.ndarray, ...]:
    """Map a recorded event state to physical-leg coordinates."""

    state = (
        orbit.jump_plus[event_offset]
        if post_impact
        else orbit.jump_minus[event_offset]
    )
    phase = event_offset if post_impact else event_offset - 1
    return physical_coordinates(state, phase)


def draw_phase_panel(
    ax,
    result: RegimeResult,
    leg: str,
    limits: dict,
    show_phase_legend: bool,
) -> None:
    """Plot one physical leg over all continuous arcs of an extracted orbit."""

    period = result.expected_period
    orbit = result.orbit
    cmap = matplotlib.colormaps.get_cmap("viridis").resampled(period)

    for phase in range(period):
        state = orbit.x[arc_slice(orbit, phase)]
        theta_a, velocity_a, theta_b, velocity_b = physical_coordinates(
            state, phase
        )
        theta = theta_a if leg == "A" else theta_b
        velocity = velocity_a if leg == "A" else velocity_b
        color = cmap(phase)
        ax.plot(
            theta,
            velocity,
            color=color,
            linewidth=1.35,
            label=f"stride phase {phase + 1}",
        )
        ax.scatter(
            [theta[0]],
            [velocity[0]],
            marker="o",
            s=28,
            color=[color],
            edgecolors="black",
            linewidths=0.35,
            zorder=4,
        )
        ax.scatter(
            [theta[-1]],
            [velocity[-1]],
            marker="v",
            s=34,
            facecolors="none",
            edgecolors=[color],
            linewidths=1.0,
            zorder=4,
        )

    closing = event_leg_state(orbit, period, post_impact=True)
    closing_theta = closing[0] if leg == "A" else closing[2]
    closing_velocity = closing[1] if leg == "A" else closing[3]
    ax.scatter(
        [closing_theta],
        [closing_velocity],
        marker="o",
        s=28,
        color=[cmap(0)],
        edgecolors="black",
        linewidths=0.35,
        zorder=4,
    )

    ax.set_xlim(*limits["theta"])
    ax.set_ylim(*limits["velocity"])
    ax.set_xlabel(r"angular position $\theta$ (rad)")
    ax.set_ylabel(r"angular velocity $\dot{\theta}$ (rad/s)")
    ax.set_title(f"Physical leg {leg}")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if show_phase_legend:
        ax.legend(loc="best", fontsize=7, ncol=2, frameon=False)


def draw_time_panel(
    ax,
    result: RegimeResult,
    component: str,
    limits: dict,
    show_legend: bool,
) -> None:
    """Plot physical-leg angles or velocities without cross-reset connectors."""

    orbit = result.orbit
    period = result.expected_period
    colors = {"A": "tab:blue", "B": "tab:orange"}

    for phase in range(period):
        selection = arc_slice(orbit, phase)
        tau = orbit.tau_p1[selection]
        state = orbit.x[selection]
        theta_a, velocity_a, theta_b, velocity_b = physical_coordinates(
            state, phase
        )
        values_a = theta_a if component == "theta" else velocity_a
        values_b = theta_b if component == "theta" else velocity_b
        ax.plot(tau, values_a, color=colors["A"], linewidth=1.2)
        ax.plot(tau, values_b, color=colors["B"], linewidth=1.2)

    for event_offset, tau in enumerate(orbit.event_tau_p1):
        ax.axvline(tau, color="0.6", linewidth=0.6, linestyle=":", zorder=0)
        if event_offset == 0:
            states = ((True, "o"),)
        else:
            states = ((False, "v"), (True, "o"))
        for post_impact, marker in states:
            event = event_leg_state(orbit, event_offset, post_impact=post_impact)
            value_a = event[0] if component == "theta" else event[1]
            value_b = event[2] if component == "theta" else event[3]
            marker_kwargs = {
                "marker": marker,
                "s": 25,
                "linewidths": 0.8,
                "zorder": 4,
            }
            if post_impact:
                ax.scatter(
                    [tau],
                    [value_a],
                    color=colors["A"],
                    edgecolors="black",
                    **marker_kwargs,
                )
                ax.scatter(
                    [tau],
                    [value_b],
                    color=colors["B"],
                    edgecolors="black",
                    **marker_kwargs,
                )
            else:
                ax.scatter(
                    [tau],
                    [value_a],
                    facecolors="none",
                    edgecolors=colors["A"],
                    **marker_kwargs,
                )
                ax.scatter(
                    [tau],
                    [value_b],
                    facecolors="none",
                    edgecolors=colors["B"],
                    **marker_kwargs,
                )

    ax.set_xlim(0.0, float(orbit.event_tau_p1[-1]) * 1.025)
    ax.set_ylim(*limits[component])
    ax.set_xlabel(r"time / period-1 return time, $\tau$")
    if component == "theta":
        ax.set_ylabel(r"angle $\theta$ (rad)")
        ax.set_title("Physical-leg angles")
    else:
        ax.set_ylabel(r"angular velocity $\dot{\theta}$ (rad/s)")
        ax.set_title("Physical-leg angular velocities")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    if show_legend:
        handles = [
            Line2D([0], [0], color=colors["A"], label="physical leg A"),
            Line2D([0], [0], color=colors["B"], label="physical leg B"),
            Line2D(
                [0],
                [0],
                marker="v",
                color="none",
                markeredgecolor="0.25",
                markerfacecolor="none",
                label="pre-impact",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markeredgecolor="black",
                markerfacecolor="0.5",
                label="post-impact",
            ),
        ]
        ax.legend(handles=handles, loc="best", fontsize=7, frameon=False)


def save_figure(fig, output_stem: Path, formats: tuple[str, ...], dpi: int) -> None:
    """Save one figure in every requested format."""

    for extension in formats:
        fig.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
        )


def plot_individual(
    result: RegimeResult,
    limits: dict,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    """Write a 2x2 phase/time figure for one periodic regime."""

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.3), dpi=dpi)
    draw_phase_panel(axes[0, 0], result, "A", limits, show_phase_legend=True)
    draw_phase_panel(axes[0, 1], result, "B", limits, show_phase_legend=False)
    draw_time_panel(axes[1, 0], result, "theta", limits, show_legend=True)
    draw_time_panel(axes[1, 1], result, "velocity", limits, show_legend=False)

    ratio = result.metrics["cycle_time_ratio_to_period1"]
    fig.suptitle(
        f"Compass gait period {result.expected_period}: one steady hybrid return  "
        f"|  $\\phi$={result.metrics['phi_deg']:.2f}$^\\circ$  "
        f"|  $T_k/T_1$={ratio:.4f}  (closure modulo leg relabeling)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(
        fig,
        output_dir / f"compass_{result.regime}_steady_orbit",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)


def draw_aligned_phase(ax, result: RegimeResult, limits: dict) -> None:
    """Draw both physical legs in one shared-limit comparison panel."""

    orbit = result.orbit
    for phase in range(result.expected_period):
        state = orbit.x[arc_slice(orbit, phase)]
        theta_a, velocity_a, theta_b, velocity_b = physical_coordinates(
            state, phase
        )
        ax.plot(theta_a, velocity_a, color="tab:blue", linewidth=1.0)
        ax.plot(theta_b, velocity_b, color="tab:orange", linewidth=1.0)
        ax.scatter(
            [theta_a[-1], theta_b[-1]],
            [velocity_a[-1], velocity_b[-1]],
            marker="v",
            s=18,
            facecolors="none",
            edgecolors=["tab:blue", "tab:orange"],
            linewidths=0.7,
            zorder=4,
        )
    ax.set_xlim(*limits["theta"])
    ax.set_ylim(*limits["velocity"])
    ax.grid(True, alpha=0.22, linewidth=0.5)


def draw_aligned_angles(
    ax, result: RegimeResult, limits: dict, common_tau_max: float
) -> None:
    """Draw physical-leg angles on the common period-one time scale."""

    orbit = result.orbit
    for phase in range(result.expected_period):
        selection = arc_slice(orbit, phase)
        state = orbit.x[selection]
        theta_a, _, theta_b, _ = physical_coordinates(state, phase)
        ax.plot(orbit.tau_p1[selection], theta_a, color="tab:blue", linewidth=1.0)
        ax.plot(orbit.tau_p1[selection], theta_b, color="tab:orange", linewidth=1.0)
    for event_offset, tau in enumerate(orbit.event_tau_p1):
        ax.axvline(tau, color="0.65", linewidth=0.5, linestyle=":", zorder=0)
        if event_offset > 0:
            pre = event_leg_state(orbit, event_offset, post_impact=False)
            post = event_leg_state(orbit, event_offset, post_impact=True)
            ax.scatter(
                [tau, tau],
                [pre[0], pre[2]],
                marker="v",
                s=15,
                facecolors="none",
                edgecolors=["tab:blue", "tab:orange"],
                linewidths=0.6,
                zorder=4,
            )
            ax.scatter(
                [tau, tau],
                [post[0], post[2]],
                marker="o",
                s=13,
                color=["tab:blue", "tab:orange"],
                edgecolors="black",
                linewidths=0.25,
                zorder=4,
            )
    ax.set_xlim(0.0, common_tau_max)
    ax.set_ylim(*limits["theta"])
    ax.grid(True, alpha=0.22, linewidth=0.5)


def draw_return_section(ax, result: RegimeResult, limits: dict) -> None:
    """Draw the finite post-impact section as a diagnostic, not a topology claim."""

    orbit = result.orbit
    period = result.expected_period
    points = result.archive.jump_plus[
        orbit.start_impact_index : orbit.start_impact_index + period
    ]
    cmap = matplotlib.colormaps.get_cmap("viridis").resampled(period)
    if period > 1:
        closed = np.vstack((points, points[0]))
        ax.plot(
            closed[:, 0],
            closed[:, 2],
            color="0.65",
            linestyle="--",
            linewidth=0.7,
            zorder=1,
        )
    for phase, point in enumerate(points):
        ax.scatter(
            [point[0]],
            [point[2]],
            s=38,
            color=[cmap(phase)],
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        ax.annotate(
            str(phase + 1),
            (point[0], point[2]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlim(*limits["section_theta"])
    ax.set_ylim(*limits["section_velocity"])
    ax.grid(True, alpha=0.22, linewidth=0.5)


def plot_aligned(
    results: list[RegimeResult],
    limits: dict,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    """Write the four-regime shared-axis comparison figure."""

    fig, axes = plt.subplots(4, 3, figsize=(14.0, 14.5), dpi=dpi)
    common_tau_max = max(
        float(result.orbit.event_tau_p1[-1]) for result in results
    ) * 1.015

    for row, result in enumerate(results):
        draw_aligned_phase(axes[row, 0], result, limits)
        draw_aligned_angles(axes[row, 1], result, limits, common_tau_max)
        draw_return_section(axes[row, 2], result, limits)
        ratio = result.metrics["cycle_time_ratio_to_period1"]
        axes[row, 0].set_ylabel(
            f"period {result.expected_period}\n"
            + r"$\dot{\theta}$ (rad/s)"
        )
        axes[row, 1].text(
            0.98,
            0.92,
            f"$T_k/T_1={ratio:.4f}$",
            transform=axes[row, 1].transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    axes[0, 0].set_title("Physical-leg phase portrait")
    axes[0, 1].set_title(r"Physical-leg angles on common $T_1$ clock")
    axes[0, 2].set_title("Post-impact section (return-map diagnostic)")
    for column in range(3):
        axes[-1, column].set_xlabel(
            r"angular position $\theta$ (rad)"
            if column != 1
            else r"time / period-1 return time, $\tau$"
        )
    axes[-1, 2].set_xlabel(r"post-impact $\theta_{ns}^{+}$ (rad)")
    for row in range(4):
        axes[row, 2].set_ylabel(r"post-impact $\dot{\theta}_{ns}^{+}$ (rad/s)")

    leg_handles = [
        Line2D([0], [0], color="tab:blue", label="physical leg A"),
        Line2D([0], [0], color="tab:orange", label="physical leg B"),
        Line2D(
            [0],
            [0],
            marker="v",
            color="none",
            markeredgecolor="0.25",
            markerfacecolor="none",
            label="pre-impact",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="black",
            markerfacecolor="0.5",
            label="post-impact",
        ),
    ]
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Compass-gait sampled doubling regimes: aligned reference-simulator returns\n"
        "Hybrid closure is modulo stance/nonstance leg relabeling at impacts",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.97))
    save_figure(
        fig,
        output_dir / "compass_steady_orbits_aligned",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)


def plot_period4_period8_diagnostic(
    results: list[RegimeResult],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    """Write a focused section-and-recurrence comparison for periods 4 and 8."""

    result4 = next(result for result in results if result.expected_period == 4)
    result8 = next(result for result in results if result.expected_period == 8)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=dpi)

    section4 = result4.archive.jump_plus[
        result4.orbit.start_impact_index : result4.orbit.end_impact_index
    ]
    section8 = result8.archive.jump_plus[
        result8.orbit.start_impact_index : result8.orbit.end_impact_index
    ]
    axes[0].scatter(
        section4[:, 0],
        section4[:, 2],
        marker="s",
        s=70,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=1.4,
        label="period 4",
        zorder=3,
    )
    axes[0].scatter(
        section8[:, 0],
        section8[:, 2],
        marker="o",
        s=40,
        color="tab:orange",
        edgecolors="black",
        linewidths=0.4,
        label="period 8",
        zorder=4,
    )
    for phase in range(4):
        daughter_pair = section8[[phase, phase + 4]]
        axes[0].plot(
            daughter_pair[:, 0],
            daughter_pair[:, 2],
            color="0.45",
            linewidth=0.9,
            linestyle="--",
            zorder=2,
        )
    for phase, point in enumerate(section8):
        axes[0].annotate(
            str(phase + 1),
            (point[0], point[2]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    daughter_stats = result8.metrics["proper_divisor_distance_stats"]["4"]
    axes[0].text(
        0.03,
        0.03,
        "period-8 lag-4 distances\n"
        f"{daughter_stats['min']:.4f}--{daughter_stats['max']:.4f}",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    axes[0].set_xlabel(r"post-impact $\theta_{ns}^{+}$ (rad)")
    axes[0].set_ylabel(r"post-impact $\dot{\theta}_{ns}^{+}$ (rad/s)")
    axes[0].set_title("Return-section daughter branches")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    for result, color in ((result4, "tab:blue"), (result8, "tab:orange")):
        lag_stats = result.metrics["recurrence_lag_stats"]
        lags = np.asarray([int(lag) for lag in lag_stats], dtype=int)
        medians = np.asarray(
            [lag_stats[str(lag)]["median"] for lag in lags], dtype=float
        )
        axes[1].semilogy(
            lags,
            medians,
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=color,
            label=f"period {result.expected_period}",
        )
    axes[1].axvline(4, color="tab:blue", linewidth=0.8, linestyle=":")
    axes[1].axvline(8, color="tab:orange", linewidth=0.8, linestyle=":")
    axes[1].set_xticks(np.arange(1, 17))
    axes[1].set_xlabel("impact lag")
    axes[1].set_ylabel("median 4D post-impact mismatch")
    axes[1].set_title("Fundamental return lag")
    axes[1].legend(frameon=False)
    axes[1].grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "Period 4 and period 8: similar geometry, different return lag",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_figure(
        fig,
        output_dir / "compass_period4_period8_diagnostic",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)


def save_orbit_extract(
    result: RegimeResult, output_dir: Path, reference_period: float
) -> None:
    """Write a compact, provenance-bearing NPZ for the selected orbit."""

    orbit = result.orbit
    extract_meta = {
        "regime": result.regime,
        "expected_period": result.expected_period,
        "phi_deg": result.metrics["phi_deg"],
        "source_file": result.archive.source_name,
        "source_sha256": result.archive.source_sha256,
        "start_impact_index": orbit.start_impact_index,
        "end_impact_index": orbit.end_impact_index,
        "period1_return_time": reference_period,
        "event_semantics": "jump_minus is pre-impact; jump_plus is post-impact",
        "physical_leg_anchor": "leg A is nonstance immediately after the first event",
    }
    np.savez_compressed(
        output_dir / f"compass_{result.regime}_steady_orbit.npz",
        t_abs=orbit.t_abs,
        t_rel=orbit.t_rel,
        tau_p1=orbit.tau_p1,
        x=orbit.x,
        arc_offsets=orbit.arc_offsets,
        event_indices=orbit.event_indices,
        event_times_abs=orbit.event_times_abs,
        event_times_rel=orbit.event_times_rel,
        event_tau_p1=orbit.event_tau_p1,
        jump_minus=orbit.jump_minus,
        jump_plus=orbit.jump_plus,
        meta_json=json.dumps(extract_meta, sort_keys=True),
    )


SUMMARY_COLUMNS = (
    "regime",
    "expected_period",
    "phi_deg",
    "source_file",
    "source_sha256",
    "source_dt",
    "burn_in_strides",
    "trim_impacts",
    "n_samples",
    "n_impacts",
    "post_trim_impacts",
    "detected_clusters",
    "cluster_count_matches_expected",
    "cluster_sizes",
    "start_impact_index",
    "end_impact_index",
    "start_time",
    "end_time",
    "selected_cycle_time",
    "median_cycle_time",
    "cycle_time_ratio_to_period1",
    "expected_cycle_time_ratio",
    "cycle_time_ratio_relative_error",
    "cycle_time_ratio_within_tolerance",
    "cycle_time_ratio_to_preceding",
    "expected_successive_ratio",
    "successive_ratio_relative_error",
    "adjacent_impact_dt_min",
    "adjacent_impact_dt_max",
    "closure_median",
    "closure_rms",
    "closure_q95",
    "closure_max",
    "selected_closure",
    "proper_divisor_min_distances",
    "proper_divisor_distance_stats",
    "min_intercluster_distance",
    "status",
)


def summary_row(result: RegimeResult, trim_impacts: int) -> dict:
    """Flatten one result for the human-readable CSV."""

    metrics = result.metrics
    return {
        "regime": result.regime,
        "expected_period": result.expected_period,
        "phi_deg": metrics["phi_deg"],
        "source_file": result.archive.source_name,
        "source_sha256": result.archive.source_sha256,
        "source_dt": metrics["source_dt"],
        "burn_in_strides": metrics["burn_in_strides"],
        "trim_impacts": trim_impacts,
        "n_samples": metrics["n_samples"],
        "n_impacts": metrics["n_impacts"],
        "post_trim_impacts": metrics["post_trim_impacts"],
        "detected_clusters": metrics["detected_clusters"],
        "cluster_count_matches_expected": metrics[
            "cluster_count_matches_expected"
        ],
        "cluster_sizes": json.dumps(metrics["cluster_sizes"]),
        "start_impact_index": metrics["start_impact_index"],
        "end_impact_index": metrics["end_impact_index"],
        "start_time": metrics["start_time"],
        "end_time": metrics["end_time"],
        "selected_cycle_time": metrics["selected_cycle_time"],
        "median_cycle_time": metrics["median_cycle_time"],
        "cycle_time_ratio_to_period1": metrics["cycle_time_ratio_to_period1"],
        "expected_cycle_time_ratio": metrics["expected_cycle_time_ratio"],
        "cycle_time_ratio_relative_error": metrics[
            "cycle_time_ratio_relative_error"
        ],
        "cycle_time_ratio_within_tolerance": metrics[
            "cycle_time_ratio_within_tolerance"
        ],
        "cycle_time_ratio_to_preceding": metrics[
            "cycle_time_ratio_to_preceding"
        ],
        "expected_successive_ratio": metrics["expected_successive_ratio"],
        "successive_ratio_relative_error": metrics[
            "successive_ratio_relative_error"
        ],
        "adjacent_impact_dt_min": metrics["adjacent_impact_dt_min"],
        "adjacent_impact_dt_max": metrics["adjacent_impact_dt_max"],
        "closure_median": metrics["closure_median"],
        "closure_rms": metrics["closure_rms"],
        "closure_q95": metrics["closure_q95"],
        "closure_max": metrics["closure_max"],
        "selected_closure": metrics["selected_closure"],
        "proper_divisor_min_distances": json.dumps(
            metrics["proper_divisor_min_distances"], sort_keys=True
        ),
        "proper_divisor_distance_stats": json.dumps(
            metrics["proper_divisor_distance_stats"], sort_keys=True
        ),
        "min_intercluster_distance": metrics["min_intercluster_distance"],
        "status": "pass" if all(result.checks.values()) else "fail",
    }


def write_summary_csv(
    results: list[RegimeResult], output_dir: Path, trim_impacts: int
) -> None:
    """Write one provenance and validation row per intended orbit."""

    with (output_dir / "compass_periodic_orbit_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(summary_row(result, trim_impacts=trim_impacts))


def write_checks_json(
    results: list[RegimeResult],
    output_dir: Path,
    reference_period: float,
    parameters: dict,
) -> None:
    """Write machine-readable checks and full metrics."""

    payload = {
        "status": "pass" if not validation_failures(results) else "fail",
        "period1_return_time": reference_period,
        "parameters": parameters,
        "regimes": {
            result.regime: {
                "expected_period": result.expected_period,
                "source_file": result.archive.source_name,
                "source_sha256": result.archive.source_sha256,
                "checks": result.checks,
                "metrics": result.metrics,
            }
            for result in results
        },
    }
    with (output_dir / "compass_periodic_orbit_checks.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def print_summary(results: list[RegimeResult]) -> None:
    """Print the central validation quantities to standard output."""

    header = (
        "regime   k  clusters  median T_k (s)  T_k/T_1   rel.err    "
        "closure max   status"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        metrics = result.metrics
        status = "PASS" if all(result.checks.values()) else "FAIL"
        print(
            f"{result.regime:8s} {result.expected_period:2d} "
            f"{metrics['detected_clusters']:9d} "
            f"{metrics['median_cycle_time']:15.12f} "
            f"{metrics['cycle_time_ratio_to_period1']:9.6f} "
            f"{metrics['cycle_time_ratio_relative_error']:9.3e} "
            f"{metrics['closure_max']:13.3e}   {status}"
        )


def parse_formats(raw_formats: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated image-format list."""

    formats = tuple(
        dict.fromkeys(
            item.strip().lower() for item in raw_formats.split(",") if item.strip()
        )
    )
    invalid = sorted(set(formats).difference({"png", "pdf"}))
    if not formats or invalid:
        detail = f"; unsupported: {', '.join(invalid)}" if invalid else ""
        raise ValidationError(f"formats must contain png and/or pdf{detail}")
    return formats


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""

    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Validate stored compass-gait periodic returns and extract one "
            "steady period-1/2/4/8 orbit per archive without simulation."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=package_root / "data_fine" / "compass_gait",
        help="Read-only directory containing compass_period{1,2,4,8}.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_root / "figures" / "steady_compass_orbits",
        help="Directory for derived extracts, checks, and figures.",
    )
    parser.add_argument(
        "--trim-impacts",
        type=int,
        default=10,
        help="Additional main-series impacts to discard after stored burn-in.",
    )
    parser.add_argument(
        "--tail-cycles",
        type=int,
        default=32,
        help="Late cycles used to choose a deterministic phase anchor.",
    )
    parser.add_argument(
        "--link-tol",
        type=float,
        default=0.002,
        help="Return-map single-linkage and proper-divisor separation tolerance.",
    )
    parser.add_argument(
        "--closure-atol",
        type=float,
        default=1e-8,
        help="Maximum allowed post-impact k-return closure error.",
    )
    parser.add_argument(
        "--ratio-rtol",
        type=float,
        default=0.02,
        help="Relative tolerance for T_k/T_1 versus k.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated figure formats: pdf, png, or both.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Raster output DPI.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run every validation check and write no files.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Validate inputs and optionally write the bounded derived outputs."""

    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parent.parent
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = validate_output_dir(args.output_dir, package_root, data_dir)

    if args.tail_cycles < 1:
        raise ValidationError("tail_cycles must be positive")
    if args.link_tol <= 0.0 or args.closure_atol <= 0.0 or args.ratio_rtol < 0.0:
        raise ValidationError("link/closure tolerances must be positive; ratio_rtol nonnegative")
    if args.dpi < 50:
        raise ValidationError("dpi must be at least 50")
    formats = parse_formats(args.formats)

    results = []
    for regime, period in REGIMES:
        archive = load_archive(data_dir / f"compass_{regime}.npz", repo_root)
        results.append(
            analyze_regime(
                regime=regime,
                period=period,
                archive=archive,
                trim_impacts=args.trim_impacts,
                tail_cycles=args.tail_cycles,
                link_tol=args.link_tol,
                closure_atol=args.closure_atol,
            )
        )

    reference_period, ratio_failures = add_period_ratios(
        results, ratio_rtol=args.ratio_rtol
    )
    failures = validation_failures(results)
    print_summary(results)
    if ratio_failures:
        for failure in ratio_failures:
            print(f"ratio diagnostic outside tolerance: {failure}", file=sys.stderr)
    if failures:
        for failure in failures:
            print(f"failed check: {failure}", file=sys.stderr)
        return 1

    if args.check_only:
        print("\nCHECK ONLY: all checks passed; no files written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = {
        "trim_impacts": args.trim_impacts,
        "tail_cycles": args.tail_cycles,
        "link_tol": args.link_tol,
        "closure_atol": args.closure_atol,
        "ratio_rtol": args.ratio_rtol,
        "formats": list(formats),
        "dpi": args.dpi,
    }
    for result in results:
        save_orbit_extract(result, output_dir, reference_period=reference_period)
    write_summary_csv(results, output_dir, trim_impacts=args.trim_impacts)
    write_checks_json(
        results,
        output_dir,
        reference_period=reference_period,
        parameters=parameters,
    )

    limits = plot_limits(results)
    for result in results:
        plot_individual(
            result,
            limits=limits,
            output_dir=output_dir,
            formats=formats,
            dpi=args.dpi,
        )
    plot_aligned(
        results,
        limits=limits,
        output_dir=output_dir,
        formats=formats,
        dpi=args.dpi,
    )
    plot_period4_period8_diagnostic(
        results,
        output_dir=output_dir,
        formats=formats,
        dpi=args.dpi,
    )

    print(f"\nWrote bounded derived outputs to {output_dir}")
    return 0


def main() -> int:
    """Command-line entry point."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (ValidationError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
