#!/usr/bin/env python3
"""Build certified Fourier periodic-orbit inputs for David's Rössler family.

The default action is a read-only plan check.  ``--execute`` is required to
perform continuation or write outputs.  The builder deliberately lives beside,
but does not modify or import, the frozen shared-probability driver.

The periodic calculations use Fourier pseudospectral collocation with an
analytic Newton matrix.  Natural parameter continuation follows each parent
periodic branch through its next flip bifurcation.  Every manuscript
representative is independently checked by a high-accuracy DOP853 integration
of the state and variational equations.  The chaotic representative is sampled
only after a separate long burn-in and must pass a positive largest-Lyapunov
exponent check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import solve as dense_solve


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
SAFE_OUTPUT_PARENT = CODE_ROOT / "experiments_planned" / "outputs"
DEFAULT_OUTPUT_ROOT = (
    SAFE_OUTPUT_PARENT / "roessler_david_fourier_continuation_v1"
)

BUNDLE_ID = "roessler_david_fourier_continuation_v1"
B_FIXED = 0.2
SAMPLE_DT = 0.01
ANALYSIS_DURATION = 3000.0
ANALYSIS_N_SAMPLES = int(round(ANALYSIS_DURATION / SAMPLE_DT)) + 1
MAX_SEGMENT_LENGTH_SAMPLES = 6000
CHAOS_DISPLAY_DURATION = 50.0
CHAOS_BURN_IN = 3000.0
PERIODIC_BURN_IN = 6000.0
BRANCH_NODES = 129
CERTIFICATION_NODE_SEQUENCE = (257, 513, 1025)
FLIP_BRACKET_TOLERANCE = 2.0e-8
COLLOCATION_TOLERANCE = 2.0e-11

REPRESENTATIVES = (
    ("period1", 2.82, 1),
    ("period2", 2.86, 2),
    ("period4", 4.10, 4),
    ("period8", 4.18, 8),
)
CHAOS_REPRESENTATIVE = ("chaos", 4.30)


@dataclass
class OrbitSolution:
    """A Fourier-collocation periodic orbit on an equispaced phase grid."""

    a: float
    q: int
    values: np.ndarray
    period: float
    collocation_residual: float
    newton_iterations: int
    seed_closure: float | None = None


@dataclass
class OrbitValidation:
    """Independent diagnostics for a corrected periodic orbit."""

    oversampled_residual_absolute: float
    oversampled_residual_relative: float
    closure_error_absolute: float
    closure_error_relative: float
    multipliers: np.ndarray
    trivial_multiplier_error: float
    flip_multiplier: complex
    stable_transverse: bool
    tail_energy_ratio: float
    shift_rms: dict[str, float]
    section_x: float
    section_dxdt: float


@dataclass
class BranchPoint:
    """One accepted point on a naturally continued periodic branch."""

    solution: OrbitSolution
    validation: OrbitValidation
    stage: str


class NumericalFailure(RuntimeError):
    """Raised when an advertised numerical certificate cannot be obtained."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))


def reject_symlink_components(path: Path, label: str) -> None:
    probe = path.absolute()
    while True:
        if probe.is_symlink():
            raise ValueError(f"{label} contains a symlink: {probe}")
        parent = probe.parent
        if parent == probe:
            return
        probe = parent


def guard_new_output_root(path: Path) -> Path:
    reject_symlink_components(path, "output root")
    resolved = path.resolve(strict=False)
    safe_parent = SAFE_OUTPUT_PARENT.resolve(strict=False)
    if resolved == safe_parent or not resolved.is_relative_to(safe_parent):
        raise ValueError(
            f"output root must be a named child of {safe_parent}: {resolved}"
        )
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {resolved}")
    return resolved


def roessler_field(state: np.ndarray, a: float) -> np.ndarray:
    """David's b=0.2 Rössler vector field, vectorized over leading axes."""
    state = np.asarray(state, dtype=float)
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]
    return np.stack(
        (-y - z, x + B_FIXED * y, B_FIXED + z * (x - a)),
        axis=-1,
    )


def roessler_jacobian(state: np.ndarray, a: float) -> np.ndarray:
    """Analytic state Jacobian of :func:`roessler_field`."""
    x, _, z = np.asarray(state, dtype=float)
    return np.asarray(
        (
            (0.0, -1.0, -1.0),
            (1.0, B_FIXED, 0.0),
            (z, 0.0, x - a),
        ),
        dtype=float,
    )


def phase_frequencies(n: int) -> np.ndarray:
    if n < 5 or n % 2 != 1:
        raise ValueError("Fourier collocation requires an odd node count >= 5")
    return np.rint(np.fft.fftfreq(n, d=1.0 / n)).astype(int)


def fourier_coefficients(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    modes = phase_frequencies(n)
    coefficients = np.fft.fft(values, axis=0) / n
    return modes, coefficients


def evaluate_fourier(
    values: np.ndarray,
    phases: np.ndarray,
    *,
    derivative: bool = False,
) -> np.ndarray:
    """Evaluate the trigonometric interpolant (or its phase derivative)."""
    modes, coefficients = fourier_coefficients(values)
    phases = np.asarray(phases, dtype=float).reshape(-1)
    if derivative:
        coefficients = coefficients * (2j * np.pi * modes[:, None])
    # A 3000-second stream at dt=.01 contains 300001 phases.  Constructing its
    # full phase-by-mode matrix would need several GiB for the q=8 orbit, so
    # evaluate in deterministic bounded-memory chunks.
    maximum_basis_entries = 8_000_000
    chunk_size = max(1, maximum_basis_entries // len(modes))
    result = np.empty((len(phases), coefficients.shape[1]), dtype=complex)
    for start in range(0, len(phases), chunk_size):
        stop = min(len(phases), start + chunk_size)
        basis = np.exp(
            2j * np.pi * phases[start:stop, None] * modes[None, :]
        )
        result[start:stop] = basis @ coefficients
    imaginary_error = float(np.max(np.abs(result.imag)))
    if imaginary_error > 5.0e-10:
        raise NumericalFailure(
            f"Fourier evaluation lost conjugate symmetry: {imaginary_error:.3e}"
        )
    return result.real


def fourier_differentiation_matrix(n: int) -> np.ndarray:
    """Dense exact differentiation matrix for the N-node Fourier grid."""
    modes = phase_frequencies(n)
    identity = np.eye(n)
    transformed = np.fft.fft(identity, axis=0)
    derivative = np.fft.ifft(
        (2j * np.pi * modes)[:, None] * transformed,
        axis=0,
    )
    if float(np.max(np.abs(derivative.imag))) > 5.0e-12:
        raise NumericalFailure("Fourier differentiation matrix is not real")
    return derivative.real


def resample_orbit(values: np.ndarray, n_new: int) -> np.ndarray:
    phases = np.arange(n_new, dtype=float) / n_new
    return evaluate_fourier(values, phases)


def collocation_residual(
    values: np.ndarray,
    period: float,
    a: float,
    differentiation: np.ndarray,
) -> np.ndarray:
    dynamic = differentiation @ values - period * roessler_field(values, a)
    return np.concatenate((dynamic.reshape(-1), [values[0, 0]]))


def collocation_jacobian(
    values: np.ndarray,
    period: float,
    a: float,
    differentiation: np.ndarray,
) -> np.ndarray:
    """Analytic Newton matrix for nodal states, period, and x(0)=0 phase."""
    n = len(values)
    size = 3 * n + 1
    matrix = np.zeros((size, size), dtype=float)
    matrix[: 3 * n, : 3 * n] = np.kron(differentiation, np.eye(3))
    for index, state in enumerate(values):
        block = slice(3 * index, 3 * index + 3)
        matrix[block, block] -= period * roessler_jacobian(state, a)
    matrix[: 3 * n, -1] = -roessler_field(values, a).reshape(-1)
    matrix[-1, 0] = 1.0
    return matrix


def correct_orbit(
    values_guess: np.ndarray,
    period_guess: float,
    a: float,
    q: int,
    *,
    tolerance: float = COLLOCATION_TOLERANCE,
    max_iterations: int = 18,
) -> OrbitSolution:
    """Newton-correct a Fourier collocation orbit at fixed parameter a."""
    values = np.asarray(values_guess, dtype=float).copy()
    period = float(period_guess)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("orbit guess must have shape (N, 3)")
    differentiation = fourier_differentiation_matrix(len(values))
    last_norm = math.inf
    for iteration in range(max_iterations + 1):
        residual = collocation_residual(values, period, a, differentiation)
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if residual_norm <= tolerance:
            if roessler_field(values[0], a)[0] <= 0.0:
                raise NumericalFailure(
                    f"q={q}, a={a}: phase section has nonpositive orientation"
                )
            return OrbitSolution(
                a=a,
                q=q,
                values=values,
                period=period,
                collocation_residual=residual_norm,
                newton_iterations=iteration,
            )
        if iteration == max_iterations:
            break
        matrix = collocation_jacobian(values, period, a, differentiation)
        try:
            update = dense_solve(
                matrix,
                -residual,
                assume_a="gen",
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            )
        except Exception as exc:  # scipy exposes backend-specific subclasses
            raise NumericalFailure(
                f"q={q}, a={a}: Newton linear solve failed"
            ) from exc
        state_update = update[:-1].reshape(values.shape)
        period_update = float(update[-1])
        accepted = False
        step = 1.0
        for _ in range(14):
            trial_period = period + step * period_update
            if trial_period <= 0.0 or not math.isfinite(trial_period):
                step *= 0.5
                continue
            trial_values = values + step * state_update
            trial_residual = collocation_residual(
                trial_values,
                trial_period,
                a,
                differentiation,
            )
            trial_norm = float(np.linalg.norm(trial_residual, ord=np.inf))
            if trial_norm < residual_norm:
                values = trial_values
                period = trial_period
                last_norm = trial_norm
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise NumericalFailure(
                f"q={q}, a={a}: Newton line search stalled at {residual_norm:.3e}"
            )
    raise NumericalFailure(
        f"q={q}, a={a}: correction did not converge; last residual {last_norm:.3e}"
    )


def poincare_event(_: float, state: np.ndarray) -> float:
    return float(state[0])


poincare_event.direction = 1.0
poincare_event.terminal = False


def integrate_state(
    initial: np.ndarray,
    a: float,
    interval: tuple[float, float],
    *,
    t_eval: np.ndarray | None = None,
    dense_output: bool = False,
    events: Any = None,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
) -> Any:
    solution = solve_ivp(
        lambda _t, state: roessler_field(state, a),
        interval,
        np.asarray(initial, dtype=float),
        method="DOP853",
        t_eval=t_eval,
        dense_output=dense_output,
        events=events,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise NumericalFailure(f"DOP853 failed at a={a}: {solution.message}")
    return solution


def stable_orbit_seed(a: float, q: int, n: int) -> OrbitSolution:
    """Extract q consecutive settled Poincare returns and correct them."""
    initial = np.asarray((1.0, 1.0, 0.0), dtype=float)
    burn = integrate_state(
        initial,
        a,
        (0.0, PERIODIC_BURN_IN),
        rtol=2.0e-11,
        atol=2.0e-13,
    )
    capture_time = max(300.0, 20.0 * q * 6.0)
    capture = integrate_state(
        burn.y[:, -1],
        a,
        (0.0, capture_time),
        dense_output=True,
        events=poincare_event,
    )
    event_times = np.asarray(capture.t_events[0], dtype=float)
    if len(event_times) < q + 3:
        raise NumericalFailure(
            f"a={a}, q={q}: only {len(event_times)} Poincare returns"
        )
    start_time = float(event_times[-q - 1])
    end_time = float(event_times[-1])
    period_guess = end_time - start_time
    start_state = capture.sol(start_time)
    end_state = capture.sol(end_time)
    seed_closure = float(np.linalg.norm(end_state - start_state))
    phases = np.arange(n, dtype=float) / n
    values_guess = capture.sol(start_time + period_guess * phases).T
    corrected = correct_orbit(values_guess, period_guess, a, q)
    corrected.seed_closure = seed_closure
    return corrected


def integrate_variational(solution: OrbitSolution) -> tuple[float, float, np.ndarray]:
    """Independently integrate state plus monodromy over one full period."""
    initial_state = solution.values[0]
    augmented_initial = np.concatenate((initial_state, np.eye(3).reshape(-1)))

    def augmented_rhs(_time: float, augmented: np.ndarray) -> np.ndarray:
        state = augmented[:3]
        fundamental = augmented[3:].reshape(3, 3)
        return np.concatenate(
            (
                roessler_field(state, solution.a),
                (roessler_jacobian(state, solution.a) @ fundamental).reshape(-1),
            )
        )

    integrated = solve_ivp(
        augmented_rhs,
        (0.0, solution.period),
        augmented_initial,
        method="DOP853",
        rtol=2.0e-12,
        atol=2.0e-14,
    )
    if not integrated.success:
        raise NumericalFailure(
            f"variational DOP853 failed at a={solution.a}: {integrated.message}"
        )
    final_state = integrated.y[:3, -1]
    closure_absolute = float(np.linalg.norm(final_state - initial_state))
    closure_relative = closure_absolute / max(1.0, float(np.linalg.norm(initial_state)))
    monodromy = integrated.y[3:, -1].reshape(3, 3)
    multipliers = np.linalg.eigvals(monodromy)
    return closure_absolute, closure_relative, multipliers


def select_floquet_multipliers(
    multipliers: np.ndarray,
) -> tuple[complex, float, bool]:
    multipliers = np.asarray(multipliers, dtype=complex)
    trivial_index = int(np.argmin(np.abs(multipliers - 1.0)))
    trivial_error = float(abs(multipliers[trivial_index] - 1.0))
    transverse = np.delete(multipliers, trivial_index)
    # The branch-critical multiplier is the dominant transverse multiplier,
    # including before it becomes negative.  Selecting the value already
    # closest to -1 instead mislabels the strongly contracting multiplier on
    # the early period-2 branch, where the critical multiplier is still
    # positive and later crosses -1 under continuation.
    flip = complex(transverse[int(np.argmax(np.abs(transverse)))])
    stable = bool(np.all(np.abs(transverse) < 1.0 + 2.0e-7))
    return flip, trivial_error, stable


def oversampled_residual(solution: OrbitSolution) -> tuple[float, float]:
    n_test = 4 * len(solution.values) + 1
    phases = np.arange(n_test, dtype=float) / n_test
    values = evaluate_fourier(solution.values, phases)
    derivative = evaluate_fourier(solution.values, phases, derivative=True)
    rhs = solution.period * roessler_field(values, solution.a)
    residual = derivative - rhs
    absolute = float(np.max(np.abs(residual)))
    relative = absolute / max(1.0, float(np.max(np.abs(rhs))))
    return absolute, relative


def fourier_tail_energy_ratio(values: np.ndarray) -> float:
    modes, coefficients = fourier_coefficients(values)
    maximum_mode = int(np.max(np.abs(modes)))
    cutoff = max(1, int(math.floor(0.75 * maximum_mode)))
    energy = np.sum(np.abs(coefficients) ** 2)
    tail = np.sum(np.abs(coefficients[np.abs(modes) >= cutoff]) ** 2)
    return float(tail / max(float(energy), np.finfo(float).tiny))


def primitive_shift_distances(solution: OrbitSolution) -> dict[str, float]:
    if solution.q == 1:
        return {}
    phases = np.arange(len(solution.values), dtype=float) / len(solution.values)
    distances: dict[str, float] = {}
    divisor = 2
    while divisor <= solution.q:
        shifted = evaluate_fourier(
            solution.values,
            np.mod(phases + 1.0 / divisor, 1.0),
        )
        rms = float(np.sqrt(np.mean((shifted - solution.values) ** 2)))
        distances[f"shift_1_over_{divisor}"] = rms
        divisor *= 2
    return distances


def validate_orbit(solution: OrbitSolution) -> OrbitValidation:
    residual_absolute, residual_relative = oversampled_residual(solution)
    closure_absolute, closure_relative, multipliers = integrate_variational(solution)
    flip, trivial_error, stable = select_floquet_multipliers(multipliers)
    return OrbitValidation(
        oversampled_residual_absolute=residual_absolute,
        oversampled_residual_relative=residual_relative,
        closure_error_absolute=closure_absolute,
        closure_error_relative=closure_relative,
        multipliers=np.asarray(multipliers, dtype=complex),
        trivial_multiplier_error=trivial_error,
        flip_multiplier=flip,
        stable_transverse=stable,
        tail_energy_ratio=fourier_tail_energy_ratio(solution.values),
        shift_rms=primitive_shift_distances(solution),
        section_x=float(solution.values[0, 0]),
        section_dxdt=float(roessler_field(solution.values[0], solution.a)[0]),
    )


def flip_measure(validation: OrbitValidation) -> float:
    if abs(validation.flip_multiplier.imag) > 2.0e-5:
        raise NumericalFailure(
            "candidate flip multiplier is not real: "
            f"{validation.flip_multiplier.real:+.8f}"
            f"{validation.flip_multiplier.imag:+.3e}i"
        )
    return float(validation.flip_multiplier.real + 1.0)


def interpolate_solution(
    left: OrbitSolution,
    right: OrbitSolution,
    a: float,
) -> tuple[np.ndarray, float]:
    if left.q != right.q or left.values.shape != right.values.shape:
        raise ValueError("cannot interpolate incompatible orbit solutions")
    weight = (a - left.a) / (right.a - left.a)
    values = (1.0 - weight) * left.values + weight * right.values
    period = (1.0 - weight) * left.period + weight * right.period
    return values, float(period)


def continuation_target(
    current: OrbitSolution,
    target_a: float,
    *,
    minimum_step: float = 2.0e-6,
) -> list[OrbitSolution]:
    """Reach target_a, recursively halving a failed natural-continuation step."""
    try:
        candidate = correct_orbit(
            current.values,
            current.period,
            target_a,
            current.q,
        )
        return [candidate]
    except NumericalFailure:
        if abs(target_a - current.a) <= minimum_step:
            raise
        midpoint = 0.5 * (current.a + target_a)
        first = continuation_target(current, midpoint, minimum_step=minimum_step)
        second = continuation_target(first[-1], target_a, minimum_step=minimum_step)
        return first + second


def trace_parent_to_flip(
    representative: OrbitSolution,
    *,
    nominal_step: float,
    a_limit: float,
) -> tuple[list[BranchPoint], tuple[BranchPoint, BranchPoint]]:
    """Naturally continue a parent branch until its -1 multiplier crosses."""
    initial_validation = validate_orbit(representative)
    points = [BranchPoint(representative, initial_validation, "representative")]
    previous = points[-1]
    previous_measure = flip_measure(previous.validation)
    if previous_measure <= 0.0:
        raise NumericalFailure(
            f"q={representative.q}: representative is already past its flip"
        )
    target = representative.a + nominal_step
    while target <= a_limit + 1.0e-14:
        candidates = continuation_target(previous.solution, min(target, a_limit))
        for candidate in candidates:
            validation = validate_orbit(candidate)
            point = BranchPoint(candidate, validation, "continuation")
            points.append(point)
            measure = flip_measure(validation)
            print(
                f"q={candidate.q} a={candidate.a:.9f} "
                f"T={candidate.period:.9f} mu_flip={validation.flip_multiplier.real:+.8f}",
                flush=True,
            )
            if previous_measure * measure <= 0.0:
                return points, (previous, point)
            previous = point
            previous_measure = measure
        target += nominal_step
    raise NumericalFailure(
        f"q={representative.q}: no flip found before a={a_limit}"
    )


def refine_flip_bracket(
    left: BranchPoint,
    right: BranchPoint,
) -> tuple[list[BranchPoint], BranchPoint, BranchPoint]:
    """Bisect a Floquet -1 crossing while correcting the parent orbit."""
    left_measure = flip_measure(left.validation)
    right_measure = flip_measure(right.validation)
    if left.solution.a > right.solution.a:
        left, right = right, left
        left_measure, right_measure = right_measure, left_measure
    if left_measure * right_measure > 0.0:
        raise ValueError("flip endpoints do not bracket zero")
    refinements: list[BranchPoint] = []
    while right.solution.a - left.solution.a > FLIP_BRACKET_TOLERANCE:
        midpoint_a = 0.5 * (left.solution.a + right.solution.a)
        guess_values, guess_period = interpolate_solution(
            left.solution,
            right.solution,
            midpoint_a,
        )
        midpoint_solution = correct_orbit(
            guess_values,
            guess_period,
            midpoint_a,
            left.solution.q,
        )
        midpoint_validation = validate_orbit(midpoint_solution)
        midpoint = BranchPoint(midpoint_solution, midpoint_validation, "flip_bisection")
        refinements.append(midpoint)
        midpoint_measure = flip_measure(midpoint_validation)
        print(
            f"q={midpoint_solution.q} flip bracket "
            f"[{left.solution.a:.10f}, {right.solution.a:.10f}] "
            f"mu_mid={midpoint_validation.flip_multiplier.real:+.10f}",
            flush=True,
        )
        if left_measure * midpoint_measure <= 0.0:
            right = midpoint
            right_measure = midpoint_measure
        else:
            left = midpoint
            left_measure = midpoint_measure
    return refinements, left, right


def refine_and_certify(solution: OrbitSolution) -> tuple[OrbitSolution, OrbitValidation]:
    """Increase Fourier order until independent certification thresholds pass."""
    current = solution
    last_validation: OrbitValidation | None = None
    node_counts = [
        node_count
        for node_count in CERTIFICATION_NODE_SEQUENCE
        if node_count > len(current.values)
    ]
    # A branch may already have been continued at its certification order.
    # Validate that representation first rather than needlessly downsampling.
    if len(current.values) >= CERTIFICATION_NODE_SEQUENCE[0]:
        node_counts.insert(0, len(current.values))
    for node_count in node_counts:
        if node_count != len(current.values):
            values_guess = resample_orbit(current.values, node_count)
            current = correct_orbit(
                values_guess,
                current.period,
                current.a,
                current.q,
                tolerance=8.0e-12,
                max_iterations=20,
            )
            current.seed_closure = solution.seed_closure
        validation = validate_orbit(current)
        last_validation = validation
        minimum_shift = min(validation.shift_rms.values(), default=math.inf)
        passes = (
            validation.oversampled_residual_relative <= 2.0e-9
            and validation.closure_error_relative <= 2.0e-8
            and validation.trivial_multiplier_error <= 2.0e-6
            and validation.tail_energy_ratio <= 2.0e-11
            and validation.section_dxdt > 0.0
            and minimum_shift > 2.0e-5
        )
        if passes:
            return current, validation
    assert last_validation is not None
    raise NumericalFailure(
        f"q={solution.q}, a={solution.a}: certification failed at "
        f"N={len(current.values)}; residual="
        f"{last_validation.oversampled_residual_relative:.3e}, closure="
        f"{last_validation.closure_error_relative:.3e}, tail="
        f"{last_validation.tail_energy_ratio:.3e}"
    )


def refine_to_node_count(solution: OrbitSolution, target_nodes: int) -> OrbitSolution:
    """Refine through intermediate spectral orders to a requested odd size."""
    if target_nodes < len(solution.values):
        raise ValueError("refinement target cannot be smaller than current order")
    current = solution
    seed_closure = solution.seed_closure
    steps = [
        node_count
        for node_count in CERTIFICATION_NODE_SEQUENCE
        if len(current.values) < node_count <= target_nodes
    ]
    if not steps or steps[-1] != target_nodes:
        steps.append(target_nodes)
    for node_count in steps:
        if node_count == len(current.values):
            continue
        current = correct_orbit(
            resample_orbit(current.values, node_count),
            current.period,
            current.a,
            current.q,
            tolerance=8.0e-12,
            max_iterations=20,
        )
        current.seed_closure = seed_closure
    return current


def multiplier_records(multipliers: np.ndarray) -> list[dict[str, float]]:
    ordered = sorted(
        (complex(value) for value in multipliers),
        key=lambda value: (abs(value - 1.0), value.real, value.imag),
    )
    return [
        {
            "real": float(value.real),
            "imag": float(value.imag),
            "magnitude": float(abs(value)),
        }
        for value in ordered
    ]


def largest_lyapunov_exponent(
    initial_state: np.ndarray,
    a: float,
    *,
    warmup: float = 100.0,
    duration: float = 2000.0,
    block_duration: float = 2.0,
) -> dict[str, Any]:
    """Benettin largest-LLE estimate using the analytic variational equation."""
    tangent = np.asarray((1.0, math.sqrt(2.0), math.sqrt(3.0)), dtype=float)
    tangent /= np.linalg.norm(tangent)
    state = np.asarray(initial_state, dtype=float).copy()

    def block(state_in: np.ndarray, tangent_in: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray, float]:
        initial = np.concatenate((state_in, tangent_in))

        def rhs(_time: float, augmented: np.ndarray) -> np.ndarray:
            point = augmented[:3]
            vector = augmented[3:]
            return np.concatenate(
                (roessler_field(point, a), roessler_jacobian(point, a) @ vector)
            )

        integrated = solve_ivp(
            rhs,
            (0.0, length),
            initial,
            method="DOP853",
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        if not integrated.success:
            raise NumericalFailure(f"LLE block failed: {integrated.message}")
        state_out = integrated.y[:3, -1]
        tangent_out = integrated.y[3:, -1]
        norm = float(np.linalg.norm(tangent_out))
        if not math.isfinite(norm) or norm <= 0.0:
            raise NumericalFailure("invalid tangent norm in LLE calculation")
        return state_out, tangent_out / norm, math.log(norm)

    elapsed = 0.0
    while elapsed < warmup - 1.0e-14:
        length = min(block_duration, warmup - elapsed)
        state, tangent, _ = block(state, tangent, length)
        elapsed += length

    increments: list[float] = []
    elapsed = 0.0
    while elapsed < duration - 1.0e-14:
        length = min(block_duration, duration - elapsed)
        state, tangent, log_growth = block(state, tangent, length)
        increments.append(log_growth)
        elapsed += length
    midpoint = len(increments) // 2
    full = float(sum(increments) / duration)
    second_duration = (len(increments) - midpoint) * block_duration
    second = float(sum(increments[midpoint:]) / second_duration)
    return {
        "method": "Benettin_DOP853_analytic_variational",
        "warmup": warmup,
        "duration": duration,
        "renormalization_interval": block_duration,
        "initial_tangent": [1.0, math.sqrt(2.0), math.sqrt(3.0)],
        "largest_lyapunov_exponent": full,
        "second_half_largest_lyapunov_exponent": second,
        "positive_threshold": 1.0e-3,
        "positive": bool(full > 1.0e-3 and second > 1.0e-3),
    }


def json_dump(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_dict_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_fourier_csv(path: Path, values: np.ndarray) -> int:
    modes, coefficients = fourier_coefficients(values)
    order = np.argsort(modes)
    rows = []
    for index in order:
        coefficient = coefficients[index]
        rows.append(
            {
                "mode": int(modes[index]),
                "x_real": f"{coefficient[0].real:.17g}",
                "x_imag": f"{coefficient[0].imag:.17g}",
                "y_real": f"{coefficient[1].real:.17g}",
                "y_imag": f"{coefficient[1].imag:.17g}",
                "z_real": f"{coefficient[2].real:.17g}",
                "z_imag": f"{coefficient[2].imag:.17g}",
            }
        )
    return write_dict_csv(
        path,
        ["mode", "x_real", "x_imag", "y_real", "y_imag", "z_real", "z_imag"],
        rows,
    )


def dense_orbit_rows(solution: OrbitSolution) -> list[dict[str, str]]:
    n_rows = max(4097, int(math.ceil(solution.period / 0.002)) + 1)
    phases = np.linspace(0.0, 1.0, n_rows)
    wrapped = np.mod(phases, 1.0)
    values = evaluate_fourier(solution.values, wrapped)
    tangents = roessler_field(values, solution.a)
    rows: list[dict[str, str]] = []
    for phase, value, tangent in zip(phases, values, tangents):
        rows.append(
            {
                "phase": f"{phase:.17g}",
                "time": f"{phase * solution.period:.17g}",
                "x": f"{value[0]:.17g}",
                "y": f"{value[1]:.17g}",
                "z": f"{value[2]:.17g}",
                "dx": f"{tangent[0]:.17g}",
                "dy": f"{tangent[1]:.17g}",
                "dz": f"{tangent[2]:.17g}",
            }
        )
    return rows


def periodic_analysis_stream(solution: OrbitSolution) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(ANALYSIS_N_SAMPLES, dtype=float) * SAMPLE_DT
    phases = np.mod(times / solution.period, 1.0)
    positions = evaluate_fourier(solution.values, phases)
    tangents = roessler_field(positions, solution.a)
    return positions, tangents


def write_periodic_case(
    root: Path,
    case_id: str,
    solution: OrbitSolution,
    validation: OrbitValidation,
) -> dict[str, Any]:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    coefficients_path = case_dir / "fourier_coefficients.csv"
    dense_path = case_dir / "orbit_dense.csv"
    positions_path = case_dir / "analysis_positions.csv"
    tangents_path = case_dir / "analysis_tangents.csv"
    certificate_path = case_dir / "certificate.json"

    n_modes = write_fourier_csv(coefficients_path, solution.values)
    dense_rows = dense_orbit_rows(solution)
    write_dict_csv(
        dense_path,
        ["phase", "time", "x", "y", "z", "dx", "dy", "dz"],
        dense_rows,
    )
    positions, tangents = periodic_analysis_stream(solution)
    np.savetxt(positions_path, positions, fmt="%.17g")
    np.savetxt(tangents_path, tangents, fmt="%.17g")

    dependent_hashes = {
        "fourier_coefficients_sha256": sha256(coefficients_path),
        "orbit_dense_sha256": sha256(dense_path),
        "analysis_positions_sha256": sha256(positions_path),
        "analysis_tangents_sha256": sha256(tangents_path),
    }
    certificate = {
        "schema_version": 1,
        "case_id": case_id,
        "kind": "periodic",
        "status": "certified",
        "a": solution.a,
        "q": solution.q,
        "sample_dt": SAMPLE_DT,
        "analysis_duration": ANALYSIS_DURATION,
        "analysis_n_samples": ANALYSIS_N_SAMPLES,
        "collocation_nodes": len(solution.values),
        "fourier_signed_mode_count": n_modes,
        "period": solution.period,
        "seed_poincare_closure_error": solution.seed_closure,
        "collocation_residual_infinity_norm": solution.collocation_residual,
        "oversampled_residual_absolute": validation.oversampled_residual_absolute,
        "oversampled_residual_relative": validation.oversampled_residual_relative,
        "dop853_closure_error_absolute": validation.closure_error_absolute,
        "dop853_closure_error_relative": validation.closure_error_relative,
        "floquet_multipliers": multiplier_records(validation.multipliers),
        "trivial_multiplier_error": validation.trivial_multiplier_error,
        "flip_multiplier": {
            "real": validation.flip_multiplier.real,
            "imag": validation.flip_multiplier.imag,
            "magnitude": abs(validation.flip_multiplier),
        },
        "stable_transverse": validation.stable_transverse,
        "fourier_tail_energy_ratio": validation.tail_energy_ratio,
        "primitive_shift_rms": validation.shift_rms,
        "poincare_section": {
            "definition": "x=0 with dx/dt>0",
            "x_at_phase_zero": validation.section_x,
            "dxdt_at_phase_zero": validation.section_dxdt,
        },
        "integration_validation": {
            "method": "DOP853",
            "rtol": 2.0e-12,
            "atol": 2.0e-14,
            "variational_jacobian": "analytic",
        },
        "analysis_stream": {
            "construction": "phase_continuous_certified_Fourier_orbit_tiling",
            "maximum_supported_segment_length_samples": MAX_SEGMENT_LENGTH_SAMPLES,
        },
        **dependent_hashes,
    }
    json_dump(certificate_path, certificate)
    return {
        "id": case_id,
        "kind": "periodic",
        "a": solution.a,
        "q": solution.q,
        "analysis": {
            "positions": str(positions_path.relative_to(root)),
            "tangents": str(tangents_path.relative_to(root)),
            "sample_dt": SAMPLE_DT,
            "n_samples": ANALYSIS_N_SAMPLES,
            "positions_sha256": dependent_hashes["analysis_positions_sha256"],
            "tangents_sha256": dependent_hashes["analysis_tangents_sha256"],
        },
        "display": {
            "kind": "certified_primitive_periodic_orbit",
            "path": str(dense_path.relative_to(root)),
            "sha256": dependent_hashes["orbit_dense_sha256"],
            "n_rows": len(dense_rows),
            "duration": solution.period,
        },
        "fourier": {
            "path": str(coefficients_path.relative_to(root)),
            "sha256": dependent_hashes["fourier_coefficients_sha256"],
            "n_modes": n_modes,
        },
        "certificate": {
            "path": str(certificate_path.relative_to(root)),
            "sha256": sha256(certificate_path),
        },
    }


def write_chaos_case(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id, a = CHAOS_REPRESENTATIVE
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    positions_path = case_dir / "analysis_positions.csv"
    tangents_path = case_dir / "analysis_tangents.csv"
    display_path = case_dir / "display_segment.csv"
    certificate_path = case_dir / "certificate.json"

    initial = np.asarray((1.0, 1.0, 0.0), dtype=float)
    burn = integrate_state(
        initial,
        a,
        (0.0, CHAOS_BURN_IN),
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    retained_initial = burn.y[:, -1]
    times = np.arange(ANALYSIS_N_SAMPLES, dtype=float) * SAMPLE_DT
    retained = integrate_state(
        retained_initial,
        a,
        (0.0, ANALYSIS_DURATION),
        t_eval=times,
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    positions = retained.y.T
    tangents = roessler_field(positions, a)
    np.savetxt(positions_path, positions, fmt="%.17g")
    np.savetxt(tangents_path, tangents, fmt="%.17g")

    display_count = int(round(CHAOS_DISPLAY_DURATION / SAMPLE_DT)) + 1
    display_rows = []
    for time, value, tangent in zip(
        times[:display_count],
        positions[:display_count],
        tangents[:display_count],
    ):
        display_rows.append(
            {
                "time": f"{time:.17g}",
                "x": f"{value[0]:.17g}",
                "y": f"{value[1]:.17g}",
                "z": f"{value[2]:.17g}",
                "dx": f"{tangent[0]:.17g}",
                "dy": f"{tangent[1]:.17g}",
                "dz": f"{tangent[2]:.17g}",
            }
        )
    write_dict_csv(
        display_path,
        ["time", "x", "y", "z", "dx", "dy", "dz"],
        display_rows,
    )
    lle = largest_lyapunov_exponent(retained_initial, a)
    if not lle["positive"]:
        raise NumericalFailure(
            f"a={a}: chaos representative failed positive-LLE validation: "
            f"{lle['largest_lyapunov_exponent']:.6g}"
        )
    dependent_hashes = {
        "analysis_positions_sha256": sha256(positions_path),
        "analysis_tangents_sha256": sha256(tangents_path),
        "display_segment_sha256": sha256(display_path),
    }
    certificate = {
        "schema_version": 1,
        "case_id": case_id,
        "kind": "chaos",
        "status": "certified",
        "a": a,
        "q": None,
        "sample_dt": SAMPLE_DT,
        "analysis_duration": ANALYSIS_DURATION,
        "analysis_n_samples": ANALYSIS_N_SAMPLES,
        "burn_in": CHAOS_BURN_IN,
        "integration": {
            "method": "DOP853",
            "rtol": 1.0e-11,
            "atol": 1.0e-13,
            "burn_in_separate_from_retained_stream": True,
        },
        "largest_lyapunov_validation": lle,
        "display_duration": CHAOS_DISPLAY_DURATION,
        "maximum_supported_segment_length_samples": MAX_SEGMENT_LENGTH_SAMPLES,
        **dependent_hashes,
    }
    json_dump(certificate_path, certificate)
    case_record = {
        "id": case_id,
        "kind": "chaos",
        "a": a,
        "q": None,
        "analysis": {
            "positions": str(positions_path.relative_to(root)),
            "tangents": str(tangents_path.relative_to(root)),
            "sample_dt": SAMPLE_DT,
            "n_samples": ANALYSIS_N_SAMPLES,
            "positions_sha256": dependent_hashes["analysis_positions_sha256"],
            "tangents_sha256": dependent_hashes["analysis_tangents_sha256"],
        },
        "display": {
            "kind": "bounded_chaos_segment",
            "path": str(display_path.relative_to(root)),
            "sha256": dependent_hashes["display_segment_sha256"],
            "n_rows": len(display_rows),
            "duration": CHAOS_DISPLAY_DURATION,
        },
        "certificate": {
            "path": str(certificate_path.relative_to(root)),
            "sha256": sha256(certificate_path),
        },
    }
    return case_record, certificate


def branch_row(point: BranchPoint) -> dict[str, Any]:
    solution = point.solution
    validation = point.validation
    return {
        "branch_q": solution.q,
        "stage": point.stage,
        "a": f"{solution.a:.17g}",
        "period": f"{solution.period:.17g}",
        "collocation_nodes": len(solution.values),
        "collocation_residual": f"{solution.collocation_residual:.17g}",
        "oversampled_residual_relative": f"{validation.oversampled_residual_relative:.17g}",
        "closure_error_relative": f"{validation.closure_error_relative:.17g}",
        "flip_multiplier_real": f"{validation.flip_multiplier.real:.17g}",
        "flip_multiplier_imag": f"{validation.flip_multiplier.imag:.17g}",
        "flip_measure": f"{flip_measure(validation):.17g}",
        "trivial_multiplier_error": f"{validation.trivial_multiplier_error:.17g}",
        "stable_transverse": str(validation.stable_transverse).lower(),
    }


def representative_row(
    case_id: str,
    solution: OrbitSolution,
    validation: OrbitValidation,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "kind": "periodic",
        "a": f"{solution.a:.17g}",
        "q": solution.q,
        "period": f"{solution.period:.17g}",
        "period_per_return": f"{solution.period / solution.q:.17g}",
        "collocation_nodes": len(solution.values),
        "oversampled_residual_relative": f"{validation.oversampled_residual_relative:.17g}",
        "closure_error_relative": f"{validation.closure_error_relative:.17g}",
        "trivial_multiplier_error": f"{validation.trivial_multiplier_error:.17g}",
        "stable_transverse": str(validation.stable_transverse).lower(),
        "fourier_tail_energy_ratio": f"{validation.tail_energy_ratio:.17g}",
    }


def output_inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"bundle_manifest.json", "BUILD_INCOMPLETE"}:
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def build_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    incomplete = root / "BUILD_INCOMPLETE"
    incomplete.write_text(
        "Bundle construction did not reach atomic manifest finalization.\n",
        encoding="utf-8",
    )
    (root / "summary").mkdir()
    (root / "cases").mkdir()

    branch_specs = {
        1: {"step": 0.005, "limit": 2.87, "nodes": 129},
        2: {"step": 0.05, "limit": 3.92, "nodes": 129},
        4: {"step": 0.01, "limit": 4.20, "nodes": 513},
        8: {"step": 0.005, "limit": 4.24, "nodes": 1025},
    }
    representative_seeds: dict[str, OrbitSolution] = {}
    all_branch_points: list[BranchPoint] = []
    flip_records: list[dict[str, Any]] = []

    for case_id, a, q in REPRESENTATIVES:
        print(f"Seeding {case_id} at a={a}, q={q}", flush=True)
        representative = stable_orbit_seed(a, q, BRANCH_NODES)
        branch_nodes = int(branch_specs[q]["nodes"])
        if branch_nodes != len(representative.values):
            representative = refine_to_node_count(representative, branch_nodes)
        representative_seeds[case_id] = representative
        points, coarse_bracket = trace_parent_to_flip(
            representative,
            nominal_step=float(branch_specs[q]["step"]),
            a_limit=float(branch_specs[q]["limit"]),
        )
        refinements, left, right = refine_flip_bracket(*coarse_bracket)
        all_branch_points.extend(points)
        all_branch_points.extend(refinements)
        flip_records.append(
            {
                "parent_q": q,
                "child_q": 2 * q,
                "a_left": f"{left.solution.a:.17g}",
                "a_right": f"{right.solution.a:.17g}",
                "a_estimate": f"{0.5 * (left.solution.a + right.solution.a):.17g}",
                "bracket_width": f"{right.solution.a - left.solution.a:.17g}",
                "mu_left_real": f"{left.validation.flip_multiplier.real:.17g}",
                "mu_right_real": f"{right.validation.flip_multiplier.real:.17g}",
                "period_left": f"{left.solution.period:.17g}",
                "period_right": f"{right.solution.period:.17g}",
                "method": "Fourier_natural_continuation_DOP853_Floquet_bisection",
            }
        )

    branch_fields = [
        "branch_q",
        "stage",
        "a",
        "period",
        "collocation_nodes",
        "collocation_residual",
        "oversampled_residual_relative",
        "closure_error_relative",
        "flip_multiplier_real",
        "flip_multiplier_imag",
        "flip_measure",
        "trivial_multiplier_error",
        "stable_transverse",
    ]
    branch_path = root / "summary" / "branch_points.csv"
    write_dict_csv(branch_path, branch_fields, map(branch_row, all_branch_points))
    flip_path = root / "summary" / "flips.csv"
    flip_fields = [
        "parent_q",
        "child_q",
        "a_left",
        "a_right",
        "a_estimate",
        "bracket_width",
        "mu_left_real",
        "mu_right_real",
        "period_left",
        "period_right",
        "method",
    ]
    write_dict_csv(flip_path, flip_fields, flip_records)

    case_records: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    for case_id, _a, _q in REPRESENTATIVES:
        print(f"Certifying and writing {case_id}", flush=True)
        certified, validation = refine_and_certify(representative_seeds[case_id])
        if not validation.stable_transverse:
            raise NumericalFailure(f"{case_id} is not transversely stable")
        case_records.append(
            write_periodic_case(root, case_id, certified, validation)
        )
        representative_rows.append(representative_row(case_id, certified, validation))

    print("Generating and validating chaos case", flush=True)
    chaos_record, chaos_certificate = write_chaos_case(root)
    case_records.append(chaos_record)
    representative_rows.append(
        {
            "case_id": "chaos",
            "kind": "chaos",
            "a": f"{CHAOS_REPRESENTATIVE[1]:.17g}",
            "q": "",
            "period": "",
            "period_per_return": "",
            "collocation_nodes": "",
            "oversampled_residual_relative": "",
            "closure_error_relative": "",
            "trivial_multiplier_error": "",
            "stable_transverse": "",
            "fourier_tail_energy_ratio": "",
        }
    )
    representative_path = root / "summary" / "representatives.csv"
    representative_fields = [
        "case_id",
        "kind",
        "a",
        "q",
        "period",
        "period_per_return",
        "collocation_nodes",
        "oversampled_residual_relative",
        "closure_error_relative",
        "trivial_multiplier_error",
        "stable_transverse",
        "fourier_tail_energy_ratio",
    ]
    write_dict_csv(
        representative_path,
        representative_fields,
        representative_rows,
    )

    summaries = {
        "branch_points": {
            "path": str(branch_path.relative_to(root)),
            "sha256": sha256(branch_path),
        },
        "flips": {
            "path": str(flip_path.relative_to(root)),
            "sha256": sha256(flip_path),
        },
        "representatives": {
            "path": str(representative_path.relative_to(root)),
            "sha256": sha256(representative_path),
        },
    }
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "bundle_id": BUNDLE_ID,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": repository_relative(script_path),
            "sha256": sha256(script_path),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "system": {
            "name": "roessler",
            "family": "david_hien_period_doubling",
            "state_order": ["x", "y", "z"],
            "equations": {
                "x": "-y-z",
                "y": "x+0.2*y",
                "z": "0.2+z*(x-a)",
            },
            "fixed_parameters": {"b": B_FIXED},
            "continuation_parameter": "a",
        },
        "sample_dt": SAMPLE_DT,
        "max_supported_segment_length_samples": MAX_SEGMENT_LENGTH_SAMPLES,
        "analysis_duration": ANALYSIS_DURATION,
        "cases": case_records,
        "summaries": summaries,
        "files": output_inventory(root),
    }
    manifest_path = root / "bundle_manifest.json"
    json_dump(manifest_path, manifest)
    incomplete.unlink()
    print(f"Completed bundle: {manifest_path}", flush=True)
    print(
        "Chaos LLE: "
        f"{chaos_certificate['largest_lyapunov_validation']['largest_lyapunov_exponent']:.9g}",
        flush=True,
    )
    return manifest_path


def plan_document(output_root: Path) -> dict[str, Any]:
    return {
        "action": "check_only",
        "writes_performed": False,
        "execute_flag_required": True,
        "bundle_id": BUNDLE_ID,
        "output_root": str(output_root),
        "system": "David Rössler family, b=0.2 and varying a",
        "periodic_representatives": [
            {"id": case_id, "a": a, "q": q}
            for case_id, a, q in REPRESENTATIVES
        ],
        "chaos_representative": {
            "id": CHAOS_REPRESENTATIVE[0],
            "a": CHAOS_REPRESENTATIVE[1],
        },
        "sample_dt": SAMPLE_DT,
        "analysis_duration": ANALYSIS_DURATION,
        "analysis_n_samples_per_case": ANALYSIS_N_SAMPLES,
        "maximum_supported_segment_length_samples": MAX_SEGMENT_LENGTH_SAMPLES,
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform continuation and create the immutable output bundle",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="new child directory below code/experiments_planned/outputs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = guard_new_output_root(args.output_root)
    if not args.execute:
        print(json.dumps(plan_document(output_root), indent=2, sort_keys=True))
        return 0
    build_bundle(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
