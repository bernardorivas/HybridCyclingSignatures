#!/usr/bin/env python3
"""Render one completed planned fine-Compass construction as a 2x5 PDF.

This is a read-only adapter for outputs of ``run_duration_c_radius.jl``.  It
does not simulate, train, or compute a cycling signature.  The horizontal
coordinate is the explicit target-duration grid frozen in ``plan.json``;
realized durations remain provenance checks and are not rebinned.  Every
manifest window contributes to its target-duration cell, including empty
birth vectors (rank zero).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

import analyze_fine_compass_c_radius_sweep as analysis


REGIMES = ("period1", "period2", "period4", "period8", "chaos")
JOB_KIND = "fine_compass_duration_c_radius_jobs"


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _load_panel_module(workspace_root: Path) -> ModuleType:
    source = (
        workspace_root
        / "paper"
        / "hybrid_cyclingsignatures"
        / "scripts"
        / "generate_period_doubling_probability_panels.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"missing shared paper renderer: {source}")
    name = "_hybrid_cycling_probability_panels"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared paper renderer: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_jobs(path: Path, plan_root: Path, code_root: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing real jobs file: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("kind") != JOB_KIND:
        raise ValueError(f"unsupported jobs document: {path}")
    recorded_plan = Path(str(document.get("plan_root", ""))).resolve()
    if recorded_plan != plan_root:
        raise ValueError(
            f"jobs document names plan root {recorded_plan}, expected {plan_root}"
        )
    recorded_working = Path(str(document.get("working_directory", ""))).resolve()
    if recorded_working != code_root:
        raise ValueError(
            f"jobs document names code root {recorded_working}, expected {code_root}"
        )
    jobs = document.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs document contains no jobs")
    if not all(isinstance(job, dict) for job in jobs):
        raise ValueError("every job must be a JSON object")
    return jobs


def _selected_jobs(
    jobs: list[dict],
    arm: str,
    c_scale: float,
    boxsize: float,
    sb_radius: int,
    plan_root: Path,
    code_root: Path,
) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    out_dirs: set[Path] = set()
    for job in jobs:
        try:
            matches = (
                _close(float(job["C"]), c_scale)
                and _close(float(job["boxsize"]), boxsize)
                and int(job["sb_radius"]) == sb_radius
                and str(job.get("arm", "")) == arm
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed construction in jobs document") from error
        if not matches:
            continue
        regime = str(job.get("regime", ""))
        if regime not in REGIMES:
            raise ValueError(f"selected job has invalid regime: {regime!r}")
        if regime in selected:
            raise ValueError(f"multiple selected jobs for {regime}")
        prefix = str(job.get("out_prefix", ""))
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", prefix) is None:
            raise ValueError(f"unsafe output prefix for {regime}: {prefix!r}")
        manifest = Path(str(job.get("manifest", ""))).resolve()
        expected_manifest = (
            plan_root / "manifests" / f"{regime}.csv"
        ).resolve()
        if manifest != expected_manifest:
            raise ValueError(f"selected job names a different manifest: {regime}")
        data_dir = Path(str(job.get("data_dir", ""))).resolve()
        expected_data_dir = (
            code_root
            / "period_doubling"
            / "data_fine"
            / "compass_gait_latent"
        ).resolve()
        if data_dir != expected_data_dir:
            raise ValueError(f"selected job names a different lift: {regime}")
        out_dir = Path(str(job.get("out_dir", ""))).resolve()
        signature_root = (plan_root / "signatures").resolve()
        if not out_dir.is_relative_to(signature_root):
            raise ValueError(
                f"selected job output lies outside plan signatures: {out_dir}"
            )
        if out_dir in out_dirs:
            raise ValueError(f"selected jobs reuse output directory: {out_dir}")
        selected[regime] = job
        out_dirs.add(out_dir)
    missing = [regime for regime in REGIMES if regime not in selected]
    if missing:
        raise ValueError(
            "construction is incomplete; missing selected jobs for "
            + ", ".join(missing)
        )
    return selected


def _result_metadata_path(
    job: dict,
    plan_root: Path,
    results_root: Path,
) -> Path:
    planned_signature_root = (plan_root / "signatures").resolve()
    planned_out_dir = Path(str(job["out_dir"])).resolve()
    relative = planned_out_dir.relative_to(planned_signature_root)
    directory = results_root / relative
    prefix = str(job["out_prefix"])
    path = directory / f"{prefix}_metadata.txt"
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing real result metadata: {path}")
    births = path.with_name(path.name.removesuffix("_metadata.txt") + "_births.csv")
    if births.is_symlink() or not births.is_file():
        raise FileNotFoundError(f"missing real result births: {births}")
    return path


def _validate_record(
    record: analysis.ResultRecord,
    job: dict,
    plan_root: Path,
    code_root: Path,
    regime: str,
    rho_grid: list[float],
) -> None:
    expected = (
        float(job["C"]),
        float(job["boxsize"]),
        int(job["sb_radius"]),
        float(job["rho_max"]),
    )
    actual = (
        record.c_scale,
        record.boxsize,
        record.sb_radius,
        record.rho_max,
    )
    if record.regime != regime or not all(
        _close(left, right) for left, right in zip(actual, expected)
    ):
        raise ValueError(f"{regime}: result metadata does not match selected job")
    expected_mode = analysis._metric_mode(job.get("metric_mode"))
    expected_cover_default = float(
        job.get("cover_default_C", float(job["boxsize"]) * int(job["sb_radius"]))
    )
    if (
        record.metric_mode != expected_mode
        or not _close(record.cover_default_c, expected_cover_default)
    ):
        raise ValueError(f"{regime}: result metric provenance differs from job")
    if not _close(record.rho_max, rho_grid[-1]):
        raise ValueError(
            f"{regime}: result stops at rho={record.rho_max:g}; "
            f"full planned grid stops at rho={rho_grid[-1]:g}"
        )
    metadata = analysis._parse_metadata(record.metadata_path)
    if metadata.get("split") != "all":
        raise ValueError(f"{regime}: figure requires combined tune and validate rows")
    manifest = (plan_root / "manifests" / f"{regime}.csv").resolve()
    if Path(metadata.get("manifest", "")).resolve() != manifest:
        raise ValueError(f"{regime}: result metadata names a different manifest")
    lift_root = (
        code_root
        / "period_doubling"
        / "data_fine"
        / "compass_gait_latent"
    ).resolve()
    expected_positions = lift_root / f"compass_{regime}_positions.csv"
    expected_tangents = Path(
        str(job.get("tangents", lift_root / f"compass_{regime}_tangents.csv"))
    ).resolve()
    if (
        Path(metadata.get("positions", "")).resolve() != expected_positions
        or Path(metadata.get("tangents", "")).resolve() != expected_tangents
    ):
        raise ValueError(f"{regime}: result metadata names different lift inputs")


def _probability_grid(
    record: analysis.ResultRecord,
    durations: list[float],
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    probability = np.empty((len(radii), len(durations)), dtype=float)
    denominators = np.empty(len(durations), dtype=int)
    rank_zero = 0
    for column, duration in enumerate(durations):
        rows = [
            row
            for key, row in record.rows.items()
            if _close(key[0], duration)
        ]
        if not rows:
            raise ValueError(
                f"{record.regime}: empty target-duration cell {duration:g}"
            )
        first_births = np.asarray(
            [row.births[0] if row.births else np.inf for row in rows], dtype=float
        )
        rank_zero += int(np.isinf(first_births).sum())
        denominators[column] = len(first_births)
        probability[:, column] = np.mean(
            first_births[np.newaxis, :] <= radii[:, np.newaxis], axis=1
        )
    return probability, denominators, rank_zero


def _orbit_panel(
    panel_module: ModuleType,
    code_root: Path,
    regime: str,
):
    raw_path = (
        code_root
        / "period_doubling"
        / "data_fine"
        / "compass_gait"
        / f"compass_{regime}.npz"
    )
    if raw_path.is_symlink() or not raw_path.is_file():
        raise FileNotFoundError(f"missing read-only raw orbit: {raw_path}")
    with np.load(raw_path, allow_pickle=False) as raw:
        metadata = json.loads(str(raw["meta_json"]))
        return_lag = panel_module.RETURN_LAGS.get(regime)
        arcs, jump_minus, jump_plus, period = panel_module.extract_compass_display(
            raw,
            return_lag if return_lag is not None else 8,
            require_closure=return_lag is not None,
        )
    return float(metadata["phi_deg"]), arcs, jump_minus, jump_plus, period


def _parser() -> argparse.ArgumentParser:
    code_root = Path(__file__).resolve().parents[1]
    workspace_root = code_root.parent
    parser = argparse.ArgumentParser(
        description=(
            "Validate one completed fine-Compass C/factorization from the "
            "prepared duration sweep and render compassgait_C*.pdf."
        )
    )
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument(
        "--jobs-file",
        type=Path,
        help="Structured jobs JSON (default: PLAN_ROOT/jobs.json).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Completed result tree (default: PLAN_ROOT/signatures).",
    )
    parser.add_argument("--c", type=float, required=True, dest="c_scale")
    parser.add_argument(
        "--arm",
        default="primary",
        help="Named experiment arm in jobs.json (default: primary).",
    )
    parser.add_argument("--boxsize", type=float, required=True)
    parser.add_argument("--sb-radius", type=int, required=True)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=code_root,
        help="Read-only code/artifact root used by the jobs and orbit row.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            workspace_root
            / "paper"
            / "hybrid_cyclingsignatures"
            / "figures"
            / "cycling_signatures"
        ),
        help="Destination for the new C-suffixed PDF.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not math.isfinite(args.c_scale) or args.c_scale <= 0:
        raise ValueError("--c must be positive and finite")
    if not math.isfinite(args.boxsize) or args.boxsize <= 0:
        raise ValueError("--boxsize must be positive and finite")
    if args.sb_radius <= 0:
        raise ValueError("--sb-radius must be positive")

    workspace_root = Path(__file__).resolve().parents[2]
    code_root = args.code_root.expanduser().absolute().resolve()
    requested_plan_root = args.plan_root.expanduser().absolute()
    if requested_plan_root.is_symlink() or not requested_plan_root.is_dir():
        raise FileNotFoundError(
            f"missing nonsymlink plan directory: {requested_plan_root}"
        )
    plan_root = requested_plan_root.resolve()
    requested_results_root = (
        args.results_root.expanduser().absolute()
        if args.results_root is not None
        else plan_root / "signatures"
    )
    if requested_results_root.is_symlink() or not requested_results_root.is_dir():
        raise FileNotFoundError(
            f"missing nonsymlink results directory: {requested_results_root}"
        )
    results_root = requested_results_root.resolve()
    requested_jobs_path = (
        args.jobs_file.expanduser().absolute()
        if args.jobs_file is not None
        else plan_root / "jobs.json"
    )
    if requested_jobs_path.is_symlink():
        raise ValueError(f"jobs file must not be a symlink: {requested_jobs_path}")
    jobs_path = requested_jobs_path.resolve()

    plan, _, c_grid, rho_grid, durations = analysis._load_plan(plan_root)
    planned_c = analysis._match_grid(args.c_scale, c_grid, "selected C")
    if plan.get("experiment_arms") is not None:
        planned_arm = analysis._experiment_arm(plan, args.arm, planned_c)
        if planned_arm is None:
            raise ValueError(f"unknown experiment arm: {args.arm}")
        planned_boxsize = float(planned_arm["boxsize"])
        planned_sb_radius = int(planned_arm["sb_radius"])
    else:
        if args.arm != "primary":
            raise ValueError("legacy plans render only the primary arm")
        _, planned_boxsize, planned_sb_radius = analysis._primary_construction(
            plan, planned_c
        )
    if (
        not _close(args.boxsize, planned_boxsize)
        or args.sb_radius != planned_sb_radius
    ):
        raise ValueError(
            "requested factorization is not the plan's primary construction: "
            f"expected boxsize={planned_boxsize:g}, "
            f"sb_radius={planned_sb_radius}"
        )
    manifests, _ = analysis._load_manifests(plan_root, plan, durations)
    jobs = _load_jobs(jobs_path, plan_root, code_root)
    selected = _selected_jobs(
        jobs,
        args.arm,
        planned_c,
        args.boxsize,
        args.sb_radius,
        plan_root,
        code_root,
    )

    panel_module = _load_panel_module(workspace_root)
    records = []
    metadata_paths: set[Path] = set()
    for regime in REGIMES:
        metadata_path = _result_metadata_path(
            selected[regime], plan_root, results_root
        )
        if metadata_path in metadata_paths:
            raise ValueError(f"mixed result selection reuses {metadata_path}")
        record = analysis._load_result_record(
            metadata_path, manifests, durations
        )
        _validate_record(
            record,
            selected[regime],
            plan_root,
            code_root,
            regime,
            rho_grid,
        )
        records.append(record)
        metadata_paths.add(metadata_path)

    actual_c = records[0].c_scale
    construction = (records[0].boxsize, records[0].sb_radius)
    metric_mode = records[0].metric_mode
    cover_default_c = records[0].cover_default_c
    if not all(
        _close(record.c_scale, actual_c)
        and record.metric_mode == metric_mode
        and _close(record.cover_default_c, cover_default_c)
        and _close(record.boxsize, construction[0])
        and record.sb_radius == construction[1]
        for record in records[1:]
    ):
        raise ValueError("five result pairs mix C values or cover factorizations")

    radii = actual_c * np.asarray(rho_grid, dtype=float)
    plot_panels = []
    rank_zero_counts: dict[str, int] = {}
    for record in records:
        probability, denominators, rank_zero = _probability_grid(
            record, durations, radii
        )
        phi_deg, arcs, jump_minus, jump_plus, orbit_period = _orbit_panel(
            panel_module, code_root, record.regime
        )
        plot_panels.append(
            panel_module.CompassProbabilityPanel(
                regime=record.regime,
                phi_deg=phi_deg,
                orbit_period=orbit_period,
                duration_centers=np.asarray(durations, dtype=float),
                radii=radii,
                probability=probability,
                n_per_duration=denominators,
                c_scale=record.c_scale,
                orbit_arcs=arcs,
                jump_minus=jump_minus,
                jump_plus=jump_plus,
            )
        )
        rank_zero_counts[record.regime] = rank_zero

    output_dir = args.output_dir.resolve()
    canonical_code_root = workspace_root / "code"
    panel_module.guard_output(canonical_code_root, output_dir)
    if code_root != canonical_code_root.resolve():
        panel_module.guard_output(code_root, output_dir)
    for read_only_root, label in (
        (plan_root, "plan"),
        (results_root, "results"),
    ):
        if output_dir == read_only_root or output_dir.is_relative_to(read_only_root):
            raise ValueError(f"output directory must not modify the {label} tree")
    paper_figure_root = (
        workspace_root
        / "paper"
        / "hybrid_cyclingsignatures"
        / "figures"
        / "cycling_signatures"
    ).resolve()
    if output_dir.is_relative_to(workspace_root) and not (
        output_dir == paper_figure_root
        or output_dir.is_relative_to(paper_figure_root)
    ):
        raise ValueError(
            "repository output must stay below the paper cycling-signature "
            f"figure directory: {paper_figure_root}"
        )
    target = output_dir / (
        "compassgait_" + panel_module.c_filename_suffix(actual_c) + ".pdf"
    )
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing figure: {target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = panel_module.render_compass_probability_panels(
        plot_panels, output_dir
    )
    if output_path != target:
        raise RuntimeError(f"unexpected output path: {output_path}")
    print(
        f"Rendered target-duration grid with C={actual_c:g}, "
        f"metric_mode={metric_mode}, cover_default_C={cover_default_c:g}, "
        f"boxsize={construction[0]:g}, sb_radius={construction[1]}, "
        f"arm={args.arm}"
    )
    for record in records:
        print(
            f"  {record.regime}: windows={record.n_windows}, "
            f"rank0={rank_zero_counts[record.regime]}, beta1(Y)={record.beta1}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
