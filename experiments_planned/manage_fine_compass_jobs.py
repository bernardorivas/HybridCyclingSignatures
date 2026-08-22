#!/usr/bin/env python3
"""Inspect or execute prepared fine-Compass signature jobs safely.

The default mode is read-only: it validates the structured job document and
prints which jobs would run, resume, or skip.  ``--execute`` is required to
launch Julia.  Each launched job writes into a private staging directory;
the manager validates the complete births/metadata pair against the frozen
manifest before atomically promoting that directory to its final name.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import shlex
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import analyze_fine_compass_c_radius_sweep as analysis
import prepare_fine_compass_c_radius_sweep as planner


REGIMES = ("period1", "period2", "period4", "period8", "chaos")
JOB_KIND = "fine_compass_duration_c_radius_jobs"


@dataclass(frozen=True)
class Job:
    job_id: str
    arm: str
    regime: str
    c_scale: float
    metric_mode: str
    cover_default_c: float
    boxsize: float
    sb_radius: int
    rho_max: float
    data_dir: Path
    positions: Path
    tangents: Path
    positions_sha256: str | None
    tangents_sha256: str | None
    manifest_sha256: str | None
    tangent_provenance: Path | None
    tangent_provenance_sha256: str | None
    tangent_provenance_regime: dict | None
    manifest: Path
    out_dir: Path
    out_prefix: str

    @property
    def births_path(self) -> Path:
        return self.out_dir / f"{self.out_prefix}_births.csv"

    @property
    def metadata_path(self) -> Path:
        return self.out_dir / f"{self.out_prefix}_metadata.txt"

    @property
    def staging_dir(self) -> Path:
        return self.out_dir.with_name(f".{self.out_dir.name}.partial")


@dataclass(frozen=True)
class Context:
    code_root: Path
    safe_root: Path
    plan_root: Path
    plan: dict
    durations: list[float]
    manifests: dict


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _require_under(path: Path, parent: Path, label: str) -> Path:
    resolved = _resolved(path)
    allowed = _resolved(parent)
    if resolved != allowed and not resolved.is_relative_to(allowed):
        raise ValueError(f"{label} must stay below {allowed}: {resolved}")
    return resolved


def _positive(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _load_jobs(path: Path) -> tuple[list[Job], Context]:
    code_root = Path(__file__).resolve().parents[1]
    safe_root = (code_root / "experiments_planned" / "outputs").resolve()
    jobs_path = _require_under(path, safe_root, "jobs file")
    if not jobs_path.is_file():
        raise FileNotFoundError(f"missing jobs file: {jobs_path}")
    document = json.loads(jobs_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("kind") != JOB_KIND:
        raise ValueError("unsupported jobs document")
    if _resolved(Path(document.get("working_directory", ""))) != code_root:
        raise ValueError("jobs document belongs to a different code checkout")

    plan_root = _require_under(
        Path(document.get("plan_root", "")), safe_root, "plan root"
    )
    if not (plan_root / "plan.json").is_file():
        raise FileNotFoundError(f"missing plan beneath {plan_root}")
    plan, _, c_grid, rho_grid, durations = analysis._load_plan(plan_root)
    manifests, _ = analysis._load_manifests(plan_root, plan, durations)
    selection = None
    if document.get("selection") is not None:
        selection_path = _require_under(
            Path(document["selection"]), safe_root, "selection"
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "frozen":
            raise ValueError("control jobs require a frozen selection")
        if _resolved(Path(selection.get("plan_root", ""))) != plan_root:
            raise ValueError("selection belongs to a different plan")

    raw_jobs = document.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("jobs document has no jobs")
    jobs: list[Job] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    data_dir_expected = (
        code_root / "period_doubling" / "data_fine" / "compass_gait_latent"
    ).resolve()
    for raw in raw_jobs:
        if not isinstance(raw, dict):
            raise ValueError("each job must be an object")
        job_id = str(raw.get("job_id", ""))
        arm = str(raw.get("arm", ""))
        regime = str(raw.get("regime", ""))
        out_prefix = str(raw.get("out_prefix", ""))
        safe_name = r"[A-Za-z0-9][A-Za-z0-9._-]*"
        if re.fullmatch(safe_name, job_id) is None or job_id in seen_ids:
            raise ValueError(f"invalid or duplicate job id: {job_id!r}")
        if re.fullmatch(safe_name, arm) is None:
            raise ValueError(f"invalid arm: {arm!r}")
        if regime not in REGIMES or re.fullmatch(safe_name, out_prefix) is None:
            raise ValueError(f"invalid regime or output prefix in {job_id}")

        c_scale = _positive(raw.get("C"), "C")
        c_scale = analysis._match_grid(c_scale, c_grid, "job C")
        metric_mode = analysis._metric_mode(raw.get("metric_mode"))
        boxsize = _positive(raw.get("boxsize"), "boxsize")
        sb_radius = int(raw.get("sb_radius", 0))
        cover_default_c = _positive(
            raw.get("cover_default_C", boxsize * sb_radius),
            "cover-default C",
        )
        rho_max = _positive(raw.get("rho_max"), "rho_max")
        if sb_radius <= 0 or not _close(cover_default_c, boxsize * sb_radius):
            raise ValueError(f"invalid comparison-cover scale in {job_id}")
        if metric_mode == "cover_default" and not _close(
            c_scale, cover_default_c
        ):
            raise ValueError(f"cover-default C != boxsize*sb_radius in {job_id}")
        c_token = re.escape(analysis._format_number(c_scale))
        if re.search(rf"(?:^|_)C_{c_token}(?:_|$)", out_prefix) is None:
            raise ValueError(f"output prefix lacks its C suffix: {job_id}")

        data_dir = _resolved(Path(raw.get("data_dir", "")))
        positions = _resolved(
            Path(
                raw.get(
                    "positions",
                    data_dir / f"compass_{regime}_positions.csv",
                )
            )
        )
        tangents = _resolved(
            Path(
                raw.get(
                    "tangents",
                    data_dir / f"compass_{regime}_tangents.csv",
                )
            )
        )
        manifest = _resolved(Path(raw.get("manifest", "")))
        out_dir = _require_under(Path(raw.get("out_dir", "")), plan_root, "out-dir")
        if data_dir != data_dir_expected:
            raise ValueError(f"unexpected data directory in {job_id}")
        expected_positions = data_dir / f"compass_{regime}_positions.csv"
        default_tangents = data_dir / f"compass_{regime}_tangents.csv"
        if positions != expected_positions or positions.is_symlink() or not positions.is_file():
            raise ValueError(f"unexpected or missing positions in {job_id}")
        if tangents.is_symlink() or not tangents.is_file():
            raise ValueError(f"missing real tangents in {job_id}")
        if tangents != default_tangents and not tangents.is_relative_to(safe_root):
            raise ValueError(f"override tangents lie outside planned outputs in {job_id}")
        positions_sha256 = raw.get("positions_sha256")
        tangents_sha256 = raw.get("tangents_sha256")
        manifest_sha256 = raw.get("manifest_sha256")
        raw_provenance = raw.get("tangent_provenance")
        tangent_provenance = (
            None
            if raw_provenance is None
            else _require_under(
                Path(str(raw_provenance)), safe_root, "tangent provenance"
            )
        )
        tangent_provenance_sha256 = raw.get("tangent_provenance_sha256")
        tangent_provenance_regime = raw.get("tangent_provenance_regime")
        if tangent_provenance_regime is not None and not isinstance(
            tangent_provenance_regime, dict
        ):
            raise ValueError(f"malformed tangent provenance in {job_id}")
        if tangent_provenance is not None:
            if tangent_provenance.is_symlink() or not tangent_provenance.is_file():
                raise FileNotFoundError(
                    f"missing real tangent provenance in {job_id}"
                )
            if str(tangent_provenance_sha256) != analysis._sha256(
                tangent_provenance
            ):
                raise ValueError(f"tangent provenance hash mismatch in {job_id}")
        for label, expected_hash, path in (
            ("positions", positions_sha256, positions),
            ("tangents", tangents_sha256, tangents),
        ):
            if expected_hash is not None and str(expected_hash) != analysis._sha256(path):
                raise ValueError(f"{label} hash mismatch in {job_id}")
        expected_manifest = plan_root / "manifests" / f"{regime}.csv"
        if manifest != expected_manifest.resolve() or not manifest.is_file():
            raise ValueError(f"unexpected or missing manifest in {job_id}")
        if manifest_sha256 is not None and str(manifest_sha256) != analysis._sha256(manifest):
            raise ValueError(f"manifest hash mismatch in {job_id}")
        if out_dir in seen_outputs:
            raise ValueError(f"duplicate output directory: {out_dir}")

        experiment_entries = plan.get("experiment_arms")
        if experiment_entries is not None:
            planned_arm = analysis._experiment_arm(plan, arm, c_scale)
            if planned_arm is None:
                raise ValueError(f"unplanned experiment arm in {job_id}: {arm}")
            expected_tangents = analysis._details_tangents_path(
                planned_arm, regime
            )
            if (
                not _close(boxsize, float(planned_arm["boxsize"]))
                or sb_radius != int(planned_arm["sb_radius"])
                or metric_mode
                != analysis._metric_mode(planned_arm.get("metric_mode"))
                or tangents != expected_tangents
            ):
                raise ValueError(f"diagnostic arm mismatch in {job_id}")
            if str(planned_arm.get("tangent_source", "default")) == "override":
                planned_provenance = planned_arm.get("tangent_provenance")
                if not isinstance(planned_provenance, dict):
                    raise ValueError(f"override arm lacks provenance in {job_id}")
                expected_regime_provenance = planned_provenance["regimes"][regime]
                if (
                    tangent_provenance
                    != Path(str(planned_provenance["path"])).resolve()
                    or tangent_provenance_sha256 != planned_provenance["sha256"]
                    or tangent_provenance_regime != expected_regime_provenance
                ):
                    raise ValueError(f"job provenance differs from arm in {job_id}")
            elif tangent_provenance is not None:
                raise ValueError(f"default-tangent arm names provenance in {job_id}")
            if not _close(rho_max, rho_grid[-1]):
                raise ValueError(f"diagnostic rho_max mismatch in {job_id}")
        else:
            _, primary_boxsize, primary_sb = analysis._primary_construction(
                plan, c_scale
            )
            if arm == "primary":
                details = analysis._primary_details(plan, c_scale)
                if (
                    not _close(boxsize, primary_boxsize)
                    or sb_radius != primary_sb
                    or metric_mode != analysis._metric_mode(details.get("metric_mode"))
                ):
                    raise ValueError(f"primary construction mismatch in {job_id}")
                if not _close(rho_max, rho_grid[-1]):
                    raise ValueError(f"primary rho_max mismatch in {job_id}")
            else:
                if selection is None:
                    raise ValueError("control job document lacks its selection")
                selected = selection["selected_cell"]
                if not _close(c_scale, float(selected["c_scale"])):
                    raise ValueError(f"control C differs from selection in {job_id}")
                if not _close(rho_max, float(selected["rho"])):
                    raise ValueError(f"control rho differs from selection in {job_id}")
                controls = {
                    label: (candidate_boxsize, candidate_sb)
                    for label, candidate_boxsize, candidate_sb
                    in analysis._control_constructions(plan, c_scale)
                }
                if arm not in controls:
                    raise ValueError(f"unplanned control arm in {job_id}: {arm}")
                expected_boxsize, expected_sb = controls[arm]
                if not _close(boxsize, expected_boxsize) or sb_radius != expected_sb:
                    raise ValueError(f"control construction mismatch in {job_id}")

        jobs.append(
            Job(
                job_id,
                arm,
                regime,
                c_scale,
                metric_mode,
                cover_default_c,
                boxsize,
                sb_radius,
                rho_max,
                data_dir,
                positions,
                tangents,
                None if positions_sha256 is None else str(positions_sha256),
                None if tangents_sha256 is None else str(tangents_sha256),
                None if manifest_sha256 is None else str(manifest_sha256),
                tangent_provenance,
                (
                    None
                    if tangent_provenance_sha256 is None
                    else str(tangent_provenance_sha256)
                ),
                (
                    None
                    if tangent_provenance_regime is None
                    else dict(tangent_provenance_regime)
                ),
                manifest,
                out_dir,
                out_prefix,
            )
        )
        seen_ids.add(job_id)
        seen_outputs.add(out_dir)

    context = Context(
        code_root, safe_root, plan_root, plan, durations, manifests
    )
    _validate_tangent_provenances(context)
    _validate_external_comparators(context)
    return jobs, context


def _validate_tangent_provenances(context: Context) -> None:
    arms = context.plan.get("experiment_arms")
    if arms is None:
        return
    manifest_hashes = {
        regime: analysis._sha256(
            context.plan_root / "manifests" / f"{regime}.csv"
        )
        for regime in REGIMES
    }
    validated_directories: dict[Path, dict[str, object]] = {}
    for arm in arms:
        if str(arm.get("tangent_source", "default")) != "override":
            if arm.get("tangent_provenance") is not None:
                raise ValueError("default-tangent arm unexpectedly names provenance")
            continue
        directory = Path(str(arm.get("tangents_dir", ""))).resolve()
        current = validated_directories.get(directory)
        if current is None:
            current = planner.validate_tangent_provenance(
                context.code_root,
                directory,
                manifest_hashes,
            )
            validated_directories[directory] = current
        if arm.get("tangent_provenance") != current:
            raise ValueError(
                f"tangent provenance changed for arm {arm.get('label')!r}"
            )


def _validate_external_comparators(context: Context) -> None:
    comparators = context.plan.get("external_comparators", [])
    if not isinstance(comparators, list):
        raise ValueError("external comparators must be a list")
    seen_labels: set[str] = set()
    for comparator in comparators:
        if not isinstance(comparator, dict):
            raise ValueError("external comparator must be an object")
        label = str(comparator.get("label", ""))
        if label in seen_labels or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", label
        ) is None:
            raise ValueError(f"invalid external comparator label: {label!r}")
        seen_labels.add(label)
        results = comparator.get("results")
        if not isinstance(results, dict) or set(results) != set(REGIMES):
            raise ValueError(f"external comparator {label} lacks five regimes")
        for regime, entry in results.items():
            if not isinstance(entry, dict):
                raise ValueError(f"malformed external comparator result: {regime}")
            paths = {
                key: _require_under(
                    Path(str(entry.get(key, ""))), context.safe_root, key
                )
                for key in ("metadata", "births", "manifest")
            }
            lift_root = (
                context.code_root
                / "period_doubling"
                / "data_fine"
                / "compass_gait_latent"
            ).resolve()
            for key in ("positions", "tangents"):
                path = _resolved(Path(str(entry.get(key, ""))))
                if not (
                    path.is_relative_to(lift_root)
                    or path.is_relative_to(context.safe_root)
                ):
                    raise ValueError(f"external {key} lies outside allowed inputs")
                paths[key] = path
            for key, path in paths.items():
                if path.is_symlink() or not path.is_file():
                    raise FileNotFoundError(f"missing real external {key}: {path}")
                expected = str(entry.get(f"{key}_sha256", ""))
                if expected != analysis._sha256(path):
                    raise ValueError(f"external comparator {key} hash changed: {path}")
            record = analysis._load_result_record(
                paths["metadata"], context.manifests, context.durations
            )
            if (
                record.regime != regime
                or not _close(record.c_scale, float(comparator["C"]))
                or not _close(record.boxsize, float(comparator["boxsize"]))
                or record.sb_radius != int(comparator["sb_radius"])
            ):
                raise ValueError(f"external comparator metadata mismatch: {regime}")
            metadata = analysis._parse_metadata(paths["metadata"])
            if metadata.get("split") != "all":
                raise ValueError(f"external comparator is not split=all: {regime}")


def _curve_bounds(path: Path) -> dict[tuple[str, ...], float]:
    rows: dict[tuple[str, ...], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            key = tuple(
                raw[name]
                for name in (
                    "target_duration",
                    "split",
                    "run_index",
                    "start_index",
                    "end_index",
                )
            )
            if key in rows:
                raise ValueError(f"duplicate curve-bound row: {path}, {key}")
            rows[key] = float(raw["curve_bound"])
    return rows


def _invariant_path(
    name: str,
    regime: str,
    jobs_by_arm: dict[tuple[str, str], Job],
    context: Context,
) -> Path | None:
    job = jobs_by_arm.get((name, regime))
    if job is not None:
        state, detail = _state(job, context)
        if state == "complete":
            return job.births_path
        if state == "invalid-final":
            raise ValueError(
                f"cannot check invariant against invalid {job.job_id}: {detail}"
            )
        return None
    comparators = {
        str(item["label"]): item
        for item in context.plan.get("external_comparators", [])
    }
    comparator = comparators.get(name)
    if comparator is None:
        raise ValueError(f"curve-bound invariant names unknown arm {name!r}")
    return Path(str(comparator["results"][regime]["births"])).resolve()


def _check_curve_bound_invariants(
    jobs: list[Job],
    context: Context,
) -> tuple[bool, list[str]]:
    invariants = context.plan.get("curve_bound_invariants", [])
    if not isinstance(invariants, list):
        raise ValueError("curve-bound invariants must be a list")
    jobs_by_arm = {(job.arm, job.regime): job for job in jobs}
    messages: list[str] = []
    all_pass = True
    for invariant in invariants:
        left = str(invariant.get("left", ""))
        right = str(invariant.get("right", ""))
        pending = False
        for regime in REGIMES:
            left_path = _invariant_path(left, regime, jobs_by_arm, context)
            right_path = _invariant_path(right, regime, jobs_by_arm, context)
            if left_path is None or right_path is None:
                pending = True
                continue
            left_rows = _curve_bounds(left_path)
            right_rows = _curve_bounds(right_path)
            if left_rows.keys() != right_rows.keys() or any(
                left_rows[key] != right_rows[key] for key in left_rows
            ):
                all_pass = False
                messages.append(
                    f"failed curve_bound invariant {left}:{right} for {regime}"
                )
        if pending:
            messages.append(f"pending curve_bound invariant {left}:{right}")
        elif all(
            not message.startswith(f"failed curve_bound invariant {left}:{right}")
            for message in messages
        ):
            messages.append(f"passed curve_bound invariant {left}:{right}")
    return all_pass, messages


def _validate_job_inputs(job: Job, context: Context) -> None:
    for label, expected_hash, path in (
        ("positions", job.positions_sha256, job.positions),
        ("tangents", job.tangents_sha256, job.tangents),
        ("manifest", job.manifest_sha256, job.manifest),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing real {label} input for {job.job_id}")
        if expected_hash is not None and analysis._sha256(path) != expected_hash:
            raise ValueError(f"{label} input hash changed for {job.job_id}")
    if job.tangent_provenance is None:
        return
    if (
        job.tangent_provenance.is_symlink()
        or not job.tangent_provenance.is_file()
        or analysis._sha256(job.tangent_provenance)
        != job.tangent_provenance_sha256
    ):
        raise ValueError(f"tangent provenance changed for {job.job_id}")
    record = job.tangent_provenance_regime
    if record is None:
        raise ValueError(f"missing regime provenance for {job.job_id}")
    if (
        Path(str(record["tangents"])).resolve() != job.tangents
        or record["tangents_sha256"] != job.tangents_sha256
        or Path(str(record["positions"])).resolve() != job.positions
        or record["positions_sha256"] != job.positions_sha256
        or record["manifest_sha256"] != job.manifest_sha256
    ):
        raise ValueError(f"job inputs differ from provenance for {job.job_id}")
    planned_arm = analysis._experiment_arm(context.plan, job.arm, job.c_scale)
    if planned_arm is None:
        raise ValueError(f"missing planned provenance arm for {job.job_id}")
    provenance = planned_arm.get("tangent_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"missing planned provenance document for {job.job_id}")
    exporter_script = Path(str(provenance["exporter_script"])).resolve()
    if (
        exporter_script.is_symlink()
        or not exporter_script.is_file()
        or analysis._sha256(exporter_script)
        != provenance["exporter_script_sha256"]
    ):
        raise ValueError(f"tangent exporter changed for {job.job_id}")
    code_repo_head = subprocess.run(
        ["git", "-C", str(context.code_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if code_repo_head != provenance["code_repo_head"]:
        raise ValueError(f"code HEAD changed for {job.job_id}")
    for label, path_key, hash_key in (
        ("source archive", "source_archive", "source_archive_sha256"),
        (
            "source encoder-JVP tangents",
            "source_encoder_jvp_tangents",
            "source_encoder_jvp_tangents_sha256",
        ),
        ("checkpoint", "checkpoint", "checkpoint_sha256"),
        ("config", "config", "config_sha256"),
    ):
        path = Path(str(record[path_key])).resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing {label} for {job.job_id}")
        if analysis._sha256(path) != record[hash_key]:
            raise ValueError(f"{label} hash changed for {job.job_id}")


def _validate_output(job: Job, directory: Path, context: Context) -> None:
    _validate_job_inputs(job, context)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"output is not a real directory: {directory}")
    births_path = directory / f"{job.out_prefix}_births.csv"
    metadata_path = directory / f"{job.out_prefix}_metadata.txt"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (births_path, metadata_path)
    ):
        raise ValueError("complete births/metadata pair is absent")
    record = analysis._load_result_record(
        metadata_path, context.manifests, context.durations
    )
    if (
        record.regime != job.regime
        or not _close(record.c_scale, job.c_scale)
        or record.metric_mode != job.metric_mode
        or not _close(record.cover_default_c, job.cover_default_c)
        or not _close(record.boxsize, job.boxsize)
        or record.sb_radius != job.sb_radius
        or not _close(record.rho_max, job.rho_max)
    ):
        raise ValueError("validated result does not match its job")
    metadata = analysis._parse_metadata(metadata_path)
    if metadata.get("split") != "all":
        raise ValueError("prepared job must contain both tune and validate rows")
    if _resolved(Path(metadata.get("manifest", ""))) != job.manifest:
        raise ValueError("result names a different manifest")
    if job.tangent_provenance is None:
        if (
            record.tangent_provenance_path is not None
            or "tangent_provenance" in metadata
            or "tangent_provenance_sha256" in metadata
        ):
            raise ValueError("default-tangent result unexpectedly names provenance")
    elif (
        record.tangent_provenance_path != job.tangent_provenance
        or record.tangent_provenance_sha256
        != job.tangent_provenance_sha256
        or metadata.get("tangent_provenance") != str(job.tangent_provenance)
        or metadata.get("tangent_provenance_sha256")
        != job.tangent_provenance_sha256
    ):
        raise ValueError("result tangent provenance differs from its job")
    if (
        record.positions_path != job.positions
        or record.tangents_path != job.tangents
    ):
        raise ValueError("result names different lift inputs")
    for key, expected_hash, path in (
        ("positions_sha256", job.positions_sha256, job.positions),
        ("tangents_sha256", job.tangents_sha256, job.tangents),
        ("manifest_sha256", job.manifest_sha256, job.manifest),
    ):
        recorded_hash = metadata.get(key)
        actual_hash = analysis._sha256(path)
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError(f"planned {key} no longer matches its input")
        if job.metric_mode == "explicit" and recorded_hash != actual_hash:
            raise ValueError(f"explicit-metric result lacks exact {key} provenance")


def _state(job: Job, context: Context) -> tuple[str, str]:
    if job.out_dir.exists() or job.out_dir.is_symlink():
        try:
            _validate_output(job, job.out_dir, context)
        except Exception as error:
            return "invalid-final", str(error)
        return "complete", "validated output pair"
    if job.staging_dir.exists() or job.staging_dir.is_symlink():
        try:
            _validate_output(job, job.staging_dir, context)
        except Exception as error:
            return "incomplete-stage", str(error)
        return "ready-to-promote", "validated staged output pair"
    return "pending", "no output"


def _command(job: Job, out_dir: Path, context: Context) -> list[str]:
    command = [
        "julia",
        f"--project={context.code_root / 'period_doubling' / 'julia'}",
        str(context.code_root / "experiments_planned" / "run_duration_c_radius.jl"),
        "--data-dir", str(job.data_dir),
        "--base", f"compass_{job.regime}",
        "--manifest", str(job.manifest),
        "--boxsize", f"{job.boxsize:.12g}",
        "--sb-radius", str(job.sb_radius),
    ]
    if job.metric_mode == "explicit":
        command.extend(["--metric-c", f"{job.c_scale:.12g}"])
    default_tangents = job.data_dir / f"compass_{job.regime}_tangents.csv"
    if job.tangents != default_tangents:
        command.extend(["--tangents", str(job.tangents)])
    if job.tangent_provenance is not None:
        command.extend(
            ["--tangent-provenance", str(job.tangent_provenance)]
        )
    command.extend(
        [
            "--rho-max", f"{job.rho_max:.12g}",
            "--out-dir", str(out_dir),
            "--out-prefix", job.out_prefix,
        ]
    )
    return command


def _quarantine(stage: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = stage.with_name(f"{stage.name}.failed-{stamp}-{os.getpid()}")
    if target.exists():
        raise FileExistsError(f"quarantine target already exists: {target}")
    stage.rename(target)
    return target


def _terminate_process(process: subprocess.Popen, timeout: float = 10.0) -> None:
    """Terminate one private Julia process group, escalating if necessary."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def _run_job(
    job: Job,
    context: Context,
    julia_threads: int,
    cancel_event: threading.Event,
    active_processes: dict[str, subprocess.Popen],
    process_lock: threading.Lock,
) -> tuple[str, str]:
    if cancel_event.is_set():
        return job.job_id, "cancelled before launch"
    _validate_job_inputs(job, context)
    job.out_dir.parent.mkdir(parents=True, exist_ok=True)
    job.staging_dir.mkdir(exist_ok=False)
    command = _command(job, job.staging_dir, context)
    environment = os.environ.copy()
    environment["JULIA_NUM_THREADS"] = str(julia_threads)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    log_path = job.staging_dir / "job.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        if cancel_event.is_set():
            return job.job_id, "cancelled before launch; empty stage retained"
        process = subprocess.Popen(
            command,
            cwd=context.code_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with process_lock:
            active_processes[job.job_id] = process
        if cancel_event.is_set():
            _terminate_process(process)
        try:
            returncode = process.wait()
        finally:
            with process_lock:
                if active_processes.get(job.job_id) is process:
                    del active_processes[job.job_id]
    if cancel_event.is_set():
        return job.job_id, f"cancelled; incomplete stage retained at {job.staging_dir}"
    if returncode != 0:
        return job.job_id, f"failed with exit {returncode}; see {log_path}"
    try:
        _validate_job_inputs(job, context)
        _validate_output(job, job.staging_dir, context)
    except Exception as error:
        return job.job_id, f"failed output validation: {error}; see {log_path}"
    with process_lock:
        if cancel_event.is_set():
            return (
                job.job_id,
                f"cancelled; validated stage retained at {job.staging_dir}",
            )
        if job.out_dir.exists() or job.out_dir.is_symlink():
            return (
                job.job_id,
                f"failed: final output appeared concurrently: {job.out_dir}",
            )
        job.staging_dir.rename(job.out_dir)
    return job.job_id, "completed and atomically promoted"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and optionally execute structured fine-Compass jobs. "
            "Without --execute this is a read-only dry run."
        )
    )
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument(
        "--arm", action="append",
        help="Limit to one arm; repeat for multiple arms.",
    )
    parser.add_argument(
        "--regime", action="append", choices=REGIMES,
        help="Limit to one regime; repeat for multiple regimes.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help="Bound simultaneous Julia processes (default: 5; maximum: 8).",
    )
    parser.add_argument(
        "--julia-threads", type=int, default=1,
        help="Threads inside each Julia process (default: 1).",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Launch pending jobs; otherwise only validate and print the plan.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("--max-workers must lie in [1,8]")
    if not 1 <= args.julia_threads <= 8:
        raise ValueError("--julia-threads must lie in [1,8]")
    jobs, context = _load_jobs(args.jobs_file)
    all_jobs = list(jobs)
    known_arms = {job.arm for job in jobs}
    selected_arms = set(args.arm or known_arms)
    unknown_arms = selected_arms - known_arms
    if unknown_arms:
        raise ValueError(f"unknown arm(s): {', '.join(sorted(unknown_arms))}")
    selected_regimes = set(args.regime or REGIMES)
    jobs = [
        job
        for job in jobs
        if job.arm in selected_arms and job.regime in selected_regimes
    ]
    if not jobs:
        raise ValueError("filters selected no jobs")

    states = {job.job_id: _state(job, context) for job in jobs}
    for job in jobs:
        state, detail = states[job.job_id]
        print(f"{state:>16}  {job.job_id}  {detail}")
    counts = {
        state: sum(candidate_state == state for candidate_state, _ in states.values())
        for state in ("complete", "ready-to-promote", "incomplete-stage",
                      "invalid-final", "pending")
    }
    print("summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    if counts["invalid-final"]:
        print("refusing to touch invalid final outputs", flush=True)
        return 2
    invariants_pass, invariant_messages = _check_curve_bound_invariants(
        all_jobs, context
    )
    for message in invariant_messages:
        print(message)
    if not invariants_pass:
        print("curve-bound invariant validation failed", flush=True)
        return 2
    if not args.execute:
        print("dry run only; no Julia process was launched")
        return 0

    jobs_path = _resolved(args.jobs_file)
    lock_path = jobs_path.with_name(f".{jobs_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another manager is executing this jobs file")

        pending: list[Job] = []
        failures = 0
        for job in jobs:
            state, _ = states[job.job_id]
            if state == "complete":
                continue
            if state == "ready-to-promote":
                if job.out_dir.exists() or job.out_dir.is_symlink():
                    print(
                        f"failed promotion  {job.job_id}  final output appeared: "
                        f"{job.out_dir}"
                    )
                    failures += 1
                    continue
                job.staging_dir.rename(job.out_dir)
                print(f"promoted          {job.job_id}")
                continue
            if state == "incomplete-stage":
                quarantined = _quarantine(job.staging_dir)
                print(f"quarantined       {job.job_id}  {quarantined}")
            pending.append(job)

        cancel_event = threading.Event()
        active_processes: dict[str, subprocess.Popen] = {}
        process_lock = threading.Lock()
        pool = ThreadPoolExecutor(max_workers=args.max_workers)
        interrupted = False
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        futures = {}
        try:
            futures = {
                pool.submit(
                    _run_job,
                    job,
                    context,
                    args.julia_threads,
                    cancel_event,
                    active_processes,
                    process_lock,
                ): job
                for job in pending
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    job_id, message = future.result()
                except Exception as error:
                    job_id = job.job_id
                    message = f"failed before completion: {error}"
                print(f"result             {job_id}  {message}", flush=True)
                failures += message.startswith("failed")
        except KeyboardInterrupt:
            interrupted = True
            with process_lock:
                cancel_event.set()
                processes = list(active_processes.values())
            for future in futures:
                future.cancel()
            for process in processes:
                _terminate_process(process)
            print(
                "execution interrupted; active Julia processes terminated and "
                "incomplete stages retained for quarantine on the next run",
                flush=True,
            )
        finally:
            try:
                pool.shutdown(wait=True, cancel_futures=interrupted)
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
        if interrupted:
            return 130
        _validate_tangent_provenances(context)
        invariants_pass, invariant_messages = _check_curve_bound_invariants(
            all_jobs, context
        )
        for message in invariant_messages:
            print(message)
        if not invariants_pass:
            failures += 1
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
