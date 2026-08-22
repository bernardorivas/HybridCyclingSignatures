#!/usr/bin/env python3
"""Render provenance-bound 2 x 5 Compass v3 full- and low-radius figures.

The renderer consumes only the immutable plan, validated existing results,
the refined orbit bundle, and ``compact_summary_v2``.  It does not rerun a
signature and does not copy either figure into the manuscript.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
HERE = SCRIPT_PATH.parent
WORKSPACE_ROOT = SCRIPT_PATH.parents[3]
CODE_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(HERE))

import compass_probability_v3 as driver  # noqa: E402
import summarize_compass_probability_v3_science_v2 as science  # noqa: E402


EXPECTED_ANALYSIS_ID = "compass_refined_v3_probability_linf_C0p75"
EXPECTED_PLAN_SHA256 = science.EXPECTED_PLAN_SHA256
EXPECTED_BUNDLE_SHA256 = science.EXPECTED_BUNDLE_SHA256
EXPECTED_SUMMARY_MANIFEST_SHA256 = (
    "5d88709ca664484d8e2a08e5d435c7be36f557bacce421f53e3fe4eba22fc07f"
)
EXPECTED_CASE_IDS = science.EXPECTED_CASE_IDS
EXPECTED_FIGURE_FILENAME = "compassgait_C0p75.pdf"
EXPECTED_PREVIEW_FILENAME = "compassgait_C0p75.png"
EXPECTED_LOWR_FIGURE_FILENAME = "compassgait_C0p75_lowr.pdf"
EXPECTED_LOWR_PREVIEW_FILENAME = "compassgait_C0p75_lowr.png"
SUMMARY_DIRNAME = science.SUMMARY_DIRNAME
RENDERER_ID = "compass_refined_v3_probability_figure_renderer_v1"
FIGURE_WIDTH_IN = 9.0
FIGURE_HEIGHT_IN = 4.86
POINTS_PER_INCH = 72.0
MINIMUM_FONT_SIZE_PT = 7.0
PREVIEW_DPI = 220
FULL_Y_LIMIT = (0.0, 0.5)
LOWR_Y_LIMIT = (0.0, 0.02)
X_LIMIT = (0.2, 12.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def reject_output_target(path: Path) -> None:
    probe = path.absolute()
    while probe != probe.parent:
        if probe.is_symlink():
            raise ValueError(f"output path contains a symlink: {probe}")
        probe = probe.parent
    if path.exists() and not path.is_file():
        raise ValueError(f"output target is not a regular file: {path}")


def verify_summary(root: Path, plan: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    summary_root = root / SUMMARY_DIRNAME
    if summary_root.is_symlink() or not summary_root.is_dir():
        raise ValueError("missing validated compact_summary_v2")
    manifest_path = summary_root / "summary_manifest.json"
    hash_path = summary_root / "summary_manifest.sha256"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("missing summary manifest")
    if hash_path.is_symlink() or not hash_path.is_file():
        raise ValueError("missing summary hash sidecar")
    manifest_hash = sha256(manifest_path)
    if "__BIND_" in EXPECTED_SUMMARY_MANIFEST_SHA256:
        raise RuntimeError("renderer is not bound to the final summary")
    if manifest_hash != EXPECTED_SUMMARY_MANIFEST_SHA256:
        raise ValueError("renderer is bound to another summary manifest")
    if hash_path.read_text(encoding="ascii") != (
        f"{manifest_hash}  summary_manifest.json\n"
    ):
        raise ValueError("summary hash sidecar disagrees with manifest")
    manifest = load_json(manifest_path)
    if (
        manifest.get("summary_id") != science.SUMMARY_ID
        or manifest.get("status") != "complete"
        or manifest.get("analysis_id") != EXPECTED_ANALYSIS_ID
        or manifest.get("plan", {}).get("sha256") != EXPECTED_PLAN_SHA256
        or manifest.get("bundle_manifest", {}).get("sha256")
        != EXPECTED_BUNDLE_SHA256
        or manifest.get("no_signature_recomputation") is not True
    ):
        raise ValueError("summary scientific binding changed")
    if manifest.get("summarizer", {}).get("sha256") != sha256(science.SCRIPT_PATH):
        raise ValueError("summary postprocessor changed after summary creation")
    inventory = manifest.get("files")
    if not isinstance(inventory, list):
        raise ValueError("summary inventory is missing")
    expected_names = {"summary_manifest.json", "summary_manifest.sha256"}
    for record in inventory:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("malformed summary inventory record")
        name = str(record["path"])
        if Path(name).name != name:
            raise ValueError("nested summary inventory paths are forbidden")
        path = summary_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing summary file: {name}")
        if path.stat().st_size != int(record["bytes"]) or sha256(path) != record["sha256"]:
            raise ValueError(f"summary file hash/size changed: {name}")
        expected_names.add(name)
    actual_names = {
        path.name for path in summary_root.iterdir() if path.is_file() and not path.is_symlink()
    }
    if actual_names != expected_names:
        raise ValueError("summary directory contains missing or extra files")
    return summary_root, manifest


def verify_inputs(
    plan_path: Path, julia_bin: str
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    list[dict[str, Any]],
]:
    plan_path = plan_path.resolve()
    if sha256(plan_path) != EXPECTED_PLAN_SHA256:
        raise ValueError("renderer is bound to another immutable plan")
    root, plan = driver.load_plan(plan_path, julia_bin)
    if (
        plan.get("analysis_id") != EXPECTED_ANALYSIS_ID
        or plan.get("bundle_manifest_sha256") != EXPECTED_BUNDLE_SHA256
        or plan.get("figure_filename") != EXPECTED_FIGURE_FILENAME
        or plan_path != root / "plan.json"
        or root != science.DEFAULT_ROOT
    ):
        raise ValueError("plan/output binding changed")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or tuple(job.get("id") for job in jobs) != EXPECTED_CASE_IDS:
        raise ValueError("layout requires ordered period1/2/4/8/chaos jobs")
    summary_root, manifest = verify_summary(root, plan)
    bindings = [driver.validate_binding(job, plan) for job in jobs]
    starts_hashes = {
        binding["raw_results"]["starts"]["sha256"] for binding in bindings
    }
    if len(starts_hashes) != 1:
        raise ValueError("result bindings do not share paired starts")
    return root, plan, summary_root, manifest, bindings


def expected_grids(plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lengths = science.segment_lengths(plan["protocol"])
    durations = lengths.astype(float) * float(plan["protocol"]["effective_sample_dt"])
    radii = np.linspace(
        float(plan["protocol"]["r_min"]),
        float(plan["protocol"]["r_max"]),
        int(plan["protocol"]["r_subdivisions"]),
    )
    return durations, radii


def load_probability_csv(
    path: Path, expected_durations: np.ndarray, expected_radii: np.ndarray
) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing summary probability matrix: {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != len(expected_radii) + 1 or not rows:
        raise ValueError(f"probability row count changed: {path.name}")
    try:
        header_durations = np.asarray([float(value) for value in rows[0][1:]], dtype=float)
        values = np.asarray(
            [[float(value) for value in row] for row in rows[1:]], dtype=float
        )
    except ValueError as error:
        raise ValueError(f"nonnumeric probability matrix: {path.name}") from error
    if rows[0][0] != "radius" or not np.allclose(
        header_durations, expected_durations, rtol=0.0, atol=5e-14
    ):
        raise ValueError(f"probability duration grid changed: {path.name}")
    if values.shape != (len(expected_radii), len(expected_durations) + 1):
        raise ValueError(f"probability shape changed: {path.name}")
    if not np.allclose(values[:, 0], expected_radii, rtol=0.0, atol=1e-14):
        raise ValueError(f"probability radius grid changed: {path.name}")
    probability = values[:, 1:]
    if np.any(~np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError(f"probability values outside [0,1]: {path.name}")
    return probability


def load_probabilities(
    plan: dict[str, Any], summary_root: Path
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    durations, radii = expected_grids(plan)
    full: list[np.ndarray] = []
    lowr: list[np.ndarray] = []
    for job in plan["jobs"]:
        case_id = job["id"]
        full_matrix = load_probability_csv(
            summary_root / f"probability_pooled_{case_id}.csv", durations, radii
        )
        stored_durations, stored_radii, stored_probability = (
            driver.shared.read_probability_matrix(job, plan["protocol"])
        )
        if (
            not np.array_equal(stored_durations, durations)
            or not np.array_equal(stored_radii, radii)
            or not np.allclose(full_matrix, stored_probability, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"{case_id}: full summary matrix disagrees with raw result")
        full.append(full_matrix)

        lowr_matrix = load_probability_csv(
            summary_root / f"probability_lowr_pooled_{case_id}.csv",
            durations,
            science.LOW_R_RADII,
        )
        first_births, vectors = science.load_births(job, plan)
        reconstructed = science.probability_matrix(
            science.CaseData(
                job=job,
                binding={},
                durations=durations,
                radii=radii,
                first_births=first_births,
                birth_vectors=vectors,
            ),
            science.LOW_R_RADII,
            science.SPLITS["pooled"],
        )
        if not np.allclose(lowr_matrix, reconstructed, rtol=0.0, atol=1e-12):
            raise ValueError(f"{case_id}: low-r matrix is not exact birth thresholding")
        lowr.append(lowr_matrix)
    return durations, radii, full, science.LOW_R_RADII.copy(), lowr


def load_display(job: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    display = job.get("display_extract")
    if not isinstance(display, dict):
        raise ValueError(f"{job['id']}: missing display extract")
    path = Path(display["path"])
    if path.is_symlink() or not path.is_file() or sha256(path) != display["sha256"]:
        raise ValueError(f"{job['id']}: display extract changed")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    dimension = int(job["dimension"])
    header = rows[0] if rows else []
    coordinate_names = [f"z{index}" for index in range(dimension)]
    if not header or header[0] != "nominal_suspension_time":
        raise ValueError(f"{job['id']}: display clock column changed")
    try:
        coordinate_indices = [header.index(name) for name in coordinate_names]
    except ValueError as error:
        raise ValueError(f"{job['id']}: missing embedded display coordinates") from error
    if len(rows) - 1 != int(display["n_rows"]):
        raise ValueError(f"{job['id']}: display row count changed")
    try:
        time = np.asarray([float(row[0]) for row in rows[1:]], dtype=float)
        positions = np.asarray(
            [[float(row[index]) for index in coordinate_indices] for row in rows[1:]],
            dtype=float,
        )
    except ValueError as error:
        raise ValueError(f"{job['id']}: nonnumeric display extract") from error
    if (
        positions.shape != (int(display["n_rows"]), dimension)
        or len(positions) < 2
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(time))
        or np.any(np.diff(time) <= 0.0)
        or not math.isclose(float(time[0]), 0.0, abs_tol=1e-14)
    ):
        raise ValueError(f"{job['id']}: malformed display extract")
    duration = float(time[-1])
    closure: float | None
    if job["q"] is None:
        closure = None
        if not math.isclose(duration, 50.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("chaos display duration changed")
    else:
        if not math.isclose(
            duration,
            float(job["refined_continuous_period"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(f"{job['id']}: display no longer spans its refined period")
        closure = float(np.linalg.norm(positions[-1] - positions[0]))
        tolerance = max(1e-10, 1e-8 * float(np.max(np.ptp(positions, axis=0))))
        if closure > tolerance:
            raise ValueError(f"{job['id']}: display orbit is not closed")
    return positions[:, :3], {
        "case_id": job["id"],
        "phi_deg": float(job["phi_deg"]),
        "q": job["q"],
        "kind": display["kind"],
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "n_rows": len(positions),
        "display_duration": duration,
        "plotted_source_columns": ["z0", "z1", "z2"],
        "display_axis_labels": ["z1", "z2", "z3"],
        "full_dimension": dimension,
        "full_state_endpoint_closure_l2": closure,
    }


def centers_to_edges(values: np.ndarray, lower: float | None = None) -> np.ndarray:
    if values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("cell centers must be strictly increasing")
    middle = (values[:-1] + values[1:]) / 2.0
    left = values[0] - (values[1] - values[0]) / 2.0
    if lower is not None:
        left = lower
    right = values[-1] + (values[-1] - values[-2]) / 2.0
    return np.concatenate(([left], middle, [right]))


def style_embedded_axes(axis: Any, positions: np.ndarray) -> None:
    axis.set_xlabel(r"$z_1$", labelpad=-11.0)
    axis.set_ylabel(r"$z_2$", labelpad=-11.0)
    axis.set_zlabel(r"$z_3$", labelpad=-9.0)
    axis.tick_params(axis="x", pad=-3.0, width=0.45, length=2.0)
    axis.tick_params(axis="y", pad=-3.0, width=0.45, length=2.0)
    axis.tick_params(axis="z", pad=-3.0, width=0.45, length=2.0)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_major_locator(MaxNLocator(nbins=2))
        coordinate_axis.set_major_formatter(NullFormatter())
        coordinate_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        coordinate_axis.pane.set_edgecolor((0.58, 0.58, 0.58, 0.72))
        coordinate_axis._axinfo["grid"]["linewidth"] = 0.0
        coordinate_axis._axinfo["axisline"]["linewidth"] = 0.45
        coordinate_axis._axinfo["axisline"]["color"] = (0.35, 0.35, 0.35, 1.0)
    spans = np.maximum(np.ptp(positions, axis=0), 1e-12)
    axis.set_box_aspect(tuple(spans))
    axis.view_init(elev=24.0, azim=-56.0)


def parameter_title(phi_deg: float) -> str:
    return rf"$\phi = {phi_deg:.2f}^\circ$"


def render_figure(
    plan: dict[str, Any],
    durations: np.ndarray,
    radii: np.ndarray,
    probability: list[np.ndarray],
    *,
    low_radius: bool,
) -> tuple[bytes, list[dict[str, Any]]]:
    if len(probability) != len(plan["jobs"]):
        raise ValueError("one probability matrix is required per case")
    duration_edges = centers_to_edges(durations)
    radius_edges = centers_to_edges(radii, lower=0.0)
    y_limit = LOWR_Y_LIMIT if low_radius else FULL_Y_LIMIT
    y_ticks = (
        [0.0, 0.005, 0.010, 0.015, 0.020]
        if low_radius
        else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    rc = {
        "font.size": MINIMUM_FONT_SIZE_PT,
        "axes.titlesize": 8.25,
        "axes.labelsize": 7.25,
        "xtick.labelsize": MINIMUM_FONT_SIZE_PT,
        "ytick.labelsize": MINIMUM_FONT_SIZE_PT,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
    }
    displays: list[dict[str, Any]] = []
    with plt.rc_context(rc):
        figure = plt.figure(
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=False
        )
        grid = figure.add_gridspec(
            2,
            6,
            left=0.067,
            right=0.940,
            bottom=0.140,
            top=0.955,
            wspace=0.10,
            hspace=0.03,
            width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.055],
            height_ratios=[1.08, 1.0],
        )
        for column, (job, matrix) in enumerate(zip(plan["jobs"], probability)):
            top = figure.add_subplot(grid[0, column], projection="3d")
            positions, display = load_display(job)
            top.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color="#0072B2",
                linewidth=0.74,
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            style_embedded_axes(top, positions)
            top.set_title(parameter_title(float(job["phi_deg"])), pad=-0.5)

            bottom = figure.add_subplot(grid[1, column])
            bottom.pcolormesh(
                duration_edges,
                radius_edges,
                matrix,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                shading="flat",
                edgecolors="none",
                linewidth=0.0,
                antialiased=False,
                rasterized=False,
            )
            vertical_guide: float | None = None
            if job["q"] is not None:
                vertical_guide = float(job["refined_continuous_period"])
                bottom.axvline(
                    vertical_guide,
                    color="white",
                    linestyle=(0, (3.0, 2.2)),
                    linewidth=0.80,
                    dash_capstyle="butt",
                )
            bottom.set_xlim(*X_LIMIT)
            bottom.set_ylim(*y_limit)
            bottom.set_xticks([0.2, 4.0, 8.0, 12.0])
            # The five panels share an identical x scale.  Label the global
            # left/right endpoints only at the outside edges so adjacent
            # ``12`` and ``0.2`` strings cannot collide between panels.
            bottom.set_xticklabels(
                [
                    "0.2" if column == 0 else "",
                    "4",
                    "8",
                    "12" if column == len(plan["jobs"]) - 1 else "",
                ]
            )
            bottom.set_yticks(y_ticks)
            bottom.tick_params(axis="both", pad=1.7)
            if column:
                bottom.tick_params(labelleft=False)
            display["vertical_refined_continuous_period_guide"] = vertical_guide
            display["horizontal_radius_guide"] = None
            displays.append(display)

        figure.text(
            0.515,
            0.035,
            "segment duration (t)",
            ha="center",
            va="center",
            fontsize=8.0,
        )
        figure.text(
            0.018,
            0.310,
            "filtration radius (r)",
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.0,
        )
        color_axis = figure.add_subplot(grid[1, -1])
        colorbar = figure.colorbar(
            ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis"),
            cax=color_axis,
        )
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
        colorbar.set_ticks([0.0, 0.5, 1.0])
        colorbar.set_label("probability", fontsize=8.0, labelpad=1.5)
        colorbar.ax.tick_params(labelsize=MINIMUM_FONT_SIZE_PT, pad=1.7)

        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="pdf",
            metadata={
                "Title": (
                    f"{EXPECTED_ANALYSIS_ID}_low_radius"
                    if low_radius
                    else EXPECTED_ANALYSIS_ID
                ),
                "Creator": str(SCRIPT_PATH),
                "Subject": (
                    "Exact-birth low-radius diagnostic; below all curve bounds"
                    if low_radius
                    else "Computed-grid refined Compass cycling probabilities"
                ),
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)
    return buffer.getvalue(), displays


def inspect_pdf(path: Path) -> dict[str, Any]:
    required = {name: shutil.which(name) for name in ("pdfinfo", "pdfimages", "pdftotext")}
    if any(value is None for value in required.values()):
        raise ValueError("Poppler pdfinfo, pdfimages, and pdftotext are required")
    info = subprocess.run(
        [str(required["pdfinfo"]), str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    page_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", info, re.MULTILINE
    )
    pages_match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if page_match is None or pages_match is None or int(pages_match.group(1)) != 1:
        raise ValueError("figure PDF must contain exactly one parseable page")
    width_pt, height_pt = (float(value) for value in page_match.groups())
    if not math.isclose(width_pt, FIGURE_WIDTH_IN * POINTS_PER_INCH, abs_tol=0.01) or not math.isclose(
        height_pt, FIGURE_HEIGHT_IN * POINTS_PER_INCH, abs_tol=0.01
    ):
        raise ValueError(f"unexpected PDF page size: {width_pt} x {height_pt}")
    image_listing = subprocess.run(
        [str(required["pdfimages"]), "-list", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    image_rows = [
        line for line in image_listing.splitlines() if re.match(r"^\s*\d+\s+\d+\s+", line)
    ]
    if image_rows:
        raise ValueError("figure PDF contains raster image objects")
    extracted = subprocess.run(
        [str(required["pdftotext"]), str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for label in ("segment duration (t)", "filtration radius (r)", "probability"):
        if label not in extracted:
            raise ValueError(f"PDF text audit did not find {label!r}")
    return {
        "pages": 1,
        "width_points": width_pt,
        "height_points": height_pt,
        "width_inches": width_pt / POINTS_PER_INCH,
        "height_inches": height_pt / POINTS_PER_INCH,
        "raster_image_objects": 0,
        "shared_axis_labels_text_audited": True,
    }


def render_preview(pdf_path: Path, output_dir: Path, name: str) -> Path:
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise ValueError("Poppler pdftoppm is required for visual QA")
    prefix = output_dir / name
    subprocess.run(
        [
            executable,
            "-png",
            "-singlefile",
            "-r",
            str(PREVIEW_DPI),
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    preview = prefix.with_suffix(".png")
    if not preview.is_file() or preview.stat().st_size == 0:
        raise ValueError("Poppler did not produce a preview")
    return preview


def prepare_destination_temp(target: Path, data: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.{RENDERER_ID}.",
        suffix=".tmp",
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def promote_transaction(artifacts: list[tuple[Path, bytes]], replace: bool) -> None:
    for target, _ in artifacts:
        reject_output_target(target)
        if target.exists() and not replace:
            raise FileExistsError(f"refusing to overwrite without --replace: {target}")
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path | None]] = []
    try:
        for target, data in artifacts:
            staged.append((target, prepare_destination_temp(target, data)))
        for target, _ in staged:
            backup: Path | None = None
            if target.exists():
                descriptor, name = tempfile.mkstemp(
                    dir=target.parent,
                    prefix=f".{target.name}.{RENDERER_ID}.backup.",
                )
                os.close(descriptor)
                backup = Path(name)
                backup.unlink()
                os.link(target, backup)
            backups.append((target, backup))
        promoted = 0
        try:
            for target, source in staged:
                os.replace(source, target)
                promoted += 1
        except BaseException:
            for target, backup in reversed(backups[:promoted]):
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            raise
    finally:
        for _, source in staged:
            source.unlink(missing_ok=True)
        for _, backup in backups:
            if backup is not None:
                backup.unlink(missing_ok=True)


def result_records(plan: dict[str, Any], bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job, binding in zip(plan["jobs"], bindings):
        records.append(
            {
                "case_id": job["id"],
                "result_binding": {
                    "path": str(driver.result_binding_path(job).resolve()),
                    "sha256": sha256(driver.result_binding_path(job)),
                },
                "raw_results": binding["raw_results"],
                "global_curve_bound_h": float(job["global_curve_bound"]),
                "refined_continuous_period": job["refined_continuous_period"],
            }
        )
    return records


def provenance_bytes(
    *,
    low_radius: bool,
    plan_path: Path,
    plan: dict[str, Any],
    summary_root: Path,
    summary_manifest: dict[str, Any],
    bindings: list[dict[str, Any]],
    displays: list[dict[str, Any]],
    geometry: dict[str, Any],
    pdf_path: Path,
    pdf_hash: str,
    png_path: Path,
    png_hash: str,
) -> bytes:
    grid = (
        summary_manifest["grids"]["low_radius_diagnostic"]
        if low_radius
        else summary_manifest["grids"]["computed"]
    )
    matrix_prefix = "probability_lowr_pooled_" if low_radius else "probability_pooled_"
    matrix_records = [
        {
            "case_id": job["id"],
            "path": str((summary_root / f"{matrix_prefix}{job['id']}.csv").resolve()),
            "sha256": sha256(summary_root / f"{matrix_prefix}{job['id']}.csv"),
        }
        for job in plan["jobs"]
    ]
    minimum_bound = min(float(job["global_curve_bound"]) for job in plan["jobs"])
    document = {
        "schema_version": 1,
        "renderer_id": RENDERER_ID,
        "scope": "display_only_no_signature_recomputation_no_paper_copy",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "view": "low_radius_exact_birth_diagnostic" if low_radius else "full_computed_grid",
        "analysis_id": EXPECTED_ANALYSIS_ID,
        "plan": {"path": str(plan_path.resolve()), "sha256": sha256(plan_path.resolve())},
        "bundle_manifest": {
            "path": plan["bundle_manifest"],
            "sha256": plan["bundle_manifest_sha256"],
        },
        "frozen_v3_driver": {
            "path": str(driver.SCRIPT_PATH),
            "sha256": sha256(driver.SCRIPT_PATH),
        },
        "summary_manifest": {
            "path": str((summary_root / "summary_manifest.json").resolve()),
            "sha256": sha256(summary_root / "summary_manifest.json"),
        },
        "display_renderer": {"path": str(SCRIPT_PATH), "sha256": sha256(SCRIPT_PATH)},
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
        },
        "statistic": "P(rank > 0) = fraction of 20 pooled trials with first birth <= r",
        "probability_processing": (
            "exact thresholding of stored births on a finer requested radius grid; "
            "no interpolation, smoothing, or signature rerun"
            if low_radius
            else "exact stored computed grid; no interpolation or smoothing"
        ),
        "grid": grid,
        "probability_matrices": matrix_records,
        "layout": (
            "2x5; phi-only titles; first-three embedded coordinates with z1/z2/z3 "
            "display axes; shared lower-axis labels"
        ),
        "axis_limits": {
            "segment_duration": list(X_LIMIT),
            "filtration_radius": list(LOWR_Y_LIMIT if low_radius else FULL_Y_LIMIT),
            "probability_color": [0.0, 1.0],
        },
        "horizontal_radius_guide": None,
        "periodic_vertical_guides": [
            float(job["refined_continuous_period"])
            for job in plan["jobs"]
            if job["q"] is not None
        ],
        "low_radius_scientific_scope": {
            "applies": low_radius,
            "minimum_global_curve_bound_h": minimum_bound,
            "maximum_displayed_radius": LOWR_Y_LIMIT[1] if low_radius else None,
            "entire_view_strictly_below_every_h": bool(
                low_radius and LOWR_Y_LIMIT[1] < minimum_bound
            ),
            "interpretation": (
                "entire diagnostic is empirical and below every certified curve-resolution bound"
                if low_radius
                else None
            ),
        },
        "displays": displays,
        "results": result_records(plan, bindings),
        "pdf_geometry": geometry,
        "preview_dpi": PREVIEW_DPI,
        "output_pdf": str(pdf_path.resolve()),
        "output_pdf_sha256": pdf_hash,
        "output_preview_png": str(png_path.resolve()),
        "output_preview_png_sha256": png_hash,
        "paper_copy": None,
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def run(plan_path: Path, julia_bin: str, replace: bool) -> tuple[Path, Path]:
    root, plan, summary_root, summary_manifest, bindings = verify_inputs(
        plan_path, julia_bin
    )
    durations, full_radii, full, lowr_radii, lowr = load_probabilities(
        plan, summary_root
    )
    full_pdf_bytes, full_displays = render_figure(
        plan, durations, full_radii, full, low_radius=False
    )
    lowr_pdf_bytes, lowr_displays = render_figure(
        plan, durations, lowr_radii, lowr, low_radius=True
    )

    pdf_path = root / EXPECTED_FIGURE_FILENAME
    png_path = root / EXPECTED_PREVIEW_FILENAME
    render_path = pdf_path.with_suffix(".render.json")
    hash_path = pdf_path.with_suffix(".sha256")
    lowr_pdf_path = root / EXPECTED_LOWR_FIGURE_FILENAME
    lowr_png_path = root / EXPECTED_LOWR_PREVIEW_FILENAME
    lowr_render_path = lowr_pdf_path.with_suffix(".render.json")
    lowr_hash_path = lowr_pdf_path.with_suffix(".sha256")

    temp_parent = WORKSPACE_ROOT / "tmp" / "pdfs"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent, prefix=f".{RENDERER_ID}.") as name:
        staging = Path(name)
        staged_full = staging / EXPECTED_FIGURE_FILENAME
        staged_full.write_bytes(full_pdf_bytes)
        full_geometry = inspect_pdf(staged_full)
        staged_full_png = render_preview(staged_full, staging, "full_preview")
        full_png_bytes = staged_full_png.read_bytes()

        staged_lowr = staging / EXPECTED_LOWR_FIGURE_FILENAME
        staged_lowr.write_bytes(lowr_pdf_bytes)
        lowr_geometry = inspect_pdf(staged_lowr)
        staged_lowr_png = render_preview(staged_lowr, staging, "lowr_preview")
        lowr_png_bytes = staged_lowr_png.read_bytes()

    full_pdf_hash = sha256_bytes(full_pdf_bytes)
    full_png_hash = sha256_bytes(full_png_bytes)
    lowr_pdf_hash = sha256_bytes(lowr_pdf_bytes)
    lowr_png_hash = sha256_bytes(lowr_png_bytes)
    full_provenance = provenance_bytes(
        low_radius=False,
        plan_path=plan_path,
        plan=plan,
        summary_root=summary_root,
        summary_manifest=summary_manifest,
        bindings=bindings,
        displays=full_displays,
        geometry=full_geometry,
        pdf_path=pdf_path,
        pdf_hash=full_pdf_hash,
        png_path=png_path,
        png_hash=full_png_hash,
    )
    lowr_provenance = provenance_bytes(
        low_radius=True,
        plan_path=plan_path,
        plan=plan,
        summary_root=summary_root,
        summary_manifest=summary_manifest,
        bindings=bindings,
        displays=lowr_displays,
        geometry=lowr_geometry,
        pdf_path=lowr_pdf_path,
        pdf_hash=lowr_pdf_hash,
        png_path=lowr_png_path,
        png_hash=lowr_png_hash,
    )
    full_hash_bytes = (
        f"{full_pdf_hash}  {pdf_path.name}\n"
        f"{full_png_hash}  {png_path.name}\n"
        f"{sha256_bytes(full_provenance)}  {render_path.name}\n"
    ).encode("ascii")
    lowr_hash_bytes = (
        f"{lowr_pdf_hash}  {lowr_pdf_path.name}\n"
        f"{lowr_png_hash}  {lowr_png_path.name}\n"
        f"{sha256_bytes(lowr_provenance)}  {lowr_render_path.name}\n"
    ).encode("ascii")
    promote_transaction(
        [
            (pdf_path, full_pdf_bytes),
            (png_path, full_png_bytes),
            (render_path, full_provenance),
            (hash_path, full_hash_bytes),
            (lowr_pdf_path, lowr_pdf_bytes),
            (lowr_png_path, lowr_png_bytes),
            (lowr_render_path, lowr_provenance),
            (lowr_hash_path, lowr_hash_bytes),
        ],
        replace,
    )
    for path, expected in (
        (pdf_path, full_pdf_hash),
        (png_path, full_png_hash),
        (render_path, sha256_bytes(full_provenance)),
        (lowr_pdf_path, lowr_pdf_hash),
        (lowr_png_path, lowr_png_hash),
        (lowr_render_path, sha256_bytes(lowr_provenance)),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"promoted artifact hash mismatch: {path}")
    print(f"output_pdf={pdf_path}")
    print(f"output_pdf_sha256={full_pdf_hash}")
    print(f"output_png={png_path}")
    print(f"output_png_sha256={full_png_hash}")
    print(f"lowr_pdf={lowr_pdf_path}")
    print(f"lowr_pdf_sha256={lowr_pdf_hash}")
    print(f"lowr_png={lowr_png_path}")
    print(f"lowr_png_sha256={lowr_png_hash}")
    print("paper_copy=none")
    return pdf_path, lowr_pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=science.DEFAULT_PLAN)
    parser.add_argument("--julia-bin", default="julia")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.plan, args.julia_bin, args.replace)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"ERROR: {error}") from error
