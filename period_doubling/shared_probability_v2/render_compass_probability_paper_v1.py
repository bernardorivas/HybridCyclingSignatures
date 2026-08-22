#!/usr/bin/env python3
"""Paper-scale renderer for the frozen Fourier-closed Compass v2 result.

The renderer is intentionally display-only.  It is bound to one immutable
analysis plan, asks the frozen Compass orchestrator to revalidate that plan and
each raw result, and then produces a vector 2 x 5 PDF plus a PNG preview.  It
does not recompute cycling signatures or modify any Rössler artifact.
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
import tempfile
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np

import compass_probability_v2 as frozen


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[3]

# These three bindings are filled only after the driver has materialized the
# final Fourier-closed plan.  Leaving a sentinel makes it impossible to render
# an earlier frozen-path draft accidentally.
EXPECTED_ANALYSIS_ID = "compass_fourier_embedded_probability_linf_v2_david_grid"
EXPECTED_PLAN_SHA256 = (
    "0a1d18864234ca8482b7587e74fb68db161f9fc0f373340b9058148c4368677e"
)
EXPECTED_BUNDLE_ID = "compass_embedded_fourier_orbits_v1"

EXPECTED_FIGURE_FILENAME = "compassgait_C5p0.pdf"
EXPECTED_PREVIEW_FILENAME = "compassgait_C5p0.png"
PAPER_COPY = (
    WORKSPACE_ROOT
    / "paper"
    / "hybrid_cyclingsignatures"
    / "figures"
    / "cycling_signatures"
    / EXPECTED_FIGURE_FILENAME
)

RENDERER_ID = "compass_fourier_closed_probability_paper_renderer_v1"
FIGURE_WIDTH_IN = 9.0
FIGURE_HEIGHT_IN = 4.86
POINTS_PER_INCH = 72.0
MINIMUM_FONT_SIZE_PT = 7.0
PREVIEW_DPI = 220
EXPECTED_CASE_IDS = ("period1", "period2", "period4", "period8", "chaos")
EXPECTED_GUIDES = (0.880, 1.760, 3.520, 7.045, None)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def reject_output_target(path: Path) -> None:
    """Reject symlinks and non-files without resolving an output through them."""
    probe = path.absolute()
    while probe != probe.parent:
        if probe.is_symlink():
            raise ValueError(f"output path contains a symlink: {probe}")
        probe = probe.parent
    if path.exists() and not path.is_file():
        raise ValueError(f"output target is not a regular file: {path}")


def verify_bound_plan(plan_path: Path, julia_bin: str) -> tuple[Path, dict[str, Any]]:
    plan_path = plan_path.resolve()
    if "__FINAL_" in EXPECTED_PLAN_SHA256:
        raise RuntimeError("renderer has not yet been bound to the final plan")
    if sha256(plan_path) != EXPECTED_PLAN_SHA256:
        raise ValueError("renderer is bound to another immutable Compass plan")
    root, plan = frozen.load_plan(plan_path, julia_bin)
    if plan.get("analysis_id") != EXPECTED_ANALYSIS_ID:
        raise ValueError("refusing an unrelated Compass analysis plan")
    if plan.get("bundle_id") != EXPECTED_BUNDLE_ID:
        raise ValueError("refusing a non-Fourier-closed Compass bundle")
    if plan.get("figure_filename") != EXPECTED_FIGURE_FILENAME:
        raise ValueError("refusing a plan with a different figure filename")
    if root.name != EXPECTED_ANALYSIS_ID or plan_path != root / "plan.json":
        raise ValueError("plan is outside its canonical analysis root")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or tuple(job.get("id") for job in jobs) != EXPECTED_CASE_IDS:
        raise ValueError("paper layout requires period1/2/4/8 and chaos in order")
    for job, expected_guide in zip(jobs, EXPECTED_GUIDES):
        guide = job.get("nominal_suspension_period")
        if expected_guide is None:
            if guide is not None or job.get("q") is not None:
                raise ValueError("chaos must have no nominal-period guide")
        elif job.get("q") is None or not math.isclose(
            float(guide), expected_guide, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{job.get('id')}: nominal-period guide changed")
    return root, plan


def binding_path(job: dict[str, Any]) -> Path:
    return Path(job["output_dir"]) / f"{job['id']}_v2_result.json"


def validate_result_binding(job: dict[str, Any], plan: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    expected = frozen.validate_raw_result(job, plan)
    path = binding_path(job)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{job['id']}: missing validated v2 result binding")
    recorded = load_json(path)
    comparison = dict(recorded)
    created = comparison.pop("created_utc", None)
    if not isinstance(created, str) or comparison != expected:
        raise ValueError(f"{job['id']}: result binding does not match raw results")
    return recorded, path


def load_probability(
    job: dict[str, Any], plan: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Path]:
    binding, binding_file = validate_result_binding(job, plan)
    rank0_path = frozen.result_paths(job)["rank0"]
    with rank0_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    lengths = np.asarray(frozen.segment_lengths(plan["protocol"]), dtype=int)
    if not rows or rows[0] != ["radius", *[str(value) for value in lengths]]:
        raise ValueError(f"{job['id']}: unexpected rank-zero heatmap header")
    if len(rows) != int(plan["protocol"]["r_subdivisions"]) + 1:
        raise ValueError(f"{job['id']}: unexpected rank-zero heatmap row count")
    try:
        values = np.asarray([[float(cell) for cell in row] for row in rows[1:]], dtype=float)
    except ValueError as error:
        raise ValueError(f"{job['id']}: nonnumeric rank-zero heatmap") from error
    if values.shape != (int(plan["protocol"]["r_subdivisions"]), len(lengths) + 1):
        raise ValueError(f"{job['id']}: rank-zero heatmap shape changed")
    radii = values[:, 0]
    expected_radii = np.linspace(
        float(plan["protocol"]["r_min"]),
        float(plan["protocol"]["r_max"]),
        int(plan["protocol"]["r_subdivisions"]),
    )
    if not np.allclose(radii, expected_radii, rtol=0.0, atol=1e-12):
        raise ValueError(f"{job['id']}: radius grid changed")
    counts = values[:, 1:]
    n_runs = int(plan["protocol"]["n_runs"])
    if (
        np.any(~np.isfinite(counts))
        or np.any(counts < 0)
        or np.any(counts > n_runs)
        or not np.array_equal(counts, np.rint(counts))
    ):
        raise ValueError(f"{job['id']}: invalid rank-zero counts")
    durations = lengths.astype(float) * float(plan["protocol"]["effective_sample_dt"])
    probability = 1.0 - counts / n_runs
    return durations, radii, probability, binding, binding_file


def load_display(job: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    display = job.get("display")
    if display is None:
        display = job.get("display_extract")
    if not isinstance(display, dict):
        raise ValueError(f"{job['id']}: missing display extract")
    path = Path(display["path"])
    if path.is_symlink() or not path.is_file() or sha256(path) != display["sha256"]:
        raise ValueError(f"{job['id']}: display extract changed")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    dimension = int(job["dimension"])
    header = rows[0] if rows else []
    coordinate_names = [f"z{index}" for index in range(dimension)]
    if not header or header[0] != "nominal_suspension_time":
        raise ValueError(f"{job['id']}: display extract clock column changed")
    try:
        coordinate_indices = [header.index(name) for name in coordinate_names]
    except ValueError as error:
        raise ValueError(f"{job['id']}: display extract lacks embedded coordinates") from error
    if len(set(coordinate_indices)) != dimension:
        raise ValueError(f"{job['id']}: duplicate embedded-coordinate column")
    if len(rows) - 1 != int(display["n_rows"]):
        raise ValueError(f"{job['id']}: display extract row count changed")
    try:
        time = np.asarray([float(row[0]) for row in rows[1:]], dtype=float)
        positions = np.asarray(
            [[float(row[index]) for index in coordinate_indices] for row in rows[1:]],
            dtype=float,
        )
    except ValueError as error:
        raise ValueError(f"{job['id']}: nonnumeric display extract") from error
    if positions.ndim != 2 or positions.shape[1] != dimension or len(positions) < 2:
        raise ValueError(f"{job['id']}: malformed display extract")
    if (
        np.any(~np.isfinite(time))
        or np.any(~np.isfinite(positions))
        or not np.all(np.diff(time) > 0)
        or not math.isclose(float(time[0]), 0.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"{job['id']}: invalid display samples")
    duration = float(time[-1])
    planned_duration = display.get("nominal_suspension_duration")
    if planned_duration is None:
        planned_duration = 50.0 if job["id"] == "chaos" else job["nominal_suspension_period"]
    if not math.isclose(duration, float(planned_duration), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{job['id']}: display duration changed")
    kind = str(display["kind"]).lower()
    if job["id"] == "chaos":
        if "chaos" not in kind and "nonperiodic" not in kind:
            raise ValueError("chaos display is not explicitly nonperiodic")
        if not 0.0 < duration <= 50.0 + 1e-12:
            raise ValueError("chaos display segment is not bounded")
        closure = None
    else:
        if "fourier" not in kind or "closed" not in kind:
            raise ValueError(f"{job['id']}: display is not Fourier-closed")
        closure = float(np.linalg.norm(positions[-1] - positions[0]))
        tolerance = max(1e-10, 1e-8 * float(np.max(np.ptp(positions, axis=0))))
        if closure > tolerance:
            raise ValueError(f"{job['id']}: Fourier display is not closed")
    return positions[:, :3], {
        "case_id": job["id"],
        "phi_deg": float(job["phi_deg"]),
        "q": job["q"],
        "kind": display["kind"],
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "n_rows": int(len(positions)),
        "nominal_suspension_duration": duration,
        "coordinates": ["z1", "z2", "z3"],
        "full_dimension": dimension,
        "full_state_endpoint_closure_l2": closure,
    }


def centers_to_edges(values: np.ndarray, lower: float) -> np.ndarray:
    if values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("cell centers must be a strictly increasing vector")
    middle = (values[:-1] + values[1:]) / 2.0
    upper = values[-1] + (values[-1] - values[-2]) / 2.0
    return np.concatenate(([lower], middle, [upper]))


def parameter_title(phi_deg: float) -> str:
    return rf"$\phi = {phi_deg:.2f}^\circ$"


def style_embedded_axes(axis: Any, positions: np.ndarray) -> None:
    axis.set_xlabel(r"$z_1$", labelpad=-11.0)
    axis.set_ylabel(r"$z_2$", labelpad=-11.0)
    axis.set_zlabel(r"$z_3$", labelpad=-9.0)
    axis.tick_params(axis="x", pad=-3.0, width=0.45, length=2.0)
    axis.tick_params(axis="y", pad=-3.0, width=0.45, length=2.0)
    axis.tick_params(axis="z", pad=-3.0, width=0.45, length=2.0)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_major_locator(MaxNLocator(nbins=2))
        # Keep the coordinate triad and tick marks, but omit tiny numeric tick
        # strings that collide across five manuscript-width 3D panels.
        coordinate_axis.set_major_formatter(NullFormatter())
        coordinate_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        coordinate_axis.pane.set_edgecolor((0.58, 0.58, 0.58, 0.72))
        coordinate_axis._axinfo["grid"]["linewidth"] = 0.0
        coordinate_axis._axinfo["axisline"]["linewidth"] = 0.45
        coordinate_axis._axinfo["axisline"]["color"] = (0.35, 0.35, 0.35, 1.0)
    spans = np.maximum(np.ptp(positions, axis=0), 1e-12)
    axis.set_box_aspect(tuple(spans))
    axis.view_init(elev=24.0, azim=-56.0)


def render_pdf(
    plan: dict[str, Any],
    loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Path]],
) -> tuple[bytes, list[dict[str, Any]]]:
    jobs = plan["jobs"]
    durations = loaded[0][0]
    radii = loaded[0][1]
    if not all(np.array_equal(durations, item[0]) for item in loaded):
        raise ValueError("duration grids differ across Compass cases")
    if not all(np.array_equal(radii, item[1]) for item in loaded):
        raise ValueError("radius grids differ across Compass cases")
    duration_edges = centers_to_edges(durations, lower=0.0)
    radius_edges = centers_to_edges(radii, lower=0.0)

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
    display_records: list[dict[str, Any]] = []
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

        for column, (job, item) in enumerate(zip(jobs, loaded)):
            top = figure.add_subplot(grid[0, column], projection="3d")
            positions, display_record = load_display(job)
            display_records.append(display_record)
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
                item[2],
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
                vertical_guide = float(job["nominal_suspension_period"])
                bottom.axvline(
                    vertical_guide,
                    color="white",
                    linestyle=(0, (3.0, 2.2)),
                    linewidth=0.80,
                    dash_capstyle="butt",
                )
            bottom.set_xlim(0.0, 12.0)
            bottom.set_ylim(0.0, 5.0)
            bottom.set_xticks([0.0, 4.0, 8.0, 12.0])
            bottom.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            bottom.tick_params(axis="both", pad=1.7)
            if column:
                bottom.tick_params(labelleft=False)
            display_record["vertical_nominal_suspension_guide"] = vertical_guide
            display_record["horizontal_radius_guide"] = None

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
                "Title": EXPECTED_ANALYSIS_ID,
                "Creator": str(SCRIPT_PATH),
                "Subject": (
                    "Fourier-closed embedded Compass periodic orbits and "
                    "validated shared-control cycling-signature probabilities"
                ),
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)
    return buffer.getvalue(), display_records


def inspect_pdf(path: Path) -> dict[str, Any]:
    pdfinfo_bin = shutil.which("pdfinfo")
    pdfimages_bin = shutil.which("pdfimages")
    pdftotext_bin = shutil.which("pdftotext")
    if pdfinfo_bin is None or pdfimages_bin is None or pdftotext_bin is None:
        raise ValueError("Poppler pdfinfo, pdfimages, and pdftotext are required")
    info = subprocess.run(
        [pdfinfo_bin, str(path)], check=True, capture_output=True, text=True
    ).stdout
    page_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", info, re.MULTILINE
    )
    pages_match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if page_match is None or pages_match is None:
        raise ValueError("could not parse PDF page geometry")
    width_pt, height_pt = (float(value) for value in page_match.groups())
    if int(pages_match.group(1)) != 1:
        raise ValueError("paper figure must be exactly one page")
    if not math.isclose(
        width_pt, FIGURE_WIDTH_IN * POINTS_PER_INCH, abs_tol=0.01
    ) or not math.isclose(
        height_pt, FIGURE_HEIGHT_IN * POINTS_PER_INCH, abs_tol=0.01
    ):
        raise ValueError(f"unexpected PDF page size: {width_pt} x {height_pt} points")
    image_listing = subprocess.run(
        [pdfimages_bin, "-list", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    image_rows = [
        line for line in image_listing.splitlines() if re.match(r"^\s*\d+\s+\d+\s+", line)
    ]
    if image_rows:
        raise ValueError("paper figure contains raster image objects")
    extracted = subprocess.run(
        [pdftotext_bin, str(path), "-"], check=True, capture_output=True, text=True
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


def render_preview(pdf_path: Path, output_dir: Path) -> Path:
    pdftoppm_bin = shutil.which("pdftoppm")
    if pdftoppm_bin is None:
        raise ValueError("Poppler pdftoppm is required for visual QA")
    prefix = output_dir / "preview"
    subprocess.run(
        [
            pdftoppm_bin,
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
        raise ValueError("Poppler did not produce the PNG preview")
    return preview


def result_records(
    plan: dict[str, Any],
    loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Path]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job, item in zip(plan["jobs"], loaded):
        raw_paths = frozen.result_paths(job)
        records.append(
            {
                "case_id": job["id"],
                "raw_results": {
                    name: {"path": str(path.resolve()), "sha256": sha256(path)}
                    for name, path in raw_paths.items()
                },
                "v2_result_binding": str(item[4].resolve()),
                "v2_result_binding_sha256": sha256(item[4]),
                "global_curve_bound": float(job["global_curve_bound"]),
                "beta1_Y": int(item[3]["beta1_Y"]),
            }
        )
    return records


def sustained_p50_onset(
    durations: np.ndarray, probability_at_one_radius: np.ndarray
) -> float | None:
    above = probability_at_one_radius >= 0.5
    sustained = np.logical_and.accumulate(above[::-1])[::-1]
    indices = np.flatnonzero(sustained)
    return None if not len(indices) else float(durations[int(indices[0])])


def prepare_destination_temp(target: Path, data: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.{RENDERER_ID}.", suffix=".tmp"
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


def promote_transaction(artifacts: list[tuple[Path, bytes]], *, replace: bool) -> None:
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


def run(plan_path: Path, julia_bin: str, replace: bool) -> Path:
    root, plan = verify_bound_plan(plan_path, julia_bin)
    loaded = [load_probability(job, plan) for job in plan["jobs"]]
    figure_path = root / EXPECTED_FIGURE_FILENAME
    preview_path = root / EXPECTED_PREVIEW_FILENAME
    provenance_path = figure_path.with_suffix(".render.json")
    pdf_bytes, displays = render_pdf(plan, loaded)

    with tempfile.TemporaryDirectory(dir=root, prefix=f".{RENDERER_ID}.inspect.") as name:
        staging = Path(name)
        staged_pdf = staging / EXPECTED_FIGURE_FILENAME
        staged_pdf.write_bytes(pdf_bytes)
        geometry = inspect_pdf(staged_pdf)
        staged_preview = render_preview(staged_pdf, staging)
        preview_bytes = staged_preview.read_bytes()
        pdf_hash = sha256(staged_pdf)
        preview_hash = sha256(staged_preview)

    curve_bounds = [float(job["global_curve_bound"]) for job in plan["jobs"]]
    common_curve_bound = max(curve_bounds)
    common_radii = loaded[0][1]
    valid_grid_radii = common_radii[common_radii > common_curve_bound]
    if not len(valid_grid_radii):
        raise ValueError("no plotted radius is strictly above the common curve bound")
    low_radius_onsets = {
        job["id"]: sustained_p50_onset(item[0], item[2][0])
        for job, item in zip(plan["jobs"], loaded)
    }
    provenance = {
        "schema_version": 1,
        "renderer_id": RENDERER_ID,
        "scope": "display_only_no_signature_recomputation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": plan["analysis_id"],
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": sha256(Path(plan_path).resolve()),
        "frozen_orchestrator": str(frozen.SCRIPT_PATH),
        "frozen_orchestrator_sha256": sha256(frozen.SCRIPT_PATH),
        "display_renderer": str(SCRIPT_PATH),
        "display_renderer_sha256": sha256(SCRIPT_PATH),
        "cases": plan["cases_path"],
        "cases_sha256": plan["cases_sha256"],
        "protocol": plan["protocol_path"],
        "protocol_sha256": plan["protocol_sha256"],
        "protocol_id": plan["protocol"]["protocol_id"],
        "bundle_manifest": plan["bundle_manifest"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "bundle_id": plan["bundle_id"],
        "scientific_scope": plan["scientific_scope"],
        "probability_statistic": "P(rank > 0) = 1 - rank0 / 20",
        "duration_grid": "1:0.2:12",
        "radius_grid": "0:0.025:5",
        "probability_rendering": "exact flat cells; no interpolation",
        "layout": (
            "2x5; parameter-only phi titles; first-three-coordinate z1/z2/z3 "
            "axes; shared lower-axis labels"
        ),
        "minimum_font_size_points": MINIMUM_FONT_SIZE_PT,
        "horizontal_radius_guide": None,
        "periodic_vertical_guides": [value for value in EXPECTED_GUIDES[:-1]],
        "displays": displays,
        "results": result_records(plan, loaded),
        "curve_resolution": {
            "per_case_global_curve_bounds": curve_bounds,
            "common_curve_valid_radius_strictly_above": common_curve_bound,
            "first_common_curve_resolved_plotted_radius": float(valid_grid_radii[0]),
            "interpretation": (
                "Any low-radius period-doubling cascade visible below the common "
                "curve bound is empirical/provisional and is not a curve-resolved "
                "topology claim."
            ),
        },
        "low_radius_exploratory_staircase": {
            "radius": 0.0,
            "threshold": "sustained P(rank > 0) >= 0.5",
            "onsets_by_case": low_radius_onsets,
            "interpretation": (
                "The 1.0, 1.8, 3.6, and 7.2 onset staircase is a discrete "
                "low-radius exploratory result; it lies below the common "
                "curve-resolved radius band."
            ),
        },
        "pdf_geometry": geometry,
        "output_pdf": str(figure_path.resolve()),
        "output_pdf_sha256": pdf_hash,
        "output_preview_png": str(preview_path.resolve()),
        "output_preview_png_sha256": preview_hash,
        "preview_dpi": PREVIEW_DPI,
        "paper_review_copy": str(PAPER_COPY),
        "paper_review_copy_sha256": pdf_hash,
    }
    provenance_bytes = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8")
    promote_transaction(
        [
            (figure_path, pdf_bytes),
            (preview_path, preview_bytes),
            (provenance_path, provenance_bytes),
            (PAPER_COPY, pdf_bytes),
        ],
        replace=replace,
    )
    for target, expected_hash in (
        (figure_path, pdf_hash),
        (preview_path, preview_hash),
        (PAPER_COPY, pdf_hash),
    ):
        if sha256(target) != expected_hash:
            raise RuntimeError(f"promoted artifact hash mismatch: {target}")
    print(f"render_provenance_sha256={sha256(provenance_path)}")
    print(f"output_pdf={figure_path}")
    print(f"output_pdf_sha256={pdf_hash}")
    print(f"output_preview_png={preview_path}")
    print(f"output_preview_png_sha256={preview_hash}")
    print(f"paper_review_copy={PAPER_COPY}")
    print(f"page_size={geometry['width_inches']:.2f} x {geometry['height_inches']:.2f} inches")
    print("raster_image_objects=0")
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-scale display-only renderer for one frozen Compass v2 plan."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--julia", default="julia")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="explicitly replace the code PDF/PNG/sidecar and paper review copy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.plan, args.julia, args.replace)


if __name__ == "__main__":
    main()
