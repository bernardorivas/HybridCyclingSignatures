#!/usr/bin/env python3
"""Display-only paper renderer for the frozen David-family v2 result.

This renderer is intentionally bound to one immutable plan.  It imports the
frozen v2 loader/validators, changes no numerical input or result, and only
replaces the rendered PDF, its render provenance, and the paper review copy.
Existing artifacts are replaced only when ``--replace`` is supplied.
"""

from __future__ import annotations

import argparse
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

import roessler_probability_v2 as frozen


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[3]
EXPECTED_ANALYSIS_ID = "roessler_david_fourier_probability_linf_v2_david_grid"
EXPECTED_PLAN_SHA256 = (
    "e9a350772fb8462a295057585cb1247dbf0cfdebfdacd69eca8871545a74a9b8"
)
EXPECTED_FIGURE_FILENAME = "roessler_C5p0.pdf"
PAPER_COPY = (
    WORKSPACE_ROOT
    / "paper"
    / "hybrid_cyclingsignatures"
    / "figures"
    / "cycling_signatures"
    / EXPECTED_FIGURE_FILENAME
)

RENDERER_ID = "roessler_probability_paper_renderer_v1"
FIGURE_WIDTH_IN = 9.0
FIGURE_HEIGHT_IN = 4.65
POINTS_PER_INCH = 72.0
MINIMUM_FONT_SIZE_PT = 7.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_output_target(path: Path) -> None:
    """Reject symlinks and non-files without resolving an output through them."""
    probe = path.absolute()
    while probe != probe.parent:
        if probe.is_symlink():
            raise ValueError(f"output path contains a symlink: {probe}")
        probe = probe.parent
    if path.exists() and not path.is_file():
        raise ValueError(f"output target is not a regular file: {path}")


def verify_bound_plan(
    plan_path: Path, julia_bin: str
) -> tuple[
    Path,
    dict[str, Any],
    frozen.Configuration,
    frozen.ValidatedBundle,
    list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]],
]:
    plan_path = plan_path.resolve()
    if sha256(plan_path) != EXPECTED_PLAN_SHA256:
        raise ValueError(
            "paper renderer is bound to the completed David-grid plan hash"
        )
    root, plan, configuration, bundle = frozen.load_plan(plan_path, julia_bin)
    if plan["analysis_id"] != EXPECTED_ANALYSIS_ID:
        raise ValueError("refusing an unrelated analysis plan")
    if plan["figure_filename"] != EXPECTED_FIGURE_FILENAME:
        raise ValueError("refusing a plan with a different output filename")
    if root.name != EXPECTED_ANALYSIS_ID:
        raise ValueError("refusing a plan outside its canonical analysis root")
    loaded = [
        frozen.validate_result(job, configuration.protocol, plan)
        for job in plan["jobs"]
    ]
    start_hashes = {item[3]["segment_starts_sha256"] for item in loaded}
    if len(start_hashes) != 1:
        raise ValueError("validated cases do not share the planned random windows")
    return root, plan, configuration, bundle, loaded


def render_pdf(
    plan: dict[str, Any],
    loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]],
) -> tuple[bytes, list[dict[str, Any]]]:
    jobs = plan["jobs"]
    if [job["id"] for job in jobs] != [
        "period1",
        "period2",
        "period4",
        "period8",
        "chaos",
    ]:
        raise ValueError("paper layout requires period1/2/4/8 and chaos in order")
    durations = loaded[0][0]
    radii = loaded[0][1]
    if not all(np.array_equal(durations, item[0]) for item in loaded):
        raise ValueError("duration grids differ across cases")
    if not all(np.array_equal(radii, item[1]) for item in loaded):
        raise ValueError("radius grids differ across cases")
    duration_edges = frozen.centers_to_edges(durations, lower=0.0)
    radius_edges = frozen.centers_to_edges(radii, lower=0.0)

    rc = {
        "font.size": MINIMUM_FONT_SIZE_PT,
        "axes.titlesize": 8.25,
        "axes.labelsize": 8.0,
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
            bottom=0.145,
            top=0.955,
            wspace=0.10,
            hspace=0.08,
            width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.055],
            height_ratios=[0.95, 1.10],
        )

        for column, (job, item) in enumerate(zip(jobs, loaded)):
            top = figure.add_subplot(grid[0, column], projection="3d")
            positions = frozen.display_positions(job)
            top.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color="#0072B2",
                linewidth=0.72,
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            # The coordinate cage is illegible in a five-column paper panel;
            # the trajectory itself and parameter-only title carry the display.
            top.set_axis_off()
            top.set_title(job["title"], pad=-1.5)
            top.view_init(elev=25, azim=-55)
            spans = np.ptp(positions, axis=0)
            top.set_box_aspect(tuple(np.maximum(spans, 1e-12)))

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
            if job["kind"] == "periodic":
                vertical_guide = float(job["display"]["duration"])
                bottom.axvline(
                    vertical_guide,
                    color="white",
                    linestyle=(0, (3.0, 2.2)),
                    linewidth=0.80,
                    dash_capstyle="butt",
                )
            bottom.set_xlim(0.0, 60.0)
            bottom.set_ylim(0.0, 5.0)
            bottom.set_xticks([0.0, 20.0, 40.0, 60.0])
            bottom.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            bottom.tick_params(axis="both", pad=1.7)
            if column:
                bottom.tick_params(labelleft=False)
            display_records.append(
                {
                    "case_id": job["id"],
                    "a": float(job["a"]),
                    "kind": job["kind"],
                    "title": job["title"],
                    "display": job["display"]["absolute_path"],
                    "display_sha256": job["display"]["sha256"],
                    "display_rows": int(job["display"]["n_rows"]),
                    "display_duration": float(job["display"]["duration"]),
                    "certificate": job["certificate"]["absolute_path"],
                    "certificate_sha256": job["certificate"]["sha256"],
                    "vertical_period_guide": vertical_guide,
                    "horizontal_radius_guide": None,
                }
            )

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
            0.325,
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
        # Matplotlib rasterizes long colorbars by default; keep this paper PDF
        # wholly vector just like the five exact-cell probability meshes.
        colorbar.solids.set_rasterized(False)
        colorbar.set_ticks([0.0, 0.5, 1.0])
        colorbar.solids.set_edgecolor("face")
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
                    "Paper-scale display of certified Rössler orbits and "
                    "validated cycling-signature probabilities"
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
    if pdfinfo_bin is None or pdfimages_bin is None:
        raise ValueError("Poppler pdfinfo and pdfimages are required")
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
        raise ValueError(
            f"unexpected PDF page size: {width_pt} x {height_pt} points"
        )
    image_listing = subprocess.run(
        [pdfimages_bin, "-list", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    image_rows = [
        line
        for line in image_listing.splitlines()
        if re.match(r"^\s*\d+\s+\d+\s+", line)
    ]
    if image_rows:
        raise ValueError("paper figure contains raster image objects")
    return {
        "pages": 1,
        "width_points": width_pt,
        "height_points": height_pt,
        "width_inches": width_pt / POINTS_PER_INCH,
        "height_inches": height_pt / POINTS_PER_INCH,
        "raster_image_objects": 0,
    }


def result_records(
    plan: dict[str, Any],
    loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job, item in zip(plan["jobs"], loaded):
        raw_paths = frozen.result_paths(job)
        binding = frozen.result_binding_path(job)
        records.append(
            {
                "case_id": job["id"],
                "raw_results": {
                    name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in raw_paths.items()
                },
                "v2_result_binding": str(binding),
                "v2_result_binding_sha256": sha256(binding),
                "segment_starts_sha256": item[3]["segment_starts_sha256"],
                "global_curve_bound": float(item[3]["global_curve_bound"]),
            }
        )
    return records


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


def promote_transaction(
    artifacts: list[tuple[Path, bytes]], *, replace: bool
) -> None:
    for target, _ in artifacts:
        reject_output_target(target)
        if target.exists() and not replace:
            raise FileExistsError(
                f"refusing to overwrite without --replace: {target}"
            )
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
    root, plan, configuration, bundle, loaded = verify_bound_plan(
        plan_path, julia_bin
    )
    figure_path = root / EXPECTED_FIGURE_FILENAME
    provenance_path = figure_path.with_suffix(".render.json")
    pdf_bytes, displays = render_pdf(plan, loaded)

    with tempfile.TemporaryDirectory(
        dir=root, prefix=f".{RENDERER_ID}.inspect."
    ) as staging_name:
        staged_pdf = Path(staging_name) / EXPECTED_FIGURE_FILENAME
        staged_pdf.write_bytes(pdf_bytes)
        geometry = inspect_pdf(staged_pdf)
        pdf_hash = sha256(staged_pdf)

    starts_hashes = {item[3]["segment_starts_sha256"] for item in loaded}
    provenance = {
        "schema_version": 1,
        "renderer_id": RENDERER_ID,
        "scope": "display_only_no_signature_recomputation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": sha256(Path(plan_path).resolve()),
        "analysis_id": plan["analysis_id"],
        "frozen_orchestrator": str(frozen.SCRIPT_PATH),
        "frozen_orchestrator_sha256": sha256(frozen.SCRIPT_PATH),
        "display_renderer": str(SCRIPT_PATH),
        "display_renderer_sha256": sha256(SCRIPT_PATH),
        "cases": str(configuration.cases_path),
        "cases_sha256": sha256(configuration.cases_path),
        "protocol": str(configuration.protocol_path),
        "protocol_sha256": sha256(configuration.protocol_path),
        "protocol_id": configuration.protocol["protocol_id"],
        "bundle_manifest": str(bundle.manifest_path),
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "probability_statistic": "P(rank > 0) = 1 - rank0 / 20",
        "duration_grid": "1:0.2:60",
        "radius_grid": "0:0.025:5",
        "probability_rendering": "exact flat cells; no interpolation",
        "layout": "2x5; parameter-only top titles; shared lower-axis labels",
        "minimum_font_size_points": MINIMUM_FONT_SIZE_PT,
        "horizontal_radius_guide": None,
        "periodic_vertical_guides": "manifest display.duration",
        "displays": displays,
        "results": result_records(plan, loaded),
        "shared_segment_starts_sha256": next(iter(starts_hashes)),
        "pdf_geometry": geometry,
        "output_pdf": str(figure_path),
        "output_pdf_sha256": pdf_hash,
        "paper_review_copy": str(PAPER_COPY),
        "paper_review_copy_sha256": pdf_hash,
    }
    provenance_bytes = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    promote_transaction(
        [
            (figure_path, pdf_bytes),
            (provenance_path, provenance_bytes),
            (PAPER_COPY, pdf_bytes),
        ],
        replace=replace,
    )
    for target in (figure_path, PAPER_COPY):
        if sha256(target) != pdf_hash:
            raise RuntimeError(f"promoted PDF hash mismatch: {target}")
    # Keep a concise terminal audit without making the sidecar self-referential.
    print(f"render_provenance_sha256={sha256(provenance_path)}")
    print(f"output_pdf={figure_path}")
    print(f"output_pdf_sha256={pdf_hash}")
    print(f"paper_review_copy={PAPER_COPY}")
    print(
        f"page_size={geometry['width_inches']:.2f} x "
        f"{geometry['height_inches']:.2f} inches"
    )
    print("raster_image_objects=0")
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-scale display-only renderer for one frozen v2 plan."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--julia", default="julia")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="explicitly replace the code PDF/sidecar and paper review copy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.plan, args.julia, args.replace)


if __name__ == "__main__":
    main()
