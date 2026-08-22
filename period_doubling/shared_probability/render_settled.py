#!/usr/bin/env python3
"""Render validated shared-protocol Rössler results with a settled tail.

The completed plans freeze ``driver.py`` by hash, so changing its display
adapter would invalidate their provenance.  This display-only renderer loads
that frozen driver, validates the plan, inputs, and signature outputs, and
changes only the top-row trajectory selection.  It draws the final
``maximum_points`` post-stride samples consecutively instead of joining a
uniform decimation of the entire retained trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import matplotlib.pyplot as plt
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
        "shared_probability_driver_for_settled_render", DRIVER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the shared probability driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def settled_positions(job: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    display = job["display"]
    if display["kind"] not in {"positions_3d", "analysis_positions_3d"}:
        raise ValueError(
            "settled-tail rendering is defined only for three-dimensional "
            f"position displays, not {display['kind']!r}"
        )
    positions_path = Path(job["positions"])
    positions = np.loadtxt(positions_path, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError(f"invalid position array for {job['case_id']}")
    stride = int(job["stride"])
    post_stride = positions[::stride]
    maximum = int(display["maximum_points"])
    selected = post_stride[-maximum:]
    if len(selected) < 2 or not np.all(np.isfinite(selected[:, :3])):
        raise ValueError(f"invalid settled display tail for {job['case_id']}")
    return selected, {
        "case_id": job["case_id"],
        "positions": str(positions_path.resolve()),
        "positions_sha256": sha256(positions_path),
        "stride": stride,
        "post_stride_sample_count": int(len(post_stride)),
        "selected_sample_count": int(len(selected)),
        "selected_post_stride_start_index_zero_based": int(
            len(post_stride) - len(selected)
        ),
        "selected_post_stride_stop_index_zero_based_exclusive": int(
            len(post_stride)
        ),
        "effective_sample_dt": float(job["effective_dt"]),
        "selected_endpoint_span": float(
            (len(selected) - 1) * float(job["effective_dt"])
        ),
    }


def temporary_path(parent: Path, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        dir=parent,
        prefix=".settled-render-",
        suffix=suffix,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def render_settled(plan_path: Path, replace_existing: bool) -> Path:
    driver = load_driver()
    root, plan = driver.load_plan(plan_path)
    protocol = plan["protocol"]
    if not all(
        job.get("source", {}).get("kind") == "roessler_rk4"
        for job in plan["jobs"]
    ):
        raise ValueError(
            "settled-tail replacement is restricted to generated Rössler plans"
        )
    figure_path = root / plan["figure_filename"]
    provenance_path = figure_path.with_suffix(".render.json")
    for path in (figure_path, provenance_path):
        if path.is_symlink():
            raise ValueError(f"refusing symlinked output: {path}")
        if path.exists() and not replace_existing:
            raise FileExistsError(
                f"refusing to overwrite {path}; pass --replace-existing"
            )

    loaded = [
        driver.load_probability(job, protocol, plan) for job in plan["jobs"]
    ]
    durations = loaded[0][0]
    radii = loaded[0][1]
    if not all(np.array_equal(durations, item[0]) for item in loaded):
        raise ValueError("cases do not share one duration grid")
    if not all(np.array_equal(radii, item[1]) for item in loaded):
        raise ValueError("cases do not share one radius grid")
    if not all(
        job["display"]["kind"] in {"positions_3d", "analysis_positions_3d"}
        for job in plan["jobs"]
    ):
        raise ValueError("this renderer requires only three-dimensional cases")

    n_cases = len(plan["jobs"])
    figure = plt.figure(
        figsize=(3.2 * n_cases + 0.7, 6.8), constrained_layout=True
    )
    grid = figure.add_gridspec(
        2,
        n_cases + 1,
        width_ratios=[1] * n_cases + [0.05],
        height_ratios=[1.0, 1.05],
        hspace=0.12,
    )
    duration_edges = driver.centers_to_edges(durations, lower=0.0)
    radius_edges = driver.centers_to_edges(radii, lower=0.0)
    display_records: list[dict[str, Any]] = []
    image = None
    for column, (job, item) in enumerate(zip(plan["jobs"], loaded)):
        top = figure.add_subplot(grid[0, column], projection="3d")
        positions, record = settled_positions(job)
        display_records.append(record)
        top.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            linewidth=0.8,
            antialiased=True,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        labels = (
            ("x", "y", "z")
            if job["display"]["kind"] == "positions_3d"
            else (r"$z_1$", r"$z_2$", r"$z_3$")
        )
        top.set_xlabel(labels[0])
        top.set_ylabel(labels[1])
        top.set_zlabel(labels[2])
        top.set_title(job["title"], fontsize=9)

        bottom = figure.add_subplot(grid[1, column])
        image = bottom.pcolormesh(
            duration_edges,
            radius_edges,
            item[2],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            shading="flat",
            rasterized=True,
        )
        guide = protocol.get("horizontal_radius_guide")
        if guide == "sample_radius":
            bottom.axhline(
                float(item[3]["sample_radius"]),
                color="white",
                linestyle="--",
                linewidth=0.9,
            )
        elif isinstance(guide, (int, float)):
            bottom.axhline(
                float(guide), color="white", linestyle="--", linewidth=0.9
            )
        bottom.set_title(job["title"], fontsize=9)
        bottom.set_xlabel("segment duration (t)")
        if column == 0:
            bottom.set_ylabel("filtration radius (r)")
        else:
            bottom.set_yticklabels([])

    if image is None:
        raise RuntimeError("no probability panels")
    color_axis = figure.add_subplot(grid[1, -1])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("probability")

    temporary_pdf = temporary_path(root, ".pdf")
    temporary_provenance = temporary_path(root, ".json")
    try:
        figure.savefig(
            temporary_pdf,
            format="pdf",
            bbox_inches="tight",
            metadata={
                "Title": plan["analysis_id"],
                "Creator": str(SCRIPT_PATH),
                "Subject": (
                    "Validated shared-protocol heatmaps with a consecutive "
                    "settled-tail trajectory display"
                ),
            },
        )
        plt.close(figure)
        provenance = {
            "schema_version": 1,
            "analysis_id": plan["analysis_id"],
            "scope": "display_only_no_signature_recomputation",
            "plan": str(plan_path.resolve()),
            "plan_sha256": sha256(plan_path.resolve()),
            "frozen_analysis_driver": str(DRIVER_PATH),
            "frozen_analysis_driver_sha256": plan["driver_sha256"],
            "display_renderer": str(SCRIPT_PATH),
            "display_renderer_sha256": sha256(SCRIPT_PATH),
            "selection_rule": (
                "last maximum_points consecutive post-stride samples; no "
                "interpolation, spline smoothing, or whole-trajectory "
                "uniform decimation"
            ),
            "cases": display_records,
            "output_pdf": str(figure_path.resolve()),
            "output_pdf_sha256": sha256(temporary_pdf),
        }
        temporary_provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_pdf, figure_path)
        os.replace(temporary_provenance, provenance_path)
    except BaseException:
        plt.close(figure)
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        raise
    print(f"Wrote {figure_path}")
    print(f"Wrote {provenance_path}")
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a completed three-dimensional shared-protocol plan using "
            "a consecutive settled trajectory tail in the top row."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="atomically replace the stable PDF and its render record",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_settled(args.plan, args.replace_existing)


if __name__ == "__main__":
    main()
