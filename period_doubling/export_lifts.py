"""Export timeseries to CSV and NPZ formats for downstream analysis.

Provides write_lift for exporting Timeseries and HybridTimeseries objects to
space-delimited CSVs (positions and tangents) and compressed NPZ archives with
full metadata. Includes load_npz for reading back NPZ files.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def write_lift(out_dir, base, ts):
    """Write timeseries to CSV and NPZ files.

    Parameters
    ----------
    out_dir : str or Path
        Output directory. Created if necessary.
    base : str
        Base filename (without extension).
    ts : Timeseries or HybridTimeseries
        Timeseries object with .t, .x, .v, .meta, and optional
        .impact_times, .jump_minus, .jump_plus attributes.

    Returns
    -------
    dict
        Dictionary with keys 'positions', 'tangents', 'npz', 'report'
        mapping to Path objects of written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV files (space-delimited, rows=samples, cols=dimensions)
    pos_path = out_dir / f"{base}_positions.csv"
    tan_path = out_dir / f"{base}_tangents.csv"
    npz_path = out_dir / f"{base}.npz"
    report_path = out_dir / f"{base}_report.txt"

    # CSVs: shape (N, d) with rows=samples, space-separated
    np.savetxt(pos_path, ts.x, fmt="%.18e", delimiter=" ")
    np.savetxt(tan_path, ts.v, fmt="%.18e", delimiter=" ")

    # NPZ with metadata as JSON string
    meta_json = json.dumps(ts.meta, default=str)
    npz_data = {
        "t": ts.t,
        "x": ts.x,
        "v": ts.v,
        "meta_json": meta_json,
    }

    # Include hybrid-specific fields if present
    if hasattr(ts, "impact_times"):
        npz_data["impact_times"] = ts.impact_times
    if hasattr(ts, "jump_minus"):
        npz_data["jump_minus"] = ts.jump_minus
    if hasattr(ts, "jump_plus"):
        npz_data["jump_plus"] = ts.jump_plus

    np.savez_compressed(npz_path, **npz_data)

    # Write human-readable report
    n_samples, dim = ts.x.shape
    dt = ts.t[1] - ts.t[0] if len(ts.t) > 1 else np.nan
    t_span = ts.t[-1] - ts.t[0]

    with open(report_path, "w") as f:
        f.write(f"n_samples: {n_samples}\n")
        f.write(f"dim: {dim}\n")
        f.write(f"dt: {dt:.6e}\n")
        f.write(f"t_span: {t_span:.6e}\n")
        for key, val in ts.meta.items():
            f.write(f"{key}: {val}\n")
        if hasattr(ts, "impact_times"):
            f.write(f"n_impacts: {len(ts.impact_times)}\n")

    return {
        "positions": pos_path,
        "tangents": tan_path,
        "npz": npz_path,
        "report": report_path,
    }


def load_npz(path):
    """Load NPZ file back into a SimpleNamespace.

    Parameters
    ----------
    path : str or Path
        Path to .npz file written by write_lift.

    Returns
    -------
    SimpleNamespace
        Object with attributes .t, .x, .v, .meta, and optional
        .impact_times, .jump_minus, .jump_plus.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    # Parse metadata from JSON string
    meta_json = str(data["meta_json"])
    meta = json.loads(meta_json)

    # Build namespace with required fields
    result = SimpleNamespace(
        t=data["t"],
        x=data["x"],
        v=data["v"],
        meta=meta,
    )

    # Add optional hybrid fields
    if "impact_times" in data:
        result.impact_times = data["impact_times"]
    if "jump_minus" in data:
        result.jump_minus = data["jump_minus"]
    if "jump_plus" in data:
        result.jump_plus = data["jump_plus"]

    return result
