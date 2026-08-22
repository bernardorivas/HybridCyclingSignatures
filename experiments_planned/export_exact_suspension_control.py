#!/usr/bin/env python3
"""Export an exact-seam analytic suspension control for compass trajectories.

The base slice is the identity embedding ``x -> (x, 0)``.  Each recorded
guard/reset pair is joined by a deterministic arch whose endpoints are
exactly ``(g, 0)`` and ``(r(g), 0)``.  This isolates exact gluing from both
the discontinuous raw control and the learned encoder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = CODE_ROOT / "period_doubling" / "data" / "compass_gait"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "outputs" / "exact_suspension"
)
JULIA_PROJECT = CODE_ROOT / "period_doubling" / "julia"
JULIA_DRIVER = (
    CODE_ROOT / "chyll_v2" / "cycling_signature" / "julia" / "run_subsegments.jl"
)


def load_hybrid_npz(path: Path) -> SimpleNamespace:
    archive = np.load(path, allow_pickle=False)
    required = {
        "t",
        "x",
        "v",
        "impact_times",
        "jump_minus",
        "jump_plus",
        "meta_json",
    }
    missing = required.difference(archive.files)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return SimpleNamespace(
        t=archive["t"],
        x=archive["x"],
        v=archive["v"],
        impact_times=archive["impact_times"],
        jump_minus=archive["jump_minus"],
        jump_plus=archive["jump_plus"],
        meta=json.loads(str(archive["meta_json"])),
    )


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms < eps):
        raise ValueError("analytic lift contains a near-zero tangent")
    return values / norms[:, None]


def build_analytic_lift(
    ts: SimpleNamespace,
    n_bridge: int,
    bridge_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interleave identity-embedded arcs with exact-endpoint bridge arches."""
    if n_bridge < 2:
        raise ValueError("--n-bridge must be at least 2")
    d = ts.x.shape[1]
    arc_index = np.searchsorted(ts.impact_times, ts.t, side="left")
    positions: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    kinds: list[np.ndarray] = []

    for impact_index in range(len(ts.impact_times) + 1):
        mask = arc_index == impact_index
        if np.any(mask):
            arc_positions = np.column_stack(
                [ts.x[mask], np.zeros(int(mask.sum()))]
            )
            arc_tangents = np.column_stack(
                [ts.v[mask], np.zeros(int(mask.sum()))]
            )
            positions.append(arc_positions)
            tangents.append(arc_tangents)
            kinds.append(np.zeros(int(mask.sum()), dtype=np.uint8))

        if impact_index >= len(ts.impact_times):
            continue
        guard = ts.jump_minus[impact_index]
        reset = ts.jump_plus[impact_index]
        # Use the same interior-only bridge convention as the trained cascade
        # exporter.  The analytic formula still has exact limiting endpoints
        # (g, 0) and (r(g), 0).
        s_values = np.linspace(0.0, 1.0, n_bridge + 2)[1:-1]
        base = (1.0 - s_values[:, None]) * guard + s_values[:, None] * reset
        height = bridge_height * np.sin(np.pi * s_values)
        bridge_positions = np.column_stack([base, height])
        derivative = np.column_stack(
            [
                np.repeat((reset - guard)[None, :], len(s_values), axis=0),
                bridge_height * np.pi * np.cos(np.pi * s_values),
            ]
        )
        positions.append(bridge_positions)
        tangents.append(normalize_rows(derivative))
        kinds.append(np.ones(len(s_values), dtype=np.uint8))

    return (
        np.vstack(positions),
        normalize_rows(np.vstack(tangents)),
        np.concatenate(kinds),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--regimes",
        default="period1,period2,period4,period8,chaos",
        help="comma-separated compass-gait regime names",
    )
    parser.add_argument(
        "--n-bridge",
        type=int,
        default=8,
        help="interior bridge samples; 8 matches the stored coarse lifts",
    )
    parser.add_argument("--bridge-height", type=float, default=1.0)
    parser.add_argument(
        "--run-signatures",
        action="store_true",
        help="after export, run the matched Julia signature grid",
    )
    parser.add_argument("--signature-seed", type=int, default=20260512)
    parser.add_argument("--boxsize", type=float, default=0.45)
    parser.add_argument("--r-max", type=float, default=0.45)
    parser.add_argument("--eval-radius", type=float, default=0.1125)
    parser.add_argument("--segment-lengths", default="20:20:800")
    parser.add_argument("--n-runs", type=int, default=150)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    regimes = [item.strip() for item in args.regimes.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bases = []
    for regime in regimes:
        source = args.input_dir / f"compass_{regime}.npz"
        ts = load_hybrid_npz(source)
        positions, tangents, piece_kind = build_analytic_lift(
            ts, args.n_bridge, args.bridge_height
        )
        dt = float(ts.meta["dt"])
        times = np.arange(len(positions), dtype=np.float64) * dt
        base = f"compass_{regime}_exact_suspension"
        bases.append(base)
        np.savetxt(
            args.output_dir / f"{base}_positions.csv", positions, delimiter=" "
        )
        np.savetxt(
            args.output_dir / f"{base}_tangents.csv", tangents, delimiter=" "
        )
        metadata = {
            **ts.meta,
            "control": "analytic_exact_seam_suspension",
            "source": str(source.resolve()),
            "base_embedding": "x -> (x, 0)",
            "bridge": "((1-s)g + s*r(g), h*sin(pi*s))",
            "bridge_height": args.bridge_height,
            "n_bridge_interior": args.n_bridge,
            "n_output_samples": len(positions),
            "time_convention": "each exported sample costs source dt",
        }
        np.savez_compressed(
            args.output_dir / f"{base}.npz",
            t=times,
            x=positions,
            v=tangents,
            piece_kind=piece_kind,
            meta_json=json.dumps(metadata),
        )
        report = [
            "Exact-seam analytic suspension control",
            f"source={source.resolve()}",
            f"samples={len(positions)}",
            f"dimension={positions.shape[1]}",
            f"impacts={len(ts.impact_times)}",
            f"n_bridge_interior={args.n_bridge}",
            f"bridge_height={args.bridge_height}",
            "seam_endpoint_error=0 by construction",
            (
                "interpretation=matched diagnostic control; the sampled arch "
                "is not a claim of a global embedding theorem"
            ),
        ]
        (args.output_dir / f"{base}_report.txt").write_text(
            "\n".join(report) + "\n"
        )
        print(f"wrote {base} to {args.output_dir}")

    if args.run_signatures:
        for base in bases:
            command = [
                "julia",
                f"--project={JULIA_PROJECT}",
                str(JULIA_DRIVER),
                "--data-dir",
                str(args.output_dir),
                "--base",
                base,
                "--boxsize",
                str(args.boxsize),
                "--sb-radius",
                "1",
                "--r-max",
                str(args.r_max),
                "--r-subdivisions",
                "101",
                "--eval-radius",
                str(args.eval_radius),
                "--segment-lengths",
                args.segment_lengths,
                "--n-runs",
                str(args.n_runs),
                "--seed",
                str(args.signature_seed),
                "--max-rank",
                "3",
                "--progress",
                "true",
            ]
            subprocess.run(command, cwd=CODE_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
