#!/usr/bin/env python
"""Generate period-doubling timeseries for Roessler and Compass Gait systems.

Generates timeseries for all periodic and chaotic regimes, detects periodicity,
validates against expected periods, and writes lifts to CSV/NPZ formats for
downstream analysis.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Insert this package into sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import roessler
import compass
import periodicity
import export_lifts


ROESSLER_REGIME_ORDER = ["period1", "period2", "period4", "period8", "chaos"]
COMPASS_REGIME_ORDER = ["period1", "period2", "period4", "period8", "chaos"]


def get_roessler_params_for_summary(regime):
    """Return parameter string for summary table (e.g., 'c=4.0')."""
    return f"c={regime.c:.1f}"


def get_compass_params_for_summary(regime):
    """Return parameter string for summary table (e.g., 'phi=4.00deg')."""
    return f"phi={regime.phi_deg:.2f}deg"


def generate_and_process_roessler(regime, t_span, dt, burn_in, quick=False):
    """Generate Roessler timeseries and detect period.

    Returns
    -------
    tuple
        (ts, elapsed_sec, n_samples, n_impacts, detected_clusters, is_ok)
    """
    t0 = time.time()
    ts = roessler.generate_timeseries(
        regime,
        t_span=t_span,
        dt=dt,
        burn_in=burn_in,
    )
    elapsed = time.time() - t0

    n_samples = len(ts.t)
    n_impacts = None

    # Detect period
    detection = periodicity.detect_roessler_period(ts)
    n_clusters = detection["n_clusters"]

    # Check against expected
    is_ok = periodicity.check_period(n_clusters, regime.expected_period)

    return ts, elapsed, n_samples, n_impacts, n_clusters, is_ok


def generate_and_process_compass(regime, t_span, dt, burn_in_strides, quick=False):
    """Generate Compass timeseries and detect period.

    Returns
    -------
    tuple
        (ts, elapsed_sec, n_samples, n_impacts, detected_clusters, is_ok)
    """
    t0 = time.time()
    ts = compass.generate_timeseries(
        regime,
        t_span=t_span,
        dt=dt,
        burn_in_strides=burn_in_strides,
    )
    elapsed = time.time() - t0

    n_samples = len(ts.t)
    n_impacts = len(ts.impact_times) if hasattr(ts, "impact_times") else 0

    # Detect period
    detection = periodicity.detect_compass_period(ts)
    n_clusters = detection["n_clusters"]

    # Check against expected
    is_ok = periodicity.check_period(n_clusters, regime.expected_period)

    return ts, elapsed, n_samples, n_impacts, n_clusters, is_ok


def main():
    parser = argparse.ArgumentParser(
        description="Generate period-doubling timeseries for analysis."
    )
    parser.add_argument(
        "--system",
        choices=["roessler", "compass", "both"],
        default="both",
        help="Which system(s) to generate.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick parameters (shorter t_span and burn_in).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root directory. Defaults to {package}/data.",
    )
    parser.add_argument(
        "--t-span",
        type=float,
        default=None,
        help="Override t_span for all systems.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override dt for all systems.",
    )

    args = parser.parse_args()

    # Determine output root
    if args.out_root is None:
        out_root = Path(__file__).resolve().parent / "data"
    else:
        out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    # Collect results for summary table
    summary_rows = []

    # ==================== Roessler ====================
    if args.system in ("roessler", "both"):
        roessler_dir = out_root / "roessler"
        roessler_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating Roessler regime series...")

        for regime_key in ROESSLER_REGIME_ORDER:
            regime = roessler.ROESSLER_REGIMES[regime_key]

            # Determine parameters
            if args.quick:
                t_span = 200.0
                burn_in = 1000.0
            else:
                t_span = 500.0
                burn_in = 2000.0

            if args.t_span is not None:
                t_span = args.t_span
            if args.dt is not None:
                dt = args.dt
            else:
                dt = 0.02

            ts, elapsed, n_samples, n_impacts, n_clusters, is_ok = (
                generate_and_process_roessler(regime, t_span, dt, burn_in, quick=args.quick)
            )

            parameter = get_roessler_params_for_summary(regime)
            ok_str = "OK" if is_ok else "MISMATCH"

            # Write the validated series (no re-generation)
            base = f"roessler_{regime_key}"
            export_lifts.write_lift(roessler_dir, base, ts)

            dt_actual = ts.t[1] - ts.t[0] if len(ts.t) > 1 else np.nan

            print(
                f"  {regime.label:12s} {parameter:10s} "
                f"t={elapsed:6.2f}s n_samples={n_samples:6d} "
                f"clusters={n_clusters:3d} (expected {regime.expected_period}) {ok_str}"
            )

            summary_rows.append({
                "system": "roessler",
                "regime": regime_key,
                "parameter": parameter,
                "expected_period": regime.expected_period if regime.expected_period is not None else "",
                "detected_clusters": n_clusters,
                "ok": "TRUE" if is_ok else "FALSE",
                "n_samples": n_samples,
                "dt": dt_actual,
                "t_span": t_span,
                "n_impacts": n_impacts if n_impacts is not None else "",
            })

            # Check for periodic regime mismatch
            if regime.expected_period is not None and not is_ok:
                print(f"ERROR: Periodic regime '{regime_key}' mismatch!")
                sys.exit(1)

    # ==================== Compass Gait ====================
    if args.system in ("compass", "both"):
        compass_dir = out_root / "compass_gait"
        compass_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating Compass Gait regime series...")

        for regime_key in COMPASS_REGIME_ORDER:
            regime = compass.COMPASS_REGIMES[regime_key]

            # Determine parameters
            if args.quick:
                t_span = 120.0
                burn_in_strides = 40
            else:
                t_span = 400.0
                burn_in_strides = 80

            if args.t_span is not None:
                t_span = args.t_span
            if args.dt is not None:
                dt = args.dt
            else:
                dt = 0.02

            ts, elapsed, n_samples, n_impacts, n_clusters, is_ok = (
                generate_and_process_compass(regime, t_span, dt, burn_in_strides, quick=args.quick)
            )

            parameter = get_compass_params_for_summary(regime)
            ok_str = "OK" if is_ok else "MISMATCH"

            # Write the validated series (no re-generation)
            base = f"compass_{regime_key}"
            export_lifts.write_lift(compass_dir, base, ts)

            dt_actual = ts.t[1] - ts.t[0] if len(ts.t) > 1 else np.nan

            print(
                f"  {regime.label:12s} {parameter:12s} "
                f"t={elapsed:6.2f}s n_samples={n_samples:6d} n_impacts={n_impacts:4d} "
                f"clusters={n_clusters:3d} (expected {regime.expected_period}) {ok_str}"
            )

            summary_rows.append({
                "system": "compass",
                "regime": regime_key,
                "parameter": parameter,
                "expected_period": regime.expected_period if regime.expected_period is not None else "",
                "detected_clusters": n_clusters,
                "ok": "TRUE" if is_ok else "FALSE",
                "n_samples": n_samples,
                "dt": dt_actual,
                "t_span": t_span,
                "n_impacts": n_impacts,
            })

            # Check for periodic regime mismatch
            if regime.expected_period is not None and not is_ok:
                print(f"ERROR: Periodic regime '{regime_key}' mismatch!")
                sys.exit(1)

    # Write summary CSV
    summary_csv = out_root / "summary.csv"
    with open(summary_csv, "w") as f:
        # Header
        header = [
            "system",
            "regime",
            "parameter",
            "expected_period",
            "detected_clusters",
            "ok",
            "n_samples",
            "dt",
            "t_span",
            "n_impacts",
        ]
        f.write(",".join(header) + "\n")

        # Rows
        for row in summary_rows:
            values = [str(row.get(h, "")) for h in header]
            f.write(",".join(values) + "\n")

    print(f"\nWrote summary to {summary_csv}")

    # Print aligned summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    # Calculate column widths
    col_widths = {
        "system": max(6, max(len(str(row["system"])) for row in summary_rows)),
        "regime": max(6, max(len(str(row["regime"])) for row in summary_rows)),
        "parameter": max(9, max(len(str(row["parameter"])) for row in summary_rows)),
        "expected": max(8, max(len(str(row["expected_period"])) for row in summary_rows)),
        "detected": max(8, max(len(str(row["detected_clusters"])) for row in summary_rows)),
        "ok": 2,
        "n_samples": 9,
        "n_impacts": 9,
    }

    # Print header
    fmt = (
        f"  {{system:<{col_widths['system']}}}  "
        f"{{regime:<{col_widths['regime']}}}  "
        f"{{parameter:<{col_widths['parameter']}}}  "
        f"{{expected:>{col_widths['expected']}}}  "
        f"{{detected:>{col_widths['detected']}}}  "
        f"{{ok:>{col_widths['ok']}}}  "
        f"{{n_samples:>{col_widths['n_samples']}}}  "
        f"{{n_impacts:>{col_widths['n_impacts']}}}"
    )

    header_str = fmt.format(
        system="System",
        regime="Regime",
        parameter="Parameter",
        expected="Expected",
        detected="Detected",
        ok="OK",
        n_samples="N Samples",
        n_impacts="N Impacts",
    )
    print(header_str)
    print("-" * len(header_str))

    # Print rows
    for row in summary_rows:
        print(
            fmt.format(
                system=row["system"],
                regime=row["regime"],
                parameter=row["parameter"],
                expected=row["expected_period"],
                detected=row["detected_clusters"],
                ok="Y" if row["ok"] == "TRUE" else "N",
                n_samples=row["n_samples"],
                n_impacts=row["n_impacts"],
            )
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
