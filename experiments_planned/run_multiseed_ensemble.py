#!/usr/bin/env python3
"""Plan or explicitly execute the CHyLL v2 multi-seed ensemble.

Without ``--execute`` this script only writes and prints the command plan.
The plan covers rimless Phase A/B, canonical compass phi=0.07, and one
configurable cascade slope, followed by matching lift exports and Julia
cycling-signature runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
PLAN_ROOT = Path(__file__).resolve().parent
TRAIN_RIMLESS = CODE_ROOT / "chyll_v2" / "scripts" / "train_rimless.py"
TRAIN_COMPASS = CODE_ROOT / "chyll_v2" / "scripts" / "train_compass.py"
EXPORT_RIMLESS = (
    CODE_ROOT
    / "chyll_v2"
    / "cycling_signature"
    / "export"
    / "prepare_rimless_lift.py"
)
EXPORT_COMPASS = (
    CODE_ROOT
    / "chyll_v2"
    / "cycling_signature"
    / "export"
    / "prepare_compass_lift.py"
)
EXPORT_MATCHED_CASCADE = (
    CODE_ROOT / "period_doubling" / "export_latent_lifts.py"
)
JULIA_DRIVER = (
    CODE_ROOT / "chyll_v2" / "cycling_signature" / "julia" / "run_subsegments.jl"
)
JULIA_PROJECT = CODE_ROOT / "period_doubling" / "julia"

CASCADE_MATCHED_DESIGN = {
    "phi_1": ("period2", "compass_gait_phi_1_4.75deg"),
    "phi_2": ("period4", "compass_gait_phi_2_5deg"),
    "phi_3": ("period8", "compass_gait_phi_3_5.02deg"),
    "phi_4_cloud": ("chaos", "compass_gait_phi_4_cloud_5.2deg"),
}


def command_record(label: str, seed: int, stage: str, argv: list[str]):
    return {"experiment": label, "seed": seed, "stage": stage, "argv": argv}


def python_command(script: Path, *args: object) -> list[str]:
    return [sys.executable, str(script), *[str(item) for item in args]]


def julia_command(
    data_dir: Path,
    base: str,
    seed: int,
    *,
    boxsize: float,
    r_max: float,
    eval_radius: float,
    segment_lengths: str,
    n_runs: int,
):
    return [
        "julia",
        f"--project={JULIA_PROJECT}",
        str(JULIA_DRIVER),
        "--data-dir",
        str(data_dir),
        "--base",
        base,
        "--boxsize",
        str(boxsize),
        "--sb-radius",
        "1",
        "--segment-lengths",
        segment_lengths,
        "--n-runs",
        str(n_runs),
        "--seed",
        str(20260819 + seed),
        "--r-max",
        str(r_max),
        "--eval-radius",
        str(eval_radius),
        "--max-rank",
        "3",
        "--progress",
        "true",
    ]


def build_plan(
    seeds: list[int],
    output_root: Path,
    device: str,
    cascade_slope: str,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for seed in seeds:
        seed_tag = f"seed_{seed:03d}"

        rimless_root = output_root / "rimless" / seed_tag
        phase_a = rimless_root / "phase_a"
        phase_b = rimless_root / "phase_b"
        rimless_lift = rimless_root / "lift"
        plan.append(
            command_record(
                "rimless",
                seed,
                "train_phase_a",
                python_command(
                    TRAIN_RIMLESS,
                    "--seed",
                    seed,
                    "--device",
                    device,
                    "--w-v",
                    0.0,
                    "--run-dir",
                    phase_a,
                    "--figure-dir",
                    phase_a / "figures",
                ),
            )
        )
        plan.append(
            command_record(
                "rimless",
                seed,
                "train_phase_b",
                python_command(
                    TRAIN_RIMLESS,
                    "--seed",
                    seed,
                    "--device",
                    device,
                    "--w-v",
                    1.0,
                    "--load-from",
                    phase_a / "model.pt",
                    "--curriculum-horizons",
                    "50,100",
                    "--steps-per-horizon",
                    500,
                    "--run-dir",
                    phase_b,
                    "--figure-dir",
                    phase_b / "figures",
                ),
            )
        )
        rimless_base = f"rimless_{seed_tag}"
        plan.append(
            command_record(
                "rimless",
                seed,
                "export",
                python_command(
                    EXPORT_RIMLESS,
                    "--model",
                    phase_b / "model.pt",
                    "--config",
                    phase_b / "config.json",
                    "--out-dir",
                    rimless_lift,
                    "--base",
                    rimless_base,
                    "--tangent-mode",
                    "tagaware",
                ),
            )
        )
        plan.append(
            command_record(
                "rimless",
                seed,
                "signature",
                julia_command(
                    rimless_lift,
                    rimless_base,
                    seed,
                    boxsize=0.3,
                    r_max=0.002,
                    eval_radius=0.001,
                    segment_lengths="20:10:300",
                    n_runs=100,
                ),
            )
        )

        phi007_root = output_root / "phi007" / seed_tag
        phi007_run = phi007_root / "run"
        phi007_lift = phi007_root / "lift"
        plan.append(
            command_record(
                "phi007",
                seed,
                "train",
                python_command(
                    TRAIN_COMPASS,
                    "--seed",
                    seed,
                    "--device",
                    device,
                    "--run-dir",
                    phi007_run,
                    "--figure-dir",
                    phi007_run / "figures",
                ),
            )
        )
        phi007_base = f"compass_phi007_{seed_tag}"
        plan.append(
            command_record(
                "phi007",
                seed,
                "export",
                python_command(
                    EXPORT_COMPASS,
                    "--model",
                    phi007_run / "model.pt",
                    "--config",
                    phi007_run / "config.json",
                    "--out-dir",
                    phi007_lift,
                    "--base",
                    phi007_base,
                    "--tangent-mode",
                    "tagaware",
                ),
            )
        )
        plan.append(
            command_record(
                "phi007",
                seed,
                "signature",
                julia_command(
                    phi007_lift,
                    phi007_base,
                    seed,
                    boxsize=0.3,
                    r_max=0.5,
                    eval_radius=0.05,
                    segment_lengths="20:10:600",
                    n_runs=200,
                ),
            )
        )

        regime, model_run_name = CASCADE_MATCHED_DESIGN[cascade_slope]
        cascade_root = output_root / cascade_slope / seed_tag
        matched_runs = cascade_root / "runs"
        cascade_run = matched_runs / model_run_name
        cascade_lift = cascade_root / "lift"
        plan.append(
            command_record(
                cascade_slope,
                seed,
                "train",
                python_command(
                    TRAIN_COMPASS,
                    "--seed",
                    seed,
                    "--device",
                    device,
                    "--slope-config",
                    cascade_slope,
                    "--run-dir",
                    cascade_run,
                    "--figure-dir",
                    cascade_run / "figures",
                ),
            )
        )
        plan.append(
            command_record(
                cascade_slope,
                seed,
                "export_matched_fixed_data",
                python_command(
                    EXPORT_MATCHED_CASCADE,
                    "--runs-dir",
                    matched_runs,
                    "--regimes",
                    regime,
                    "--out-dir",
                    cascade_lift,
                ),
            )
        )
        plan.append(
            command_record(
                cascade_slope,
                seed,
                "signature",
                julia_command(
                    cascade_lift,
                    f"compass_{regime}",
                    seed,
                    boxsize=0.45,
                    r_max=0.45,
                    eval_radius=0.1125,
                    segment_lengths="20:20:800",
                    n_runs=150,
                ),
            )
        )
    return plan


def summarize(plan: list[dict[str, object]], output_root: Path) -> None:
    rows = []
    for record in plan:
        if record["stage"] != "signature":
            continue
        argv = record["argv"]
        data_dir = Path(argv[argv.index("--data-dir") + 1])
        base = argv[argv.index("--base") + 1]
        n_runs = int(argv[argv.index("--n-runs") + 1])
        path = data_dir / f"subsegments_{base}_rank_at_radius.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            data = list(csv.DictReader(handle))
        onset = None
        for row in data:
            rank1 = int(row.get("rank1", 0))
            if not 0 <= rank1 <= n_runs:
                raise ValueError(f"invalid rank1 count in {path}: {rank1}")
            rank1_fraction = rank1 / n_runs
            if rank1_fraction >= 0.95:
                onset = int(row["segment_length"])
                break
        last = data[-1]
        final_rank1 = int(last.get("rank1", 0))
        if not 0 <= final_rank1 <= n_runs:
            raise ValueError(f"invalid final rank1 count in {path}: {final_rank1}")
        rows.append(
            {
                "experiment": record["experiment"],
                "seed": record["seed"],
                "rank1_onset_95pct": onset,
                "final_segment_length": int(last["segment_length"]),
                "final_rank1_fraction": final_rank1 / n_runs,
                "source": str(path),
            }
        )
    if not rows:
        print("no completed signature outputs found; summary not written")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "ensemble_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cascade-slope",
        choices=tuple(CASCADE_MATCHED_DESIGN),
        default="phi_3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PLAN_ROOT / "outputs" / "multiseed",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run every training/export/signature command; default is plan only",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="summarize any already-completed outputs without running commands",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain distinct integers")
    plan = build_plan(seeds, args.output_root, args.device, args.cascade_slope)
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    plan_path = PLAN_ROOT / "multiseed-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote {len(plan)} commands to {plan_path}")
    for record in plan:
        label = f"{record['experiment']} seed={record['seed']} {record['stage']}"
        print(f"[{label}] {shlex.join(record['argv'])}")

    if args.summarize_existing:
        summarize(plan, args.output_root)
    if not args.execute:
        print("plan only; no training, export, or Julia analysis was launched")
        return 0
    for record in plan:
        subprocess.run(record["argv"], cwd=CODE_ROOT, check=True)
    summarize(plan, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
