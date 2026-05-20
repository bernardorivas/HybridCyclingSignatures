"""Regenerate successful CHyLL v2 lift files from bundled models.

Run from this bundle root:

    python export_all_lifts.py

Outputs are written under:

    regenerated/<example>/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source" / "chyll_v2" / "cycling_signature" / "export"


TASKS = [
    {
        "folder": "rimless_wheel_phaseB",
        "script": SRC / "prepare_rimless_lift.py",
        "base": "continuous_lift_chyll_v2_phaseB",
        "model": ROOT / "models" / "rimless_wheel_phaseB" / "model.pt",
        "config": ROOT / "models" / "rimless_wheel_phaseB" / "config.json",
        "args": ["--n-impacts", "5", "--tangent-source", "diff", "--tangent-mode", "tagaware"],
    },
    {
        "folder": "bouncing_ball_phaseB_5impacts",
        "script": SRC / "prepare_bouncing_ball_lift.py",
        "base": "continuous_lift_chyll_v2_bb_phaseB",
        "model": ROOT / "models" / "bouncing_ball_phaseB" / "model.pt",
        "config": ROOT / "models" / "bouncing_ball_phaseB" / "config.json",
        "args": [
            "--n-impacts",
            "5",
            "--alpha",
            "0.8",
            "--tangent-source",
            "diff",
            "--tangent-mode",
            "tagaware",
        ],
    },
    {
        "folder": "bouncing_ball_phaseB_15impacts",
        "script": SRC / "prepare_bouncing_ball_lift.py",
        "base": "continuous_lift_chyll_v2_bb_phaseB_long",
        "model": ROOT / "models" / "bouncing_ball_phaseB" / "model.pt",
        "config": ROOT / "models" / "bouncing_ball_phaseB" / "config.json",
        "args": [
            "--n-impacts",
            "15",
            "--alpha",
            "0.8",
            "--tangent-source",
            "diff",
            "--tangent-mode",
            "tagaware",
        ],
    },
    {
        "folder": "compass_period2_4p75deg_phi1",
        "script": SRC / "prepare_compass_lift.py",
        "base": "continuous_lift_compass_phi1_vfield",
        "model": ROOT / "models" / "compass_period2_4p75deg_phi1" / "model.pt",
        "config": ROOT / "models" / "compass_period2_4p75deg_phi1" / "config.json",
        "args": [
            "--n-impacts",
            "10",
            "--tangent-source",
            "vfield",
            "--phi",
            "0.08290313946973066",
            "--ic-offset",
            "-0.298886808092,0.298886808092,-0.153332476821,-1.15367407837",
        ],
    },
    {
        "folder": "compass_period4_5p00deg_phi2",
        "script": SRC / "prepare_compass_lift.py",
        "base": "continuous_lift_compass_phi2_vfield",
        "model": ROOT / "models" / "compass_period4_5p00deg_phi2" / "model.pt",
        "config": ROOT / "models" / "compass_period4_5p00deg_phi2" / "config.json",
        "args": [
            "--n-impacts",
            "10",
            "--tangent-source",
            "vfield",
            "--phi",
            "0.08726646259971647",
            "--ic-offset",
            "-0.305081657436,0.305081657437,-0.108006761939,-1.16144488886",
        ],
    },
    {
        "folder": "compass_period8_5p02deg_phi3",
        "script": SRC / "prepare_compass_lift.py",
        "base": "continuous_lift_compass_phi3_vfield",
        "model": ROOT / "models" / "compass_period8_5p02deg_phi3" / "model.pt",
        "config": ROOT / "models" / "compass_period8_5p02deg_phi3" / "config.json",
        "args": [
            "--n-impacts",
            "10",
            "--tangent-source",
            "vfield",
            "--phi",
            "0.08761552845011533",
            "--ic-offset",
            "-0.307031367853,0.307031367852,-0.097321131256,-1.16219899681",
        ],
    },
    {
        "folder": "compass_chaotic_cloud_5p20deg_phi4",
        "script": SRC / "prepare_compass_lift.py",
        "base": "continuous_lift_compass_phi4_cloud_vfield",
        "model": ROOT / "models" / "compass_chaotic_cloud_5p20deg_phi4" / "model.pt",
        "config": ROOT / "models" / "compass_chaotic_cloud_5p20deg_phi4" / "config.json",
        "args": [
            "--n-impacts",
            "10",
            "--tangent-source",
            "vfield",
            "--phi",
            "0.09075712110370514",
            "--ic-offset",
            "-0.310335587349,0.310335587349,-0.0879631222879,-1.17023777335",
        ],
    },
]


def main() -> int:
    for task in TASKS:
        out_dir = ROOT / "regenerated" / task["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(task["script"]),
            "--model",
            str(task["model"]),
            "--config",
            str(task["config"]),
            "--out-dir",
            str(out_dir),
            "--base",
            task["base"],
            *task["args"],
        ]
        print("\n==>", task["folder"])
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
