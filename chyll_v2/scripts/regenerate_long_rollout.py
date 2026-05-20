"""Regenerate a long-horizon truth-vs-rollout figure for a saved checkpoint.

Loads a CHyLL v2 run's config + model, simulates one ground-truth trajectory
at the requested length, runs the learned latent rollout for the full
horizon, and writes ``rollout_vs_truth_long.png`` next to the run's other
figures. Defaults are tuned for the rimless wheel Phase-B fine-tune.

Usage:
    python chyll_v2/scripts/regenerate_long_rollout.py \
        --run-dir chyll_v2/runs/rimless_wheel_phaseB_finetune \
        --figure-dir chyll_v2/figures/rimless_wheel_phaseB_finetune \
        --system rimless_wheel --steps 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from chyll_v2.chyll_v2.config import CHyLLv2Config, make_default  # noqa: E402
from chyll_v2.chyll_v2.data import generate_trajectories  # noqa: E402
from chyll_v2.chyll_v2.networks import CHyLLv2Networks  # noqa: E402
from chyll_v2.chyll_v2.systems.bouncing_ball import BouncingBall  # noqa: E402
from chyll_v2.chyll_v2.systems.rimless_wheel import RimlessWheel  # noqa: E402
from chyll_v2.chyll_v2.train import resolve_device  # noqa: E402
from chyll_v2.chyll_v2.visualize import plot_rollout_vs_truth  # noqa: E402


def load_config(path: Path) -> CHyLLv2Config:
    payload = json.loads(path.read_text())
    allowed = {f.name for f in fields(CHyLLv2Config)}
    payload = {k: v for k, v in payload.items() if k in allowed}
    return CHyLLv2Config(**payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument(
        "--system", choices=("rimless_wheel", "bouncing_ball"), required=True,
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-name", type=str, default="rollout_vs_truth_long.png")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="bouncing-ball coefficient of restitution")
    args = parser.parse_args()

    cfg = load_config(args.run_dir / "config.json")
    cfg.n_trajectories = 1
    cfg.trajectory_steps = args.steps
    cfg.seed = args.seed

    if args.system == "rimless_wheel":
        system = RimlessWheel()
    else:
        system = BouncingBall(alpha=args.alpha)

    trajs = generate_trajectories(
        system=system,
        n_trajectories=cfg.n_trajectories,
        tau=cfg.tau,
        trajectory_steps=cfg.trajectory_steps,
        seed=cfg.seed,
        sim_rtol=cfg.sim_rtol,
        sim_atol=cfg.sim_atol,
    )
    traj = trajs[0]
    print(f"truth trajectory length: {len(traj)}")

    device = resolve_device(cfg.device)
    nets = CHyLLv2Networks(cfg).to(device)
    state = torch.load(args.run_dir / "model.pt", map_location=device, weights_only=True)
    nets.load_state_dict(state)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.figure_dir / args.out_name
    plot_rollout_vs_truth(
        cfg=cfg, nets=nets, trajectory=traj, out_path=out_path,
        rollout_horizon=len(traj) - 1,
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
