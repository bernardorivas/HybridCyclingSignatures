"""Train the CHyLL v2 baseline on the bouncing ball (mapping-cylinder data).

Mirrors ``train_rimless.py`` but for the bouncing-ball system. Default
``w_v = 1.0`` (the canonical two-seam form has been validated on rimless
and is the framework's standard configuration going forward).

Usage:
    python chyll_v2/scripts/train_bouncing_ball.py [--smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``chyll_v2`` importable as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chyll_v2.chyll_v2.config import make_default
from chyll_v2.chyll_v2.data import generate_trajectories
from chyll_v2.chyll_v2.systems.bouncing_ball import BouncingBall
from chyll_v2.chyll_v2.train import train
from chyll_v2.chyll_v2.visualize import plot_loss_history, plot_rollout_vs_truth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny curriculum for sanity")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the deterministic training/data seed")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ode-method", type=str, default=None)
    parser.add_argument("--no-adjoint", action="store_true")
    parser.add_argument("--w-v", type=float, default=1.0,
                        help="seam-velocity weight (default 1.0 = two-seam canonical)")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="coefficient of restitution (1.0 = perfectly elastic)")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--figure-dir", type=str, default=None)
    parser.add_argument(
        "--load-from", type=str, default=None,
        help="path to a saved state_dict to initialise from (Phase-B fine-tune mode)",
    )
    parser.add_argument(
        "--curriculum-horizons", type=str, default=None,
        help="comma-separated override, e.g. '50,100'",
    )
    parser.add_argument(
        "--steps-per-horizon", type=int, default=None,
    )
    args = parser.parse_args()

    cfg = make_default("bouncing_ball")
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.w_v = args.w_v
    if args.device is not None:
        cfg.device = args.device
    if args.ode_method is not None:
        cfg.ode_method = args.ode_method
    if args.no_adjoint:
        cfg.ode_use_adjoint = False
    if args.run_dir is not None:
        cfg.run_dir = args.run_dir
    if args.figure_dir is not None:
        cfg.figure_dir = args.figure_dir
    if args.curriculum_horizons is not None:
        cfg.curriculum_horizons = tuple(int(t) for t in args.curriculum_horizons.split(","))
    if args.steps_per_horizon is not None:
        cfg.steps_per_horizon = args.steps_per_horizon
    if args.smoke:
        cfg.n_trajectories = 16
        cfg.trajectory_steps = 200
        cfg.curriculum_horizons = (5, 10)
        cfg.steps_per_horizon = 30
        cfg.batch_size = 8
        cfg.log_every = 5
        cfg.save_every = 0
        cfg.ode_method = "rk4"
        cfg.ode_use_adjoint = False
        cfg.run_dir = "chyll_v2/runs/bouncing_ball_smoke"
        cfg.figure_dir = "chyll_v2/figures/bouncing_ball_smoke"

    print(f"[chyll_v2] system={cfg.system_name} device={cfg.device} seed={cfg.seed}")
    print(
        f"[chyll_v2] base_dim={cfg.base_dim} state_dim(aug)={cfg.state_dim} "
        f"latent_dim={cfg.latent_dim}"
    )
    print(
        f"[chyll_v2] ode_method={cfg.ode_method} adjoint={cfg.ode_use_adjoint} "
        f"tau={cfg.tau}  w_v={cfg.w_v}"
    )
    print(
        f"[chyll_v2] curriculum={cfg.curriculum_horizons} "
        f"steps/h={cfg.steps_per_horizon}"
    )

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
    on_cyl = sum(int((t.states[:, -1] > 0).sum()) for t in trajs)
    n_total = sum(len(t) for t in trajs)
    print(
        f"[chyll_v2] {len(trajs)} trajectories x {cfg.trajectory_steps} steps; "
        f"cylinder fraction = {on_cyl / max(1, n_total):.3f}"
    )

    nets = train(cfg=cfg, system=system, trajectories=trajs, load_from=args.load_from)

    plot_loss_history(
        jsonl_path=Path(cfg.run_dir) / "train_log.jsonl",
        out_path=Path(cfg.figure_dir) / "loss_history.png",
    )
    plot_rollout_vs_truth(
        cfg=cfg,
        nets=nets,
        trajectory=trajs[0],
        out_path=Path(cfg.figure_dir) / "rollout_vs_truth.png",
        rollout_horizon=cfg.trajectory_steps - 1,
    )
    print(f"[chyll_v2] done; artifacts in {cfg.run_dir} and {cfg.figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
