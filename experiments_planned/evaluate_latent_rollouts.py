#!/usr/bin/env python3
"""Evaluate learned-latent-flow rollouts on held-out initial conditions.

This is a prepared experiment, not part of the stored-results pipeline.  It
integrates the learned autonomous latent vector field, decodes the result for
hybrid-event diagnostics, and exports each latent trajectory in the
position/tangent CSV format consumed by ``run_subsegments.jl``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from chyll_v2.chyll_v2.config import CHyLLv2Config  # noqa: E402
from chyll_v2.chyll_v2.networks import CHyLLv2Networks  # noqa: E402
from chyll_v2.chyll_v2.systems.compass_gait import CompassGait  # noqa: E402
from chyll_v2.chyll_v2.systems.compass_gait_slope_configs import (  # noqa: E402
    GOSWAMI_COMPASS_SLOPE_CONFIGS,
)
from chyll_v2.chyll_v2.systems.rimless_wheel import RimlessWheel  # noqa: E402


def load_config(path: Path) -> CHyLLv2Config:
    payload = json.loads(path.read_text())
    allowed = {item.name for item in fields(CHyLLv2Config)}
    return CHyLLv2Config(**{k: v for k, v in payload.items() if k in allowed})


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_system(cfg: CHyLLv2Config, slope_name: str | None):
    if cfg.system_name == "rimless_wheel":
        return RimlessWheel(), 1
    if cfg.system_name != "compass_gait":
        raise ValueError(
            "planned rollout evaluation currently supports rimless_wheel and "
            "compass_gait"
        )

    saved_slope = cfg.system_params.get("slope_config")
    slope_name = slope_name or saved_slope
    if slope_name is None:
        return CompassGait(phi=0.07), 1
    slope_cfg = GOSWAMI_COMPASS_SLOPE_CONFIGS[slope_name]
    return CompassGait(slope_config=slope_cfg), slope_cfg.expected_period


def gluing_event_indices(
    states: np.ndarray,
    s_high: float,
    s_low: float,
) -> np.ndarray:
    """Indices immediately after a decoded cylinder-top to base transition."""
    s = states[:, -1]
    return np.flatnonzero((s[:-1] >= s_high) & (s[1:] <= s_low)) + 1


def infer_return_period(
    postimpact_states: np.ndarray,
    candidates: tuple[int, ...] = (1, 2, 4, 8),
    relative_slack: float = 0.1,
    absolute_slack: float = 1e-3,
) -> tuple[int | None, dict[str, float]]:
    """Choose the smallest lag statistically tied with the best recurrence."""
    scores: dict[str, float] = {}
    for lag in candidates:
        if len(postimpact_states) < 2 * lag + 1:
            continue
        delta = postimpact_states[lag:] - postimpact_states[:-lag]
        scores[str(lag)] = float(np.median(np.linalg.norm(delta, axis=1)))
    if not scores:
        return None, scores
    best = min(scores.values())
    cutoff = best * (1.0 + relative_slack) + absolute_slack
    inferred = min(int(lag) for lag, score in scores.items() if score <= cutoff)
    return inferred, scores


def unit_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms < eps):
        raise ValueError("latent vector field produced a near-zero tangent")
    return values / norms[:, None]


def timing_metrics(
    truth_times: np.ndarray,
    predicted_times: np.ndarray,
) -> dict[str, float | int | None]:
    matched = min(len(truth_times), len(predicted_times))
    if matched:
        errors = np.abs(truth_times[:matched] - predicted_times[:matched])
        mean_error: float | None = float(errors.mean())
        max_error: float | None = float(errors.max())
    else:
        mean_error = None
        max_error = None
    return {
        "truth_impact_count": int(len(truth_times)),
        "predicted_impact_count": int(len(predicted_times)),
        "impact_count_error": int(len(predicted_times) - len(truth_times)),
        "matched_impact_count": int(matched),
        "mean_abs_impact_time_error": mean_error,
        "max_abs_impact_time_error": max_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="CHyLL v2 run containing config.json and model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: experiments_planned/outputs/latent_rollouts/<run-name>",
    )
    parser.add_argument(
        "--slope-config",
        choices=tuple(GOSWAMI_COMPASS_SLOPE_CONFIGS),
        default=None,
        help="override/infer the compass-gait slope record",
    )
    parser.add_argument("--expected-period", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--n-trajectories", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--held-out-seed", type=int, default=20260819)
    parser.add_argument(
        "--period-selection-rtol",
        type=float,
        default=0.1,
        help="relative slack for selecting the smallest near-best return lag",
    )
    parser.add_argument(
        "--period-selection-atol",
        type=float,
        default=1e-3,
        help="absolute state-distance slack for near-best return lags",
    )
    parser.add_argument(
        "--allow-training-seed",
        action="store_true",
        help="permit held-out seed to equal the saved training seed",
    )
    parser.add_argument("--s-high", type=float, default=0.9)
    parser.add_argument("--s-low", type=float, default=0.1)
    parser.add_argument(
        "--state-bound",
        type=float,
        default=10.0,
        help="absolute bound for all decoded physical state coordinates",
    )
    parser.add_argument(
        "--compass-angle-bound",
        type=float,
        default=1.5,
        help="additional absolute bound on compass-gait angles",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Keep ``--help`` usable in lean environments.  A real rollout requires
    # the pinned torchdiffeq dependency declared in code/requirements.txt.
    from chyll_v2.chyll_v2.ode import make_ode_func, rollout

    run_dir = args.run_dir.resolve()
    cfg = load_config(run_dir / "config.json")
    if args.held_out_seed == cfg.seed and not args.allow_training_seed:
        raise ValueError(
            "--held-out-seed equals the saved training seed; choose a distinct "
            "seed or pass --allow-training-seed explicitly"
        )
    if args.n_trajectories < 1 or args.steps < 3:
        raise ValueError("need at least one trajectory and three rollout steps")
    if args.period_selection_rtol < 0 or args.period_selection_atol < 0:
        raise ValueError("period-selection tolerances must be nonnegative")

    cfg.device = "cpu"
    cfg.ode_use_adjoint = False
    torch.manual_seed(args.held_out_seed)
    np.random.seed(args.held_out_seed)
    nets = CHyLLv2Networks(cfg)
    nets.load_state_dict(load_checkpoint(run_dir / "model.pt"))
    nets.eval()
    ode_func = make_ode_func(nets.vfield)

    system, saved_period = build_system(cfg, args.slope_config)
    expected_period = args.expected_period or saved_period
    trajectories = system.generate_dataset(
        n_trajectories=args.n_trajectories,
        tau=cfg.tau,
        n_steps=args.steps,
        seed=args.held_out_seed,
        rtol=cfg.sim_rtol,
        atol=cfg.sim_atol,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent
            / "outputs"
            / "latent_rollouts"
            / run_dir.name
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    times_t = torch.arange(args.steps, dtype=torch.float32) * cfg.tau
    rows: list[dict[str, object]] = []
    bounded_count = 0
    period_correct_count = 0
    period_evaluable_count = 0

    for index, truth in enumerate(trajectories):
        x0 = torch.as_tensor(truth.states[0:1], dtype=torch.float32)
        with torch.no_grad():
            z0 = nets.encoder(x0)
            z = rollout(ode_func, z0, times_t, cfg)[:, 0, :]
            decoded = nets.decoder(z)
            velocity = nets.vfield(z)
        z_np = z.cpu().numpy().astype(np.float64)
        decoded_np = decoded.cpu().numpy().astype(np.float64)
        tangents = unit_rows(velocity.cpu().numpy().astype(np.float64))

        truth_events = gluing_event_indices(
            truth.states, args.s_high, args.s_low
        )
        predicted_events = gluing_event_indices(
            decoded_np, args.s_high, args.s_low
        )
        truth_event_times = truth.times[truth_events]
        predicted_event_times = truth.times[predicted_events]
        metrics = timing_metrics(truth_event_times, predicted_event_times)

        physical = decoded_np[:, : cfg.base_dim]
        bounded = bool(
            np.isfinite(decoded_np).all()
            and np.max(np.abs(physical)) <= args.state_bound
        )
        if cfg.system_name == "compass_gait":
            bounded = bounded and bool(
                np.max(np.abs(physical[:, :2])) <= args.compass_angle_bound
            )
        bounded_count += int(bounded)

        postimpact = physical[predicted_events]
        inferred_period, period_scores = infer_return_period(
            postimpact,
            relative_slack=args.period_selection_rtol,
            absolute_slack=args.period_selection_atol,
        )
        period_correct: bool | None = None
        if expected_period is not None and inferred_period is not None:
            period_correct = inferred_period == expected_period
            period_evaluable_count += 1
            period_correct_count += int(period_correct)

        base = f"latent_rollout_{index:03d}"
        np.savetxt(output_dir / f"{base}_positions.csv", z_np, delimiter=" ")
        np.savetxt(output_dir / f"{base}_tangents.csv", tangents, delimiter=" ")
        metadata = {
            "run_dir": str(run_dir),
            "checkpoint": str(run_dir / "model.pt"),
            "config": str(run_dir / "config.json"),
            "training_seed": cfg.seed,
            "held_out_seed": args.held_out_seed,
            "trajectory_index": index,
            "tau": cfg.tau,
            "expected_period": expected_period,
            "inferred_period": inferred_period,
            "period_scores": period_scores,
            "period_selection_rule": (
                "smallest lag within best*(1+rtol)+atol"
            ),
            "period_selection_rtol": args.period_selection_rtol,
            "period_selection_atol": args.period_selection_atol,
            "bounded": bounded,
            **metrics,
        }
        np.savez_compressed(
            output_dir / f"{base}.npz",
            times=truth.times,
            truth_states=truth.states,
            latent_states=z_np,
            decoded_states=decoded_np,
            latent_tangents=tangents,
            truth_event_indices=truth_events,
            predicted_event_indices=predicted_events,
            meta_json=json.dumps(metadata),
        )
        row = {
            "base": base,
            "bounded": bounded,
            "expected_period": expected_period,
            "inferred_period": inferred_period,
            "period_correct": period_correct,
            **metrics,
        }
        rows.append(row)

    manifest_path = output_dir / "rollout_metrics.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_dir": str(run_dir),
        "training_seed": cfg.seed,
        "held_out_seed": args.held_out_seed,
        "n_trajectories": args.n_trajectories,
        "steps": args.steps,
        "tau": cfg.tau,
        "bounded_rollout_rate": bounded_count / args.n_trajectories,
        "expected_period": expected_period,
        "period_selection_rule": "smallest lag within best*(1+rtol)+atol",
        "period_selection_rtol": args.period_selection_rtol,
        "period_selection_atol": args.period_selection_atol,
        "return_period_evaluable_count": period_evaluable_count,
        "return_period_accuracy": (
            period_correct_count / period_evaluable_count
            if period_evaluable_count
            else None
        ),
        "signature_status": (
            "position/tangent inputs exported; Julia signature analysis not run"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(rows)} held-out rollout exports to {output_dir}")
    print("Julia cycling-signature analysis remains a separate, unrun stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
