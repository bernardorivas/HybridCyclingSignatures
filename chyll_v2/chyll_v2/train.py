"""Training loop (Algorithm 1 of Sangli et al.) on mapping-cylinder data.

A single stream of supervised data drives every gradient step: length-
``horizon`` slices of the tau-semiflow trajectory in ``X'``. Each slice
carries a boolean ``glue_mask`` for gluing exits and a ``seam_mask`` for
both cylinder-entry and gluing-exit seams. ``L_g`` consumes ``glue_mask``;
``L_v`` consumes ``seam_mask``; ``L_x``, ``L_z``, ``L_c`` consume the slice
itself.

The latent flow is integrated by ``torchdiffeq.odeint_adjoint``. The
integration times are the real times of the cylinder trajectory --
``[0, tau, 2 tau, ..., (horizon - 1) tau]`` -- so the Neural ODE
learns the same time scale as the data.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import CHyLLv2Config
from .data import make_slice_loader
from .losses import LossOutputs, compute_losses
from .networks import CHyLLv2Networks
from .ode import make_ode_func, rollout
from .systems.base import BaseHybridSystem, Trajectory


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if (
        requested == "mps"
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def seed_training(seed: int) -> None:
    """Apply one config seed to all RNGs used by training and loading."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "cudnn", None) is not None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def forward_step(
    nets: CHyLLv2Networks,
    cfg: CHyLLv2Config,
    states: torch.Tensor,           # (B, T, n + 1)
    glue_mask: torch.Tensor,        # (B, T - 1) bool
    seam_mask: torch.Tensor,        # (B, T - 1) bool
    ode_func,
    integration_times: torch.Tensor,# (T,)
) -> LossOutputs:
    # 1) Encode all observed augmented states.
    z_encoded = nets.encoder(states)                         # (B, T, m)
    # 2) Integrate the latent flow from z_0 at the trajectory times.
    z0 = z_encoded[:, 0, :]
    z_traj = rollout(
        ode_func=ode_func, z0=z0, times=integration_times, cfg=cfg
    )                                                         # (T, B, m)
    z_predicted = z_traj.transpose(0, 1).contiguous()        # (B, T, m)
    # 3) Decode the integrated latent back to augmented state space.
    states_decoded = nets.decoder(z_predicted)               # (B, T, n + 1)
    return compute_losses(
        cfg=cfg,
        states=states,
        z_predicted=z_predicted,
        z_encoded=z_encoded,
        states_decoded=states_decoded,
        glue_mask=glue_mask,
        seam_mask=seam_mask,
    )


def train(
    cfg: CHyLLv2Config,
    system: BaseHybridSystem,
    trajectories: List[Trajectory],
    load_from: str | None = None,
) -> CHyLLv2Networks:
    seed_training(cfg.seed)
    device = resolve_device(cfg.device)
    nets = CHyLLv2Networks(cfg).to(device)
    if load_from is not None:
        state = torch.load(load_from, map_location=device, weights_only=True)
        nets.load_state_dict(state)
        print(f"[chyll_v2] initialised from checkpoint: {load_from}", flush=True)
    ode_func = make_ode_func(nets.vfield).to(device)

    optim = AdamW(nets.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_updates = cfg.steps_per_horizon * len(cfg.curriculum_horizons)
    scheduler = (
        CosineAnnealingLR(optim, T_max=total_updates)
        if cfg.cosine_schedule
        else None
    )

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"
    log_file = log_path.open("w")
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str)
    )

    global_step = 0
    t_start = time.time()

    try:
        for horizon in cfg.curriculum_horizons:
            loader = make_slice_loader(
                trajectories=trajectories,
                horizon=horizon,
                batch_size=cfg.batch_size,
                s_high=cfg.glue_s_high,
                s_low=cfg.glue_s_low,
                seed=cfg.seed,
            )
            integration_times = torch.linspace(
                0.0, (horizon - 1) * cfg.tau, horizon, device=device
            )
            iterator = iter(loader)
            for _ in range(cfg.steps_per_horizon):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)

                states = batch.states.to(device)
                glue_mask = batch.glue_mask.to(device)
                seam_mask = batch.seam_mask.to(device)

                losses = forward_step(
                    nets=nets,
                    cfg=cfg,
                    states=states,
                    glue_mask=glue_mask,
                    seam_mask=seam_mask,
                    ode_func=ode_func,
                    integration_times=integration_times,
                )
                optim.zero_grad(set_to_none=True)
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(nets.parameters(), cfg.grad_clip)
                optim.step()
                if scheduler is not None:
                    scheduler.step()

                if global_step % cfg.log_every == 0:
                    record = {
                        "step": global_step,
                        "horizon": horizon,
                        "lr": optim.param_groups[0]["lr"],
                        "total": float(losses.total.item()),
                        "L_x": float(losses.L_x.item()),
                        "L_z": float(losses.L_z.item()),
                        "L_g": float(losses.L_g.item()),
                        "L_v": float(losses.L_v.item()),
                        "L_c": float(losses.L_c.item()),
                        "elapsed": time.time() - t_start,
                    }
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    print(
                        f"[step {global_step:6d} | H={horizon:3d}] "
                        f"total={record['total']:.4f} "
                        f"L_x={record['L_x']:.4f} L_z={record['L_z']:.4f} "
                        f"L_g={record['L_g']:.4f} L_v={record['L_v']:.4f} "
                        f"L_c={record['L_c']:.4f}",
                        flush=True,
                    )

                if (
                    cfg.save_every > 0
                    and global_step > 0
                    and global_step % cfg.save_every == 0
                ):
                    torch.save(nets.state_dict(), run_dir / "model.pt")

                global_step += 1
    finally:
        log_file.close()

    torch.save(nets.state_dict(), run_dir / "model.pt")
    return nets
