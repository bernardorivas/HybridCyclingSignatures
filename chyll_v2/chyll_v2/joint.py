"""Joint (multi-slope) CHyLL v2 training with a shared encoder/decoder.

Motivation (see notes/slope-frame-check.md). In the slope-fixed frame

    u = theta + phi

the compass gait's guard  ``u_ns + u_s = 0``  and its reset are BOTH
phi-independent; only the vector field depends on phi, through gravity
``sin(u - phi)``. So the hybrid suspension

    Sigma = X u (G x [0,1]) / ~ ,    (g,1) ~ (r(g), 0)

is ONE space for every slope, and a single encoder/decoder pair is well posed:
all slopes impose the same gluing relation ``E(g,1) = E(r(g),0)`` rather than
five competing ones. The phi-dependence is carried entirely by the latent
vector field, which reads phi as a continuous input:

    E(x, s)        shared
    D(z)           shared
    V(z, phi)      phi-conditioned      <- the only phi-dependent piece

Because phi is a continuous input (not an index into five separate networks),
the model is defined at slopes it never saw, which is what makes the held-out
interpolation test possible.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .config import CHyLLv2Config
from .data import identify_gluing_bool, identify_seam_bool
from .losses import compute_losses
from .networks import Decoder, Encoder, VectorField, build_mlp
from .ode import rollout
from .systems.base import Trajectory
from .train import resolve_device


# ----------------------------------------------------------------------
# phi normalisation
# ----------------------------------------------------------------------
#
# Raw phi spans only [0.0698, 0.0908] rad across the cascade. Feeding that
# straight into an MLP gives an input that barely varies, and the network
# cannot resolve the dependence. We map the TRAINING range affinely onto
# about [-1, 1]. The scaling is stored in the checkpoint so evaluation and
# interpolation use exactly the same map.

class PhiScaler:
    def __init__(self, phis: Sequence[float]):
        lo, hi = float(min(phis)), float(max(phis))
        self.center = 0.5 * (lo + hi)
        self.half = 0.5 * (hi - lo) if hi > lo else 1.0

    def __call__(self, phi):
        return (np.asarray(phi, dtype=np.float64) - self.center) / self.half

    def to_dict(self) -> Dict[str, float]:
        return {"center": self.center, "half": self.half}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PhiScaler":
        s = cls([0.0, 1.0])
        s.center, s.half = d["center"], d["half"]
        return s


# ----------------------------------------------------------------------
# phi-conditioned vector field
# ----------------------------------------------------------------------


class PhiVectorField(nn.Module):
    """``V : R^m x R -> R^m``, i.e. dz/dt = V(z, phi_normalised)."""

    def __init__(self, cfg: CHyLLv2Config):
        super().__init__()
        self.mlp = build_mlp(
            in_dim=cfg.latent_dim + 1,
            out_dim=cfg.latent_dim,
            hidden=cfg.vfield_hidden,
            n_layers=cfg.vfield_layers,
            use_sin_last=False,
            sin_omega=cfg.sin_omega,
        )

    def forward(self, z: torch.Tensor, phi_n: torch.Tensor) -> torch.Tensor:
        # z: (..., m)   phi_n: (..., 1) broadcastable
        return self.mlp(torch.cat([z, phi_n.expand(*z.shape[:-1], 1)], dim=-1))


class JointNetworks(nn.Module):
    """Shared E and D; phi-conditioned V."""

    def __init__(self, cfg: CHyLLv2Config):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.vfield = PhiVectorField(cfg)
        self.decoder = Decoder(cfg)


class MultiHeadNetworks(nn.Module):
    """Shared E and D; ONE INDEPENDENT vector field per slope.

    This is the cleaner ablation of the two. Against the baseline of five fully
    separate models (E_j, V_j, D_j) it changes exactly one thing -- E and D
    become shared -- whereas the phi-conditioned model also collapses the five
    fields into one network, so a regression there could be either the chart or
    the field's capacity to interpolate in phi.

    The heads are a lookup table: phi_j -> V_j, with no structure relating them
    and nothing defined between the training slopes.
    """

    def __init__(self, cfg: CHyLLv2Config, n_heads: int):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.heads = nn.ModuleList([VectorField(cfg) for _ in range(n_heads)])
        self.decoder = Decoder(cfg)
        self.n_heads = n_heads


class _HeadWrapper(nn.Module):
    """torchdiffeq adapter for one head (autonomous, so t is ignored)."""

    def __init__(self, vfield: nn.Module):
        super().__init__()
        self.vfield = vfield

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.vfield(z)


class _PhiVFieldWrapper(nn.Module):
    """torchdiffeq wants ``forward(t, z)``. phi is constant along a rollout,
    so it is stashed on the wrapper for the duration of one batch."""

    def __init__(self, vfield: PhiVectorField):
        super().__init__()
        self.vfield = vfield
        self.phi_n: torch.Tensor | None = None

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.vfield(z, self.phi_n)


# ----------------------------------------------------------------------
# dataset over (trajectory, phi) pairs
# ----------------------------------------------------------------------


class PhiSliceDataset(Dataset):
    """Length-``horizon`` slices pooled across slopes; each carries its phi.

    A batch therefore mixes slopes, so every gradient step constrains the
    shared encoder/decoder with data from all of them at once.
    """

    def __init__(
        self,
        traj_by_phi: List[Tuple[float, List[Trajectory]]],
        horizon: int,
        scaler: PhiScaler,
        s_high: float = 0.9,
        s_low: float = 0.1,
    ):
        if horizon < 2:
            raise ValueError("horizon must be >= 2")
        self.horizon = horizon
        self.trajectories: List[Trajectory] = []
        self.phi_n: List[float] = []
        for phi, trajs in traj_by_phi:
            pn = float(scaler(phi))
            for t in trajs:
                self.trajectories.append(t)
                self.phi_n.append(pn)
        self._glue = [identify_gluing_bool(t, s_high=s_high, s_low=s_low)
                      for t in self.trajectories]
        self._seam = [identify_seam_bool(t, s_high=s_high, s_low=s_low)
                      for t in self.trajectories]
        self._starts = [len(t) - horizon + 1 for t in self.trajectories]
        self._cum = np.cumsum([0] + self._starts)

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, idx: int):
        ti = int(np.searchsorted(self._cum, idx, side="right") - 1)
        st = idx - int(self._cum[ti])
        en = st + self.horizon
        tr = self.trajectories[ti]
        return (
            torch.from_numpy(tr.states[st:en]).float(),
            torch.from_numpy(self._glue[ti][st:en - 1].copy()),
            torch.from_numpy(self._seam[ti][st:en - 1].copy()),
            torch.tensor([self.phi_n[ti]], dtype=torch.float32),
        )


def _collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
        torch.stack([b[3] for b in batch]),
    )


# ----------------------------------------------------------------------
# u-frame conversion
# ----------------------------------------------------------------------


def to_u_frame(trajs: List[Trajectory], phi: float,
               angle_idx: Sequence[int] = (0, 1)) -> List[Trajectory]:
    """u = theta + phi on the angle coordinates; velocities and s untouched.

    A pure translation, so distances are preserved and every geometric
    quantity measured in theta carries over unchanged.
    """
    out = []
    for t in trajs:
        s = t.states.copy()
        for i in angle_idx:
            s[:, i] += phi
        out.append(Trajectory(states=s, times=t.times.copy()))
    return out


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------


def train_joint(
    cfg: CHyLLv2Config,
    traj_by_phi: List[Tuple[float, List[Trajectory]]],
    scaler: PhiScaler,
    load_from: str | None = None,
) -> JointNetworks:
    device = resolve_device(cfg.device)
    nets = JointNetworks(cfg).to(device)
    if load_from is not None:
        nets.load_state_dict(
            torch.load(load_from, map_location=device, weights_only=True))
        print(f"[joint] initialised from {load_from}", flush=True)
    ode_func = _PhiVFieldWrapper(nets.vfield).to(device)

    optim = AdamW(nets.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.steps_per_horizon * len(cfg.curriculum_horizons)
    sched = CosineAnnealingLR(optim, T_max=total) if cfg.cosine_schedule else None

    run_dir = Path(cfg.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str))
    (run_dir / "phi_scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2))
    log_file = (run_dir / "train_log.jsonl").open("w")

    step, t0 = 0, time.time()
    try:
        for horizon in cfg.curriculum_horizons:
            ds = PhiSliceDataset(traj_by_phi, horizon, scaler,
                                 s_high=cfg.glue_s_high, s_low=cfg.glue_s_low)
            loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                collate_fn=_collate, drop_last=True)
            times = torch.linspace(0.0, (horizon - 1) * cfg.tau, horizon,
                                   device=device)
            it = iter(loader)
            for _ in range(cfg.steps_per_horizon):
                try:
                    states, glue, seam, phi_n = next(it)
                except StopIteration:
                    it = iter(loader)
                    states, glue, seam, phi_n = next(it)
                states = states.to(device); glue = glue.to(device)
                seam = seam.to(device);     phi_n = phi_n.to(device)

                z_enc = nets.encoder(states)
                ode_func.phi_n = phi_n                       # (B, 1)
                z_traj = rollout(ode_func=ode_func, z0=z_enc[:, 0, :],
                                 times=times, cfg=cfg)
                z_pred = z_traj.transpose(0, 1).contiguous()
                dec = nets.decoder(z_pred)
                losses = compute_losses(
                    cfg=cfg, states=states, z_predicted=z_pred,
                    z_encoded=z_enc, states_decoded=dec,
                    glue_mask=glue, seam_mask=seam)

                optim.zero_grad(set_to_none=True)
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(nets.parameters(), cfg.grad_clip)
                optim.step()
                if sched is not None:
                    sched.step()

                if step % cfg.log_every == 0:
                    rec = {"step": step, "horizon": horizon,
                           "lr": optim.param_groups[0]["lr"],
                           "total": losses.total.item(),
                           "L_x": losses.L_x.item(), "L_z": losses.L_z.item(),
                           "L_g": losses.L_g.item(), "L_v": losses.L_v.item(),
                           "L_c": losses.L_c.item(),
                           "elapsed": time.time() - t0}
                    log_file.write(json.dumps(rec) + "\n"); log_file.flush()
                    print(f"[step {step:6d} | H={horizon:3d}] "
                          f"total={rec['total']:.4f} L_x={rec['L_x']:.4f} "
                          f"L_z={rec['L_z']:.4f} L_g={rec['L_g']:.4f} "
                          f"L_c={rec['L_c']:.4f}", flush=True)
                if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
                    torch.save(nets.state_dict(), run_dir / "model.pt")
                step += 1
    finally:
        log_file.close()

    torch.save(nets.state_dict(), run_dir / "model.pt")
    return nets


# ----------------------------------------------------------------------
# multi-head training (Design A)
# ----------------------------------------------------------------------


def train_multihead(
    cfg: CHyLLv2Config,
    traj_by_phi: List[Tuple[float, List[Trajectory]]],
    slope_names: Sequence[str],
    per_slope_batch: int | None = None,
    load_from: str | None = None,
) -> MultiHeadNetworks:
    """Balanced multi-head training.

    Each optimizer step draws one HOMOGENEOUS batch per slope, encodes them all
    with the shared E, rolls each out under its own head, decodes with the
    shared D, and averages the per-slope losses. Homogeneous batches are what
    make the head routing trivial: one V per sub-batch, so `rollout` is
    unchanged.

    ``per_slope_batch`` defaults to cfg.batch_size // n_slopes so that the total
    samples per optimizer step MATCH the phi-conditioned run. That keeps the
    ablation to a single variable (architecture) rather than also changing the
    batch size.
    """
    device = resolve_device(cfg.device)
    n = len(traj_by_phi)
    bs = per_slope_batch or max(1, cfg.batch_size // n)
    nets = MultiHeadNetworks(cfg, n_heads=n).to(device)
    if load_from is not None:
        nets.load_state_dict(
            torch.load(load_from, map_location=device, weights_only=True))
        print(f"[multihead] initialised from {load_from}", flush=True)
    wraps = [_HeadWrapper(h).to(device) for h in nets.heads]

    optim = AdamW(nets.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.steps_per_horizon * len(cfg.curriculum_horizons)
    sched = CosineAnnealingLR(optim, T_max=total) if cfg.cosine_schedule else None

    # The heads do not read phi. The scaler is built only so the dataset and the
    # on-disk artifacts match the conditional run's format exactly.
    scaler = PhiScaler([p for p, _ in traj_by_phi])
    run_dir = Path(cfg.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str))
    (run_dir / "head_map.json").write_text(json.dumps(
        {name: i for i, name in enumerate(slope_names)}, indent=2))
    # Written for I/O parity with the conditional run so downstream tooling can
    # load either checkpoint the same way. The heads do NOT read phi -- the map
    # is a slope -> head index lookup, recorded in head_map.json.
    (run_dir / "phi_scaler.json").write_text(
        json.dumps(scaler.to_dict(), indent=2))
    log_file = (run_dir / "train_log.jsonl").open("w")
    print(f"[multihead] {n} heads, {bs} samples/slope/step "
          f"({n*bs} total, cfg.batch_size={cfg.batch_size})", flush=True)

    step, t0 = 0, time.time()
    try:
        for horizon in cfg.curriculum_horizons:
            loaders, iters = [], []
            for phi, trajs in traj_by_phi:
                ds = PhiSliceDataset([(phi, trajs)], horizon, scaler,
                                     s_high=cfg.glue_s_high, s_low=cfg.glue_s_low)
                ld = DataLoader(ds, batch_size=bs, shuffle=True,
                                collate_fn=_collate, drop_last=True)
                loaders.append(ld); iters.append(iter(ld))
            times = torch.linspace(0.0, (horizon - 1) * cfg.tau, horizon,
                                   device=device)
            for _ in range(cfg.steps_per_horizon):
                per = []
                for j in range(n):
                    try:
                        states, glue, seam, _ = next(iters[j])
                    except StopIteration:
                        iters[j] = iter(loaders[j])
                        states, glue, seam, _ = next(iters[j])
                    states = states.to(device); glue = glue.to(device)
                    seam = seam.to(device)
                    z_enc = nets.encoder(states)
                    z_traj = rollout(ode_func=wraps[j], z0=z_enc[:, 0, :],
                                     times=times, cfg=cfg)
                    z_pred = z_traj.transpose(0, 1).contiguous()
                    per.append(compute_losses(
                        cfg=cfg, states=states, z_predicted=z_pred,
                        z_encoded=z_enc, states_decoded=nets.decoder(z_pred),
                        glue_mask=glue, seam_mask=seam))
                loss = sum(p.total for p in per) / n

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nets.parameters(), cfg.grad_clip)
                optim.step()
                if sched is not None:
                    sched.step()

                if step % cfg.log_every == 0:
                    rec = {"step": step, "horizon": horizon,
                           "lr": optim.param_groups[0]["lr"],
                           "total": float(loss.item()),
                           "L_x": float(sum(p.L_x for p in per).item() / n),
                           "L_z": float(sum(p.L_z for p in per).item() / n),
                           "L_g": float(sum(p.L_g for p in per).item() / n),
                           "L_v": float(sum(p.L_v for p in per).item() / n),
                           "L_c": float(sum(p.L_c for p in per).item() / n),
                           "elapsed": time.time() - t0}
                    # Bernardo: report every loss per angle, not only pooled
                    for name, p in zip(slope_names, per):
                        rec[f"L_g/{name}"] = float(p.L_g.item())
                        rec[f"L_x/{name}"] = float(p.L_x.item())
                    log_file.write(json.dumps(rec) + "\n"); log_file.flush()
                    print(f"[step {step:6d} | H={horizon:3d}] total={rec['total']:.4f} "
                          f"L_x={rec['L_x']:.4f} L_g={rec['L_g']:.5f} "
                          + " ".join(f"{nm}:{p.L_g.item():.5f}"
                                     for nm, p in zip(slope_names, per)),
                          flush=True)
                if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
                    torch.save(nets.state_dict(), run_dir / "model.pt")
                step += 1
    finally:
        log_file.close()
    torch.save(nets.state_dict(), run_dir / "model.pt")
    return nets
