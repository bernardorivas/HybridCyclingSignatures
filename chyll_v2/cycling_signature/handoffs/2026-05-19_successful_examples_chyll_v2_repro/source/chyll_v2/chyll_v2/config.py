"""Configuration dataclass for the CHyLL v2 baseline (cylinder-augmented)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CHyLLv2Config:
    """Config for the Sangli-style baseline applied to mapping-cylinder data.

    The encoder sees the augmented state ``(x, s) in X' = X \\cup (G x [0,1])``.
    ``base_dim`` is the physical state dimension ``n``; ``state_dim`` is the
    encoder input dimension ``n + 1`` (extra ``s`` coordinate).
    """

    # --- system ----------------------------------------------------------
    system_name: str = "rimless_wheel"
    base_dim: int = 2                   # n  (physical state)
    # state_dim is set by ``__post_init__`` to base_dim + 1.
    state_dim: int = 3                  # n + 1 (augmented with s)
    # Sangli's existence theorem on the hybrifold requires m > 2 n. On the
    # mapping cylinder X' (dimension n+1) the analogue is m > 2 (n+1).
    # We default to 2 (n+1) + 1.
    latent_dim: int = 7

    # --- networks --------------------------------------------------------
    encoder_hidden: int = 128
    encoder_layers: int = 3
    vfield_hidden: int = 128
    vfield_layers: int = 3
    decoder_hidden: int = 128
    decoder_layers: int = 3
    # Optional sin activation at the last hidden layer (their §7.4 ablation).
    use_sin_last_layer: bool = False
    sin_omega: float = 30.0

    # --- data ------------------------------------------------------------
    n_trajectories: int = 800
    trajectory_steps: int = 500
    tau: float = 0.05                   # discrete tau-semiflow step on X'
    sim_rtol: float = 1e-8
    sim_atol: float = 1e-10
    seed: int = 0
    # Thresholds on the cylinder coordinate ``s`` used by the
    # data-driven gluing-pair detector (``identify_gluing_bool``).
    glue_s_high: float = 0.9
    glue_s_low: float = 0.1

    # --- loss weights ----------------------------------------------------
    w_x: float = 1.0       # L_x  reconstruction of augmented state
    w_z: float = 1.0       # L_z  latent consistency along trajectory
    w_g: float = 3.0       # L_g  gluing on detected gluing-edge pairs
    w_v: float = 0.0       # L_v  seam-velocity compatibility (their ablation: redundant)
    w_c: float = 1.0       # L_c  per-coordinate variance floor
    # Lambda: floor on per-coord variance. With latent_dim ~ 7-11 and
    # natively-bounded states (theta in [-0.2, 0.6], omega in [-1, 2], s in
    # [0, 1]), Lambda=1.0 is too aggressive -- it dominates the loss budget
    # against the pad-identity initialisation. Drop to 0.3.
    collapse_threshold: float = 0.3     # Lambda

    # --- curriculum (Algorithm 1) ---------------------------------------
    curriculum_horizons: Tuple[int, ...] = (5, 10, 25, 50, 100)
    steps_per_horizon: int = 1000

    # --- optimizer -------------------------------------------------------
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    grad_clip: float = 2.0
    cosine_schedule: bool = True

    # --- latent Neural ODE (torchdiffeq) --------------------------------
    # ``method``: any solver torchdiffeq exposes (e.g. dopri5, rk4, euler).
    ode_method: str = "dopri5"
    # Use the adjoint method for memory-efficient backprop. Required when
    # curriculum horizons go beyond ~50 steps.
    ode_use_adjoint: bool = True
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-6
    # Fixed-step solvers ignore rtol/atol but need ``step_size``.
    ode_step_size: float = 0.01

    # --- runtime ---------------------------------------------------------
    device: str = "cuda"
    run_dir: str = "chyll_v2/runs/rimless_wheel"
    figure_dir: str = "chyll_v2/figures/rimless_wheel"
    log_every: int = 50
    save_every: int = 1000

    # --- system-specific (filled by system constructors) ---------------
    system_params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce state_dim = base_dim + 1 (mapping-cylinder coordinate).
        self.state_dim = self.base_dim + 1


def make_default(system_name: str) -> CHyLLv2Config:
    if system_name == "rimless_wheel":
        return CHyLLv2Config(
            system_name="rimless_wheel",
            base_dim=2,
            latent_dim=7,                  # > 2 (n+1) = 6
            tau=0.05,
            trajectory_steps=400,
            run_dir="chyll_v2/runs/rimless_wheel",
            figure_dir="chyll_v2/figures/rimless_wheel",
        )
    if system_name == "bouncing_ball":
        return CHyLLv2Config(
            system_name="bouncing_ball",
            base_dim=2,
            latent_dim=7,
            tau=0.02,
            trajectory_steps=500,
            run_dir="chyll_v2/runs/bouncing_ball",
            figure_dir="chyll_v2/figures/bouncing_ball",
        )
    if system_name == "compass_gait":
        return CHyLLv2Config(
            system_name="compass_gait",
            base_dim=4,
            latent_dim=11,                 # > 2 (n+1) = 10
            tau=0.05,
            trajectory_steps=600,
            run_dir="chyll_v2/runs/compass_gait",
            figure_dir="chyll_v2/figures/compass_gait",
        )
    raise ValueError(f"unknown system: {system_name}")
