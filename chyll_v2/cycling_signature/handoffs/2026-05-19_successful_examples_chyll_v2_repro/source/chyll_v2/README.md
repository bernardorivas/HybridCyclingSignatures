# chyll_v2

Standalone reimplementation of Teng, Liu, Sreenath (ICML 2026), *Embedding
Hybrid Systems into Continuous Latent Vector Fields*
(`local_docs/.../references/Sangli-ICML-Fed.18 (1).pdf`).

This directory is an **experimental baseline** built from scratch to compare
against the suspension/mapping-cylinder pipeline in `src/`. It does not import
from `src/` and shares no state; deleting `chyll_v2/` is a clean fallback.

## What this implements (and what we changed)

We test **Sangli's loss + continuous latent Neural ODE** on **our
mapping-cylinder data**. Specifically:

- **Geometry** -- mapping cylinder ``X' = X \cup (G x [0,1])``, same as the
  rest of our framework. The encoder sees the augmented state ``(x, s)``.
- **Encoder** ``f_theta : X' -> R^m`` with ``m > 2 (n+1)``.
- **Latent vector field** ``V_theta : R^m -> R^m``, integrated by
  ``torchdiffeq.odeint_adjoint`` (memory-efficient adjoint Neural ODE).
- **Decoder** ``f_theta^{-1} : R^m -> X'``.
- **Five-term loss** ``L_x + L_z + L_g + L_v + L_c`` (their eqs. 4-8) with
  ``L_g`` and ``L_v`` evaluated on *symbolic* guard pairs sampled from the
  system (no Lipschitz-threshold heuristic needed -- the gluing is exact).
- **Curriculum** on rollout horizon (Algorithm 1) with cosine LR.
- **Optional sin activation** at the last hidden layer (their §7.4
  ablation).

What is *not* present here:
- The CHyLL 2025 conformal loss (Sangli's ICML 2026 version dropped it).
- The discrete tau-map ``F`` used in `src/`.
- Their data-driven Lipschitz-threshold gluing detector -- we have the
  exact guard set ``G`` and reset map ``r``.

## How this differs from src/

| Aspect | src/ (current manuscript pipeline) | chyll_v2/ (this experiment) |
|---|---|---|
| Data manifold | Mapping cylinder ``X'`` | Mapping cylinder ``X'`` (same) |
| Latent dynamics | Discrete tau-map ``F`` | Continuous ``V_theta`` via Neural ODE |
| ODE integrator | -- | ``torchdiffeq.odeint_adjoint`` |
| Loss terms | ``dyn + glue + rec + seam + conf + coll`` (6) | ``x + z + g + v + c`` (5) |
| Seam continuity | ``L_seam`` (cosine, two summands) | ``L_v`` (MSE between latent tangents) |
| Conformal regularizer | Yes (``L_conf``) | No (dropped after CHyLL 2025) |
| Gluing identification | Symbolic (sampled ``g in G``) | Symbolic (same) |

## Layout

```
chyll_v2/
├── chyll_v2/
│   ├── config.py             # base_dim vs state_dim (= base_dim + 1)
│   ├── systems/
│   │   ├── base.py           # tau-semiflow on X' with event-driven cylinder transitions
│   │   ├── rimless_wheel.py
│   │   ├── bouncing_ball.py  (Phase 2)
│   │   └── compass_gait.py   (Phase 3)
│   ├── data.py               # slice loader + GuardSampler for L_g / L_v
│   ├── networks.py           # Encoder / V_theta / Decoder (MLPs)
│   ├── ode.py                # torchdiffeq rollout wrapper (adjoint)
│   ├── losses.py             # 5-term composite; L_v via JVP on encoder
│   ├── train.py              # Algorithm 1 with curriculum
│   └── visualize.py
├── scripts/
│   ├── train_rimless.py
│   ├── train_bouncing_ball.py  (Phase 2)
│   └── train_compass.py        (Phase 3)
├── cycling_signature/          # topology postprocessing + publication figures
│   ├── export/                 # trained-model -> lift CSV exporters
│   ├── julia/                  # David-style cycling-signature runners
│   ├── plot/                   # figure builders
│   ├── data/
│   └── figures/
├── runs/                       (created on first run)
└── figures/                    (created on first run)
```

## Running

```bash
# Activate whichever Python environment hosts torch + torchdiffeq + scipy.
python chyll_v2/scripts/train_rimless.py            # full run
python chyll_v2/scripts/train_rimless.py --smoke    # tiny sanity run
python chyll_v2/scripts/train_rimless.py --no-adjoint --ode-method rk4

# Export rimless latent UT inputs for David Hien's cycling-signature pipeline.
python chyll_v2/cycling_signature/export/prepare_rimless_lift.py
julia --project="time series/cycling_signature" \
  "time series/cycling_signature/run_cycling_signature.jl" \
  --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
  --base continuous_lift_chyll_v2

# Build the current rimless manuscript-style cycling-signature figures.
python chyll_v2/cycling_signature/plot/plot_rimless_signature_figures.py
```

Outputs:
- `chyll_v2/runs/<system>/model.pt`, `train_log.jsonl`, `config.json`.
- `chyll_v2/figures/<system>/loss_history.png`,
  `phase_portrait_truth.png`, `rollout_vs_truth.png`.
- `chyll_v2/cycling_signature/data/rimless_wheel/continuous_lift_chyll_v2*`
  and `barcode_H1_chyll_v2*.csv` for cycling-signature checks.
- `chyll_v2/cycling_signature/figures/rimless_wheel/fig_rimless_signature_rank_distribution.{pdf,png}`.
- `chyll_v2/cycling_signature/figures/rimless_wheel/fig_rimless_signature_rank_heatmaps.{pdf,png}`.

## Dependencies

Already present in `requirements.txt`: ``torch``, ``numpy``, ``scipy``,
``matplotlib``. **Additional:** ``torchdiffeq`` (verified installed at
0.2.5 at the time of writing).

## Status

- **Phase 1**: infrastructure + rimless wheel.
- **Phase 1b**: rimless cycling-signature export + rank check.
- **Phase 1c**: David-style rimless subsegment figures.
- **Phase 2** (pending): bouncing ball.
- **Phase 3** (pending): compass gait.
