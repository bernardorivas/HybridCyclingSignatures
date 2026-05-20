# Relaxed-Space Compass-Gait Cycling-Signature Pipeline — Specification

This file documents exactly what the pipeline does, in the order the bits run,
with file-and-line citations into the canonical source. For the operational
runbook see [`README.md`](README.md).

## 0. What we mean by "relaxed-space encoder" vs. "CHyLL"

The encoder `E_theta` used here is inspired by the CHyLL framework of
[Teng et al., 2024] but is **not** CHyLL. The key contrasts (draft.md:428,
Remark 4.6 of the manuscript):

| Aspect               | Original CHyLL                        | This pipeline (relaxed)                          |
|----------------------|---------------------------------------|--------------------------------------------------|
| Target space         | hybrifold `M_H = X / ~`               | relaxed space `X' = X union (G x [0,1])`         |
| Semiflow             | neural ODE, integrated                | discrete residual MLP, one tau per step          |
| Training             | curriculum on prediction horizon      | no curriculum                                    |
| Cylinder coordinate  | none                                  | `s` explicitly encoded as a state coordinate     |
| Fibres over `G`      | collapsed by construction             | preserved; collision prevention is a training-time property |

File-name suffix `_relaxed`, figure titles `Learned, N impacts`, and the
folder name `compass_gait_relaxed/` all track this choice.

## 1. System definition

`src/system.py:144` — `CompassGaitHybridSystem`. State `[theta_ns, theta_s,
theta_dot_ns, theta_dot_s]`. Physical parameters from `src/config.py:23-31`
(mass `m = 5`, hip mass `m_H = 10`, leg length `l = 1`, `a = b = 0.5`, gravity
`g = 9.81`, ground slope `phi = 0.07 rad`).

Vector field: Goswami Lagrangian `M(q) q_dot_dot + N(q, q_dot) q_dot + G(q) = 0`
assembled in `get_matrices` (`src/system.py:159`), integrated via
`scipy.integrate.solve_ivp` with `max_step = 0.01` (`src/system.py:28`).

Guard: `theta_ns + theta_s + 2 phi = 0` with `theta_ns - theta_s > 0.01`,
`guard_direction = -1` (`src/system.py:193-202`).

Reset map: leg-role swap + `Q_+ q_dot_+ = Q_- q_dot_-` from angular-momentum
conservation (`src/system.py:209-239`).

Relaxed semiflow `phi'` on `X union (G x [0, 1])` is
`BaseHybridSystem.generate_tau_timeseries` (`src/system.py:38`) with
`tau = 0.05` (`config.integration_tau`, `src/config.py:73`). The mapping
cylinder parameter `s` is tracked explicitly as the 5th state coordinate.

## 2. Networks (`src/networks.py`)

Input is `R^{n+1} = R^5` (4-D compass state plus `s`). Embedding dimension
`d = state_dim + 1 + embed_extra = 5` (`embed_extra = 0`, set in
`scripts/train_compass.py`).

All three networks are 4-layer GELU MLPs with hidden dimension 256 built by
`_build_mlp`.

### Encoder `E` (`Encoder`, `src/networks.py`)

Generic residual MLP on `R^{n+1}`:

```
E(x) = pad(x, d) + MLP(x)
```

where `pad(x, d)` zero-pads the input to dimension `d`. The final linear
layer of `MLP` is initialised with weights scaled by 0.01 and zero bias, so
that at the start of training `E` approximates the identity on the first
`n+1` output coordinates and near-zero noise on any padded dims. Nothing
architecturally pins `E` at `s = 0`; base-space injectivity is a learned
property, enforced during Phase II reconstruction training.

### Flow predictor `F` (`FlowPredictor`, `src/networks.py`)

Time-conditioned residual MLP:

```
F(y, dt) = y + MLP(concat(y, dt))
```

This realises the manuscript's `Ψ(Δt, z)`. During training `dt` is always
`integration_tau = 0.05`, but the time-conditioning permits evaluation at
other step sizes without retraining.

### Decoder `D` (`Decoder`, `src/networks.py`)

Generic residual MLP reconstructing the full `(state, s)` tuple:

```
D(y) = y[:n+1] + MLP(y)
```

No `s = 0` forcing. `s` is reconstructed freely.

## 3. Training (`scripts/train_compass.py`)

Two-phase procedure matching Algorithm 1 of the manuscript.

### Dataset

`generate_suspension_dataset` (`src/system.py`) with `num_orbits = 3000`,
`points_per_orbit = 20` steps of `tau = 0.05`. Half of the initial
conditions start in the base space with `theta ~ U(-0.5, 0.5)^2`,
`theta_dot ~ U(-2, 2)^2`; the other half start on the guard surface
`theta_ns + theta_s = -2 phi` with random `s ~ U(0.01, 1)`. Total:
~60 000 `(x_i, x_{i+1})` supervised pairs.

### Phase I (encoder + semiflow; 120 epochs)

AdamW on E + F parameters, `lr = 8e-4`, `weight_decay = 1e-5`, cosine LR
schedule, batch 1024, gradient clip 2.0. Five loss terms, no recon:

| Term    | Weight | Expression                                               |
|---------|--------|----------------------------------------------------------|
| `L_dyn` | 1.0    | `MSE(Ψ(τ, E(x_i)), E(x_{i+1}))`                          |
| `L_glue`| 3.0    | `MSE(E(g, 1), E(r(g), 0))`, 256 sampled guard states     |
| `L_conf`| 0.01   | `mean_i ||λ_i I - J_i^T J_i||_F^2` over ≤64 samples/batch |
| `L_coll`| 1.0    | `sum_k ReLU(Λ - Var_batch(E(x)_k))`, `Λ = 0.1`            |
| `L_utb` | 1.0    | `MSE(v_cyl(g, 0), v_X(g)) + MSE(v_cyl(g, 1), v_X(r(g)))` |

`v_cyl` is a finite-difference of `E` along the `s` direction with
`h = 0.02`. `v_X` is padded with zeros in the `s`-slot to match `d`.

### Phase II (decoder only; 40 epochs)

Freeze E and F. AdamW on D alone, `lr = 2e-3`, cosine schedule. Single
term:

| Term     | Weight | Expression                                            |
|----------|--------|-------------------------------------------------------|
| `L_recon`| 1.0    | `MSE(D(E(x)), x)` on samples with `s = 0` or `ε < s < 1-ε` |

Samples within `ε = 0.05` of the cylinder boundaries `s ∈ {0, 1}` are
masked, consistent with Section 4 ("In practice, L_recon is evaluated
modulo the equivalence relation...").

## 4. Lift export for cycling signature (`time series/cycling_signature/prepare_compass_cs_inputs_relaxed.py`)

### Canonical rollout

`time series/compass gait/simulate.py:98` — `simulate_compass_gait` with
`LIMIT_CYCLE_IC = [-phi - 0.27, -phi + 0.27, -0.38, -1.09]`, `rtol = 1e-10`,
`atol = 1e-12`. Returns `segments` (continuous arcs at `s = 0`) and
`jump_pairs` (pre-/post-impact states at each impact).

### Bridge parameterisation

Each bridge is 50 interior samples of the line
`{(x_minus, s) : s in (0, 1)}` passed through `E_theta`
(`prepare_compass_cs_inputs_relaxed.py:78-84`):

```
inp_br = hstack([tile(x_minus, (50, 1)), s_vals[:, None]])
z_br = E(inp_br)
```

There is no runtime enforcement of closure; closure is whatever the trained
encoder gives you via the gluing loss.

### Tangents

First differences `dz / ||dz||` with the last row duplicated
(`prepare_compass_cs_inputs_relaxed.py:97-101`). Unit-norm assertion uses
`atol = 1e-6`; Julia later renormalises to its own tolerance.

### Diagnostics report

Phase-A diagnostics written to
`data/compass_gait/report_relaxed{,_n20}_encoder_diagnostics.txt`:

1. Reconstruction error `||D(E(x)) - x||` split by arc / bridge and by
   full vector / state-only.
2. Near-collision screen: for each sample, nearest-neighbour distance in
   latent space among samples more than 20 indices away; flag points where
   `latent_dist / input_dist < 0.25` and `input_dist > 0.1`.
3. Tangent quality: pre-normalisation tangent magnitudes and junction-angle
   summary (arc to bridge, bridge to arc).
4. Bridge-to-bridge separation under David's DynamicDistance
   `d((p, v), (q, w)) = max(||p - q||, C * ||v - w||)` for
   `C in {0.05, 0.2, 0.5, 1.0}`, reported as `rho = inter_bridge_min /
   arc_step_median`.

## 5. Julia cycling signature (`time series/cycling_signature/run_cycling_signature.jl`)

### First-run setup (`run_cycling_signature.jl:22-29`)

```julia
Pkg.add(url = "https://github.com/davidhien/StepFunctions.jl")
Pkg.develop(path = "local_docs/CyclingSignatures.jl-main/CyclingSignatures.jl-main/")
Pkg.instantiate()
```

`StepFunctions.jl` is an unregistered dependency of `CyclingSignatures.jl` and
must be added by URL first.

### Tangent renormalisation

`run_cycling_signature.jl:73-86` renormalises tangents in Julia because the
CSV round-trip can drop a few ULPs below David's
`utb_trajectory_space_from_trajectory` tolerance of roughly `1.5e-8`.

### Parameter sweep

`(boxsize, sb_radius) in {0.30, 0.20, 0.10, 0.05} x {1, 2, 4}` at
`run_cycling_signature.jl:91-95`. For each cell:

- `ts = utb_trajectory_space_from_trajectory(X, TX, boxsize, sb_radius)`
- `b1 = betti_1(ts)`
- `sig = cycling_signature(ts, (1, N))`, `rank = dimension(sig)`

Full sweep is written to the CSV header of `barcode_H1_relaxed{,_n20}.csv`.
Canonical cell (first `(boxsize, sb_radius)` with `b1 > 0` and `rank > 0`) is
used for the full birth vector in the body.

## 6. Figure (`time series/cycling_signature/plot_compass_rank_heatmap.py`)

Two outputs:

- `figures/compass_gait/fig_compass_cycling_rank_heatmap.pdf/.png` — 2 panels:
  `Learned, 5 impacts`, `Learned, 20 impacts`. Figure size `6.8 x 2.9 in`.
- `figures/compass_gait/fig_compass_cycling_rank_heatmap_appendix.pdf/.png` —
  3 panels: adds `Analytic reference` (`barcode_H1_analytic_tgt025.csv`).
  Figure size `10.0 x 2.9 in`.

Discrete colormap:

| Rank        | Color           | Interpretation                       |
|-------------|-----------------|--------------------------------------|
| 0           | gray `#E5E5E5`  | cover contractible, no `H_1`         |
| 1           | green `#2CA25F` | expected dominant loop               |
| 2..4        | yellow `#FFE08A`| mild over-counting                   |
| 5..9        | orange `#F58C3B`| notable over-counting                |
| >= 10       | red `#B83030`   | heavy over-counting                  |

## What this pipeline is not doing

- No BridgeNet. `time series/bridge_net.py` is the bounded-`u` analytic bridge
  used elsewhere; it is **not** imported by
  `prepare_compass_cs_inputs_relaxed.py` and has no role in the relaxed-space pipeline.
- No per-bridge network; the same `E_theta` handles arcs and bridges,
  selected only by the value of `s`.
- No post-hoc bridge smoothing, arc-endpoint fitting, or tangent alignment
  before CSV export.
