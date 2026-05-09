# Relaxed-Space Rimless-Wheel Cycling-Signature Pipeline — Specification

This file documents the pipeline architecture, parallel to
`pipelines/compass_gait_relaxed/SPEC.md`. For the operational runbook see
[`README.md`](README.md). Anything that has not yet been ported from the
compass pipeline is flagged **TODO**.

## 0. Encoder posture

The encoder `E_theta` is the same generic Section-4 residual MLP used in the
compass pipeline. In particular:

| Aspect               | Bernardo's old `ExtrusionEncoder` (`38aebfd`) | This pipeline (relaxed, generic) |
|----------------------|-----------------------------------------------|----------------------------------|
| Base-space embedding | $E(x, 0) = (x, 0)$ identity by construction   | Generic MLP; base-space injectivity is a learned property |
| Cylinder embedding   | $E(x, s) = (x, 0) + s \cdot \mathrm{MLP}(x, s)$ | $E(x) = \mathrm{pad}(x, d) + \mathrm{MLP}(x)$, last layer near-zero init |
| Architectural prior  | Yes — base = identity, cylinder = small extrusion | None |
| Manuscript reading   | Pre-empts the injectivity question of Prop 4.1 | Lets injectivity be tested empirically |

This is the same pivot recorded in the compass `SPEC.md` and the
architecture-pivot memory.

## 1. System definition

`src/system.py` — `RimlessWheelHybridSystem`. State `[theta, omega]`.
Physical parameters from `src/config.py`: `alpha = 0.4` (spoke half-angle),
`gamma = 0.2` (slope), giving `theta_guard = alpha + gamma = 0.6` and
`theta_reset = 2*gamma - theta_guard = -0.2`. Restitution coefficient
`omega_restitution = cos(2*alpha)`.

Vector field: simple pendulum `theta_dot = omega`, `omega_dot = sin(theta)`.

Guard: `theta = theta_guard` with `omega > 0`, `guard_direction = +1`.

Reset map: `(theta, omega) -> (theta_reset, omega * cos(2*alpha))`.

Relaxed semiflow `phi'` on $X \cup (G \times [0, 1])$ is
`BaseHybridSystem.generate_tau_timeseries` with `tau = 0.05`
(`config.integration_tau`). The mapping cylinder parameter `s` is tracked
as the 3rd state coordinate.

## 2. Networks (`src/networks.py`)

Input is $\mathbb{R}^{n+1} = \mathbb{R}^{3}$ (2-D rimless state plus `s`).
Embedding dimension $d = n + 1 + \mathrm{embed\_extra} = 3$ at the baseline
(`embed_extra = 0`, set in `scripts/train_rimless.py`). An `embed7`
experiment (analogous to compass `embed9`) is the natural follow-up if
collapse pressure shows up in diagnostics.

Architecture is exactly the same `Encoder`, `FlowPredictor`, `Decoder` used
by the compass pipeline. See compass `SPEC.md` §2 for the full layer-level
details.

## 3. Training (`scripts/train_rimless.py`)

Two-phase procedure matching Algorithm 1 of the manuscript and the compass
schedule.

### Dataset

`generate_suspension_dataset` (`src/system.py`) with `num_orbits = 3500`,
`points_per_orbit = 20` steps of `tau = 0.05`. Half of the initial
conditions start in the base space with `theta ~ U(theta_bounds)`,
`omega ~ U(0, omega_bounds[1])`; the other half start on the guard surface
`theta = theta_guard` with `omega ~ U(omega_bounds)` and random
`s ~ U(0.01, 1)`.

### Phase I (encoder + semiflow; 120 epochs)

AdamW on E + F parameters, `lr = 8e-4`, `weight_decay = 1e-5`, cosine LR
schedule, batch 1024, gradient clip 2.0. Five loss terms, no recon:

| Term    | Weight | Expression                                               |
|---------|--------|----------------------------------------------------------|
| `L_dyn` | 1.0    | `MSE(Ψ(τ, E(x_i)), E(x_{i+1}))`                          |
| `L_glue`| 3.0    | `MSE(E(g, 1), E(r(g), 0))`, 256 sampled guard states     |
| `L_conf`| 1e-4   | `mean_i ||λ_i I - J_i^T J_i||_F^2` over ≤64 samples/batch (cut from default 1e-2 per compass tuning) |
| `L_coll`| 1.0    | `sum_k ReLU(Λ - Var_batch(E(x)_k))`, `Λ = 0.1`            |
| `L_utb` | 1.0    | `MSE(v_cyl(g, 0), v_X(g)) + MSE(v_cyl(g, 1), v_X(r(g)))` |

`v_cyl` is a finite-difference of `E` along the `s` direction with
`h = 0.02`. `v_X` is padded with zeros in the `s`-slot to match `d`.

For rimless, the guard set is one-dimensional (a half-line of constant
`theta = theta_guard` parameterised by `omega > 0`), so `_sample_guard_states`
only varies `omega ~ U(0, omega_bounds[1])`.

### Phase II (decoder only; 150 epochs)

Freeze E and F. AdamW on D alone, `lr = 2e-3`, cosine schedule. Single
term:

| Term     | Weight | Expression                                            |
|----------|--------|-------------------------------------------------------|
| `L_recon`| 1.0    | `MSE(D(E(x)), x)` on samples with `s = 0` or `ε < s < 1-ε`, `ε = 0.05` |

## 4. Lift export for cycling signature — **TODO**

A rimless analogue of
`time series/cycling_signature/prepare_compass_cs_inputs_relaxed.py` does
not exist yet. Once written it should:

1. Load `runs/rimless_wheel/model.pt`.
2. Simulate the canonical rimless limit-cycle rollout via
   `time series/rimless wheel/simulate.py` (verify this script exposes a
   `simulate_rimless` returning `segments` and `jump_pairs` matching the
   compass interface; if not, write a thin wrapper).
3. Encode arcs (at `s = 0`) and bridges (50 interior `s ∈ (0, 1)` samples
   per impact) through the trained encoder.
4. Write `.npy` / `.csv` artifacts and a Phase-A diagnostics report
   (reconstruction, collapse screen, tangent quality, inter-bridge
   separation under DynamicDistance).

## 5. Julia cycling signature

`time series/cycling_signature/run_cycling_signature.jl` is system-agnostic
once the lift CSVs exist. Same `(boxsize, sb_radius) ∈ {0.30, 0.20, 0.10,
0.05} × {1, 2, 4}` sweep as compass.

## 6. Heatmap figure — **TODO**

Adapt `time series/cycling_signature/plot_compass_rank_heatmap.py` to
read the rimless barcodes and render
`figures/rimless_wheel/fig_rimless_cycling_rank_heatmap.{pdf,png}` and
its appendix variant against the analytic baseline.

## What this pipeline is not doing

- No model-aware structural pinning. The encoder is a generic MLP.
- No bouncing-ball merge. Bernardo proposed rolling the bouncing-ball
  example into the rimless section; that is a §4 narrative decision, not
  a pipeline change.
- No analytic bridge fallback. Whatever the trained encoder gives you on
  the cylinder is what gets passed to Julia.
