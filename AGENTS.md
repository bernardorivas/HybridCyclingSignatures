# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Computational topology / machine learning research project implementing "Hybrid Suspension" -- learning continuous manifold embeddings of discontinuous hybrid dynamical systems. The code models the **Rimless Wheel**, **Bouncing Ball**, and **Compass-Gait Biped**. In the manuscript, the stable canonical compass orbit is Example 2 and the varying-slope compass bifurcation analysis is Example 3. Companion code for the `suspension.tex` manuscript.

## Running the Project

```bash
source venv/bin/activate

# Rimless Wheel training (primary Section-4 pipeline)
python scripts/train_rimless.py

# Compass-Gait training (experimental)
python scripts/train_compass.py

# Regenerate figures from saved models
python scripts/replot.py

# Exact (non-learned) suspension visualization
python scripts/true_suspension.py [--height 0.5 --azim -60 --elev 30]

# Interactive 3D model viewer
python scripts/explore.py [--azim 30 --elev 45]

# Long time-series generation + error analysis
python scripts/generate_series.py          # Rimless (1M steps, tau=0.001)
python scripts/generate_series_compass.py  # Compass (400 steps, tau=0.05)
```

No build step, test framework, or linting. Dependencies: PyTorch, NumPy, SciPy, Matplotlib (in `venv/`). Default device is MPS (Apple Silicon); falls back to CUDA then CPU.

**Do not compile LaTeX.**

## Repository Layout

```
src/                  # Core library (on sys.path via scripts)
  config.py           # HybridSuspensionConfig dataclass + SystemType enum
  system.py           # BaseHybridSystem, RimlessWheelHybridSystem, CompassGaitHybridSystem
  networks.py         # E (encoder), F (flow predictor), D (decoder), SuspensionNetworks
  losses.py           # Two-phase Section-4 losses, including seam compatibility
  visualize.py        # Publication-quality plotting (Okabe-Ito palette, PUB_STYLE)
scripts/              # Entry points -- all prepend src/ to sys.path
  train.py            # Disabled legacy entry point; use train_rimless.py
  train_rimless.py    # Rimless Wheel two-phase training loop
  train_compass.py    # Compass-Gait training loop
  replot.py           # Reload saved model, regenerate figures
  true_suspension.py  # Analytic (no NN) suspension embedding
  explore.py          # Interactive 3D matplotlib viewer
  generate_series.py  # Long time-series generation + error analysis (rimless)
  generate_series_compass.py  # Same for compass gait
  hybrid_manifold.py  # Izhikevich neuron demo (separate hybrid system, illustrative)
figures/              # Output PNG/PDF figures (rimless_wheel/, compass_gait/)
runs/                 # Saved model.pt files (rimless_wheel/, compass_gait/)
data/                 # Generated .npy/.csv datasets + error reports
CyclingSignatures.jl/ # Separate Julia package for topological data analysis (persistent homology, cycling signatures). Not coupled to the Python training code.
references/           # Reference papers (Goswami compass gait, etc.)
```

## Architecture

Three cooperating networks learn the embedding `(suspension state) -> (continuous manifold) -> (suspension state)`:

- **E (Encoder)**: `R^{n+1} -> R^d`. A generic residual MLP initialized near the identity-padded input; it is not pinned at `s=0`.
- **F (FlowPredictor)**: `R_+ x R^d -> R^d`. A time-conditioned residual MLP learning the discrete semiflow.
- **D (Decoder)**: `R^d -> R^{n+1}`. A generic residual decoder that reconstructs the cylinder coordinate as well as the physical state.

All three use GELU MLPs. Dimensions adapt to `config.state_dim` (2 for rimless wheel, 4 for compass gait), with `embed_dim = state_dim + 1 + embed_extra`. Archived checkpoints are loaded through explicit legacy network classes selected from their state-dict shapes.

### Loss Function (losses.py)

| Loss | Default weight | Purpose |
|------|----------------|---------|
| Dynamics | 1.0 | `F(dt, E(x_i)) = E(x_{i+1})` |
| Gluing | 3.0 | `E(guard, 1) = E(reset, 0)` |
| Seam | 1.0 | Align arc and bridge tangents at both seams |
| Conformal | 0.01 | Control local metric distortion of `E` |
| Anti-collapse | 1.0 | Maintain latent-coordinate variance |
| Reconstruction | 1.0 | `D(E(x)) = x` away from quotient-boundary ambiguity |

The legacy one-sided UTB anchor remains available with default weight zero. Phase I trains `E` and `F` without reconstruction; Phase II freezes them and trains `D` on masked reconstruction.

### Hybrid Systems (system.py)

`BaseHybridSystem` provides the generic mapping-cylinder semiflow (`generate_tau_timeseries`). The semiflow augments state with `s in [0,1]` tracking position on the mapping cylinder. Each step has two phases: (1) if `s > 0`, advance along cylinder until `s=1`, apply reset, continue in base space; (2) if `s ~ 0`, integrate ODE, and if guard is hit before tau expires, enter cylinder for the remainder.

Subclasses define:
- `base_vector_field` -- continuous ODE (integrated via `scipy.integrate.solve_ivp`)
- `is_guard_hit` -- event function for impact detection
- `reset_map` -- instantaneous state transformation at impact
- `is_moving_forward` -- directionality check
- `guard_direction` -- which direction the event function crosses zero (+1 rimless, -1 compass)

**Rimless Wheel** (2D state): simple pendulum + spoke ground-strike. State: `[theta, omega]`.

**Compass Gait** (4D state): Lagrangian biped with mass matrix dynamics + angular momentum impact map. State: `[theta_ns, theta_s, dtheta_ns, dtheta_s]`. Uses `get_matrices(q, dq)` returning (M, N, G) 2x2 arrays. Impact map derived from angular momentum conservation with leg-role swap.

### Data Generation

`generate_suspension_dataset()` samples initial conditions (half from base space at `s=0`, half from the cylinder at `s>0`), integrates `generate_tau_timeseries` to produce `(x_i, x_{i+1})` pairs for supervised training.

### Training Pipeline

`train_rimless.py` and `train_compass.py` generate data, train `E` and `F` with the Phase-I objective, freeze them, train `D` with masked reconstruction, validate, plot, and save to `runs/{system}/model.pt`. They use AdamW, cosine scheduling, and gradient clipping at `max_norm=2.0`.

## Key Design Decisions

- **Singleton config**: `config = HybridSuspensionConfig()` in `config.py` is mutated directly by scripts. Training scripts expose selected overrides through CLI flags and write them back to this singleton before execution.
- **sys.path injection**: Scripts prepend `src/` to `sys.path` at the top; imports within `src/` use bare module names (`from config import config`).
- **Visualization**: Okabe-Ito colorblind-safe palette and shared `PUB_STYLE` dict. Functions are system-specific: `plot_hybrid_suspension_rimless`, `plot_hybrid_suspension_compass`, `compute_deep_crossing_validation_rimless`.
- **Output paths**: Model weights to `runs/{system}/model.pt`; figures to `figures/{system}/`; data to `data/{system}/`.
- **Device fallback**: config default is `"mps"`; scripts check `torch.backends.mps.is_available()` then CUDA then CPU.

## Known Issues

- `scripts/train.py` is retained as an explicitly non-runnable historical entry point. Use `scripts/train_rimless.py`.

## Mathematical Correspondence

The code realizes the hybrid suspension semiflow from `suspension.tex`:

| Math | Code |
|------|------|
| H = (X, phi, G, r) | `BaseHybridSystem` subclasses |
| X' = X ∪ (G x [0,1]) (mapping cylinder) | `s` parameter in state vector |
| ~ (quotient: (g,1) ~ r(g)) | Gluing loss |
| embed: Sigma_H(X) -> R^d | E (Encoder) |
| Phi_H(tau, .) | F (FlowPredictor) |
| learned inverse of the embedding | D (Decoder) |
| F . E = E . phi' (Thm 4.1) | Commutativity loss |
| (g,1) ~ iota(r(g)) (Def 3.4) | Gluing loss |
