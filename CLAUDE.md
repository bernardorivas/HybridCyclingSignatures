# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computational topology / machine learning research project implementing "Hybrid Suspension" -- learning continuous manifold embeddings of discontinuous hybrid dynamical systems. Two example systems: the **Rimless Wheel** (stable) and the **Compass-Gait Biped** (in progress). Companion code for the `suspension.tex` manuscript.

## Running the Project

```bash
source venv/bin/activate

# Rimless Wheel training (primary, stable)
python scripts/train.py

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
  losses.py           # Composite loss: commutativity + gluing + reconstruction
  visualize.py        # Publication-quality plotting (Okabe-Ito palette, PUB_STYLE)
scripts/              # Entry points -- all prepend src/ to sys.path
  train.py            # Rimless Wheel training loop
  train_compass.py    # Compass-Gait training loop
  replot.py           # Reload saved model, regenerate figures
  true_suspension.py  # Analytic (no NN) suspension embedding
  explore.py          # Interactive 3D matplotlib viewer
  generate_series.py  # Long time-series generation + error analysis (rimless)
  generate_series_compass.py  # Same for compass gait
  hybrid_manifold.py  # Izhikevich neuron demo (separate hybrid system, illustrative)
  temp/               # Throwaway debugging scripts (limit-cycle finders, physics debuggers)
figures/              # Output PNG/PDF figures (rimless_wheel/, compass_gait/)
runs/                 # Saved model.pt files (rimless_wheel/, compass_gait/)
data/                 # Generated .npy/.csv datasets + error reports
CyclingSignatures.jl/ # Separate Julia package for topological data analysis (persistent homology, cycling signatures). Not coupled to the Python training code.
references/           # Reference papers (Goswami compass gait, etc.)
```

## Architecture

Three cooperating networks learn the embedding `(state space) -> (continuous manifold) -> (state space)`:

- **E (ExtrusionEncoder)**: `[state, s] -> R^{d+1}`. Identity at `s=0` (base space); learned deformation scaled by `s` on the mapping cylinder (`s > 0`).
- **F (FlowPredictor)**: `R^{d+1} -> R^{d+1}`. Residual network learning the time-tau discrete semiflow in embedded space.
- **D (StabilizationDecoder)**: `R^{d+1} -> state`. Inverts E via small residual correction; forces `s=0` in output.

All three are MLPs built by `_build_mlp()` with GELU activations. Dimensions adapt to `config.state_dim` (2 for rimless wheel, 4 for compass gait) via the `+1` for the cylinder parameter `s`. When `embed_extra > 0`, embedding dimension is `state_dim + 1 + embed_extra`.

### Loss Function (losses.py)

| Loss | Weight | Purpose |
|------|--------|---------|
| Commutativity | 1.0 | `F(E(x_i)) = E(x_{i+1})` -- dynamics commute with embedding |
| Gluing | 3.0 | `E(guard, 1) = E(reset, 0)` -- enforces quotient topology |
| Reconstruction | 0.1 | `D(E(x)) = x` -- prevents manifold collapse |

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

Both `train.py` and `train_compass.py` follow: generate dataset -> DataLoader -> AdamW + CosineAnnealingLR -> training loop with `calculate_composite_losses` -> validation -> plot -> save to `runs/{system}/model.pt`. Gradient clipping at `max_norm=2.0`.

## Key Design Decisions

- **Singleton config**: `config = HybridSuspensionConfig()` in `config.py` is mutated directly by scripts. `train_compass.py` sets `config.system_type = SystemType.COMPASS_GAIT` before importing system/network classes. No CLI argument parsing.
- **sys.path injection**: Scripts prepend `src/` to `sys.path` at the top; imports within `src/` use bare module names (`from config import config`).
- **Visualization**: Okabe-Ito colorblind-safe palette and shared `PUB_STYLE` dict. Functions are system-specific: `plot_hybrid_suspension_rimless`, `plot_hybrid_suspension_compass`, `compute_deep_crossing_validation_rimless`.
- **Output paths**: Model weights to `runs/{system}/model.pt`; figures to `figures/{system}/`; data to `data/{system}/`.
- **Device fallback**: config default is `"mps"`; scripts check `torch.backends.mps.is_available()` then CUDA then CPU.

## Known Issues

- `train.py` imports `compute_deep_crossing_validation` and `plot_hybrid_suspension` but `visualize.py` exports `compute_deep_crossing_validation_rimless` and `plot_hybrid_suspension_rimless` (suffixed names). This is a stale import that will cause ImportError.

## Mathematical Correspondence

The code realizes the hybrid suspension semiflow from `suspension.tex`:

| Math | Code |
|------|------|
| H = (X, phi, G, r) | `BaseHybridSystem` subclasses |
| X' = X ∪ (G x [0,1]) (mapping cylinder) | `s` parameter in state vector |
| ~ (quotient: (g,1) ~ r(g)) | Gluing loss |
| embed: Sigma_H(X) -> R^{d+1} | E (ExtrusionEncoder) |
| Phi_H(tau, .) | F (FlowPredictor) |
| pi_0 . embed^{-1} | D (StabilizationDecoder) |
| F . E = E . phi' (Thm 4.1) | Commutativity loss |
| (g,1) ~ iota(r(g)) (Def 3.4) | Gluing loss |
