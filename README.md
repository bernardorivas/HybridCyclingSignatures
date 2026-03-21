# Hybrid Suspension

Learning continuous manifold embeddings of discontinuous hybrid dynamical systems via neural networks trained on a mapping-cylinder construction. Companion code for the *suspension.tex* manuscript.

Given a hybrid system $H = (X, \varphi, G, r)$ with state space $X$, continuous flow $\varphi$, guard $G$, and reset map $r$, the code learns an embedding of the hybrid suspension $\Sigma_H(X) = X \cup (G \times [0,1]) / {\sim}$ into $\mathbb{R}^{d+1}$ such that the discrete semiflow $\Phi_H(\tau, \cdot)$ lifts to a continuous map in the embedded space.

## Systems

- **Rimless Wheel** (2D state: $[\theta, \omega]$) -- stable, primary example.
- **Compass-Gait Biped** (4D state: $[\theta_{ns}, \theta_s, \dot\theta_{ns}, \dot\theta_s]$) -- experimental.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. Default compute device is MPS (Apple Silicon); falls back to CUDA, then CPU.

## Usage

```bash
# Train
python scripts/train.py            # Rimless wheel
python scripts/train_compass.py    # Compass gait

# Regenerate figures from a saved model
python scripts/replot.py

# Exact (non-learned) suspension visualization
python scripts/true_suspension.py [--height 0.5 --azim -60 --elev 30]

# Interactive 3D viewer
python scripts/explore.py [--azim 30 --elev 45]

# Long time-series generation + error analysis
python scripts/generate_series.py          # Rimless (1M steps, tau=0.001)
python scripts/generate_series_compass.py  # Compass (400 steps, tau=0.05)
```

Model weights are saved to `runs/{system}/model.pt`; figures to `figures/{system}/`.

## Architecture

Three cooperating networks learn the round-trip `state space -> continuous manifold -> state space`:

| Network | Signature | Role |
|---------|-----------|------|
| **E** (ExtrusionEncoder) | $[\mathbf{x}, s] \to \mathbb{R}^{d+1}$ | Identity at $s=0$; learned deformation scaled by $s$ on the mapping cylinder |
| **F** (FlowPredictor) | $\mathbb{R}^{d+1} \to \mathbb{R}^{d+1}$ | Residual network learning the time-$\tau$ discrete semiflow |
| **D** (StabilizationDecoder) | $\mathbb{R}^{d+1} \to \mathbf{x}$ | Inverts E; forces $s=0$ in output |

All are MLPs with GELU activations. Embedding dimension is `state_dim + 1` (plus optional `embed_extra`).

### Loss

| Term | Weight | Enforces |
|------|--------|----------|
| Commutativity | 1.0 | $F \circ E(x_i) = E(x_{i+1})$ -- dynamics commute with embedding |
| Gluing | 3.0 | $E(g, 1) = E(r(g), 0)$ -- quotient topology of the mapping cylinder |
| Reconstruction | 0.1 | $D \circ E(x) = x$ -- prevents collapse |

## Repository Layout

```
src/
  config.py         HybridSuspensionConfig dataclass, SystemType enum
  system.py         BaseHybridSystem, RimlessWheelHybridSystem, CompassGaitHybridSystem
  networks.py       E, F, D networks and SuspensionNetworks container
  losses.py         Composite loss computation
  visualize.py      Publication-quality plotting (Okabe-Ito palette)
scripts/
  train.py          Rimless wheel training loop
  train_compass.py  Compass-gait training loop
  replot.py         Reload saved model, regenerate figures
  true_suspension.py  Analytic suspension visualization
  explore.py        Interactive 3D matplotlib viewer
  generate_series.py / generate_series_compass.py  Long time-series + error analysis
  hybrid_manifold.py  Izhikevich neuron demo (illustrative)
figures/            Output PNG/PDF (rimless_wheel/, compass_gait/)
runs/               Saved model checkpoints
data/               Generated datasets and error reports
CyclingSignatures.jl/  Julia package for persistent homology (independent)
references/         Reference papers
```
