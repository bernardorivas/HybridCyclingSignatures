# chyll_v2

A standalone implementation of the method of Teng, Liu, and Sreenath
(ICML 2026), *Embedding Hybrid Systems into Continuous Latent Vector Fields*.

A hybrid dynamical system mixes smooth motion with sudden jumps, for example a
walking robot whose leg swings forward and then strikes the ground. The jumps
make such a system hard to model directly. This code learns a continuous
representation instead: an encoder maps each state into a latent space, a
single smooth vector field advances it in time, and a decoder maps it back.
The jumps are absorbed into the smooth latent flow.

This directory is self-contained. It shares no code with the `src/` directory
(the main pipeline of this project), so the two approaches can be compared
directly and `chyll_v2/` can be removed without affecting `src/`.

## Example systems

Three hybrid systems are included, in increasing order of difficulty.

- **Rimless wheel** — a spoked wheel rolling down a slope. Stable, two state
  variables.
- **Bouncing ball** — a ball bouncing under gravity, losing energy at each
  impact. Two state variables.
- **Compass-gait biped** — a two-legged walking model, four state variables.
  As the slope steepens it walks with period 1, then 2, 4, 8, and finally
  chaotically, a period-doubling route to chaos.

## Layout

```
chyll_v2/
  chyll_v2/            core library: systems, networks, training, ODE solver
  scripts/             training entry points and the cascade analysis
  cycling_signature/   topology postprocessing (has its own README)
  compass_analysis/    compass-gait cascade analysis outputs
  runs/                trained models
  figures/             training figures
```

## Training

Each system has one training script. It writes the model, its configuration,
and a training log to `runs/<name>/`, and loss and rollout figures to
`figures/<name>/`.

```bash
python chyll_v2/scripts/train_rimless.py
python chyll_v2/scripts/train_bouncing_ball.py
python chyll_v2/scripts/train_compass.py
```

The compass-gait period-doubling cascade is studied separately. This script
clusters the learned latent states at a sequence of slopes and reports how
many distinct gait patterns each slope produces:

```bash
python chyll_v2/scripts/analyze_compass_cascade.py
```

## Results

The trained models reproduce each system's motion, including the post-impact
jumps, over long rollouts. For the compass-gait biped the cascade analysis
recovers the period-doubling sequence exactly: the learned latent states
separate into 2, 4, and 8 clusters at the period-2, -4, and -8 gaits, and
disperse into many clusters under chaos.

The topology of the learned embeddings is analyzed in `cycling_signature/`.
See that folder's README.

## Requirements

Python with `torch`, `numpy`, `scipy`, `matplotlib`, and `torchdiffeq`. The
continuous latent flow is integrated with `torchdiffeq.odeint_adjoint`.
