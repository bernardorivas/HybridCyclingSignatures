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

Every run writes its model, configuration, and training log to `runs/<name>/`,
and loss and rollout figures to `figures/<name>/`.

The rimless wheel and bouncing ball train in two phases. Phase A trains from a
random start. Phase B reloads the Phase-A model and fine-tunes it on a shorter
schedule with the seam-velocity loss enabled (`--w-v 1.0`). The models kept in
this repository are the Phase-B results, so reproducing them takes both steps:

```bash
# Rimless wheel: Phase A, then Phase B fine-tuned from the Phase-A model
python chyll_v2/scripts/train_rimless.py
python chyll_v2/scripts/train_rimless.py \
  --load-from chyll_v2/runs/rimless_wheel/model.pt --w-v 1.0 \
  --curriculum-horizons 50,100 --steps-per-horizon 500 \
  --run-dir chyll_v2/runs/rimless_wheel_phaseB_finetune \
  --figure-dir chyll_v2/figures/rimless_wheel_phaseB_finetune
```

The bouncing ball follows the same two steps with `train_bouncing_ball.py`:
Phase A with `--w-v 0 --run-dir chyll_v2/runs/bouncing_ball_phaseA_wv0`, then
Phase B fine-tuning from that model into `bouncing_ball_phaseB_finetune`.

The compass-gait biped trains in a single phase, one command per slope. Each
slope is one gait in the period-doubling cascade:

```bash
python chyll_v2/scripts/train_compass.py --slope-config phi_1        # period 2
python chyll_v2/scripts/train_compass.py --slope-config phi_2        # period 4
python chyll_v2/scripts/train_compass.py --slope-config phi_3        # period 8
python chyll_v2/scripts/train_compass.py --slope-config phi_4_cloud  # chaotic
```

Once the compass runs exist, this script clusters their learned latent states
and reports how many distinct gaits each slope produces:

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
