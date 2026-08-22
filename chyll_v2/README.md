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

## Modeled systems

Three hybrid systems are included, in increasing order of difficulty.

- **Rimless wheel** — a spoked wheel rolling down a slope. Stable, two state
  variables.
- **Bouncing ball** — a ball bouncing under gravity, losing energy at each
  impact. Two state variables.
- **Compass-gait biped** — a two-legged walking model, four state variables.
  As the slope steepens it walks with period 1, then 2, 4, 8, and finally
  chaotically, a period-doubling route to chaos.

The numbered manuscript studies are not in one-to-one correspondence with
these system implementations.  The stable canonical compass gait is Example 2;
the varying-slope compass-gait bifurcation analysis is Example 3.  The Rössler
study under `../period_doubling/` is an auxiliary continuous-system comparison.

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

The stored default-slope `compass_gait_phi007` model supports the stable gait in
Example 2.  Example 3 uses a separate compass model for every sampled slope:

```bash
python chyll_v2/scripts/train_compass.py --slope-config phi_1        # period 2
python chyll_v2/scripts/train_compass.py --slope-config phi_2        # period 4
python chyll_v2/scripts/train_compass.py --slope-config phi_3        # period 8
python chyll_v2/scripts/train_compass.py --slope-config phi_4_cloud  # chaotic
```

Once the compass runs exist, this script applies DBSCAN to the post-impact
return-map states after encoding:

```bash
python chyll_v2/scripts/analyze_compass_cascade.py
```

This is a return-map diagnostic, not a cycling-signature computation or other
topological test.  The raw return-map states give the same 2, 4, and 8 cluster
counts, and the script uses one fixed relative DBSCAN tolerance rather than a
tolerance sweep.

## Results

The trained models reproduce each system's motion, including the post-impact
jumps, over long rollouts.  The canonical `phi007` cycling-signature result for
Example 2 reaches rank one on sufficiently long stored segments, but its
comparison space is scale-sensitive.  For Example 3, the DBSCAN return-map
diagnostic separates the sampled periodic regimes into 2, 4, and 8 clusters
and gives many clusters for the chaotic regime.  These counts diagnose temporal
recurrence on the chosen section; they are not evidence that the learned
embedding preserves topology.

Topology is analyzed separately with cycling signatures.  See
`cycling_signature/README.md`; the current matched-design cascade study and
its raw and untrained controls live under `../period_doubling/`.

## Requirements

Python with `torch`, `numpy`, `scipy`, `matplotlib`, and `torchdiffeq`. The
continuous latent flow is integrated with `torchdiffeq.odeint_adjoint`.
