# Cycling signatures of the chyll_v2 embeddings

This folder checks whether the learned embeddings preserve the *topology* of
the original hybrid systems, using the cycling-signature method
(`CyclingSignatures.jl`, by David Hien).

A cycling signature measures how a trajectory loops through space: it counts
the independent cycles traced by a curve. A periodic system traces one cycle,
so a faithful embedding of it should also have one cycle. Comparing the two is
a test of whether the learned representation kept the system's structure.

This folder is separate from the training code in `chyll_v2/chyll_v2/` and
`chyll_v2/scripts/`.

## Layout

```
cycling_signature/
  export/        trained model -> latent-lift CSV files
  julia/         cycling-signature computations (Julia)
  plot/          figure builders
  diagnostics/   tangent-coherence checks
  data/          exported lifts and computed rank summaries
  figures/       generated figures
```

## Pipeline

The analysis has three stages. The rimless wheel is shown here as a worked
example; the bouncing ball and compass-gait biped follow the same steps with
`prepare_bouncing_ball_lift.py` and `prepare_compass_lift.py`.

**1. Export** a latent lift (positions and unit tangents) from a trained model:

```bash
python chyll_v2/cycling_signature/export/prepare_rimless_lift.py \
  --model chyll_v2/runs/rimless_wheel_phaseB_finetune/model.pt \
  --config chyll_v2/runs/rimless_wheel_phaseB_finetune/config.json \
  --base continuous_lift_chyll_v2_phaseB
```

**2. Compute** the cycling signature over random subsegments of the lift.
Additional flags set the cover scale and the number of samples; see the
script for defaults.

```bash
julia --project="time series/cycling_signature" \
  chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
  --base continuous_lift_chyll_v2_phaseB
```

**3. Plot** the rank distribution and heatmaps:

```bash
python chyll_v2/cycling_signature/plot/plot_signature_figures.py
```

## Results

- **Rimless wheel** — the embedding recovers the expected single cycle.
- **Compass-gait biped** — each gait of the period-doubling cascade is
  recovered, and the chaotic gait is distinguished from the periodic ones.
- **Bouncing ball** — the ball's orbit spirals slowly toward rest rather than
  closing into a loop, so its cycling signature is only partially recovered.

A circle (`data/sanity_circle/`) is included as a known single-cycle baseline.
