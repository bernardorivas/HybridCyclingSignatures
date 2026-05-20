# Successful CHyLL v2 Examples Repro Bundle

This bundle contains only the successful CHyLL v2 examples so far:

- rimless wheel Phase-B fine-tune
- bouncing ball Phase-B fine-tune
- compass gait incline cascade: period 2, period 4, period 8, chaotic/cloud

It intentionally excludes smoke runs, failed runs, obsolete top-level
`data/compass_gait` artifacts, and intermediate ablations.

## Quick Map

See `examples.csv` for the machine-readable table. The data folders are:

```text
data/rimless_wheel_phaseB/
data/bouncing_ball_phaseB_5impacts/
data/bouncing_ball_phaseB_15impacts/
data/compass_period2_4p75deg_phi1/
data/compass_period4_5p00deg_phi2/
data/compass_period8_5p02deg_phi3/
data/compass_chaotic_cloud_5p20deg_phi4/
```

Each data folder has:

- `latent_lift_positions.npy`
- `latent_lift_tangents.npy`
- `latent_lift_positions.csv`
- `latent_lift_tangents.csv`
- `export_report.txt`
- `cycling_signature_outputs/`

Compass folders also have:

- `postimpact_states.npy`
- `postimpact_latent_z.npy`
- `postimpact_labels.npy`

The chaotic compass case also includes its return cloud:

- `return_cloud.npy`
- `return_cloud_summary.txt`

## Models

The matching trained models are under `models/`:

```text
models/rimless_wheel_phaseB/
models/bouncing_ball_phaseB/
models/compass_period2_4p75deg_phi1/
models/compass_period4_5p00deg_phi2/
models/compass_period8_5p02deg_phi3/
models/compass_chaotic_cloud_5p20deg_phi4/
```

Each model folder contains at least:

- `model.pt`
- `config.json`
- `train_log.jsonl`

## Meeting Slides

The Beamer walkthrough deck for a Thursday user-guide meeting is:

```text
user_guide_beamer.pdf
user_guide_beamer.tex
```

It is written as a meeting guide for colleagues who know the mathematics but
have not used this repository: concrete about file names/data layout, and
technical about the learned maps, losses, checkpoints, and regeneration path.

The PDF is precompiled. To rebuild it on a machine with LaTeX installed, run
this from the bundle folder:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error user_guide_beamer.tex
pdflatex -interaction=nonstopmode -halt-on-error user_guide_beamer.tex
```

## Load Data

```python
import numpy as np

case = "compass_period4_5p00deg_phi2"
Z = np.load(f"data/{case}/latent_lift_positions.npy")
T = np.load(f"data/{case}/latent_lift_tangents.npy")

print(Z.shape, T.shape)
```

For compass post-impact section data:

```python
states = np.load(f"data/{case}/postimpact_states.npy")
section_z = np.load(f"data/{case}/postimpact_latent_z.npy")
labels = np.load(f"data/{case}/postimpact_labels.npy")
```

## Regenerate Lift Data From Models

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python export_all_lifts.py
```

This regenerates lift files under `regenerated/<case>/` using the bundled
source code in `source/` and model weights in `models/`.

The rimless and bouncing-ball exported tangents use tag-aware finite
differences. The compass incline exported tangents use the learned latent
vector field (`vfield`), matching the files in `data/`.

## Rerun Cycling-Signature Subsegment Computations

The existing outputs are already included in each folder's
`cycling_signature_outputs/`.

To rerun, use:

```text
source/chyll_v2/cycling_signature/julia/run_subsegments.jl
```

That script expects David's `CyclingSignatures.jl` Julia environment. In the
original repo, that environment is under `time series/cycling_signature/`.

Example:

```bash
julia --project="time series/cycling_signature" \
  source/chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir regenerated/rimless_wheel_phaseB \
  --base continuous_lift_chyll_v2_phaseB \
  --boxsize 0.3 \
  --sb-radius 1 \
  --segment-lengths 20:10:300 \
  --n-runs 100
```

## Included Figures

The `figures/` folder includes rollout overlays for every successful model:

- `rimless_wheel_phaseB_rollout_vs_truth.png`
- `rimless_wheel_phaseB_rollout_vs_truth_index.png`
- `rimless_wheel_phaseB_rollout_vs_truth_long.png`
- `bouncing_ball_phaseB_rollout_vs_truth.png`
- `bouncing_ball_phaseB_rollout_vs_truth_index.png`
- `compass_period2_4p75deg_phi1_rollout_vs_truth.png`
- `compass_period4_5p00deg_phi2_rollout_vs_truth.png`
- `compass_period8_5p02deg_phi3_rollout_vs_truth.png`
- `compass_chaotic_cloud_5p20deg_phi4_rollout_vs_truth.png`

It also includes training loss-history plots for every successful training:

- `rimless_wheel_phaseB_loss_history.png`
- `bouncing_ball_phaseB_loss_history.png`
- `compass_period2_4p75deg_phi1_loss_history.png`
- `compass_period4_5p00deg_phi2_loss_history.png`
- `compass_period8_5p02deg_phi3_loss_history.png`
- `compass_chaotic_cloud_5p20deg_phi4_loss_history.png`

It also includes selected orientation plots:

- rank heatmaps for rimless and bouncing ball
- compass incline latent cluster panel
- beta-1 comparison-space sweep

## Verification Before Packaging

The staged bundle was checked before zipping:

- rimless lift shape: `(960, 7)`
- bouncing-ball 5-impact lift shape: `(1069, 7)`
- bouncing-ball 15-impact lift shape: `(2004, 7)`
- compass lift shapes: `(2032, 11)`, `(2036, 11)`, `(2036, 11)`, `(2031, 11)`
- all tangent rows are unit norm up to floating-point roundoff
- all model folders contain `model.pt` and `config.json`
- rollout figures are included for every successful model
- loss-history figures are included for every successful training

`MANIFEST.csv` contains SHA-256 hashes for the packaged files.
