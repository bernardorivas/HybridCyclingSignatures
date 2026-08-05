# Goswami Compass-Gait Simulations

This folder contains self-contained Python and MATLAB implementations of the
nominal compass-gait model used for the period-doubling examples in Goswami,
Thuilot, and Espiau (1998).

The included cases are:

- 4.00 degrees: period 1
- 4.75 degrees: period 2
- 5.00 degrees: period 4
- 5.02 degrees: period 8
- 5.20 degrees: chaotic gait

## Python

From the repository root, run the exact Python installation already verified
on this Mac:

```bash
/usr/bin/python3 scripts/goswami_period_doubling/compass_goswami.py
```

To create MP4 animations as well:

```bash
/usr/bin/python3 scripts/goswami_period_doubling/compass_goswami.py --video
```

All five videos use the same 0.00-12.00 second physical-time window at 50 fps.

Outputs are written to `compass_goswami_output/` in the repository root.
The script also creates `figure10_modern_phase_portraits.png` and `.pdf`, a
five-panel phase-plane comparison of the period-1, 2, 4, 8, and chaotic gaits.

If `/usr/bin/python3` is unavailable on another computer, create a local
environment instead:

```bash
python3 -m venv .venv-compass
source .venv-compass/bin/activate
python -m pip install numpy scipy matplotlib pillow
python scripts/goswami_period_doubling/compass_goswami.py --video
```

## MATLAB

Open `compass_goswami.m` in MATLAB and press **Run**, or execute:

```matlab
run('scripts/goswami_period_doubling/compass_goswami.m')
```

MATLAB outputs are written to `compass_goswami_output_matlab/`.

## Learned continuous latent trajectories

`export_learned_latents.py` consumes the generated Python CSVs directly.  It
keeps every physical flow arc, inserts the mapping-cylinder coordinate
`s = 0 ... 1` at each impact, encodes the ordered augmented trajectory with
the corresponding trained CHyLL-v2 checkpoint, and evaluates the learned
latent vector field along it.

Run all five cases from the repository root:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl-compass \
  /Users/kaitoi/.venvs/sci/bin/python \
  scripts/goswami_period_doubling/export_learned_latents.py
```

Run only selected cases or change the bridge resolution:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl-compass \
  /Users/kaitoi/.venvs/sci/bin/python \
  scripts/goswami_period_doubling/export_learned_latents.py \
  --cases period2,period4,period8 --n-bridge 100
```

The outputs live under
`chyll_v2/cycling_signature/data/compass_gait_goswami_csv/<case>/`.
For each case, the important files are:

- `*_positions.csv`: ordered 11-dimensional latent positions, one sample per
  row, space-delimited for Julia.
- `*_tangents.csv`: unit learned-vector-field tangents in the same format.
- `*_latent_velocity.npy`: unnormalized learned latent vector field `V(z)`.
- `*_diff_tangents.npy`: unit forward tangents of the encoded trajectory,
  computed without chords across piece boundaries.
- `*_augmented_inputs.npy`: the source `(theta_ns, theta_s, dtheta_ns,
  dtheta_s, s)` samples.
- `*_metadata.csv`: sample order, relaxed time, physical time, arc/bridge tag,
  step, impact number, and cylinder coordinate.
- `*_successor_edges.npy/.csv`: the directed `i -> i+1` trajectory graph,
  with flow, bridge-entry, and quotient-seam transitions labeled in the CSV.
- `*_postimpact_latent.npy`: the latent Poincare-return sequence.
- `report_*.json`: gluing, reconstruction, vector-field alignment, and period
  closure checks.

`manifest.json` records the physical inputs and checkpoint used for each
case.  `validation_summary.csv` collects the main checks across all five.
The period-2/4/8/chaotic trajectories use slope-matched checkpoints.  The
period-1 trajectory at 4.00 degrees uses the nearby checkpoint trained at
`phi = 0.07 rad = 4.0107 degrees`; the manifest records this explicitly.

### Inputs for downstream cycling-signature computation

The exported position/tangent pairs already match the input format of
`run_subsegments.jl`.  For example:

```bash
julia --project="time series/cycling_signature" \
  chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir chyll_v2/cycling_signature/data/compass_gait_goswami_csv/period2 \
  --base continuous_lift_goswami_period2_vfield
```

Change `period2` in both places to `period1`, `period4`, `period8`, or
`chaos` for the other systems.

### Conley-oriented data

Use `*_positions.csv` or the same-named `.npy` file as the sampled latent
state set, `*_latent_velocity.npy` as the vector field on that set, and
`*_successor_edges.npy/.csv` as the initial directed combinatorial map.
`*_metadata.csv` can isolate individual flow/bridge pieces or select the
Poincare section.  The manifest links each data set back to its model and
config so the latent vector field can also be evaluated at new points.

The learned quotient gluing is approximate, not algebraically exact.  Its
residual `||E(g,1)-E(r(g),0)||` is reported rather than silently snapping or
altering the learned coordinates.
