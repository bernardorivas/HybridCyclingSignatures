# Rimless-Wheel Relaxed-Space Cycling-Signature Pipeline — NON-RUNNABLE/PARTIAL

> **NON-RUNNABLE orchestrator.** `run_pipeline.sh` exits before executing any
> stage. Stages 1–4 can be invoked individually, but stage 5 and an end-to-end
> validation are still missing.

Parallel of `pipelines/compass_gait_relaxed/`, applied to the rimless-wheel
hybrid system. The end goal is the same: train the generic Section-4 encoder
on the relaxed semiflow, embed a canonical limit-cycle rollout, run David
Hien's `CyclingSignatures.jl` over a `(boxsize, sb_radius)` grid, and render
the heatmap figure.

The full architecture spec is in [`SPEC.md`](SPEC.md). This file is the
runbook. Anything below marked **TODO** is a stage that has not been ported
from the compass pipeline yet.

## Status

| Stage | Status |
|-------|--------|
| 1. Train relaxed-space encoder       | **live** (`scripts/train_rimless.py`) |
| 2. Decoded-rollout evaluation        | **live** (`scripts/evaluate_decoded_rollout_rimless.py`, plus `_basin.py` for off-LC ICs) |
| 3. Encoded-lift export + diagnostics | **live** (`time series/cycling_signature/prepare_rimless_cs_inputs_relaxed.py`) |
| 4. Julia cycling signature           | reusable as-is (script is system-agnostic) |
| 5. Heatmap figure                    | **TODO** — adapt `plot_compass_rank_heatmap.py` |

## How to run today

From the repository root:

```bash
python scripts/train_rimless.py
```

Produces `runs/rimless_wheel/model.pt` after Phase I (E + F) and Phase II
(D alone) and writes
`figures/rimless_wheel/fig_rimless_optimization_losses.png`.

The partial [`run_pipeline.sh`](run_pipeline.sh) is deliberately disabled and
does not train or write artifacts. Invoke the live stages individually while
the heatmap renderer and end-to-end validation remain outstanding.

## What this pipeline produces (eventually)

Main manuscript figure (parallel to compass):

- `figures/rimless_wheel/fig_rimless_cycling_rank_heatmap.{pdf,png}` —
  panels for representative impact counts.

Supporting artifacts from the individual live stages:

- `runs/rimless_wheel/model.pt`
- `data/rimless_wheel/continuous_lift_relaxed{,_n*}.{npy,csv}`
- `data/rimless_wheel/continuous_lift_relaxed{,_n*}_tangents.{npy,csv}`
- `data/rimless_wheel/report_relaxed_encoder_diagnostics.txt`
- `data/rimless_wheel/barcode_H1_relaxed{,_n*}.csv`

## Where the code actually lives

| Role                            | File                                                                       |
|---------------------------------|----------------------------------------------------------------------------|
| Physical system + guard         | `src/system.py` (`RimlessWheelHybridSystem`)                               |
| Network architecture            | `src/networks.py` (`Encoder`, `FlowPredictor`, `Decoder`)                  |
| Loss                            | `src/losses.py` (handles rimless via `_sample_guard_states`)               |
| Training script                 | `scripts/train_rimless.py`                                                 |
| Canonical rollout simulator     | `time series/rimless wheel/simulate.py` (TODO: confirm interface matches `simulate_compass_gait`) |
| Lift export                     | `time series/cycling_signature/prepare_rimless_cs_inputs_relaxed.py`        |
| Julia cycling signature         | `time series/cycling_signature/run_cycling_signature.jl` (system-agnostic) |
| Heatmap figure                  | TODO — `time series/cycling_signature/plot_rimless_rank_heatmap.py`        |

The Julia runner uses the locked `period_doubling/julia/` project; it does
not resolve the Windows-pathed Manifest under `time series/cycling_signature/`.

## Notes

The encoder is the same generic Section-4 residual MLP as the compass
pipeline:

```
E(x) = pad(x, d) + MLP(x)
```

with the last layer initialised near zero. There is no architectural prior
that fixes the base space to the identity (this is the explicit pivot away
from Bernardo's older `ExtrusionEncoder`); base-space injectivity is a
learned property.
