# time series/

Support code for constructing suspension data and cycling-signature inputs from
hybrid-system time series.

## Shared modules

| File | Purpose |
|------|---------|
| `data_construction.py` | System-agnostic jump detection and augmented suspension construction from a base trajectory and jump pairs |

## System simulators

### `rimless wheel/`

| File | Purpose |
|------|---------|
| `simulate.py` | Runs the rimless-wheel ODE with impact detection and returns smooth segments plus jump pairs |

### `compass gait/`

| File | Purpose |
|------|---------|
| `simulate.py` | Runs the Compass-gait hybrid dynamics used by downstream exporters and period-doubling analyses |

## Cycling signatures

`cycling_signature/` contains the retained Python preparation/plotting tools
and Julia environment used by the relaxed-space pipelines.
