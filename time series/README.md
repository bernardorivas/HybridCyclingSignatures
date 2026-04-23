# time series/

Data-driven suspension construction from time series data (see `seed_context.md`).

## Shared modules

| File | Purpose |
|------|---------|
| `data_construction.py` | `build_augmented_dataset()` — system-agnostic: takes a base trajectory + jump pairs, produces the augmented suspension dataset (base points at s=0, bridge samples) |
| `plot_suspension.py` | `plot_augmented_suspension()` — 3D sanity-check plot of base trajectory + bridges in (state, s) space |

## Examples

### `rimless wheel/`

| File | Purpose |
|------|---------|
| `simulate.py` | `simulate_rimless_wheel()` — runs the rimless wheel ODE with ground-truth impact detection, returns smooth segments + jump pairs. Also has `extract_jump_pairs()` to concatenate segments into a base trajectory. |
| `example.py` | Minimal runnable demo: simulate 6 impacts, build augmented dataset, plot. Run with `python "time series/rimless wheel/example.py"` |
