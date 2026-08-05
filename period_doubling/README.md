# Period-Doubling Cascade Package

## Purpose

This package generates fixed-time-span trajectories for two systems exhibiting period-doubling routes to chaos: the Roessler attractor and the passive compass-gait biped. The data serves as reference datasets for the cycling-signatures analysis, enabling side-by-side comparison of continuous and hybrid dynamical systems undergoing period-doubling bifurcations.

## Systems

### Roessler Attractor

Continuous dissipative 3D system governed by:

```
dx/dt = -y - z
dy/dt = x + a*y
dz/dt = b + z*(x - c)
```

| Period | c   | a   | b   | Expected Period |
|--------|-----|-----|-----|-----------------|
| 1      | 4.0 | 0.1 | 0.1 | 1               |
| 2      | 6.0 | 0.1 | 0.1 | 2               |
| 4      | 8.5 | 0.1 | 0.1 | 4               |
| 8      | 8.7 | 0.1 | 0.1 | 8               |
| Chaotic| 9.0 | 0.1 | 0.1 | None            |

Initial condition: x0 = (1.0, 1.0, 0.0) for all regimes.

### Passive Compass Gait

Hybrid 4D biped with continuous swing phase (Lagrangian dynamics) and instantaneous impact map. State: [theta_ns, theta_s, dtheta_ns, dtheta_s] where ns/s denote non-stance/stance legs.

Nominal model parameters: mu = 2, beta = 1, l = 1 m (corresponds to M=5 kg, M_H=10 kg, L=1 m, A=0.5 m, B=0.5 m, G=9.81 m/s²).

| Period | phi (deg) | Expected Period |
|--------|-----------|-----------------|
| 1      | 4.00      | 1               |
| 2      | 4.75      | 2               |
| 4      | 5.00      | 4               |
| 8      | 5.02      | 8               |
| Chaotic| 5.20      | None            |

Initial conditions are post-impact states recorded from Goswami's return-map fixed points (see chyll_v2/systems/compass_gait_slope_configs.py).

## Design: Fixed Time Span

All five regimes of each system integrate for identical T and dt, ensuring uniform segment length across the cascade. This design enables direct comparison of trajectory complexity: periodic orbits produce shorter symbolic sequences, while chaotic regimes fill the available space.

- Roessler: T = 500 s, dt = 0.02 s → 25,001 samples
- Compass gait: T = 400 s, dt = 0.02 s → 20,001 samples (between impacts)

## Usage

### Generate Data

```bash
cd /Users/bdoprad/Work/Projects/hybrid-cycling-signatures/code/period_doubling

# Full run (slow)
python generate_all.py --system both

# Quick test (reduced time span)
python generate_all.py --system both --quick

# Roessler only
python generate_all.py --system roessler

# Compass gait only
python generate_all.py --system compass
```

### Plot Atlases

```bash
python plot_atlas.py

# Or specify custom paths:
python plot_atlas.py --data-root /path/to/data --fig-dir /path/to/figures
```

Produces:
- `figures/roessler_atlas.png` — 2×3 phase-plane grid (x-y projections)
- `figures/roessler_data_colored.png` — 1×5 time-colored 3D scatters
- `figures/compass_atlas_check.png` — 2×3 dual-leg phase planes

### Downstream Julia Analysis

Extract cycling signatures using the period_doubling lifts as input to the CHyLL pipeline:

```bash
cd /Users/bdoprad/Work/Projects/hybrid-cycling-signatures

# Roessler period-2 orbit
julia --project="code/chyll_v2/cycling_signature" \
  code/chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir code/period_doubling/data/roessler \
  --base roessler_period2 \
  --boxsize 0.3 --sb-radius 1 \
  --segment-lengths 20:10:300 --n-runs 200

# Compass period-4 orbit (hybrid case with learned latent embedding)
julia --project="code/chyll_v2/cycling_signature" \
  code/chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir code/period_doubling/data/compass_gait \
  --base compass_period4 \
  --boxsize 0.3 --sb-radius 1 \
  --segment-lengths 20:10:300 --n-runs 200
```

For the compass gait, the CHyLL latent embedding route (learned from the hybrid suspension network) is available under `code/chyll_v2/cycling_signature/`.

## Cycling-Signature Computation

Once time-series data (positions and tangents CSVs) are generated, compute cycling signatures via the Julia driver:

```bash
cd /Users/bdoprad/Work/Projects/hybrid-cycling-signatures

# Roessler all regimes
for regime in period1 period2 period4 period8 chaos; do
  julia --project="code/chyll_v2/cycling_signature" \
    code/chyll_v2/cycling_signature/julia/run_subsegments.jl \
    --data-dir code/period_doubling/data/roessler \
    --base roessler_${regime} \
    --boxsize 0.3 --sb-radius 1 \
    --segment-lengths 20:10:300 --n-runs 200
done

# Output: code/period_doubling/data/roessler/signatures/
#   subsegments_roessler_{regime}_rank_at_radius.csv
#   subsegments_roessler_{regime}_rank{k}_spaces_at_radius.csv (k=1,2,...)
#   subsegments_roessler_{regime}_s21_inclusion.csv
#   subsegments_roessler_{regime}_rank_heatmap_rank{k}.csv
#   subsegments_roessler_{regime}_metadata.txt (includes stride)
```

**Data conventions:**
- Segment lengths are in post-stride samples (multiply by `dt * stride` to get time span)
- `dt` is read from `roessler_{regime}.npz` meta_json field
- Stride is read from metadata.txt line `stride=N` (default 1 if absent)
- Radii and segment lengths are axes of rank heatmaps

**Multi-radius evaluation:**
When calling the Julia script with `--eval-radius r_value`, variants of CSV files are written with suffix `_r{value_with_dots_as_p}`:
- `subsegments_roessler_{regime}_rank_at_radius_r0p8.csv`
- `subsegments_roessler_{regime}_rank1_spaces_at_radius_r0p8.csv`
- `subsegments_roessler_{regime}_s21_inclusion_r0p8.csv`

(Heatmap files are always generated with full radius range and do not use suffixes.)

### Plotting Cycling Signatures

Generate matplotlib reproductions of paper Figures 10, 11, 13, 14, 15:

```bash
cd /Users/bdoprad/Work/Projects/hybrid-cycling-signatures/code/period_doubling

# Default: plot all regimes at default evaluation radius
python plot_signatures.py

# Specify custom data and figure directories
python plot_signatures.py --data-root /path/to/data --fig-dir /path/to/figures

# Plot multi-radius variant (e.g., evaluation at radius 0.8)
python plot_signatures.py --eval-radius-suffix r0p8
```

**Output figures:**
- `roessler_fig10_rank_stacked.png` — Rank distribution (stacked bar chart)
- `roessler_fig11_sig1_stacked.png` — Rank-1 cycling spaces (per-system stacked bars, V1/V2/... labels)
- `roessler_fig13_sig2_stacked.png` — Rank-2 cycling spaces (per-system stacked bars, W1/W2/... labels)
- `roessler_fig14_inclusion.png` — Inclusion graphs (bipartite: V's ↔ W's)
- `roessler_fig15_rank_heatmaps.png` — Rank heatmaps grid (5 regimes × 3 ranks, shared colorbar)

For multi-radius evaluation, output filenames include the suffix: `roessler_fig10_rank_stacked_r0p8.png`, etc.

## Data Format

All regimes export three file types per system to `data/roessler/` and `data/compass_gait/`:

### CSV Lifts (space-delimited, no header)

- `{base}_positions.csv` — (N, d) matrix of state samples, loadable by Julia `readdlm(path, ' ', Float64)` 
- `{base}_tangents.csv` — (N, d) matrix of unit tangent vectors

### NPZ Archives

- `{base}.npz` — compressed NumPy archive containing:
  - `t` — (N,) time array
  - `x` — (N, d) positions
  - `v` — (N, d) unit tangents
  - `meta_json` — JSON string with system parameters and metadata
  - `impact_times`, `jump_minus`, `jump_plus` (compass gait only)

### Reports

- `{base}_report.txt` — human-readable summary (sample count, dimensions, dt, parameters)

### Summary

- `data/summary.csv` — one row per regime with columns: system, regime, parameter (c or phi_deg), expected_period, detected_clusters, ok (bool), n_samples, dt, t_span, n_impacts

## Periodicity Detection

The package includes post-hoc numerical checks:

- **Roessler**: `detect_roessler_period()` clusters local maxima of x-coordinate to count oscillations.
- **Compass gait**: `detect_compass_period()` clusters post-impact states.

`generate_all.py` runs these checks and exits with code 1 if any periodic regime does not match its expected period (chaos is exempt). This validates that the fixed-time-span data actually captures the intended dynamical regime.

## Raw compass-gait signatures (negative control)

Cycling signatures were computed directly on the raw (discontinuous) compass-gait state data, per regime:

```bash
julia --project=julia julia/run_signatures.jl \
  --data-dir data/compass_gait \
  --base compass_{period1,period2,period4,period8,chaos} \
  --boxsize 0.3 --sb-radius 1 --stride 2 \
  --segment-lengths 10,20,...,300 \
  --n-runs 150 \
  --eval-radius 0.1,0.2,0.3 \
  --out-dir data/compass_gait/signatures
```

Observed `beta1_Y` (first Betti number of the ambient cubical complex at boxsize 0.3): period1 = 0, period2 = 1, period4 = 0, period8 = 0, chaos = 0.

Result: in period1, period4, period8, and chaos the ambient complex carries no 1-cycles at all, so every subsegment has cycling rank 0 at all three evaluation radii — vanilla cycling signatures on the raw hybrid state see no cycling, because the reset discontinuity prevents the arcs from closing. This is the negative-control baseline for the forthcoming suspension-embedding comparison, where the same data lifted to the suspension is expected to exhibit the period-doubling cascade. One caveat: in the period2 regime the boxsize-0.3 cover incidentally closes a single loop (`beta1_Y = 1`), and rank 1 then dominates for segment time spans above roughly 1 at all three radii (rank-1 fractions 3907/4500, 4001/4500, 4018/4500 at r = 0.1, 0.2, 0.3); this is a resolution artifact of the cubical cover closing a loop that the trajectory itself does not close, not a recovered hybrid cycle, and it is absent in the other four regimes. A box-size probe confirms the artifact reading: `beta1_Y` for period2 is 5, 3, 0, 1 at boxsize 0.15, 0.2, 0.25, 0.3 respectively (`signatures_pilot/probe_period2_*`) — non-monotone in the cover scale, unlike the Roessler comparison spaces, which are stable across box sizes. On raw hybrid data the vanilla method thus produces both a false negative (the genuine period-1 limit cycle is invisible because its arc never closes) and a scale-unstable false positive (period2).

## References

- Roessler, O. E. (1976). "An equation for continuous chaos." *Phys. Lett. A*, 57(5), 397–398.
- Goswami, A., Thuilot, B., & Espiau, B. (1998). "A study of the passive gait of a compass-like biped robot: Symmetry and chaos." *Int. J. Robot. Res.*, 7(2), 19–42.
- Conley, C. (1978). *Isolated invariant sets and the Morse index*. CBMS Regional Conference Series in Mathematics.

## License

Research code for hybrid-cycling-signatures project.
