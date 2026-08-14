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

## Latent suspension lifts (positive hybrid computation)

`export_latent_lifts.py` lifts the SAME compass-gait trajectories used for the
raw negative control into the continuous latent suspension space of the
per-slope trained CHyLL v2 models (`chyll_v2/runs/compass_gait_phi*`, 11-dim
latent). Flow samples enter as augmented states (x, s=0); at each recorded
impact a mapping-cylinder bridge (jump_minus, s), s in (0,1), n_s = 8 samples,
is inserted, so the lift is a single continuous polyline once the learned
gluing E(g,1) = E(r(g),0) holds. Every sample (arc or bridge) costs one dt of
suspension time. Tangents are the encoder pushforward (JVP) of the physical
unit tangents (arcs) and of e_s (bridges); a tag-aware finite-difference
tangent set is written alongside as a cross-check (mean |cos| agreement
>= 0.99; learned latent vector field agrees with the JVP tangents at
mean |cos| 0.97-0.99). period1 uses the phi=0.07 rad (4.0107 deg) model on
4.00 deg data (mismatch 2e-4 rad, noted in the report).

Model map: period1 -> phi007, period2 -> phi_1, period4 -> phi_2,
period8 -> phi_3, chaos -> phi_4_cloud.

Diagnostics per regime (`report_compass_{regime}.txt`): symbolic gluing seam
errors 0.072-0.128 (about a quarter to a third of the median latent arc step
0.29); reconstruction error concentrates on just-post-impact samples, the
expected quotient effect (decoder must choose a seam preimage); only the
encoder enters the signature computation.

Pilot (`julia/pilot_beta1_latent.jl`): beta_1(Y) = 1 for ALL five regimes,
stable across boxsizes 0.3-1.0 and strides 1-2 — the suspension closes the
loop the raw hybrid state cannot (raw: beta_1 = 0), with none of the raw
period2 scale instability.

Production runs (2026-08-13): boxsize 0.45, sb-radius 1, stride 1, segment
lengths 20:20:800, n-runs 150, eval radii 0.1/0.15/0.3/0.45; outputs in
`data/compass_gait_latent/signatures/`, figures
`figures/compass_latent_fig{10,11,13,14,15}_r*.png`.

Results:
- beta_1(Y) = 1 in every regime; a single 1-dim cycling space V1 everywhere
  (rank never exceeds 1, consistent with the suspension carrying one closed
  orbit per regime).
- The cascade discriminator is the minimal closing length: at eval radii in
  the resolving band r in [0.09, 0.14], rank-1 onset (>=135/150 runs) is
  60 / 100 / 160 / 220 samples for period1/2/4/8, i.e. about 1.3 / 2.2 /
  3.5 / 4.8 strides — the tau_min staircase. period8 is only partially
  resolved: its branch separation (~0.02 latent) lies below the reachable
  radius floor.
- Radius floor: below r ~ 0.09 periodic regimes stop closing entirely
  (sampling-phase drift: returns miss sampled points by up to half a latent
  sample step ~0.15). Chaos is qualitatively distinct — its rank-1 boundary
  is ragged and descends toward r -> 0 at long segment spans (recurrence),
  where periodic regimes are hard rank-0.
- Untrained-encoder control (`--untrained`, same architecture, seed 0, same
  bridge construction): seam gap 2.3-3.2 (the physical state jump, unglued),
  beta_1(Y) = 0, rank 0 at all lengths and radii
  (`data/compass_gait_latent_untrained/`). Together with the raw control this
  isolates the TRAINED gluing as the operative ingredient.

### Fine variant (dt = 0.005) — Roessler-comparable figures

The dt = 0.02 lift caps the analysis in the fully-merged regime: latent arc
steps ~0.29 force boxsize >= ~0.3, every doubled branch merges, beta_1(Y) = 1
everywhere, rank never exceeds 1, and figs 13/14 are empty — visibly poorer
than the Roessler set. The dt = 0.005 regeneration (`data_fine/`, same ICs
and t_span; arc steps ~0.072, max gap 0.14; uniform n_s = 26) lowers the
usable boxsize to 0.2 and lands in the partially-resolved regime the Roessler
study occupies.

Production (2026-08-13): boxsize 0.2, sb-radius 1, stride 2, lengths
40:40:1600, n-runs 150, eval radii 0.05/0.1/0.15/0.2; outputs in
`data_fine/compass_gait_latent/signatures/`, figures in `figures_fine/`.

Results (cf. Roessler at boxsize 1.5: beta_1 = 1/2/5/3/2):
- beta_1(Y) = 1 / 4 / 2 / 4 / 1 for period1/2/4/8/chaos, stable at strides
  1-2; 4/4/9 at boxsize 0.15.
- Rank takeovers mirror the Roessler figures: period1 saturates at rank 1
  (tau ~ 1); period2 passes rank 1 -> rank 2 -> rank 3 (tau ~ 2.5 and ~5-6 at
  r = 0.1); period4 and period8 saturate at rank 2 from tau ~ 2.5-3; the
  fig15 heatmaps show the nested rank-1/rank-2 staircase tiers with
  radius-dependent onsets.
- Multiple 1-dim cycling spaces (V1-V3) appear in the pre-saturation band
  and the V -> W inclusion graphs (fig14) are nontrivial for period2/4/8;
  period1 and chaos have a single V1 and no rank-2 segments.
- Chaos differs from Roessler chaos: its latent band does not fragment at
  boxsize >= 0.1 (beta_1 stays 1), so it keeps the rank <= 1
  recurrence-limited diffuse boundary instead of a rank-1/2 mixture.

## References

- Roessler, O. E. (1976). "An equation for continuous chaos." *Phys. Lett. A*, 57(5), 397–398.
- Goswami, A., Thuilot, B., & Espiau, B. (1998). "A study of the passive gait of a compass-like biped robot: Symmetry and chaos." *Int. J. Robot. Res.*, 7(2), 19–42.
- Conley, C. (1978). *Isolated invariant sets and the Morse index*. CBMS Regional Conference Series in Mathematics.

## License

Research code for hybrid-cycling-signatures project.
